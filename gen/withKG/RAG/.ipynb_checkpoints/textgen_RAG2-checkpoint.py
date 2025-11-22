#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextGen with KG H1/H2 path hints (adapter-only), evaluated RAW vs KG.

Key changes in this version:
- KG hints are *paths*, not ranked diagnoses:
  [KG CONTEXT — H1 NEIGHBORS] and [KG CONTEXT — H2 PATHS]
- H2 endpoints get tiny context-only ICD-9 anchors:
  [ICD-9: 410.90 — Acute MI]  (title from code2name.pkl or icd9_profiles.json)
- Output remains free-text diagnoses; mapping is done after with your ICDMapper.

Inputs you already have:
- Dataset pickle with structured signals (ndc=ATC, pro_code=ICD-9-Proc, lab_test=LOINC) + notes.
- UMLS KG CSVs (kg_nodes.csv, kg_edges.csv) with columns you shared.
- Evidence maps: ATC/LOINC/ICD-9-Proc → CUI (pkl).
- code2name.pkl for concise ICD-9 titles (diagnoses).
- Optional: icd9_profiles.json for fallback short titles (left side of "code :: title").

No external FAISS KG index is used; we mine H1/H2 directly from CSVs.
"""

import os, re, json, time, argparse, pickle, glob, sys, math
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
import torch.distributed as dist
import networkx as nx

# ====== shared utils expected in your repo ======
from common_textgen import (
    log, is_main_process, world_size, local_rank,
    serialize_structured_readable, serialize_notes,
    ICDMapper, to_list, format_icd9, is_valid_icd9,
    restrict_to, multihot, eval_pack, add_parent_macro_f1,
    get_icd9_parent
)

# ------------------------------- robust OUTPUT parsing -------------------------------
_OUTPUT_RE = re.compile(r"\[OUTPUT\]", flags=re.IGNORECASE)

def _coerce_text(x) -> str:
    if x is None: return ""
    if isinstance(x, (list, tuple)): return "\n".join(_coerce_text(y) for y in x if y is not None)
    if isinstance(x, bytes):
        try: return x.decode("utf-8", errors="ignore")
        except Exception: return str(x)
    return str(x)

def extract_after_output(generation, n_max: int = 12) -> List[str]:
    s = _coerce_text(generation).replace("</s>", "").replace("<s>", "").strip()
    m = _OUTPUT_RE.search(s)
    block = s[m.end():] if m else s
    out=[]
    for raw in block.splitlines():
        t = raw.strip()
        if not t: continue
        if t.startswith("[") and not re.match(r"^\[\s*OUTPUT\s*\]$", t, flags=re.I): break
        t = re.sub(r"^(?:[-*]\s*|\d+\.\s*|\(\d+\)\s*)", "", t)
        if t: out.append(t)
        if len(out) >= n_max: break
    return out

def _safe_extract_batch(texts, nmax, label):
    out=[]; misses=0
    for t in texts:
        s = _coerce_text(t)
        if not _OUTPUT_RE.search(s): misses += 1
        out.append(extract_after_output(s, nmax))
    if misses:
        log.info(f"[{label}] generations missing [OUTPUT] tag: {misses}/{len(texts)} (parsed best-effort)")
    return out

# ------------------------------- parent mapping (safe) -------------------------------
def get_icd9_parent_safe(code: str) -> str:
    try:
        return get_icd9_parent(code)
    except Exception:
        c = (code or "").upper()
        if not c: return c
        if c[0].isdigit(): return c.split('.')[0][:3]
        if c[0] == 'V':    return c.split('.')[0][:3]
        if c[0] == 'E':    return c.split('.')[0][:4]
        return c

def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
    g = [[get_icd9_parent_safe(c) for c in lst] for lst in gold_lists]
    p = [[get_icd9_parent_safe(c) for c in lst] for lst in pred_lists]
    labels = sorted({x for lst in g for x in lst})
    Yg = multihot(g, labels); Yp = multihot(p, labels)
    metrics_dict.update({
        "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
        "recall_macro_parent":    float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent":        float(f1_score(Yg, Yp, average="macro", zero_division=0)),
        "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
        "recall_micro_parent":    float(recall_score(Yg, Yp, average="micro", zero_division=0)),
        "f1_micro_parent":        float(f1_score(Yg, Yp, average="micro", zero_division=0)),
        "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
        "recall_samples_parent":    float(recall_score(Yg, Yp, average="samples", zero_division=0)),
        "f1_samples_parent":        float(f1_score(Yg, Yp, average="samples", zero_division=0)),
    })
    return labels, Yg, Yp

# ------------------------------- dist helpers -------------------------------
def maybe_init_dist():
    if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl", timeout=torch.distributed.elastic.utils.get_default_timeout())
    return dist.is_initialized()

def shard_indices(N:int, rank:int, W:int):
    return list(range(rank, N, W))

def barrier():
    if dist.is_available() and dist.is_initialized():
        try: dist.barrier()
        except Exception: pass

def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        try: dist.destroy_process_group()
        except Exception: pass

# ------------------------------- token budgeting -------------------------------
def count_tokens(tok, text: str) -> int:
    return int(tok(text, add_special_tokens=False, return_length=True)["length"][0])

def trim_to_token_budget(tok, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text: return ""
    if count_tokens(tok, text) <= max_tokens: return text
    lo, hi = 0, len(text); best = ""
    while lo <= hi:
        mid = (lo + hi)//2
        cand = text[:mid]
        if count_tokens(tok, cand) <= max_tokens:
            best = cand; lo = mid + 1
        else:
            hi = mid - 1
    return best

# ------------------------------- KG loading & evidence -------------------------------
_DX_SEMTYPE_HINTS = ("Disease", "Neoplastic", "Syndrome", "Injury", "Infect", "Disorder")
def looks_like_dx_semtype(sem: str) -> bool:
    s = (sem or "")
    return any(k.lower() in s.lower() for k in _DX_SEMTYPE_HINTS)

def _norm(s: str) -> str:
    return (s or "").strip().upper()

def _norm_keepdigits(s: str) -> str:
    return re.sub(r"[^0-9A-Z\.]", "", (s or "").upper())

def load_kg(nodes_csv: str, edges_csv: str):
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # Build node dict
    node_name: Dict[str,str] = {}
    node_sem: Dict[str,str] = {}
    # For anchoring: CUI -> list of (ICD9 code, title)
    cui_to_icd9: Dict[str, List[Tuple[str,str]]] = defaultdict(list)

    for _, r in nodes_df.iterrows():
        cui = str(r.get("cui","")).strip()
        nm  = str(r.get("name","")).strip()
        sem = str(r.get("semantic_type","") or "").strip()
        sab = str(r.get("sab","")).strip()
        code= str(r.get("code","") or "").strip()
        if cui and nm:
            # prefer first seen name; keep simple
            node_name.setdefault(cui, nm)
        if cui and sem:
            node_sem.setdefault(cui, sem)
        # collect ICD9CM codes for anchoring
        if sab == "ICD9CM" and code:
            # title: try this node's name as fallback; cleaned below later via code2name/profiles
            title = nm or ""
            cui_to_icd9[cui].append((code, title))

    # Build directed graph with attributes
    G = nx.DiGraph()
    for cui, nm in node_name.items():
        G.add_node(cui, name=nm, sem=node_sem.get(cui, ""))

    # Edge columns: ['cui_start','name_start','sab_start','codes_start','rel','rela','sab_relation','cui_target',...]
    for _, e in edges_df.iterrows():
        u = str(e.get("cui_start","")).strip()
        v = str(e.get("cui_target","")).strip()
        if not u or not v: continue
        rel = str(e.get("rel","") or "").strip()
        rela= str(e.get("rela","") or "").strip()
        # add nodes if missing (defensive)
        if u not in G: G.add_node(u, name=str(e.get("name_start","") or ""), sem="")
        if v not in G: G.add_node(v, name=str(e.get("name_target","") or ""), sem="")
        if not G.has_edge(u,v):
            G.add_edge(u, v, rel=rel, rela=rela)

    return G, node_name, node_sem, cui_to_icd9

def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    """
    Map structured signals to CUIs.
    Returns (src2cuis dict, union set).
    """
    src2cuis = {}
    ev = set()

    # ATC via 'ndc'
    for c in to_list(row.get("ndc", [])):
        key = _norm_keepdigits(c)
        cuis = atc_map.get(key, [])
        if cuis:
            src2cuis[f"ATC:{key}"] = list(dict.fromkeys(cuis))
            ev.update(cuis)

    # LOINC via 'lab_test'
    for c in to_list(row.get("lab_test", [])):
        key = _norm_keepdigits(c)
        cuis = loinc_map.get(key, [])
        if cuis:
            src2cuis[f"LOINC:{key}"] = list(dict.fromkeys(cuis))
            ev.update(cuis)

    # ICD-9 PROC via 'pro_code' (normalize like 54.91)
    def _norm_proc(x: str) -> str:
        s = re.sub(r"[^0-9]", "", str(x))
        if not s: return ""
        return (s[:2] + "." + s[2:]) if len(s) >= 3 else s

    for c in to_list(row.get("pro_code", [])):
        cc = _norm_proc(c)
        if not cc: continue
        cuis = icd9_proc_map.get(cc, [])
        if cuis:
            src2cuis[f"PROC:{cc}"] = list(dict.fromkeys(cuis))
            ev.update(cuis)

    return src2cuis, set(ev)

# ------------------------------- H1/H2 path mining -------------------------------
def mine_paths_h1_h2(G: nx.DiGraph,
                     node_name: Dict[str,str],
                     node_sem: Dict[str,str],
                     cui_to_icd9: Dict[str, List[Tuple[str,str]]],
                     ev_cuis: Set[str],
                     notes_text: str,
                     rel_whitelist: Optional[Set[str]] = None,
                     rela_whitelist: Optional[Set[str]] = None,
                     h1_per_src: int = 3,
                     h2_per_h1: int = 3,
                     prefer_dx_like: bool = True) -> Dict[str, List[dict]]:
    """
    Return structure:
    {
      "H1": [ {u, v, rel, rela, name_u, name_v}, ... ],
      "H2": [ {u, v, w, rel1, rela1, rel2, rela2, name_u, name_v, name_w, anchors:[(code,title),...]}, ... ]
    }
    """
    notes_lower = (notes_text or "").lower()
    def ok_edge_rel(rel, rela):
        if rel_whitelist and (rel or "").strip() not in rel_whitelist: return False
        if rela_whitelist and (rela or "").strip() not in rela_whitelist: return False
        return True

    def score_node(n_cui: str, n_name: str) -> float:
        # simple overlap with notes gives a tiny boost
        score = 0.0
        if n_name and n_name.strip():
            key = n_name.strip().lower()
            if key and key in notes_lower:
                score += 1.0
        if prefer_dx_like and looks_like_dx_semtype(node_sem.get(n_cui, "")):
            score += 0.5
        return score

    out = {"H1": [], "H2": []}
    seen_h1 = set()
    seen_h2 = set()

    for u in ev_cuis:
        if u not in G: continue
        # H1: u -> v
        cand_h1=[]
        for v in G.successors(u):
            d = G[u][v]
            rel  = (d.get("rel")  or "").strip()
            rela = (d.get("rela") or "").strip()
            if not ok_edge_rel(rel, rela): continue
            nm_u = node_name.get(u, "Unknown")
            nm_v = node_name.get(v, "Unknown")
            s = score_node(v, nm_v)
            cand_h1.append((s, u, v, rel, rela, nm_u, nm_v))
        cand_h1.sort(key=lambda x: x[0], reverse=True)
        for _, u_, v_, rel_, rela_, nmu_, nmv_ in cand_h1[:h1_per_src]:
            key = (u_, v_)
            if key in seen_h1: continue
            seen_h1.add(key)
            out["H1"].append({
                "u": u_, "v": v_, "rel": rel_, "rela": rela_,
                "name_u": nmu_, "name_v": nmv_
            })

            # H2: v -> w
            cand_h2=[]
            if v_ in G:
                for w in G.successors(v_):
                    d2 = G[v_][w]
                    rel2  = (d2.get("rel")  or "").strip()
                    rela2 = (d2.get("rela") or "").strip()
                    if not ok_edge_rel(rel2, rela2): continue
                    nmw = node_name.get(w, "Unknown")
                    s2 = score_node(w, nmw)
                    # small extra if endpoint has ICD9 anchors
                    if cui_to_icd9.get(w): s2 += 0.75
                    cand_h2.append((s2, u_, v_, w, rel_, rela_, rel2, rela2))
            cand_h2.sort(key=lambda x: x[0], reverse=True)
            for _, u2, v2, w2, r1, ra1, r2, ra2 in cand_h2[:h2_per_h1]:
                key2 = (u2, v2, w2)
                if key2 in seen_h2: continue
                seen_h2.add(key2)
                out["H2"].append({
                    "u": u2, "v": v2, "w": w2,
                    "rel1": r1, "rela1": ra1,
                    "rel2": r2, "rela2": ra2,
                    "name_u": node_name.get(u2, "Unknown"),
                    "name_v": node_name.get(v2, "Unknown"),
                    "name_w": node_name.get(w2, "Unknown"),
                })
    # attach anchors lazily (so we can title them later)
    for item in out["H2"]:
        w = item["w"]
        item["anchors"] = cui_to_icd9.get(w, [])
    return out

# ------------------------------- ICD-9 anchor title resolution -------------------------------
def load_profiles_json(path: str) -> Dict[str,str]:
    if not path: return {}
    try:
        with open(path, "r") as f:
            J = json.load(f)
        if isinstance(J, dict): return J
        return {}
    except Exception:
        return {}

def profile_short_title(code: str, profiles: Dict[str,str]) -> Optional[str]:
    """Return the text after 'code :: ' from profiles, if present."""
    if not profiles: return None
    k = code.strip()
    if k not in profiles: return None
    raw = profiles[k]
    # Expect pattern "CODE :: Title ...". Split once on '::'
    parts = [p.strip() for p in str(raw).split("::", 1)]
    if len(parts) == 2:
        # return just the right side's first clause (before ' ; ' if present)
        right = parts[1]
        return right.split(" ; ")[0].strip() or None
    # If value starts with "CODE ::" but key mismatch, still try splitting
    if "::" in str(raw):
        right = str(raw).split("::",1)[1].strip()
        return right.split(" ; ")[0].strip() or None
    return None

def make_anchor_label(code: str, code2name: Dict[str,str], profiles: Dict[str,str]) -> str:
    c = format_icd9(code)
    title = None
    # prefer your curated code2name title (usually clean and short)
    if c in code2name:
        title = str(code2name[c]).strip()
    # fallback to profiles short title (left-side extraction)
    if not title:
        title = profile_short_title(c, profiles)
    # final fallback: show just code
    return f"{c} — {title}" if title else c

# ------------------------------- Render KG CONTEXT -------------------------------
def render_kg_context_paths(tok,
                            mined: Dict[str, List[dict]],
                            code2name: Dict[str,str],
                            profiles: Dict[str,str],
                            budget_tokens: int,
                            h2_first_ratio: float = 0.7) -> str:
    if budget_tokens <= 0: return ""
    H1 = mined.get("H1", [])
    H2 = mined.get("H2", [])

    # Prepare H2 lines with ICD-9 anchors (compact)
    h2_lines=[]
    for it in H2:
        w_anchors = []
        for (code, _title_from_node) in it.get("anchors", [])[:3]:
            lab = make_anchor_label(code, code2name, profiles)
            w_anchors.append(lab)
        anchor_txt = ""
        if w_anchors:
            # keep to ≤ 2 anchors for compactness
            anchor_txt = " [ICD-9: " + " | ".join(w_anchors[:2]) + "]"
        line = f"- {it['name_u']} → {it['name_v']} → {it['name_w']}{anchor_txt}"
        h2_lines.append(line)

    # H1 lines
    h1_lines=[]
    for it in H1:
        reltxt = it['rela'] if it.get('rela') else it.get('rel','')
        reltxt = reltxt or ""
        if reltxt: reltxt = f" --{reltxt}→ "
        else:     reltxt = " → "
        line = f"- {it['name_u']}{reltxt}{it['name_v']}"
        h1_lines.append(line)

    # Assemble under budget: spend h2_first_ratio on H2
    target_h2 = int(budget_tokens * h2_first_ratio)
    target_h1 = budget_tokens - target_h2

    parts=[]
    # H2 first
    if h2_lines:
        head = "[KG CONTEXT — H2 PATHS]"
        body=""
        for ln in h2_lines:
            trial = head + ("\n" + body + ("\n" if body else "") + ln)
            if count_tokens(tok, trial) <= target_h2:
                body = (body + ("\n" if body else "") + ln)
            else:
                break
        if body:
            parts.append(head + "\n" + body)

    # Then H1
    if h1_lines and target_h1 > 0:
        head = "[KG CONTEXT — H1 NEIGHBORS]"
        body=""
        for ln in h1_lines:
            trial = "\n\n".join(parts + [head + ("\n" + body + ("\n" if body else "") + ln)])
            extra = count_tokens(tok, trial) - count_tokens(tok, "\n\n".join(parts))
            if extra <= target_h1:
                body = (body + ("\n" if body else "") + ln)
            else:
                break
        if body:
            parts.append(head + "\n" + body)

    if not parts:
        return ""
    text = "\n\n".join(parts)
    # last-resort clamp if we overshot
    if count_tokens(tok, text) > budget_tokens:
        text = trim_to_token_budget(tok, text, budget_tokens)
    return text

# ------------------------------- Prompt builders -------------------------------
def build_tail(N_max_terms:int) -> str:
    lines = [
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]"
    ]
    return "\n".join(lines)

def build_prompts_for_row(row: pd.Series,
                          tok,
                          kg_text: str,
                          notes_soft_budget: int,
                          N_max_terms:int) -> Tuple[str, str, dict]:
    header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
    full_notes  = serialize_notes(row)
    notes  = trim_to_token_budget(tok, full_notes, notes_soft_budget) if notes_soft_budget > 0 else ""
    tail   = build_tail(N_max_terms)

    raw_prompt = "\n".join([x for x in [header, notes, tail] if x])
    kg_prompt  = "\n".join([x for x in [header, notes, kg_text, tail] if x])

    dbg = {
        "header_tokens": count_tokens(tok, header),
        "notes_tokens":  count_tokens(tok, notes),
        "kg_tokens":     count_tokens(tok, kg_text) if kg_text else 0,
        "tail_tokens":   count_tokens(tok, tail),
        "total_raw":     count_tokens(tok, raw_prompt),
        "total_kg":      count_tokens(tok, kg_prompt),
    }
    return raw_prompt, kg_prompt, dbg

# ------------------------------- Generation -------------------------------
def build_generate_kwargs(decoding: str,
                          max_new: int,
                          eos_id: int,
                          pad_id: int,
                          num_beams: int = 2,
                          temperature: float = 1.0,
                          top_p: float = 0.95,
                          top_k: int = 50,
                          no_repeat_ngram: int = 0):
    if decoding == "greedy":
        return dict(max_new_tokens=max_new, do_sample=False, eos_token_id=eos_id, pad_token_id=pad_id,
                    no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))
    if decoding == "beam":
        return dict(max_new_tokens=max_new, num_beams=num_beams, do_sample=False, eos_token_id=eos_id, pad_token_id=pad_id,
                    no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))
    return dict(max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                eos_token_id=eos_id, pad_token_id=pad_id, no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))

@torch.no_grad()
def generate_texts(model, tok, prompts: List[str], max_len: int, gen_kwargs: dict, batch_size: int = 8, device=None) -> List[str]:
    device = device or (model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outs = []
    bs = max(1, int(batch_size))
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False)
        for k in enc: enc[k] = enc[k].to(device)
        gen = model.generate(**enc, **{k:v for k,v in gen_kwargs.items() if v is not None})
        dec = tok.batch_decode(gen, skip_special_tokens=False)
        outs.extend(dec)
    return outs

# ------------------------------- N_max_terms dynamics -------------------------------
def compute_terms_caps(gold_lists: List[List[str]]) -> Tuple[int, int, int]:
    sizes = sorted(len(x) for x in gold_lists)
    if not sizes: return 12, 10, 18
    def pct(p):
        k = max(0, min(len(sizes)-1, int(round((p/100.0)* (len(sizes)-1)))))
        return sizes[k]
    p50 = pct(50); p90 = pct(90)
    base_N = int(max(6, min(24, round(p50 + 2))))
    return base_N, p50, p90

def row_cap_from_candidates(base_N: int, p90: int, allowed_count: int) -> int:
    bonus = 0
    if   allowed_count >= 30: bonus = 6
    elif allowed_count >= 20: bonus = 4
    elif allowed_count >= 10: bonus = 2
    return int(max(6, min(p90, base_N + bonus)))

# ------------------------------- Pretty printer -------------------------------
def _pretty_print_block(title: str, d: dict):
    log.info(f"--- {title} ---")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            log.info(f"{k:>28s}: {v:.6f}")
        else:
            log.info(f"{k:>28s}: {v}")

# ------------------------------- MAIN -------------------------------
def main():
    ap = argparse.ArgumentParser()

    # ========= data =========
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--subset_n", type=int, default=0)
    ap.add_argument("--print_samples", type=int, default=5)

    # ========= prompt budgets =========
    ap.add_argument("--max_len", type=int, default=4096, help="Tokenizer max length for inputs")
    ap.add_argument("--kg_hint_budget", type=int, default=600, help="Token budget for [KG CONTEXT] block")
    ap.add_argument("--N_max_terms", type=int, default=12, help="Max lines to parse after [OUTPUT]")

    # ========= decoding =========
    ap.add_argument("--gen_max_new", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--decoding", choices=["greedy","beam","sample"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_repeat_ngram", type=int, default=0)

    # ========= model/adapter =========
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--use_bf16", action="store_true")

    # ========= ICD Mapper (SapBERT) =========
    ap.add_argument("--icd_index_dir", required=True)
    ap.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--faiss_rows", type=int, default=50)
    ap.add_argument("--tau_cos", type=float, default=0.40)
    ap.add_argument("--tau_final", type=float, default=0.60)
    ap.add_argument("--w_cos", type=float, default=0.6)
    ap.add_argument("--w_fuz", type=float, default=0.4)

    # ========= eval label space =========
    ap.add_argument("--labels_space", choices=["full","head"], default="full")
    ap.add_argument("--labels_head_k", type=int, default=0)

    # ========= optional lists =========
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")

    # ========= KG inputs =========
    ap.add_argument("--kg_nodes_csv", required=True)
    ap.add_argument("--kg_edges_csv", required=True)

    # Evidence maps
    ap.add_argument("--icd9_proc_map_pkl", required=True, help="ICD-9-Proc -> CUIs")
    ap.add_argument("--loinc_map_pkl",     required=True, help="LOINC -> CUIs")
    ap.add_argument("--atc_map_pkl",       required=True, help="ATC (from ndc) -> CUIs")

    # Anchors/titles
    ap.add_argument("--code2name_pkl",     required=True, help="ICD-9 code -> short name")
    ap.add_argument("--icd9_profiles_json", default="", help="Optional profiles for fallback titles")

    # H1/H2 mining knobs
    ap.add_argument("--rel_whitelist",  default="", help="Comma-separated rel whitelist; leave empty for all")
    ap.add_argument("--rela_whitelist", default="", help="Comma-separated rela whitelist; leave empty for all")
    ap.add_argument("--h1_per_src", type=int, default=3)
    ap.add_argument("--h2_per_h1", type=int, default=3)

    # ========= distributed & outputs =========
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--tmp_dir", default="runs_textgen/test_shards")
    ap.add_argument("--out_metrics", default="runs_textgen/test_metrics_kg_rag.json")

    args = ap.parse_args()

    # ---------------- data ----------------
    try:
        df = pd.read_pickle(args.data_pickle)
    except Exception:
        with open(args.data_pickle, "rb") as f: df = pickle.load(f)

    if args.test_only:
        test_df = df.copy()
    else:
        from sklearn.model_selection import train_test_split
        subs = df[args.subject_col].dropna().unique()
        _, te_subs = train_test_split(subs, test_size=0.10, random_state=args.seed)
        test_df  = df[df[args.subject_col].isin(te_subs)].copy()

    if args.subset_n and args.subset_n > 0:
        test_df = test_df.iloc[:args.subset_n].reset_index(drop=True)

    # gold
    def extract_codes(df, label_col):
        out=[]
        for _, r in df.iterrows():
            lst = to_list(r.get(label_col, []))
            lst = [format_icd9(c) for c in lst if c]
            lst = [c for c in lst if is_valid_icd9(c)]
            out.append(lst)
        return out
    gold_codes = extract_codes(test_df, args.label_col)

    labels_full = sorted({c for lst in gold_codes for c in lst})
    labels_eval = labels_full
    head_name = None
    if args.labels_space == "head" and args.labels_head_k > 0:
        from collections import Counter
        cnt = Counter([c for lst in gold_codes for c in lst])
        labels_eval = [c for c,_ in cnt.most_common(args.labels_head_k)]
        head_name = f"HEAD_{args.labels_head_k}"

    if is_main_process():
        log.info(f"Test size: {len(test_df)}")
        log.info(f"Eval label space: {len(labels_eval)} codes ({'FULL' if head_name is None else head_name})")

    # dataset prior & caps
    def build_code_prior(gold_lists): return Counter([c for lst in gold_lists for c in lst])
    code_prior = build_code_prior(gold_codes)
    base_N, p50_codes, p90_codes = compute_terms_caps(gold_codes)
    if is_main_process():
        log.info(f"N_max_terms (dataset): base={base_N}, p50={p50_codes}, p90={p90_codes}")

    # model (adapter)
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) \
           else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.config.use_cache = True
    dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    # KG load
    G, node_name, node_sem, cui_to_icd9 = load_kg(args.kg_nodes_csv, args.kg_edges_csv)

    # maps
    icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
    loinc_map     = pickle.load(open(args.loinc_map_pkl,     "rb"))
    atc_map       = pickle.load(open(args.atc_map_pkl,       "rb"))

    # code titles
    code2name = pickle.load(open(args.code2name_pkl, "rb"))
    profiles  = load_profiles_json(args.icd9_profiles_json)

    rel_w  = {s.strip() for s in args.rel_whitelist.split(",")  if s.strip()} or None
    rela_w = {s.strip() for s in args.rela_whitelist.split(",") if s.strip()} or None

    # ======== Build prompts (RAW & KG paths) ========
    raw_prompts=[]; kg_prompts=[]; dbg_rows=[]
    for i, row in test_df.iterrows():
        # evidence CUIs
        src2cuis, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
        notes_text = serialize_notes(row)

        # mine H1/H2
        mined = mine_paths_h1_h2(
            G, node_name, node_sem, cui_to_icd9, ev_cuis, notes_text,
            rel_whitelist=rel_w, rela_whitelist=rela_w,
            h1_per_src=args.h1_per_src, h2_per_h1=args.h2_per_h1, prefer_dx_like=True
        )

        # render KG CONTEXT (token-budgeted)
        kg_text = render_kg_context_paths(
            tok, mined, code2name, profiles,
            budget_tokens=args.kg_hint_budget,
            h2_first_ratio=0.7
        )

        # dynamic N cap (use number of distinct anchored codes as a proxy)
        anchored_codes = set()
        for it in mined.get("H2", []):
            for c,_ in it.get("anchors", [])[:2]:
                if is_valid_icd9(format_icd9(c)):
                    anchored_codes.add(format_icd9(c))
        allowed_count = len(anchored_codes)
        dynamic_base = base_N if args.N_max_terms <= 0 else args.N_max_terms
        N_this = row_cap_from_candidates(dynamic_base, p90_codes, allowed_count=allowed_count)

        # prompts
        header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
        notes_trim = trim_to_token_budget(tok, serialize_notes(row), max(0, args.max_len - args.kg_hint_budget - 512))
        tail = build_tail(N_this)
        raw_p = "\n".join([x for x in [header, notes_trim, tail] if x])
        kg_p  = "\n".join([x for x in [header, notes_trim, kg_text, tail] if x])

        # final clamp to max_len
        def clamp(s: str) -> str:
            if count_tokens(tok, s) <= args.max_len: return s
            return trim_to_token_budget(tok, s, args.max_len - 8)

        raw_p = clamp(raw_p)
        kg_p  = clamp(kg_p)

        raw_prompts.append(raw_p)
        kg_prompts.append(kg_p)

        dbg_rows.append({
            "idx": i,
            "hadm_id": row.get("hadm_id",""),
            "ev_cuis": len(ev_cuis),
            "N_cap": N_this,
            "raw_tokens": count_tokens(tok, raw_p),
            "kg_tokens":  count_tokens(tok, kg_p),
            "kg_block_tokens": count_tokens(tok, kg_text) if kg_text else 0,
            "anchored_codes": len(anchored_codes)
        })

    # decoding kwargs
    gen_kwargs = build_generate_kwargs(
        decoding=args.decoding, max_new=args.gen_max_new,
        eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
        num_beams=args.num_beams, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, no_repeat_ngram=args.no_repeat_ngram
    )

    # sharding
    if args.distributed:
        maybe_init_dist()
        rank = int(os.environ.get("RANK", "0"))
        W = world_size()
        idxs = shard_indices(len(raw_prompts), rank, W)
    else:
        rank, W = 0, 1
        idxs = list(range(len(raw_prompts)))

    shard_raw = [raw_prompts[i] for i in idxs]
    shard_kg  = [kg_prompts[i]  for i in idxs]
    shard_gold= [gold_codes[i]  for i in idxs]

    # generation
    t0 = time.time()
    raw_out_texts = generate_texts(model, tok, shard_raw, max_len=args.max_len, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)
    kg_out_texts  = generate_texts(model, tok, shard_kg,  max_len=args.max_len, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)

    ncap_global = max(d["N_cap"] for d in dbg_rows) if dbg_rows else args.N_max_terms
    raw_terms = _safe_extract_batch(raw_out_texts,  ncap_global, "RAW")
    kg_terms  = _safe_extract_batch(kg_out_texts,   ncap_global, "KG")

    if is_main_process():
        per = (time.time()-t0)/max(1,len(idxs))
        log.info(f"Generation done ({per:.2f}s/sample on rank {rank}).")

    # mapping terms -> ICD-9
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    mapped_raw = mapper.map_terms(raw_terms)
    mapped_kg  = mapper.map_terms(kg_terms)

    # persist shard
    os.makedirs(args.tmp_dir, exist_ok=True)
    shard_path = os.path.join(args.tmp_dir, f"shard_{rank:03d}_of_{W:03d}.pkl")
    with open(shard_path, "wb") as f:
        pickle.dump({
            "idxs": idxs,
            "raw_texts": raw_out_texts,
            "kg_texts":  kg_out_texts,
            "raw_terms": raw_terms,
            "kg_terms":  kg_terms,
            "mapped_raw": mapped_raw,
            "mapped_kg":  mapped_kg,
            "gold": shard_gold,
            "dbg": dbg_rows,
        }, f)
    log.info(f"[Rank {rank}] wrote shard to {shard_path}")
    barrier()

    # merge & evaluate on rank 0
    if rank == 0:
        shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
        all_idx, all_rawt, all_kgt, all_rawc, all_kgc, all_gold = [], [], [], [], [], []
        for sp in shards:
            with open(sp, "rb") as f:
                D = pickle.load(f)
            all_idx.extend(D["idxs"])
            all_rawt.extend(D["raw_terms"]); all_kgt.extend(D["kg_terms"])
            all_rawc.extend(D["mapped_raw"]); all_kgc.extend(D["mapped_kg"])
            all_gold.extend(D["gold"])

        order = np.argsort(np.array(all_idx))
        raw_terms_all = [all_rawt[i] for i in order]
        kg_terms_all  = [all_kgt[i]  for i in order]
        pred_raw_all  = [all_rawc[i] for i in order]
        pred_kg_all   = [all_kgc[i]  for i in order]
        gold_all      = [all_gold[i] for i in order]

        # restrict to evaluation space
        gold_eval = restrict_to(gold_all, labels_eval)
        raw_eval  = restrict_to(pred_raw_all, labels_eval)
        kg_eval   = restrict_to(pred_kg_all,  labels_eval)

        # metrics
        Yt = multihot(gold_eval, labels_eval)
        Yr = multihot(raw_eval,  labels_eval)
        Yk = multihot(kg_eval,   labels_eval)

        m_raw = eval_pack(Yt, Yr)
        pm_raw = {}
        _ = add_parent_metrics_full(pm_raw, gold_eval, raw_eval)
        add_parent_macro_f1(pm_raw, gold_eval, raw_eval)
        m_raw.update(pm_raw)

        m_kg = eval_pack(Yt, Yk)
        pm_kg = {}
        _ = add_parent_metrics_full(pm_kg, gold_eval, kg_eval)
        add_parent_macro_f1(pm_kg, gold_eval, kg_eval)
        m_kg.update(pm_kg)

        # per-label CSV
        out_dir = os.path.dirname(os.path.abspath(args.out_metrics))
        os.makedirs(out_dir, exist_ok=True)
        def per_label_table(y_true, y_pred, labels, out_csv_path=None):
            p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            df = pd.DataFrame({"code": labels,"precision": p,"recall": r,"f1": f1,"support": support})
            if out_csv_path: df.to_csv(out_csv_path, index=False)
            return df
        per_label_table(Yt, Yr, labels_eval, os.path.join(out_dir, "per_label_FULL_RAW.csv"))
        per_label_table(Yt, Yk, labels_eval, os.path.join(out_dir, "per_label_FULL_KG.csv"))

        # optional TOP/BOTTOM/PARENTS CSVs
        def _read_first_col_codes(path) -> list:
            if not path: return []
            try:
                df = pd.read_csv(path)
                if df.shape[1] == 0: return []
                col = df.columns[0]
                vals = [format_icd9(x) for x in df[col].tolist()]
                return sorted(set([v for v in vals if is_valid_icd9(v)]))
            except Exception:
                return []

        def _read_first_col_parents(path) -> list:
            if not path: return []
            try:
                df = pd.read_csv(path)
                if df.shape[1] == 0: return []
                col = df.columns[0]
                raw = [format_icd9(x) for x in df[col].tolist()]
                return sorted(set([get_icd9_parent_safe(x) for x in raw if x]))
            except Exception:
                return []

        results_ext = {}

        top_codes = _read_first_col_codes(args.top_codes_csv)
        bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
        top_parents = _read_first_col_parents(args.top_parent_csv)

        if top_codes:
            g = restrict_to(gold_all, top_codes)
            r = restrict_to(pred_raw_all, top_codes)
            k = restrict_to(pred_kg_all,  top_codes)
            Yg = multihot(g, top_codes)
            Yr2= multihot(r, top_codes)
            Yk2= multihot(k, top_codes)
            results_ext["TOP_50_CODES_RAW"] = eval_pack(Yg, Yr2)
            results_ext["TOP_50_CODES_KG"]  = eval_pack(Yg, Yk2)
            per_label_table(Yg, Yr2, top_codes, os.path.join(out_dir, "per_label_TOP_50_CODES_RAW.csv"))
            per_label_table(Yg, Yk2, top_codes, os.path.join(out_dir, "per_label_TOP_50_CODES_KG.csv"))

        if bottom_codes:
            g = restrict_to(gold_all, bottom_codes)
            r = restrict_to(pred_raw_all, bottom_codes)
            k = restrict_to(pred_kg_all,  bottom_codes)
            Yg = multihot(g, bottom_codes)
            Yr2= multihot(r, bottom_codes)
            Yk2= multihot(k, bottom_codes)
            results_ext["BOTTOM_50_CODES_RAW"] = eval_pack(Yg, Yr2)
            results_ext["BOTTOM_50_CODES_KG"]  = eval_pack(Yg, Yk2)
            per_label_table(Yg, Yr2, bottom_codes, os.path.join(out_dir, "per_label_BOTTOM_50_CODES_RAW.csv"))
            per_label_table(Yg, Yk2, bottom_codes, os.path.join(out_dir, "per_label_BOTTOM_50_CODES_KG.csv"))

        if top_parents:
            g_par = [[get_icd9_parent_safe(c) for c in lst] for lst in gold_all]
            r_par = [[get_icd9_parent_safe(c) for c in lst] for lst in pred_raw_all]
            k_par = [[get_icd9_parent_safe(c) for c in lst] for lst in pred_kg_all]
            g_par = restrict_to(g_par, top_parents)
            r_par = restrict_to(r_par, top_parents)
            k_par = restrict_to(k_par, top_parents)
            YgP = multihot(g_par, top_parents)
            YrP = multihot(r_par, top_parents)
            YkP = multihot(k_par, top_parents)
            results_ext["TOP_50_PARENTS_RAW"] = eval_pack(YgP, YrP)
            results_ext["TOP_50_PARENTS_KG"]  = eval_pack(YgP, YkP)
            per_label_table(YgP, YrP, top_parents, os.path.join(out_dir, "per_label_TOP_50_PARENTS_RAW.csv"))
            per_label_table(YgP, YkP, top_parents, os.path.join(out_dir, "per_label_TOP_50_PARENTS_KG.csv"))

        # sample prints
        n_show = min(args.print_samples, len(raw_terms_all))
        log.info("=== Sample predictions (RAW vs KG) ===")
        for i in range(n_show):
            Gs = set(gold_all[i]); R = set(pred_raw_all[i]); K = set(pred_kg_all[i])
            # sample PRF (RAW)
            tp = len(Gs & R); fp = len(R - Gs); fn = len(Gs - R)
            pr = tp/(tp+fp) if tp+fp>0 else 0.0
            rr = tp/(tp+fn) if tp+fn>0 else 0.0
            fr = (2*pr*rr)/(pr+rr) if pr+rr>0 else 0.0
            # sample PRF (KG)
            tp2 = len(Gs & K); fp2 = len(K - Gs); fn2 = len(Gs - K)
            pr2 = tp2/(tp2+fp2) if tp2+fp2>0 else 0.0
            rr2 = tp2/(tp2+fn2) if tp2+fn2>0 else 0.0
            fr2 = (2*pr2*rr2)/(pr2+rr2) if pr2+rr2>0 else 0.0

            log.info(f"[Sample {i+1}] hadm={test_df.iloc[i].get('hadm_id','')}")
            log.info(f"  GOLD codes: {', '.join(sorted(Gs)) if Gs else '(none)'}")
            log.info(  "  RAW free-text terms:")
            for t in raw_terms_all[i][:ncap_global]:
                log.info(f"    - {t}")
            log.info(f"  RAW mapped ICD-9: {', '.join(sorted(R)) if R else '(none)'}  | P/R/F1 = {pr:.3f}/{rr:.3f}/{fr:.3f}")
            log.info(  "  KG  free-text terms:")
            for t in kg_terms_all[i][:ncap_global]:
                log.info(f"    - {t}")
            log.info(f"  KG  mapped ICD-9: {', '.join(sorted(K)) if K else '(none)'}  | P/R/F1 = {pr2:.3f}/{rr2:.3f}/{fr2:.3f}")

        payload = {
            "label_space": ("FULL" if head_name is None else head_name),
            "num_samples": len(raw_terms_all),
            "metrics_raw": m_raw,
            "metrics_kg":  m_kg,
            **results_ext
        }
        with open(args.out_metrics, "w") as f:
            json.dump(payload, f, indent=2)
        log.info(f"Metrics saved to {args.out_metrics}")

        _pretty_print_block("OVERALL RAW (code-level)", m_raw)
        _pretty_print_block("OVERALL KG  (code-level)", m_kg)
        for k,v in results_ext.items():
            _pretty_print_block(k, v)

    cleanup_dist()
    return 0

if __name__ == "__main__":
    sys.exit(main())
