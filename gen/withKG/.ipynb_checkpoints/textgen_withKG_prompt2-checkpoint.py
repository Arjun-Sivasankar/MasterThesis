#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextGen with *scored H1/H2 KG hints* in the prompt (adapter-only), evaluated RAW vs KG.

Changes vs your previous textgen:
- Uses the scored miner (rela_canon + rela_score) with degree caps K1/K2.
- Applies global thresholds TAU_H1 / TAU_H2 for selection.
- Renders "[KG CONTEXT — H2 PATHS]" then "[KG CONTEXT — H1 PATHS]" blocks.
- Token budgeting:
    * notes_soft_budget and assistant_reserve are honored.
    * If total_input_budget == 0 -> no total clamp (diagnostic mode).
    * If kg_soft_budget == 0 -> KG block is *not* trimmed (diagnostic mode).
- Per-row CSV log with H1/H2/total KG tokens and prompt sizes for downstream analysis.
"""

import os, re, json, time, argparse, pickle, glob, sys, math
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

# ====== your shared utils ======
from common_textgen import (
    log, is_main_process, world_size, local_rank,
    serialize_structured_readable, serialize_notes,
    ICDMapper, to_list, format_icd9, is_valid_icd9,
    restrict_to, multihot, eval_pack, add_parent_macro_f1,
    get_icd9_parent
)

# ------------------------------- robust output parsing -------------------------------
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
    lines_out = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line: continue
        if line.startswith("[") and not re.match(r"^\[\s*OUTPUT\s*\]$", line, flags=re.I): break
        line = re.sub(r"^(?:[-*]\s*|\d+\.\s*|\(\d+\)\s*)", "", line)
        if line: lines_out.append(line)
        if len(lines_out) >= n_max: break
    return lines_out

def _safe_extract_batch(texts, nmax, label):
    out, misses = [], 0
    for t in texts:
        s = _coerce_text(t)
        if not _OUTPUT_RE.search(s): misses += 1
        out.append(extract_after_output(s, nmax))
    if misses: log.info(f"[{label}] generations missing [OUTPUT] tag: {misses}/{len(texts)} (parsed best-effort)")
    return out

# ------------------------------- Parent helpers -------------------------------
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

# ------------------------------- Dist helpers (minimal) -------------------------------
def shard_indices(N:int, rank:int, W:int):
    return list(range(rank, N, W))

# ------------------------------- Token budgeting -------------------------------
def count_tokens(tok, text: str) -> int:
    if not text: return 0
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

# ------------------------------- Evidence → CUIs -------------------------------
def _strip(x) -> str:
    return str(x or "").strip().upper().replace(" ", "")

def format_icd9_proc_from_pro(c: str) -> str:
    s = _strip(c)
    if s.startswith("PRO_"): s = s[4:]
    s = re.sub(r"[^0-9]", "", s)
    if not s: return ""
    if len(s) >= 3: return s[:2] + "." + s[2:]
    return s

def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    src2cuis, ev = {}, set()

    # ATC via 'ndc'
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        cuis = atc_map.get(key, [])
        if cuis: src2cuis[f"ATC:{key}"] = cuis; ev.update(cuis)

    # LOINC (use 'lab_test_loinc' if present; else 'lab_test')
    lab_col = "lab_test_loinc" if "lab_test_loinc" in row.index else "lab_test"
    for c in to_list(row.get(lab_col, [])):
        key = _strip(c)
        cuis = loinc_map.get(key, [])
        if cuis: src2cuis[f"LNC:{key}"] = cuis; ev.update(cuis)

    # ICD-9 PROC via 'pro_code'
    for c in to_list(row.get("pro_code", [])):
        cc = format_icd9_proc_from_pro(c)
        if not cc: continue
        cuis = icd9_proc_map.get(cc, [])
        if cuis: src2cuis[f"PROC:{cc}"] = cuis; ev.update(cuis)

    return src2cuis, ev

# ------------------------------- Load KG (CSV + canon) -------------------------------
def load_kg_dual(nodes_csv: str, edges_csv: str, edges_canon_csv: str):
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    canon_df = pd.read_csv(edges_canon_csv)

    for need in ("cui_start","cui_target"):
        if need not in edges_df.columns:
            raise ValueError(f"Missing column '{need}' in {edges_csv}")

    keep_canon = canon_df[["cui_start","cui_target","rela_canon","rela_score"]].copy()
    merged = edges_df.merge(keep_canon, on=["cui_start","cui_target"], how="left")

    G = nx.DiGraph()
    node_name = {}
    cui_to_icd9 = defaultdict(set)

    for _, r in nodes_df.iterrows():
        cui = str(r.get("cui","")).strip()
        if not cui: continue
        if cui not in G: G.add_node(cui)
        nm  = str(r.get("name","") or "").strip(); 
        sab = str(r.get("sab","") or "").strip().upper()
        code= str(r.get("code","") or "").strip()
        if nm: node_name[cui] = nm
        if sab == "ICD9CM" and code:
            cui_to_icd9[cui].add(code)

    for _, r in merged.iterrows():
        u = str(r.get("cui_start","") or "").strip()
        v = str(r.get("cui_target","") or "").strip()
        if not u or not v: continue
        if u not in G: G.add_node(u)
        if v not in G: G.add_node(v)
        rel  = str(r.get("rel","")  or "").strip()
        rela = str(r.get("rela","") or "").strip()
        rela_canon = str(r.get("rela_canon","") or "").strip().lower()
        try: rela_score = float(r.get("rela_score", np.nan))
        except Exception: rela_score = float("nan")
        G.add_edge(u, v, rel=rel, rela=rela, rela_canon=rela_canon, rela_score=rela_score)

    return G, node_name, {k: sorted(v) for k,v in cui_to_icd9.items()}

# ------------------------------- Scored miner with degree caps -------------------------------
def _top_successors(G, u, k, min_edge_score, bucket_whitelist):
    out=[]
    if u not in G: return out
    for v in G.successors(u):
        d = G[u][v]
        bucket = (d.get("rela_canon") or "").lower()
        esc = d.get("rela_score")
        try: esc = float(esc)
        except Exception: esc = float("nan")
        if np.isnan(esc) or esc < min_edge_score: continue
        if bucket_whitelist and bucket not in bucket_whitelist: continue
        out.append((v, esc, d))
    out.sort(key=lambda z: z[1], reverse=True)
    return out[:k] if (k and k>0) else out

def mine_paths_scored_all_capped(
    G: nx.DiGraph,
    node_name: Dict[str,str],
    cui_to_icd9: Dict[str, List[str]],
    ev_cuis: Set[str],
    rel_bucket_whitelist: Set[str],
    min_edge_score: float,
    k1: int,
    k2: int,
):
    H1, H2 = [], []
    for u in ev_cuis:
        if u not in G: continue
        v_list = _top_successors(G, u, k1, min_edge_score, rel_bucket_whitelist)

        # H1
        for (v, esc, d) in v_list:
            base = esc + (0.25 if len(cui_to_icd9.get(v, [])) > 0 else 0.0)
            H1.append({
                "src_cui": u, "nbr_cui": v,
                "src_name": node_name.get(u,""), "nbr_name": node_name.get(v,""),
                "rel": (d.get("rel","") or "").strip(), "rela": (d.get("rela","") or "").strip(),
                "rela_canon": (d.get("rela_canon") or "").lower(),
                "rela_score": float(esc),
                "score": base
            })

        # H2
        for (v, esc_uv, d_uv) in v_list:
            w_list = _top_successors(G, v, k2, min_edge_score, rel_bucket_whitelist)
            for (w, esc_vw, d_vw) in w_list:
                anchors = cui_to_icd9.get(w, [])
                score = (esc_uv + esc_vw) + 1.5 * len(set(anchors))
                H2.append({
                    "u": u, "v": v, "w": w,
                    "u_name": node_name.get(u,""),
                    "v_name": node_name.get(v,""),
                    "w_name": node_name.get(w,""),
                    "rel_uv": (d_uv.get("rel","") or "").strip(),  "rela_uv": (d_uv.get("rela","") or "").strip(),
                    "rel_vw": (d_vw.get("rel","") or "").strip(),  "rela_vw": (d_vw.get("rela","") or "").strip(),
                    "w_icd9": anchors,
                    "score": float(score),
                    "s_uv": float(esc_uv), "s_vw": float(esc_vw),
                })

    H1.sort(key=lambda z: z["score"], reverse=True)
    H2.sort(key=lambda z: z["score"], reverse=True)
    return {"H1": H1, "H2": H2}

# ------------------------------- Selection & Rendering -------------------------------
def select_paths_by_threshold(paths_all: Dict[str, list], tau_h1: float, tau_h2: float):
    H1_sel = [h for h in paths_all["H1"] if float(h.get("rela_score", 0.0)) >= tau_h1]
    H2_sel = [h for h in paths_all["H2"] if float(h.get("score", 0.0))      >= tau_h2]
    return {"H1": H1_sel, "H2": H2_sel}

def _arrow_label(rela: str, rel: str) -> str:
    r = (rela or "").strip() or (rel or "").strip()
    return f" --{r}--> " if r else " → "

def short_title_for_code(code: str, code2name: Dict[str,str], profiles: Dict[str,str]) -> str:
    t = (code2name or {}).get(code)
    if t: return t
    raw = (profiles or {}).get(code, "")
    if not raw: return ""
    if "::" in raw:
        after = raw.split("::",1)[1].strip()
        return after.split(";",1)[0].strip()
    return raw.strip()

def render_h2_block_budgeted(paths: Dict[str, list], code2name: Dict[str,str], profiles: Dict[str,str], tok, budget_tokens: int):
    lines = ["[KG CONTEXT — H2 PATHS]"]
    if budget_tokens == 0:
        # unlimited
        for c in paths.get("H2", []):
            anchors=[]
            for code in (c.get("w_icd9", []) or []):
                t = short_title_for_code(code, code2name, profiles)
                anchors.append(f"{code} — {t}" if t else code)
            a_str = " | ".join(anchors) if anchors else "-"
            u = c.get("u_name") or c.get("u") or ""
            v = c.get("v_name") or c.get("v") or ""
            w = c.get("w_name") or c.get("w") or ""
            lines.append(f"- {u}{_arrow_label(c.get('rela_uv'), c.get('rel_uv'))}{v}{_arrow_label(c.get('rela_vw'), c.get('rel_vw'))}{w} [ICD-9: {a_str}]")
        return "\n".join(lines)

    # budgeted
    for c in paths.get("H2", []):
        anchors=[]
        for code in (c.get("w_icd9", []) or []):
            t = short_title_for_code(code, code2name, profiles)
            anchors.append(f"{code} — {t}" if t else code)
        a_str = " | ".join(anchors) if anchors else "-"
        u = c.get("u_name") or c.get("u") or ""
        v = c.get("v_name") or c.get("v") or ""
        w = c.get("w_name") or c.get("w") or ""
        line = f"- {u}{_arrow_label(c.get('rela_uv'), c.get('rel_uv'))}{v}{_arrow_label(c.get('rela_vw'), c.get('rel_vw'))}{w} [ICD-9: {a_str}]"
        trial = "\n".join(lines + [line])
        if count_tokens(tok, trial) <= budget_tokens:
            lines.append(line)
        else:
            break
    return "\n".join(lines) if len(lines) > 1 else ""

def render_h1_block_budgeted(paths: Dict[str, list], tok, budget_tokens: int):
    lines = ["[KG CONTEXT — H1 PATHS]"]
    if budget_tokens == 0:
        for c in paths.get("H1", []):
            u = c.get("src_name") or c.get("src_cui") or ""
            v = c.get("nbr_name") or c.get("nbr_cui") or ""
            lines.append(f"- {u}{_arrow_label(c.get('rela'), c.get('rel'))}{v}")
        return "\n".join(lines)

    for c in paths.get("H1", []):
        u = c.get("src_name") or c.get("src_cui") or ""
        v = c.get("nbr_name") or c.get("nbr_cui") or ""
        line = f"- {u}{_arrow_label(c.get('rela'), c.get('rel'))}{v}"
        trial = "\n".join(lines + [line])
        if count_tokens(tok, trial) <= budget_tokens:
            lines.append(line)
        else:
            break
    return "\n".join(lines) if len(lines) > 1 else ""

def build_tail(N_max_terms:int) -> str:
    return "\n".join([
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]"
    ])

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
        return dict(max_new_tokens=max_new, num_beams=num_beams, do_sample=False,
                    eos_token_id=eos_id, pad_token_id=pad_id,
                    no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))
    return dict(max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                eos_token_id=eos_id, pad_token_id=pad_id,
                no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))

@torch.no_grad()
def generate_texts(model, tok, prompts: List[str], max_len: int, gen_kwargs: dict, batch_size: int = 8, device=None) -> List[str]:
    device = device or (model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outs = []
    bs = max(1, int(batch_size))
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
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

# ------------------------------- Main -------------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--subset_n", type=int, default=0)
    ap.add_argument("--print_samples", type=int, default=5)

    # prompt budgets
    ap.add_argument("--total_input_budget", type=int, default=4096, help="0 = unlimited (diagnostic)")
    ap.add_argument("--assistant_reserve",  type=int, default=256)
    ap.add_argument("--notes_soft_budget",  type=int, default=3008)
    ap.add_argument("--kg_soft_budget",     type=int, default=832, help="0 = unlimited (diagnostic)")
    ap.add_argument("--kg_h2_ratio",        type=float, default=0.7, help="fraction of KG budget for H2; rest to H1")

    # decoding
    ap.add_argument("--gen_max_new", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--decoding", choices=["greedy","beam","sample"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_repeat_ngram", type=int, default=0)

    # model/adapter
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--use_bf16", action="store_true")

    # mapper
    ap.add_argument("--icd_index_dir", required=True)
    ap.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--faiss_rows", type=int, default=50)
    ap.add_argument("--tau_cos", type=float, default=0.40)
    ap.add_argument("--tau_final", type=float, default=0.60)
    ap.add_argument("--w_cos", type=float, default=0.6)
    ap.add_argument("--w_fuz", type=float, default=0.4)

    # KG CSVs (for scored miner)
    ap.add_argument("--kg_nodes_csv", required=True)
    ap.add_argument("--kg_edges_csv", required=True)
    ap.add_argument("--kg_edges_canon_csv", required=True)

    # code2cui maps
    ap.add_argument("--icd9_dx_map_pkl",   required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl",     required=True)
    ap.add_argument("--atc_map_pkl",       required=True)

    # miner knobs
    ap.add_argument("--min_edge_score", type=float, default=0.8)
    ap.add_argument("--rel_bucket_whitelist", type=str, default="etiology,finding_site,morphology,pathology,proc_site,proc_method,measurement,isa,equivalent,severity,course,assoc")
    ap.add_argument("--tau_h1", type=float, default=2.0)
    ap.add_argument("--tau_h2", type=float, default=6.0)
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=30)

    # N cap
    ap.add_argument("--N_max_terms", type=int, default=0, help="0 = dynamic from dataset")

    # outputs
    ap.add_argument("--tmp_dir", default="runs_textgen/test_shards")
    ap.add_argument("--out_metrics", default="runs_textgen/test_metrics_raw_vs_kg.json")
    ap.add_argument("--out_rows_csv", default="runs_textgen/kg_prompt_rows.csv")

    args = ap.parse_args()

    # data
    try: df = pd.read_pickle(args.data_pickle)
    except Exception:
        with open(args.data_pickle, "rb") as f: df = pickle.load(f)

    if args.test_only:
        test_df = df.copy()
    else:
        from sklearn.model_selection import train_test_split
        subs = df[args.subject_col].dropna().unique()
        _, te_subs = train_test_split(subs, test_size=0.10, random_state=42)
        test_df  = df[df[args.subject_col].isin(te_subs)].copy()

    if args.subset_n and args.subset_n > 0:
        test_df = test_df.iloc[:args.subset_n].reset_index(drop=True)

    # gold codes
    def extract_codes(df, label_col):
        out=[]
        for _, r in df.iterrows():
            lst = to_list(r.get(label_col, []))
            lst = [format_icd9(c) for c in lst if c]
            lst = [c for c in lst if is_valid_icd9(c)]
            out.append(lst)
        return out

    gold_codes = extract_codes(test_df, args.label_col)
    labels_eval = sorted({c for lst in gold_codes for c in lst})

    # model
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

    # KG + maps for miner
    G, node_name, cui_to_icd9 = load_kg_dual(args.kg_nodes_csv, args.kg_edges_csv, args.kg_edges_canon_csv)
    icd9_dx_map   = pickle.load(open(args.icd9_dx_map_pkl,   "rb"))
    icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
    loinc_map     = pickle.load(open(args.loinc_map_pkl,     "rb"))
    atc_map       = pickle.load(open(args.atc_map_pkl,       "rb"))

    buckets = {s.strip().lower() for s in args.rel_bucket_whitelist.split(",") if s.strip()}
    log.info(f"[KG] Nodes={G.number_of_nodes():,} Edges={G.number_of_edges():,} | buckets={sorted(buckets)} | min_edge_score={args.min_edge_score} | k1={args.k1}, k2={args.k2}")
    log.info(f"[THR] tau_h1={args.tau_h1} tau_h2={args.tau_h2}")

    # dataset-level caps
    base_N, p50_codes, p90_codes = compute_terms_caps(gold_codes)
    if args.N_max_terms and args.N_max_terms > 0:
        base_N = p50_codes = p90_codes = args.N_max_terms

    # decoding kwargs
    gen_kwargs = build_generate_kwargs(
        decoding=args.decoding, max_new=args.gen_max_new,
        eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
        num_beams=args.num_beams, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, no_repeat_ngram=args.no_repeat_ngram
    )

    # prompt building
    rows_csv = []
    raw_prompts, kg_prompts = [], []
    max_total = args.total_input_budget if args.total_input_budget > 0 else None

    for i, row in test_df.iterrows():
        # evidence → CUIs
        src2cuis, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

        # mine paths (scored + capped)
        paths_all = mine_paths_scored_all_capped(
            G=G, node_name=node_name, cui_to_icd9=cui_to_icd9, ev_cuis=set(ev_cuis),
            rel_bucket_whitelist=buckets, min_edge_score=args.min_edge_score, k1=args.k1, k2=args.k2
        )
        # thresholds
        paths_sel = select_paths_by_threshold(paths_all, args.tau_h1, args.tau_h2)

        # render KG blocks (budgeted or unlimited)
        if args.kg_soft_budget > 0:
            h2_budget = int(args.kg_soft_budget * args.kg_h2_ratio)
            h1_budget = args.kg_soft_budget - h2_budget
            h2_text = render_h2_block_budgeted(paths_sel, {}, {}, tok, h2_budget)
            h1_text = render_h1_block_budgeted(paths_sel, tok, h1_budget)
            kg_text = "\n".join([b for b in (h2_text, h1_text) if b])
        else:
            # unlimited for diagnostics
            h2_text = render_h2_block_budgeted(paths_sel, {}, {}, tok, 0)
            h1_text = render_h1_block_budgeted(paths_sel, tok, 0)
            kg_text = "\n".join([b for b in (h2_text, h1_text) if b])

        # prompts
        header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
        notes_full = serialize_notes(row)
        notes = trim_to_token_budget(tok, notes_full, args.notes_soft_budget) if args.notes_soft_budget > 0 else notes_full

        # row N cap (dynamic unless forced)
        N_this = row_cap_from_candidates(base_N, p90_codes, allowed_count=len(paths_sel["H2"]) + len(paths_sel["H1"]))

        tail   = build_tail(N_this)
        raw_p  = "\n".join([x for x in (header, notes, tail) if x])
        kg_p   = "\n".join([x for x in (header, notes, kg_text, tail) if x])

        # optional total clamp (only if total_input_budget > 0)
        if max_total is not None:
            # try to shrink KG block first if over
            if count_tokens(tok, kg_p) > (max_total - args.assistant_reserve):
                # shrink KG by the overage if any
                over = count_tokens(tok, kg_p) - (max_total - args.assistant_reserve)
                shrink_target = max(0, (args.kg_soft_budget if args.kg_soft_budget>0 else count_tokens(tok, kg_text)) - over)
                kg_text2 = trim_to_token_budget(tok, kg_text, shrink_target)
                kg_p2 = "\n".join([x for x in (header, notes, kg_text2, tail) if x])
                if count_tokens(tok, kg_p2) <= (max_total - args.assistant_reserve):
                    kg_p = kg_p2
                    kg_text = kg_text2
                else:
                    # fallback: drop KG
                    kg_p = raw_p
                    kg_text = ""
            # clamp RAW in worst case if needed
            if count_tokens(tok, raw_p) > (max_total - args.assistant_reserve):
                over = count_tokens(tok, raw_p) - (max_total - args.assistant_reserve)
                new_notes = max(0, count_tokens(tok, notes) - over - 8)
                notes_trim = trim_to_token_budget(tok, notes_full, new_notes)
                raw_p = "\n".join([x for x in (header, notes_trim, tail) if x])
                if kg_text == "": kg_p = raw_p  # keep consistent if KG was dropped

        # token accounting
        row_rec = {
            "idx": i,
            "hadm_id": row.get("hadm_id",""),
            "H1_lines": len(paths_sel["H1"]),
            "H2_lines": len(paths_sel["H2"]),
            "KG_tokens": count_tokens(tok, kg_text),
            "KG_H2_tokens": count_tokens(tok, h2_text),
            "KG_H1_tokens": count_tokens(tok, h1_text),
            "RAW_tokens": count_tokens(tok, raw_p),
            "WITHKG_tokens": count_tokens(tok, kg_p),
        }
        rows_csv.append(row_rec)

        raw_prompts.append(raw_p)
        kg_prompts.append(kg_p)

    # persist per-row CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_rows_csv)), exist_ok=True)
    pd.DataFrame(rows_csv).to_csv(args.out_rows_csv, index=False)
    log.info(f"[OK] Wrote per-row prompt stats -> {args.out_rows_csv}")

    # generation
    gen_kwargs = gen_kwargs  # naming clarity
    raw_out_texts = generate_texts(model, tok, raw_prompts, max_len=(args.total_input_budget or 8192), gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)
    kg_out_texts  = generate_texts(model, tok, kg_prompts,  max_len=(args.total_input_budget or 8192), gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)

    # parse terms
    ncap = max(6, max(r["WITHKG_tokens"] for r in rows_csv) // 200)  # rough cap if not given; model still follows [OUTPUT] max
    ncap = max(ncap, 12)
    raw_terms = _safe_extract_batch(raw_out_texts, ncap, "RAW")
    kg_terms  = _safe_extract_batch(kg_out_texts,  ncap, "KG")

    # map terms -> ICD-9
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    mapped_raw = mapper.map_terms(raw_terms)
    mapped_kg  = mapper.map_terms(kg_terms)

    # restrict to eval label space
    gold_eval = restrict_to(gold_codes, labels_eval)
    raw_eval  = restrict_to(mapped_raw, labels_eval)
    kg_eval   = restrict_to(mapped_kg,  labels_eval)

    # metrics
    Yt = multihot(gold_eval, labels_eval)
    Yr = multihot(raw_eval,  labels_eval)
    Yk = multihot(kg_eval,   labels_eval)

    m_raw = eval_pack(Yt, Yr)
    pm_raw = {}; _ = add_parent_metrics_full(pm_raw, gold_eval, raw_eval); add_parent_macro_f1(pm_raw, gold_eval, raw_eval)
    m_raw.update(pm_raw)

    m_kg  = eval_pack(Yt, Yk)
    pm_kg = {}; _ = add_parent_metrics_full(pm_kg, gold_eval, kg_eval); add_parent_macro_f1(pm_kg, gold_eval, kg_eval)
    m_kg.update(pm_kg)

    # per-label CSVs (FULL)
    out_dir = os.path.dirname(os.path.abspath(args.out_metrics))
    os.makedirs(out_dir, exist_ok=True)

    def per_label_table(y_true, y_pred, labels, out_csv_path=None):
        p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        df = pd.DataFrame({"code": labels,"precision": p,"recall": r,"f1": f1,"support": support})
        if out_csv_path: df.to_csv(out_csv_path, index=False)
        return df

    per_label_table(Yt, Yr, labels_eval, os.path.join(out_dir, "per_label_FULL_RAW.csv"))
    per_label_table(Yt, Yk, labels_eval, os.path.join(out_dir, "per_label_FULL_KG.csv"))

    payload = {
        "label_space": "FULL",
        "num_samples": len(test_df),
        "metrics_raw": m_raw,
        "metrics_kg":  m_kg,
        "args": vars(args)
    }
    with open(args.out_metrics, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"[OK] Metrics saved to {args.out_metrics}")

    # sample prints
    n_show = min(args.print_samples, len(test_df))
    log.info("=== Sample predictions (RAW vs KG) ===")
    for i in range(n_show):
        Gs = set(gold_eval[i]); R = set(raw_eval[i]); K = set(kg_eval[i])
        tp = len(Gs & R); fp = len(R - Gs); fn = len(Gs - R)
        pr = tp/(tp+fp) if tp+fp>0 else 0.0
        rr = tp/(tp+fn) if tp+fn>0 else 0.0
        fr = (2*pr*rr)/(pr+rr) if (pr+rr)>0 else 0.0

        tp2 = len(Gs & K); fp2 = len(K - Gs); fn2 = len(Gs - K)
        pr2 = tp2/(tp2+fp2) if tp2+fp2>0 else 0.0
        rr2 = tp2/(tp2+fn2) if tp2+fn2>0 else 0.0
        fr2 = (2*pr2*rr2)/(pr2+rr2) if (pr2+rr2)>0 else 0.0

        log.info(f"[Sample {i+1}] hadm={test_df.iloc[i].get('hadm_id','')}")
        log.info(f"  GOLD codes: {', '.join(sorted(Gs)) if Gs else '(none)'}")
        log.info(f"  RAW mapped ICD-9: {', '.join(sorted(R)) if R else '(none)'}  | P/R/F1 = {pr:.3f}/{rr:.3f}/{fr:.3f}")
        log.info(f"  KG  mapped ICD-9: {', '.join(sorted(K)) if K else '(none)'}  | P/R/F1 = {pr2:.3f}/{rr2:.3f}/{fr2:.3f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
