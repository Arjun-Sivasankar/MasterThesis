# gen/withKG/RAG/textgen_prompt_hpaths.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextGen with KG hints in the prompt (adapter-only), evaluated RAW vs KG.

Adds:
- Optional global prompt clamp (set --total_input_budget <= 0 to disable).
- Optional KG clamp (set --kg_soft_budget <= 0 to disable).
- Notes soft cap remains (default 3008).
- KG context rendered in two blocks:
    [KG context - H2 paths]
    [KG context - H1 paths]
- Writes per-row token stats (RAW, KG total/H2/H1) to CSV when --stats_csv is set.
- --kg_h2_ratio to allocate KG budget between H2 and H1 (defaults to 1.0 = H2-first).
- Logs RAW and KG free-text terms for sample rows (like older runs).
"""

import os, re, json, time, argparse, pickle, glob, sys, math
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
import torch.distributed as dist
import networkx as nx

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
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(_coerce_text(y) for y in x if y is not None)
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return str(x)

def extract_after_output(generation, n_max: int = 12) -> List[str]:
    s = _coerce_text(generation)
    s = s.replace("</s>", "").replace("<s>", "").strip()
    m = _OUTPUT_RE.search(s)
    block = s[m.end():] if m else s
    lines_out = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and not re.match(r"^\[\s*OUTPUT\s*\]$", line, flags=re.I):
            break
        line = re.sub(r"^(?:[-*]\s*|\d+\.\s*|\(\d+\)\s*)", "", line)
        if line:
            lines_out.append(line)
        if len(lines_out) >= n_max:
            break
    return lines_out

def _safe_extract_batch(texts, nmax, label):
    out = []
    misses = 0
    for t in texts:
        s = _coerce_text(t)
        if not _OUTPUT_RE.search(s):
            misses += 1
        out.append(extract_after_output(s, nmax))
    if misses:
        log.info(f"[{label}] generations missing [OUTPUT] tag: {misses}/{len(texts)} (parsed best-effort)")
    return out

# ------------------------------- ICD-9 helpers (safe parent) -------------------------------
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

# ------------------------------- Parent metrics (full) -------------------------------
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

# ------------------------------- Minimal dist utils -------------------------------
def maybe_init_dist():
    if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl")
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

# ------------------------------- Evidence & KG plumbing -------------------------------
def _strip(x) -> str:
    return str(x or "").strip().upper().replace(" ", "")

def format_icd9_proc_from_pro(c: str) -> str:
    s = _strip(c)
    if s.startswith("PRO_"): s = s[4:]
    s = re.sub(r"[^0-9]", "", s)
    if not s: return ""
    if len(s) >= 3:
        return s[:2] + "." + s[2:]
    return s

def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    src2cuis = {}
    ev = set()

    # ATC via 'ndc'
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        cuis = atc_map.get(key, [])
        if cuis:
            src2cuis[f"ATC:{key}"] = cuis
            ev.update(cuis)

    # LOINC via 'lab_test_loinc'
    for c in to_list(row.get("lab_test_loinc", [])):
        key = _strip(c)
        cuis = loinc_map.get(key, [])
        if cuis:
            src2cuis[f"LNC:{key}"] = cuis
            ev.update(cuis)

    # ICD-9 PROC via 'pro_code'
    for c in to_list(row.get("pro_code", [])):
        cc = format_icd9_proc_from_pro(c)
        if not cc: continue
        cuis = icd9_proc_map.get(cc, [])
        if cuis:
            src2cuis[f"PROC:{cc}"] = cuis
            ev.update(cuis)

    return src2cuis, ev

def expand_cuis(G: nx.DiGraph,
                seeds: Set[str],
                hop: int) -> Set[str]:
    if hop <= 0 or not seeds or G is None:
        return set(seeds)
    frontier = set(seeds); visited = set(seeds)
    for _ in range(hop):
        nxt=set()
        for u in frontier:
            if u not in G: continue
            for v in G.successors(u):
                if v not in visited:
                    nxt.add(v)
        visited |= nxt
        frontier = nxt
        if not frontier: break
    return visited

# ------------------------------- Candidate ranking & priors -------------------------------
_HISTORY_PAT = re.compile(r"\bhistory of\b|\bfamily history\b", re.I)
_EXTERNAL_PAT = re.compile(r"\baccident\b|\btrauma\b|\binjury\b|\bfall\b|\bassault\b", re.I)

def extract_note_cues(text: str) -> Set[str]:
    s = (text or "").lower()
    cues=set()
    if _HISTORY_PAT.search(s): cues.add("history")
    if _EXTERNAL_PAT.search(s): cues.add("external")
    return cues

def build_code_prior(gold_lists: List[List[str]]) -> Counter:
    return Counter([c for lst in gold_lists for c in lst])

def top_prior_codes(code_prior: Counter, k: int = 50) -> List[str]:
    return [c for (c,_) in code_prior.most_common(k)]

def allowed_icd9_from_cuis(dx_map: Dict[str, List[str]], bag_cuis: Set[str]) -> List[str]:
    if not bag_cuis: return []
    allowed=[]
    for code, cuis in dx_map.items():
        if bag_cuis.intersection(cuis):
            c = format_icd9(code)
            if is_valid_icd9(c): allowed.append(c)
    return sorted(set(allowed))

def rank_candidates(allowed_codes: List[str],
                    icd9_dx_map: Dict[str, List[str]],
                    ev_cuis: Set[str],
                    nb_cuis: Set[str],
                    note_cues: Set[str] = None,
                    code_prior: Counter = None,
                    w_ev: float = 3.0,
                    w_nb: float = 1.0,
                    w_typ: float = 1.0,
                    w_freq: float = 0.2) -> List[str]:
    def type_bonus(code: str) -> float:
        if code.startswith("V"):
            return 0.0 if (note_cues and "history" in note_cues) else -1.5
        if code.startswith("E"):
            return 0.0 if (note_cues and "external" in note_cues) else -1.5
        return 0.0

    ev_cuis = ev_cuis or set()
    nb_cuis = nb_cuis or set()

    scored=[]
    for c in allowed_codes:
        cuis = set(icd9_dx_map.get(c, []))
        s = 0.0
        if ev_cuis:
            s += w_ev * len(cuis & ev_cuis)
        if nb_cuis:
            s += w_nb * len(cuis & nb_cuis)
        s += w_typ * type_bonus(c)
        if code_prior:
            s += w_freq * math.log1p(code_prior.get(c, 0))
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

# ------------------------------- KG paths (H2/H1) -------------------------------
def _arrow_label(rela: str, rel: str) -> str:
    r = (rela or "").strip() or (rel or "").strip()
    return f" --{r}--> " if r else " → "

def mine_hops_simple(G: nx.DiGraph,
                     ev_cuis: Set[str],
                     k1:int=30, k2:int=30) -> Tuple[List[dict], List[dict]]:
    """
    Simple (score-less) miner:
      H1: u -> v
      H2: u -> v -> w
    Degree capped by k1/k2 for speed. No bucket filters here since KG_PKL may not have scores.
    """
    H1, H2 = [], []
    for u in ev_cuis:
        if u not in G: continue

        # collect top-k1 successors by id (deterministic) if no score
        vs = list(G.successors(u))
        vs.sort()
        vs = vs[:k1] if k1 and k1>0 else vs

        # H1
        for v in vs:
            d = G[u][v]
            H1.append({
                "src_cui": u, "nbr_cui": v,
                "src_name": G.nodes[u].get("name",""),
                "nbr_name": G.nodes[v].get("name",""),
                "rel": (d.get("rel") or ""), "rela": (d.get("rela") or "")
            })

        # H2
        for v in vs:
            if v not in G: continue
            ws = list(G.successors(v))
            ws.sort()
            ws = ws[:k2] if k2 and k2>0 else ws
            for w in ws:
                d_uv = G[u][v]
                d_vw = G[v][w]
                H2.append({
                    "u": u, "v": v, "w": w,
                    "u_name": G.nodes[u].get("name",""),
                    "v_name": G.nodes[v].get("name",""),
                    "w_name": G.nodes[w].get("name",""),
                    "rel_uv": (d_uv.get("rel") or ""),  "rela_uv": (d_uv.get("rela") or ""),
                    "rel_vw": (d_vw.get("rel") or ""),  "rela_vw": (d_vw.get("rela") or ""),
                })
    return H1, H2

def render_h2_block(H2_rows: List[dict]) -> str:
    lines = ["[KG context - H2 paths]"]
    for c in H2_rows:
        u = c.get("u_name") or c.get("u") or ""
        v = c.get("v_name") or c.get("v") or ""
        w = c.get("w_name") or c.get("w") or ""
        lines.append(
            f"- {u}{_arrow_label(c.get('rela_uv'), c.get('rel_uv'))}"
            f"{v}{_arrow_label(c.get('rela_vw'), c.get('rel_vw'))}{w}"
        )
    return "\n".join(lines)

def render_h1_block(H1_rows: List[dict]) -> str:
    lines = ["[KG context - H1 paths]"]
    for c in H1_rows:
        u = c.get("src_name") or c.get("src_cui") or ""
        v = c.get("nbr_name") or c.get("nbr_cui") or ""
        lines.append(f"- {u}{_arrow_label(c.get('rela'), c.get('rel'))}{v}")
    return "\n".join(lines)

def combine_kg_blocks_with_budget(tok, h2_text: str, h1_text: str, budget: int, h2_ratio: float = 1.0):
    """
    Returns: (combined_text, total_tokens, h2_tokens, h1_tokens)

    If budget <= 0: unlimited — include full H2 then H1 without trimming.
    If budget  > 0: allocate int(budget*h2_ratio) to H2, then the remainder to H1.
    """
    if budget is None or budget <= 0:
        combined = h2_text + ("\n" + h1_text if h1_text else "")
        return combined, count_tokens(tok, combined), count_tokens(tok, h2_text), count_tokens(tok, h1_text)

    h2_quota = int(max(0, min(1.0, h2_ratio)) * budget)
    h1_quota = max(0, budget - h2_quota)

    h2_trim = trim_to_token_budget(tok, h2_text, h2_quota) if h2_quota>0 else ""
    h1_trim = trim_to_token_budget(tok, h1_text, h1_quota) if h1_quota>0 else ""

    # If H2 used less than quota, roll leftover to H1 once
    used_h2 = count_tokens(tok, h2_trim)
    leftover = max(0, budget - used_h2 - count_tokens(tok, h1_trim))
    if leftover > 0 and h1_text:
        h1_trim = trim_to_token_budget(tok, h1_text, count_tokens(tok, h1_trim) + leftover)

    combined = (h2_trim if h2_trim else h2_text[:0]) + (("\n"+h1_trim) if h1_trim else "")
    return (combined,
            count_tokens(tok, combined),
            count_tokens(tok, h2_trim),
            count_tokens(tok, h1_trim))

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
    notes  = trim_to_token_budget(tok, full_notes, notes_soft_budget) if notes_soft_budget > 0 else full_notes
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

# ------------------------------- Generation (decode-to-text) -------------------------------
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
        return dict(
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None),
        )
    if decoding == "beam":
        return dict(
            max_new_tokens=max_new,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None),
        )
    # sample
    return dict(
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None),
    )

@torch.no_grad()
def generate_texts(model, tok, prompts: List[str], max_len: int, gen_kwargs: dict, batch_size: int = 8, device=None) -> List[str]:
    device = device or (model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outs = []
    bs = max(1, int(batch_size))
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len if max_len and max_len>0 else None, add_special_tokens=False)
        for k in enc:
            enc[k] = enc[k].to(device)
        gen = model.generate(**enc, **{k:v for k,v in gen_kwargs.items() if v is not None})
        dec = tok.batch_decode(gen, skip_special_tokens=False)
        outs.extend(dec)
    return outs

# ------------------------------- N_max_terms dynamics -------------------------------
def compute_terms_caps(gold_lists: List[List[str]]) -> Tuple[int, int, int]:
    sizes = sorted(len(x) for x in gold_lists)
    if not sizes:
        return 12, 10, 18
    def pct(p):
        k = max(0, min(len(sizes)-1, int(round((p/100.0)* (len(sizes)-1)))))
        return sizes[k]
    p50 = pct(50)
    p90 = pct(90)
    base_N = int(max(6, min(24, round(p50 + 2))))
    return base_N, p50, p90

def row_cap_from_candidates(base_N: int, p90: int, allowed_count: int) -> int:
    bonus = 0
    if   allowed_count >= 30: bonus = 6
    elif allowed_count >= 20: bonus = 4
    elif allowed_count >= 10: bonus = 2
    return int(max(6, min(p90, base_N + bonus)))

# ------------------------------- Pretty printer + term block logger -------------------------------
def _pretty_print_block(title: str, d: dict):
    log.info(f"--- {title} ---")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            log.info(f"{k:>28s}: {v:.6f}")
        else:
            log.info(f"{k:>28s}: {v}")

def _log_terms_block(tag: str, terms: list):
    log.info(f"  {tag} free-text terms:")
    if not terms:
        log.info("    - (none)")
        return
    for t in terms:
        if not t:
            continue
        log.info(f"    - {t}")

# ------------------------------- MAIN -------------------------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--subset_n", type=int, default=0)
    ap.add_argument("--print_samples", type=int, default=5)

    # prompts/generation
    ap.add_argument("--N_max_terms", type=int, default=0)

    # token budgets
    ap.add_argument("--total_input_budget", type=int, default=4096,
                    help="If <=0, do NOT clamp total prompt length.")
    ap.add_argument("--assistant_reserve",  type=int, default=256)
    ap.add_argument("--notes_soft_budget",  type=int, default=3008,
                    help="Notes are always softly trimmed to this; set <=0 to keep all notes.")
    ap.add_argument("--kg_soft_budget",     type=int, default=832,
                    help="If <=0, include full H2 then H1 without trimming.")
    ap.add_argument("--kg_h2_ratio",        type=float, default=1.0,
                    help="Share of KG budget for H2 block (0..1). Remainder goes to H1.")

    # decoding
    ap.add_argument("--gen_max_new", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--decoding", choices=["greedy","beam","sample"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_repeat_ngram", type=int, default=0)

    # model/adapter (adapter-only)
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

    # eval label space
    ap.add_argument("--labels_space", choices=["full","head"], default="full")
    ap.add_argument("--labels_head_k", type=int, default=0)

    # optional lists
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")

    # KG inputs
    ap.add_argument("--kg_pkl", required=True)
    ap.add_argument("--icd9_dx_map_pkl",   required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl",     required=True)
    ap.add_argument("--atc_map_pkl",       required=True)
    ap.add_argument("--hop", type=int, default=1, choices=[0,1,2])
    ap.add_argument("--max_neighbors_show", type=int, default=24)  # kept for compatibility
    # simple miner caps
    ap.add_argument("--kg_k1", type=int, default=30, help="Max v per u for H1/H2 first hop")
    ap.add_argument("--kg_k2", type=int, default=30, help="Max w per v for H2 second hop")

    # distributed
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--tmp_dir", default="runs_textgen/test_shards")
    ap.add_argument("--out_metrics", default="runs_textgen/test_metrics_raw_vs_kg.json")

    # per-row token stats output (CSV)
    ap.add_argument("--stats_csv", default="", help="If set, write per-row token stats (RAW, KG total/H2/H1).")

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

    # label space
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

    # ---------------- dataset prior & dynamic caps ----------------
    code_prior = build_code_prior(gold_codes)
    base_N, p50_codes, p90_codes = compute_terms_caps(gold_codes)
    if is_main_process():
        log.info(f"N_max_terms (dataset): base={base_N}, p50={p50_codes}, p90={p90_codes}")

    # ---------------- model (adapter-only, LEFT padding) ----------------
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

    # ---------------- KG + maps ----------------
    KG  = pickle.load(open(args.kg_pkl, "rb"))  # expected to be nx.DiGraph
    icd9_dx_map   = pickle.load(open(args.icd9_dx_map_pkl,   "rb"))
    icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
    loinc_map     = pickle.load(open(args.loinc_map_pkl,     "rb"))
    atc_map       = pickle.load(open(args.atc_map_pkl,       "rb"))

    # ---------------- build prompts (RAW & KG), token-budgeted ----------------
    log.info("Building RAW & KG prompts...")
    raw_prompts=[]; kg_prompts=[]; dbg_rows=[]

    # Compute max allowed total only if user wants it
    max_prompt = args.total_input_budget - args.assistant_reserve if args.total_input_budget and args.total_input_budget > 0 else None

    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # evidence (with CUIs)
        src2cuis, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

        # expand CUIs to neighbors (for candidate allow-list)
        expanded_cuis = expand_cuis(KG, ev_cuis, hop=args.hop)
        bag_cuis = expanded_cuis

        # allowed from KG; if none, fallback to prior top-K
        allowed  = allowed_icd9_from_cuis(icd9_dx_map, bag_cuis)
        if not allowed:
            allowed = top_prior_codes(code_prior, k=50)

        # row-adaptive cap
        dynamic_base = base_N if args.N_max_terms == 0 else args.N_max_terms
        N_this = row_cap_from_candidates(dynamic_base, p90_codes, allowed_count=len(allowed))

        # (ranking kept for future use; not printed in KG block)
        notes_text = serialize_notes(row)
        cues = extract_note_cues(notes_text)
        nb_only = bag_cuis - ev_cuis
        _ = rank_candidates(
            allowed_codes=allowed,
            icd9_dx_map=icd9_dx_map,
            ev_cuis=ev_cuis,
            nb_cuis=nb_only,
            note_cues=cues,
            code_prior=code_prior
        )

        # Mine & render H2/H1 path blocks (simple, no scores)
        H1_rows, H2_rows = mine_hops_simple(KG, ev_cuis, k1=args.kg_k1, k2=args.kg_k2)
        h2_block = render_h2_block(H2_rows) if H2_rows else "[KG context - H2 paths]\n- (none)"
        h1_block = render_h1_block(H1_rows) if H1_rows else "[KG context - H1 paths]\n- (none)"
        kg_text_combined, kg_total_tokens, kg_h2_tokens, kg_h1_tokens = combine_kg_blocks_with_budget(
            tok, h2_block, h1_block, args.kg_soft_budget, args.kg_h2_ratio
        )

        # prompts
        raw_p, kg_p, d = build_prompts_for_row(row, tok, kg_text_combined, args.notes_soft_budget, N_this)

        # --- Clamp RAW only if total budget is active ---
        if max_prompt is not None and d["total_raw"] > max_prompt:
            over = d["total_raw"] - max_prompt
            new_notes = max(0, d["notes_tokens"] - over - 8)
            notes_trim = trim_to_token_budget(tok, serialize_notes(row), new_notes)
            header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
            tail   = build_tail(N_this)
            raw_p = "\n".join([x for x in [header, notes_trim, tail] if x])
            d["total_raw"] = count_tokens(tok, raw_p)

        # --- Clamp KG only if total budget is active ---
        if max_prompt is not None and d["total_kg"] > max_prompt:
            if args.kg_soft_budget and args.kg_soft_budget > 0:
                shrink = max(0, args.kg_soft_budget - (d["total_kg"] - max_prompt))
                kg_text2 = trim_to_token_budget(tok, kg_text_combined, shrink)
                raw_p2, kg_p2, d2 = build_prompts_for_row(row, tok, kg_text2, args.notes_soft_budget, N_this)
                if count_tokens(tok, kg_p2) <= max_prompt:
                    kg_p, d = kg_p2, d2
                    kg_total_tokens = count_tokens(tok, kg_text2)
                    # Approximated split after trim
                    kg_h2_tokens = min(kg_h2_tokens, kg_total_tokens)
                    kg_h1_tokens = max(0, kg_total_tokens - kg_h2_tokens)
                else:
                    kg_p = raw_p
                    d["kg_tokens"] = 0
                    d["total_kg"]  = d["total_raw"]
                    kg_total_tokens, kg_h2_tokens, kg_h1_tokens = 0, 0, 0
            else:
                kg_p = raw_p
                d["kg_tokens"] = 0
                d["total_kg"]  = d["total_raw"]
                kg_total_tokens, kg_h2_tokens, kg_h1_tokens = 0, 0, 0

        raw_prompts.append(raw_p)
        kg_prompts.append(kg_p)
        dbg_rows.append({
            "idx": i,
            "hadm_id": row.get("hadm_id",""),
            "ev_cuis": len(ev_cuis),
            "exp_cuis": len(bag_cuis),
            "N_cap": N_this,
            "header_tokens": d["header_tokens"],
            "notes_tokens": d["notes_tokens"],
            "tail_tokens": d["tail_tokens"],
            "RAW_tokens": d["total_raw"],
            "WITHKG_tokens": d["total_kg"],
            "KG_tokens": kg_total_tokens,
            "KG_H2_tokens": kg_h2_tokens,
            "KG_H1_tokens": kg_h1_tokens,
        })

        if i==0:
            print(...)

    log.info("Done building prompts....")

    # ---------------- decoding kwargs ----------------
    gen_kwargs = build_generate_kwargs(
        decoding=args.decoding, max_new=args.gen_max_new,
        eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
        num_beams=args.num_beams, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, no_repeat_ngram=args.no_repeat_ngram
    )

    # ---------------- distributed sharding ----------------
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

    # ---------------- generation -> decoded text -> parsed terms ----------------
    t0 = time.time()
    raw_out_texts = generate_texts(model, tok, shard_raw, max_len=args.total_input_budget, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)
    kg_out_texts  = generate_texts(model, tok, shard_kg,  max_len=args.total_input_budget, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)

    raw_terms = _safe_extract_batch(raw_out_texts,  max(d["N_cap"] for d in dbg_rows), "RAW")
    kg_terms  = _safe_extract_batch(kg_out_texts,   max(d["N_cap"] for d in dbg_rows), "KG")

    if is_main_process():
        per = (time.time()-t0)/max(1,len(idxs))
        log.info(f"Generation done ({per:.2f}s/sample on rank {rank}).")

    # ---------------- mapping terms -> ICD-9 ----------------
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    mapped_raw = mapper.map_terms(raw_terms)
    mapped_kg  = mapper.map_terms(kg_terms)

    # ---------------- persist shard ----------------
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

    # ---------------- merge & evaluate (rank 0) ----------------
    if rank == 0:
        shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
        all_idx, all_rawt, all_kgt, all_rawc, all_kgc, all_gold = [], [], [], [], [], []
        all_dbg = []
        for sp in shards:
            with open(sp, "rb") as f:
                D = pickle.load(f)
            all_idx.extend(D["idxs"])
            all_rawt.extend(D["raw_terms"]); all_kgt.extend(D["kg_terms"])
            all_rawc.extend(D["mapped_raw"]); all_kgc.extend(D["mapped_kg"])
            all_gold.extend(D["gold"])
            all_dbg.extend(D["dbg"])

        order = np.argsort(np.array(all_idx))
        raw_terms_all = [all_rawt[i] for i in order]
        kg_terms_all  = [all_kgt[i]  for i in order]
        pred_raw_all  = [all_rawc[i] for i in order]
        pred_kg_all   = [all_kgc[i]  for i in order]
        gold_all      = [all_gold[i] for i in order]
        dbg_all       = [all_dbg[i]  for i in order]

        # Write per-row stats CSV if requested
        if args.stats_csv:
            df_stats = pd.DataFrame(dbg_all)
            os.makedirs(os.path.dirname(args.stats_csv), exist_ok=True)
            df_stats.to_csv(args.stats_csv, index=False)
            log.info(f"Wrote per-row token stats -> {args.stats_csv}")

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

        # per-label CSV for FULL
        out_dir = os.path.dirname(os.path.abspath(args.out_metrics))
        os.makedirs(out_dir, exist_ok=True)
        def per_label_table(y_true, y_pred, labels, out_csv_path=None):
            p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            df = pd.DataFrame({"code": labels,"precision": p,"recall": r,"f1": f1,"support": support})
            if out_csv_path:
                df.to_csv(out_csv_path, index=False)
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

        # print a few side-by-side samples (now includes RAW/KG free-text terms)
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

            _log_terms_block("RAW", raw_terms_all[i])
            log.info(f"  RAW mapped ICD-9: {', '.join(sorted(R)) if R else '(none)'}  | P/R/F1 = {pr:.3f}/{rr:.3f}/{fr:.3f}")

            _log_terms_block("KG ", kg_terms_all[i])
            log.info(f"  KG  mapped ICD-9: {', '.join(sorted(K)) if K else '(none)'}  | P/R/F1 = {pr2:.3f}/{rr2:.3f}/{fr2:.3f}")

        # save metrics JSON
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

