#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, re, math, pickle, random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer

# ========= Project paths (adjust if needed) =========
PROJECT_DIR = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis"
sys.path.append(PROJECT_DIR)

from common_textgen import serialize_structured_readable, serialize_notes

# === Assets ===
KG_NODES_CSV       = f"{PROJECT_DIR}/KG/kg_output4/kg_nodes.csv"
KG_EDGES_CSV       = f"{PROJECT_DIR}/KG/kg_output4/kg_edges.csv"
KG_EDGES_CANON_CSV = f"{PROJECT_DIR}/KG/kg_output4/kg_edges_canon.csv"  # produced by kg_rel_canon.py

ICD9_PROC_MAP = f"{PROJECT_DIR}/KG/kg_output4/code2cui_icd9_proc.pkl"
LOINC_MAP     = f"{PROJECT_DIR}/KG/kg_output4/code2cui_loinc.pkl"
ATC_MAP       = f"{PROJECT_DIR}/KG/kg_output4/code2cui_atc.pkl"

CODE2NAME_PKL    = f"{PROJECT_DIR}/gen/withKG/RAG/kg_recommender/code2name.pkl"        # optional
ICD9_PROFILES_JS = f"{PROJECT_DIR}/gen/withKG/RAG/kg_recommender/icd9_profiles.json"   # optional

DATA_PKL   = f"{PROJECT_DIR}/dataset/final_data/test_df.pkl"
BASE_MODEL = f"{PROJECT_DIR}/models/Llama-3.1-8B-Instruct"  # tokenizer only

# === Prompt features (used only for counts / examples) ===
MAX_LEN        = 4096
KG_HINT_BUDGET = 600
N_MAX_TERMS    = 12

# === Canon buckets & floor score ===
REL_BUCKET_WHITELIST = {
    "etiology","finding_site","morphology","pathology",
    "proc_site","proc_method","measurement", "isa", "equivalent",
    "severity", "course", "assoc"
}
MIN_EDGE_SCORE = 0.8   # ↑ a bit for faster, higher-signal mining

# === Global thresholds for selection ===
TAU_H1 = 2.0
# TAU_H2 = 2.2
TAU_H2 = 6.0

# === Subset controls (change here) ===
SAMPLE_LIMIT = 100       # set to 50–100
SHUFFLE_SEED = 1337      # deterministic sampling

# === Degree caps per hop (big speed win) ===
K1 = 30   # max v per u for H1/H2 E1
K2 = 30   # max w per v for H2 E2

# === Output files (subset suffix) ===
OUT_DIR = f"{PROJECT_DIR}/gen/withKG/RAG/analysis2"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_STATS_GLOBAL = os.path.join(OUT_DIR, "kg_threshold_stats_subset.csv")
OUT_STATS_ADAPT  = os.path.join(OUT_DIR, "kg_adaptive_p75_stats_subset.csv")
OUT_EXAMPLES_DIR = os.path.join(OUT_DIR, "prompt_examples_subset")
os.makedirs(OUT_EXAMPLES_DIR, exist_ok=True)

# ==================== Helpers ====================

def _norm(x) -> str:
    if x is None: return ""
    try:
        if isinstance(x, float) and np.isnan(x): return ""
    except Exception:
        pass
    return str(x).strip()

def _strip(x: str) -> str:
    return str(x or "").strip().upper().replace(" ", "")

def count_tokens(tok, text: str) -> int:
    if not text: return 0
    enc = tok(text, add_special_tokens=False, return_length=True)
    return int(enc["length"][0])

def trim_to_token_budget(tok, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text: return ""
    if count_tokens(tok, text) <= max_tokens: return text
    lo, hi, best = 0, len(text), ""
    while lo <= hi:
        mid = (lo + hi)//2
        cand = text[:mid]
        if count_tokens(tok, cand) <= max_tokens:
            best = cand; lo = mid+1
        else:
            hi = mid-1
    return best

def to_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return []
    if isinstance(x, list): return x
    if isinstance(x, (set, tuple)): return list(x)
    if isinstance(x, str):
        if '|' in x: return [p.strip() for p in x.split('|') if p.strip()]
        if ',' in x: return [p.strip() for p in x.split(',') if p.strip()]
        return [x.strip()] if x.strip() else []
    return [x]

def format_icd9_proc_from_pro(c: str) -> str:
    s = _strip(c)
    if s.startswith("PRO_"): s = s[4:]
    digits = re.sub(r"[^0-9]", "", s)
    if not digits: return ""
    if len(digits) >= 3:
        return digits[:2] + "." + digits[2:]
    return digits

def load_profiles_json(path: str) -> Dict[str,str]:
    if not path or not os.path.exists(path): return {}
    with open(path, "r") as f: return json.load(f)

def _profile_title_fallback(code: str, profiles: Dict[str,str]) -> str:
    raw = profiles.get(code, "")
    if not raw: return ""
    if "::" in raw:
        after = raw.split("::",1)[1].strip()
        return after.split(";",1)[0].strip()
    return raw.strip()

def short_title_for_code(code: str, code2name: Dict[str,str], profiles: Dict[str,str]) -> str:
    t = (code2name or {}).get(code)
    if t: return t
    return _profile_title_fallback(code, profiles) or ""

# ---------------- KG loading (dual edges) ----------------
def load_kg_dual(nodes_csv: str, edges_csv: str, edges_canon_csv: str, min_edge_score: float = 0.0):
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    canon_df = pd.read_csv(edges_canon_csv)

    for need in ("cui_start","cui_target"):
        if need not in edges_df.columns:
            raise ValueError(f"Missing column '{need}' in {edges_csv}")

    keep_canon = canon_df[["cui_start","cui_target","rela_canon","rela_score"]].copy()
    merged = edges_df.merge(keep_canon, on=["cui_start","cui_target"], how="left")

    G = nx.DiGraph()
    node_name, node_sem = {}, {}
    cui_to_icd9 = defaultdict(set)

    for _, r in nodes_df.iterrows():
        cui = str(r.get("cui","")).strip()
        if not cui: continue
        if cui not in G: G.add_node(cui)
        nm  = str(r.get("name","") or "").strip()
        sem = str(r.get("semantic_type","") or "").strip()
        sab = str(r.get("sab","") or "").strip().upper()
        code= str(r.get("code","") or "").strip()
        if nm:  node_name[cui] = nm
        if sem: node_sem[cui]  = sem
        if sab == "ICD9CM" and code:
            cui_to_icd9[cui].add(code)

    for _, r in merged.iterrows():
        u = str(r.get("cui_start","") or "").strip()
        v = str(r.get("cui_target","") or "").strip()
        if not u or not v: continue
        if u not in G: G.add_node(u)
        if v not in G: G.add_node(v)
        rel  = str(r.get("rel","") or "").strip()
        rela = str(r.get("rela","") or "").strip()
        rela_canon = str(r.get("rela_canon","") or "").strip().lower()
        try:
            rela_score = float(r.get("rela_score", np.nan))
        except Exception:
            rela_score = float("nan")
        G.add_edge(u, v, rel=rel, rela=rela, rela_canon=rela_canon, rela_score=rela_score)

    return G, node_name, node_sem, {k: sorted(v) for k,v in cui_to_icd9.items()}

# ---------------- Evidence → CUIs ----------------
def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    src2cuis, ev = {}, set()
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        cuis = atc_map.get(key, [])
        if cuis:
            src2cuis[f"ATC:{key}"] = cuis
            ev.update(cuis)
    for c in to_list(row.get("lab_test_loinc", [])):
        key = _strip(c)
        cuis = loinc_map.get(key, [])
        if cuis:
            src2cuis[f"LNC:{key}"] = cuis
            ev.update(cuis)
    for c in to_list(row.get("pro_code", [])):
        pc = format_icd9_proc_from_pro(c)
        if not pc: continue
        cuis = icd9_proc_map.get(pc, [])
        if cuis:
            src2cuis[f"PROC:{pc}"] = cuis
            ev.update(cuis)
    return src2cuis, ev

# ---------------- Scored path mining with degree caps ----------------
def _top_successors(G, u, k, min_edge_score, rel_bucket_whitelist):
    """Return top-k successors v for u by rela_score with filters applied."""
    cands = []
    for v in G.successors(u):
        d = G[u][v]
        bucket = (d.get("rela_canon") or "").lower()
        escore = d.get("rela_score")
        try:
            escore = float(escore)
        except Exception:
            escore = float("nan")
        if np.isnan(escore) or escore < min_edge_score: 
            continue
        if rel_bucket_whitelist and bucket not in rel_bucket_whitelist:
            continue
        cands.append((v, escore, d))
    cands.sort(key=lambda z: z[1], reverse=True)
    if k is not None and k > 0:
        cands = cands[:k]
    return cands  # list of (v, escore, d)

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
        if u not in G: 
            continue

        # Top-K v for u (E1)
        v_list = _top_successors(G, u, k1, min_edge_score, rel_bucket_whitelist)

        # H1
        for (v, escore, d) in v_list:
            base = escore + (0.25 if len(cui_to_icd9.get(v, [])) > 0 else 0.0)
            H1.append({
                "src_cui": u, "nbr_cui": v,
                "src_name": node_name.get(u,""), "nbr_name": node_name.get(v,""),
                "rel": _norm(d.get("rel","")), "rela": _norm(d.get("rela","")),
                "rela_canon": (d.get("rela_canon") or "").lower(),
                "rela_score": float(escore),
                "score": base
            })

        # H2: for each v, take top-K w for v
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
                    "rel_uv": _norm(d_uv.get("rel","")),  "rela_uv": _norm(d_uv.get("rela","")),
                    "rel_vw": _norm(d_vw.get("rel","")),  "rela_vw": _norm(d_vw.get("rela","")),
                    "w_icd9": anchors,
                    "score": float(score),
                    "s_uv": float(esc_uv), "s_vw": float(esc_vw),
                })

    H1.sort(key=lambda z: z["score"], reverse=True)
    H2.sort(key=lambda z: z["score"], reverse=True)
    return {"H1": H1, "H2": H2}

# ---------------- Selection & Rendering (no budgets) ----------------
def select_paths_by_threshold(paths_all: Dict[str, list], tau_h1: float, tau_h2: float):
    H1_sel = [h for h in paths_all["H1"] if float(h.get("rela_score", 0.0)) >= tau_h1]
    H2_sel = [h for h in paths_all["H2"] if float(h.get("score", 0.0))      >= tau_h2]
    return {"H1": H1_sel, "H2": H2_sel}

def _arrow_label(rela: str, rel: str) -> str:
    r = (rela or "").strip() or (rel or "").strip()
    return f" --{r}--> " if r else " → "

def render_kg_context_no_budget(paths: Dict[str, list], code2name: Dict[str,str], profiles: Dict[str,str]) -> str:
    lines = []
    # H2 first
    lines.append("[KG CONTEXT — H2 PATHS]")
    for c in paths.get("H2", []):
        anchors=[]
        for code in (c.get("w_icd9", []) or []):
            t = short_title_for_code(code, code2name, profiles)
            anchors.append(f"{code} — {t}" if t else code)
        a_str = " | ".join(anchors) if anchors else "-"
        u = c.get("u_name") or c.get("u") or ""
        v = c.get("v_name") or c.get("v") or ""
        w = c.get("w_name") or c.get("w") or ""
        line = f"- {u}{_arrow_label(c.get('rela_uv'), c.get('rel_uv'))}{v}{_arrow_label(c.get('rela_vw'), c.get('rel_vw'))}{w} [ICD-9: {a_str}] : score={c.get('score')}"
        lines.append(line)

    # H1
    lines.append("[KG CONTEXT — H1 PATHS]")
    for c in paths.get("H1", []):
        u = c.get("src_name") or c.get("src_cui") or ""
        v = c.get("nbr_name") or c.get("nbr_cui") or ""
        # bugfix: close the format expression
        lines.append(f"- {u}{_arrow_label(c.get('rela'), c.get('rel'))}{v} : score={c.get('rela_score')}")

    return "\n".join(lines)

# ---- NEW: split renderers to count H2/H1 tokens separately ----
def render_h2_block_no_budget(paths: Dict[str, list], code2name: Dict[str,str], profiles: Dict[str,str]) -> str:
    """Render only the H2 block (with its header) — no token budget clamp."""
    lines = ["[KG CONTEXT — H2 PATHS]"]
    for c in paths.get("H2", []):
        anchors=[]
        for code in (c.get("w_icd9", []) or []):
            t = short_title_for_code(code, code2name, profiles)
            anchors.append(f"{code} — {t}" if t else code)
        a_str = " | ".join(anchors) if anchors else "-"
        u = c.get("u_name") or c.get("u") or ""
        v = c.get("v_name") or c.get("v") or ""
        w = c.get("w_name") or c.get("w") or ""
        lines.append(
            f"- {u}{_arrow_label(c.get('rela_uv'), c.get('rel_uv'))}"
            f"{v}{_arrow_label(c.get('rela_vw'), c.get('rel_vw'))}"
            f"{w} [ICD-9: {a_str}]"
        )
    return "\n".join(lines)

def render_h1_block_no_budget(paths: Dict[str, list]) -> str:
    """Render only the H1 block (with its header) — no token budget clamp."""
    lines = ["[KG CONTEXT — H1 PATHS]"]
    for c in paths.get("H1", []):
        u = c.get("src_name") or c.get("src_cui") or ""
        v = c.get("nbr_name") or c.get("nbr_cui") or ""
        lines.append(f"- {u}{_arrow_label(c.get('rela'), c.get('rel'))}{v}")
    return "\n".join(lines)

def build_tail(n_max_terms:int=N_MAX_TERMS) -> str:
    return "\n".join([
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {n_max_terms} lines",
        "[OUTPUT]"
    ])

def summarize(df, col):
    return {
        "min": float(df[col].min()),
        "p25": float(df[col].quantile(0.25)),
        "p50": float(df[col].quantile(0.50)),
        "p75": float(df[col].quantile(0.75)),
        "p90": float(df[col].quantile(0.90)),
        "max": float(df[col].max()),
        "mean": float(df[col].mean()),
    }

def propose_row_thresholds(H2_rows: List[dict]):
    if not H2_rows:
        return {"mean":0.0,"p50":0.0,"p75":0.0,"p90":0.0}
    s = np.array([h["score"] for h in H2_rows], dtype=float)
    return {
        "mean": float(s.mean()),
        "p50": float(np.quantile(s, 0.50)),
        "p75": float(np.quantile(s, 0.75)),
        "p90": float(np.quantile(s, 0.90)),
    }

# ==================== Main ====================
def main():
    print(f"[INFO] Using whitelist: {REL_BUCKET_WHITELIST} | MIN_EDGE_SCORE={MIN_EDGE_SCORE}")
    print(f"[INFO] SAMPLE_LIMIT={SAMPLE_LIMIT}, K1={K1}, K2={K2}")

    # Load KG
    print("[INFO] Loading KG...")
    G, node_name, node_sem, cui_to_icd9 = load_kg_dual(
        KG_NODES_CSV, KG_EDGES_CSV, KG_EDGES_CANON_CSV, min_edge_score=0.0
    )
    print(f"KG: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"CUIs with ICD9 anchors: {sum(1 for v in cui_to_icd9.values() if v)}")

    # Maps
    print("[INFO] Loading code2cui maps...")
    with open(ICD9_PROC_MAP, "rb") as f: icd9_proc_map = pickle.load(f)
    with open(LOINC_MAP, "rb") as f:     loinc_map     = pickle.load(f)
    with open(ATC_MAP, "rb") as f:       atc_map       = pickle.load(f)

    # Titles
    print("[INFO] Loading code2name and profiles...")
    code2name = {}
    if os.path.exists(CODE2NAME_PKL):
        with open(CODE2NAME_PKL, "rb") as f: code2name = pickle.load(f)
    profiles = load_profiles_json(ICD9_PROFILES_JS)

    # Tokenizer
    print("[INFO] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Dataset
    print("[INFO] Loading dataset...")
    try:
        df = pd.read_pickle(DATA_PKL)
    except Exception:
        with open(DATA_PKL, "rb") as f: df = pickle.load(f)
    print("Dataset shape:", df.shape)

    # --------- Choose subset rows ---------
    all_idx = list(range(len(df)))
    random.Random(SHUFFLE_SEED).shuffle(all_idx)

    # Optional sharding via env (no argparse)
    num_shards = int(os.getenv("NUM_SHARDS", "1"))
    shard_idx  = int(os.getenv("SHARD_IDX", "0"))
    shard_idx  = max(0, min(shard_idx, num_shards-1))

    shard_idxs = [i for i in all_idx if i % num_shards == shard_idx]
    subset = shard_idxs[:SAMPLE_LIMIT]
    subset = sorted(subset)

    print(f"[INFO] num_shards={num_shards} shard_idx={shard_idx}")
    print(f"[INFO] Processing subset of {len(subset)} rows: first 5 -> {subset[:5]}")

    # ---------- GLOBAL THRESHOLDS RUN (subset) ----------
    rows_stats = []
    for ridx in tqdm(subset, desc="Global thresholds (subset)"):
        row = df.iloc[ridx]

        # evidence → CUIs
        _, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

        # mine ALL (with degree caps + filters)
        paths_all = mine_paths_scored_all_capped(
            G=G,
            node_name=node_name,
            cui_to_icd9=cui_to_icd9,
            ev_cuis=set(ev_cuis),
            rel_bucket_whitelist=REL_BUCKET_WHITELIST,
            min_edge_score=MIN_EDGE_SCORE,
            k1=K1,
            k2=K2,
        )

        # select by fixed thresholds
        paths_sel = select_paths_by_threshold(paths_all, TAU_H1, TAU_H2)

        # render full KG context (no trimming)
        kg_text = render_kg_context_no_budget(paths_sel, code2name, profiles)
        # ---- NEW: split KG tokens into H2/H1
        h2_text = render_h2_block_no_budget(paths_sel, code2name, profiles)
        h1_text = render_h1_block_no_budget(paths_sel)
        kg_tokens    = count_tokens(tok, kg_text)
        kg_h2_tokens = count_tokens(tok, h2_text)
        kg_h1_tokens = count_tokens(tok, h1_text)

        # prompts for token accounting
        header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
        notes  = serialize_notes(row)
        tail   = build_tail()

        raw_prompt = "\n".join([p for p in (header, notes, tail) if p])
        kg_prompt  = "\n".join([p for p in (header, notes, kg_text, tail) if p])

        rows_stats.append({
            "row_idx": ridx,
            "hadm_id": row.get("hadm_id",""),
            "ev_cuis": len(ev_cuis),
            "H1_lines": len(paths_sel["H1"]),
            "H2_lines": len(paths_sel["H2"]),
            "KG_tokens": kg_tokens,
            "KG_H2_tokens": kg_h2_tokens,   # NEW
            "KG_H1_tokens": kg_h1_tokens,   # NEW
            "RAW_tokens": count_tokens(tok, raw_prompt),
            "WITHKG_tokens": count_tokens(tok, kg_prompt),
            "DELTA_tokens": count_tokens(tok, kg_prompt) - count_tokens(tok, raw_prompt),
        })

        # Save a couple of examples
        if len(rows_stats) <= 5:
            with open(os.path.join(OUT_EXAMPLES_DIR, f"row{ridx}_global_kg_context.txt"), "w") as f:
                f.write(kg_text)
            with open(os.path.join(OUT_EXAMPLES_DIR, f"row{ridx}_global_raw_prompt.txt"), "w") as f:
                f.write(raw_prompt)
            with open(os.path.join(OUT_EXAMPLES_DIR, f"row{ridx}_global_withkg_prompt.txt"), "w") as f:
                f.write(kg_prompt)

    STATS = pd.DataFrame(rows_stats)
    STATS.to_csv(OUT_STATS_GLOBAL, index=False)
    print(f"[OK] Wrote global threshold stats (subset) -> {OUT_STATS_GLOBAL}"))

    for col in ["H1_lines","H2_lines","KG_tokens","KG_H2_tokens","KG_H1_tokens","RAW_tokens","WITHKG_tokens","DELTA_tokens"]:
        print(col, summarize(STATS, col))
    OVER = STATS[STATS["WITHKG_tokens"] > MAX_LEN]
    print(f"Rows exceeding {MAX_LEN} tokens (no trimming): {len(OVER)}")

    # ---------- ADAPTIVE (per-row H2 @ p75) RUN (subset) ----------
    adapt_rows = []
    for ridx in tqdm(subset, desc="Adaptive p75 (subset)"):
        row = df.iloc[ridx]
        _, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

        paths_all = mine_paths_scored_all_capped(
            G=G,
            node_name=node_name,
            cui_to_icd9=cui_to_icd9,
            ev_cuis=set(ev_cuis),
            rel_bucket_whitelist=REL_BUCKET_WHITELIST,
            min_edge_score=MIN_EDGE_SCORE,
            k1=K1,
            k2=K2,
        )

        thr = propose_row_thresholds(paths_all["H2"])
        paths_sel = select_paths_by_threshold(paths_all, tau_h1=1.0, tau_h2=thr["p75"])
        kg_text = render_kg_context_no_budget(paths_sel, code2name, profiles)
        # ---- NEW: split KG tokens into H2/H1
        h2_text = render_h2_block_no_budget(paths_sel, code2name, profiles)
        h1_text = render_h1_block_no_budget(paths_sel)
        kg_tokens    = count_tokens(tok, kg_text)
        kg_h2_tokens = count_tokens(tok, h2_text)
        kg_h1_tokens = count_tokens(tok, h1_text)

        header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
        notes  = serialize_notes(row)
        tail   = build_tail()

        raw_prompt = "\n".join([p for p in (header, notes, tail) if p])
        kg_prompt  = "\n".join([p for p in (header, notes, kg_text, tail) if p])

        adapt_rows.append({
            "row_idx": ridx,
            "H1_lines": len(paths_sel["H1"]),
            "H2_lines": len(paths_sel["H2"]),
            "KG_tokens": kg_tokens,
            "KG_H2_tokens": kg_h2_tokens,   # NEW
            "KG_H1_tokens": kg_h1_tokens,   # NEW
            "RAW_tokens": count_tokens(tok, raw_prompt),
            "WITHKG_tokens": count_tokens(tok, kg_prompt),
            "H2_tau_used": thr["p75"],
        })

        if len(adapt_rows) <= 3:
            with open(os.path.join(OUT_EXAMPLES_DIR, f"row{ridx}_adapt_kg_context.txt"), "w") as f:
                f.write(kg_text)

    ADAPT = pd.DataFrame(adapt_rows)
    ADAPT.to_csv(OUT_STATS_ADAPT, index=False)
    print(f"[OK] Wrote adaptive (H2@p75) stats (subset) -> {OUT_STATS_ADAPT}")

    print("Adaptive summary (H2@p75):")
    for c in ["H1_lines","H2_lines","KG_tokens","KG_H2_tokens","KG_H1_tokens","WITHKG_tokens"]:
        print(c, summarize(ADAPT, c))

if __name__ == "__main__":
    main()
