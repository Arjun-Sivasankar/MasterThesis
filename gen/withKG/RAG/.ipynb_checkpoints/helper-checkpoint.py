#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, re, math, pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
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
print(f"[INFO] Using the rela bucket whitelist: {REL_BUCKET_WHITELIST}")
MIN_EDGE_SCORE = 0.4

# === Global thresholds for selection ===
TAU_H1 = 1.0   # keep H1 if edge rela_score >= 1.0
TAU_H2 = 2.2   # keep H2 if (s_uv + s_vw + anchor_bonus) >= 2.2

# === Output files ===
OUT_DIR = f"{PROJECT_DIR}/gen/withKG/RAG/analysis"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_STATS_GLOBAL = os.path.join(OUT_DIR, "kg_threshold_stats.csv")
OUT_STATS_ADAPT  = os.path.join(OUT_DIR, "kg_adaptive_p75_stats.csv")
OUT_EXAMPLES_DIR = os.path.join(OUT_DIR, "prompt_examples")
os.makedirs(OUT_EXAMPLES_DIR, exist_ok=True)

# ---------------- Globals built once in main() and shared by workers (via fork) ----------------
EDGES = None           # filtered edges table
NAME_MAP = None        # CUI -> name dataframe
ANCHOR_COUNTS = None   # CUI -> ICD9 anchor count series->df
cui_to_icd9 = None     # dict CUI -> [ICD-9 codes]

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

# cache + batch token counting
_TOKEN_CACHE = {}

def count_tokens_cached(tok, text: str) -> int:
    if not text: return 0
    h = hash(text)
    if h in _TOKEN_CACHE: return _TOKEN_CACHE[h]
    n = count_tokens(tok, text)
    _TOKEN_CACHE[h] = n
    return n

def count_tokens_batch(tok, texts: List[str]) -> List[int]:
    if not texts: return []
    enc = tok(texts, add_special_tokens=False, return_length=True)
    return [int(x) for x in enc["length"]]

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
    """
    Build directed KG with canonical scores merged onto edges.
    - Nodes: CUI with {name, semantic_type} and collect CUI -> ICD9 codes
    - Edges: u->v with attrs {rel, rela, rela_canon, rela_score}
    """
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    canon_df = pd.read_csv(edges_canon_csv)

    # merge '(cui_start, cui_target)' with canon fields
    for need in ("cui_start","cui_target"):
        if need not in edges_df.columns:
            raise ValueError(f"Missing column '{need}' in {edges_csv}")

    keep_canon = canon_df[["cui_start","cui_target","rela_canon","rela_score"]].copy()
    merged = edges_df.merge(
        keep_canon, on=["cui_start","cui_target"], how="left", suffixes=("","")
    )

    # Build graph
    G = nx.DiGraph()
    node_name, node_sem = {}, {}
    cui_to_icd9_local = defaultdict(set)

    # Nodes
    for _, r in nodes_df.iterrows():
        cui = str(r.get("cui","")).strip()
        if not cui: continue
        if cui not in G:
            G.add_node(cui)
        nm  = str(r.get("name","") or "").strip()
        sem = str(r.get("semantic_type","") or "").strip()
        sab = str(r.get("sab","") or "").strip().upper()
        code= str(r.get("code","") or "").strip()

        if nm:  node_name[cui] = nm
        if sem: node_sem[cui]  = sem
        if sab == "ICD9CM" and code:
            cui_to_icd9_local[cui].add(code)

    # Edges
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

    return G, node_name, node_sem, {k: sorted(v) for k,v in cui_to_icd9_local.items()}

# ---------------- Evidence → CUIs ----------------

def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    src2cuis, ev = {}, set()

    # ATC from 'ndc' (your dataset)
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        cuis = atc_map.get(key, [])
        if cuis:
            src2cuis[f"ATC:{key}"] = cuis
            ev.update(cuis)

    # LOINC
    for c in to_list(row.get("lab_test_loinc", [])):
        key = _strip(c)
        cuis = loinc_map.get(key, [])
        if cuis:
            src2cuis[f"LNC:{key}"] = cuis
            ev.update(cuis)

    # ICD9-Proc via 'pro_code'
    for c in to_list(row.get("pro_code", [])):
        pc = format_icd9_proc_from_pro(c)
        if not pc: continue
        cuis = icd9_proc_map.get(pc, [])
        if cuis:
            src2cuis[f"PROC:{pc}"] = cuis
            ev.update(cuis)

    return src2cuis, ev

# ---------------- Vectorized mining helpers (built once) ----------------

def attach_names(df: pd.DataFrame, col: str, outcol: str) -> pd.DataFrame:
    """Join NAME_MAP to df[col] -> outcol"""
    global NAME_MAP
    return df.merge(
        NAME_MAP.rename(columns={"cui": col, "name": outcol}),
        on=col, how="left"
    )

def icd_list_from_dict(cui: str) -> List[str]:
    global cui_to_icd9
    return cui_to_icd9.get(cui, [])

# ---------------- Vectorized mine ALL scored paths (no per-src caps) ----------------

def mine_paths_scored_all(
    G_unused,                   # kept for signature compatibility
    node_name_unused,
    cui_to_icd9_unused,
    ev_cuis: Set[str],
    rel_bucket_whitelist: Set[str],
    min_edge_score: float
):
    """
    Vectorized version using global EDGES/NAME_MAP/ANCHOR_COUNTS.
      H1: EDGES where u in ev_cuis, score = rela_score + 0.25*(has ICD9 on v)
      H2: self-join E1(u->v) join E2(v->w), score = s_uv + s_vw + 1.5*anchor_cnt(w)
    """
    global EDGES, ANCHOR_COUNTS

    if not ev_cuis:
        return {"H1": [], "H2": []}
    ev_cuis = set(ev_cuis)

    # --- H1
    H1 = EDGES[EDGES["u"].isin(ev_cuis)].copy()
    H1 = H1.merge(ANCHOR_COUNTS.rename(columns={"cui":"v"}), on="v", how="left")
    H1["anchor_cnt"] = H1["anchor_cnt"].fillna(0.0)
    H1["score"] = H1["rela_score"].astype(float) + (H1["anchor_cnt"] > 0).astype(float) * 0.25

    # names
    H1 = attach_names(H1, "u", "src_name")
    H1 = attach_names(H1, "v", "nbr_name")

    # --- H2
    E1 = EDGES[EDGES["u"].isin(ev_cuis)].copy()
    E2 = EDGES.copy()
    H2 = E1.merge(E2, left_on="v", right_on="u", suffixes=("_uv","_vw"))

    # Add anchor counts for w (which is E2.v -> we'll rename to v_w)
    H2 = H2.rename(columns={"u_uv":"u_uv", "v":"v", "u_vw":"u_vw", "v_vw":"v_w"})
    H2 = H2.merge(ANCHOR_COUNTS.rename(columns={"cui":"v_w"}), on="v_w", how="left")
    H2["anchor_cnt"] = H2["anchor_cnt"].fillna(0.0)
    H2["score"] = H2["rela_score_uv"].astype(float) + H2["rela_score_vw"].astype(float) + 1.5 * H2["anchor_cnt"]

    # names
    H2 = attach_names(H2, "u_uv", "u_name")
    H2 = attach_names(H2, "v",    "v_name")
    H2 = attach_names(H2, "v_w",  "w_name")

    # Prepare outputs in your dict-of-lists shape
    H1_records = []
    for r in H1.itertuples(index=False):
        H1_records.append({
            "src_cui": r.u, "nbr_cui": r.v,
            "src_name": r.src_name or "", "nbr_name": r.nbr_name or "",
            "rel": r.rel, "rela": r.rela,
            "rela_canon": r.rela_canon, "rela_score": float(r.rela_score),
            "score": float(r.score)
        })

    H2_records = []
    for r in H2.itertuples(index=False):
        w_codes = icd_list_from_dict(r.v_w)
        H2_records.append({
            "u": r.u_uv, "v": r.v, "w": r.v_w,
            "u_name": r.u_name or "", "v_name": r.v_name or "", "w_name": r.w_name or "",
            "rel_uv": r.rel_uv,  "rela_uv": r.rela_uv,
            "rel_vw": r.rel_vw,  "rela_vw": r.rela_vw,
            "w_icd9": w_codes,
            "score": float(r.score),
            "s_uv": float(r.rela_score_uv), "s_vw": float(r.rela_score_vw),
        })

    H1_records.sort(key=lambda z: z["score"], reverse=True)
    H2_records.sort(key=lambda z: z["score"], reverse=True)
    return {"H1": H1_records, "H2": H2_records}

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
        line = f"- {u}{_arrow_label(c.get('rela_uv'), c.get('rel_uv'))}{v}{_arrow_label(c.get('rela_vw'), c.get('rel_vw'))}{w} [ICD-9: {a_str}]"
        lines.append(line)

    # H1
    lines.append("[KG CONTEXT — H1 PATHS]")
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

# ---------------- Row workers (parallel) ----------------

def process_row(args):
    """
    Global thresholds worker (TAU_H1/TAU_H2).
    Returns minimal info + text for token counting done in parent.
    """
    (ridx, row_dict, icd9_proc_map, loinc_map, atc_map, code2name, profiles) = args
    row = pd.Series(row_dict)

    # evidence → CUIs
    _, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

    paths_all = mine_paths_scored_all(
        G_unused=None, node_name_unused=None, cui_to_icd9_unused=None,
        ev_cuis=set(ev_cuis),
        rel_bucket_whitelist=REL_BUCKET_WHITELIST,
        min_edge_score=MIN_EDGE_SCORE
    )

    paths_sel = select_paths_by_threshold(paths_all, TAU_H1, TAU_H2)
    kg_text = render_kg_context_no_budget(paths_sel, code2name, profiles)

    header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
    notes  = serialize_notes(row)
    tail   = build_tail()

    raw_prompt = "\n".join([p for p in (header, notes, tail) if p])
    kg_prompt  = "\n".join([p for p in (header, notes, kg_text, tail) if p])

    return {
        "ridx": ridx,
        "hadm_id": row.get("hadm_id",""),
        "ev_cuis": len(ev_cuis),
        "H1_lines": len(paths_sel["H1"]),
        "H2_lines": len(paths_sel["H2"]),
        "kg_text": kg_text,
        "raw_prompt": raw_prompt,
        "kg_prompt": kg_prompt,
    }

def process_row_adapt(args):
    """
    Adaptive per-row worker (H2@p75).
    """
    (ridx, row_dict, icd9_proc_map, loinc_map, atc_map, code2name, profiles) = args
    row = pd.Series(row_dict)

    _, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
    paths_all = mine_paths_scored_all(
        G_unused=None, node_name_unused=None, cui_to_icd9_unused=None,
        ev_cuis=set(ev_cuis),
        rel_bucket_whitelist=REL_BUCKET_WHITELIST,
        min_edge_score=MIN_EDGE_SCORE
    )
    thr = propose_row_thresholds(paths_all["H2"])
    paths_sel = select_paths_by_threshold(paths_all, tau_h1=1.0, tau_h2=thr["p75"])
    kg_text = render_kg_context_no_budget(paths_sel, code2name, profiles)

    header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
    notes  = serialize_notes(row)
    tail   = build_tail()

    raw_prompt = "\n".join([p for p in (header, notes, tail) if p])
    kg_prompt  = "\n".join([p for p in (header, notes, kg_text, tail) if p])

    return {
        "ridx": ridx,
        "H1_lines": len(paths_sel["H1"]),
        "H2_lines": len(paths_sel["H2"]),
        "kg_text": kg_text,
        "raw_prompt": raw_prompt,
        "kg_prompt": kg_prompt,
        "H2_tau_used": thr["p75"],
    }

# ==================== Main ====================

def main():
    global EDGES, NAME_MAP, ANCHOR_COUNTS, cui_to_icd9

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

    # === Prebuild vectorized edge tables (faster than NetworkX walks) ===
    print("[INFO] Prebuilding edge tables for vectorized mining...")

    edges_list = []
    for u, v, d in G.edges(data=True):
        edges_list.append({
            "u": u,
            "v": v,
            "rel": d.get("rel",""),
            "rela": d.get("rela",""),
            "rela_canon": (d.get("rela_canon","") or "").lower(),
            "rela_score": d.get("rela_score", np.nan),
        })
    EDGES = pd.DataFrame(edges_list)

    # Filter to clinically-meaningful & scored edges
    EDGES = EDGES[
        (EDGES["rela_score"].fillna(-1.0) >= MIN_EDGE_SCORE) &
        (EDGES["rela_canon"].isin(REL_BUCKET_WHITELIST))
    ].copy()

    # Map CUI -> name (df for joins)
    NAME_MAP = pd.Series(node_name, name="name").rename_axis("cui").reset_index()

    # Precompute anchor counts for every node
    anchor_counts = pd.Series({k: len(v) for k, v in cui_to_icd9.items()}, name="anchor_cnt")
    anchor_counts.index.name = "cui"
    ANCHOR_COUNTS = anchor_counts.reset_index()

    print(f"[INFO] EDGES filtered: {len(EDGES):,} rows")

    # ---------- GLOBAL THRESHOLDS RUN (parallel) ----------
    print("[INFO] Global thresholds pass (parallel)...")
    rows_stats = []
    examples_written = 0

    args_iter = [
        (int(i), df.iloc[i].to_dict(), icd9_proc_map, loinc_map, atc_map, code2name, profiles)
        for i in range(len(df))
    ]

    workers = max(1, cpu_count()-1)
    with Pool(processes=workers) as pool:
        for out in tqdm(pool.imap_unordered(process_row, args_iter, chunksize=32), total=len(args_iter)):
            kg_text     = out["kg_text"]
            raw_prompt  = out["raw_prompt"]
            kg_prompt   = out["kg_prompt"]

            # batch token count
            KG_tokens, RAW_tokens, WITHKG_tokens = count_tokens_batch(tok, [kg_text, raw_prompt, kg_prompt])

            rows_stats.append({
                "row_idx": out["ridx"],
                "hadm_id": out["hadm_id"],
                "ev_cuis": out["ev_cuis"],
                "H1_lines": out["H1_lines"],
                "H2_lines": out["H2_lines"],
                "KG_tokens": KG_tokens,
                "RAW_tokens": RAW_tokens,
                "WITHKG_tokens": WITHKG_tokens,
                "DELTA_tokens": WITHKG_tokens - RAW_tokens,
            })

            # Save a few example prompts
            if examples_written < 5:
                with open(os.path.join(OUT_EXAMPLES_DIR, f"row{out['ridx']}_global_kg_context.txt"), "w") as f:
                    f.write(kg_text)
                with open(os.path.join(OUT_EXAMPLES_DIR, f"row{out['ridx']}_global_raw_prompt.txt"), "w") as f:
                    f.write(raw_prompt)
                with open(os.path.join(OUT_EXAMPLES_DIR, f"row{out['ridx']}_global_withkg_prompt.txt"), "w") as f:
                    f.write(kg_prompt)
                examples_written += 1

    STATS = pd.DataFrame(rows_stats).sort_values("row_idx").reset_index(drop=True)
    STATS.to_csv(OUT_STATS_GLOBAL, index=False)
    print(f"[OK] Wrote global threshold stats -> {OUT_STATS_GLOBAL}")

    # Print summaries
    for col in ["H1_lines","H2_lines","KG_tokens","RAW_tokens","WITHKG_tokens","DELTA_tokens"]:
        print(col, summarize(STATS, col))
    OVER = STATS[STATS["WITHKG_tokens"] > MAX_LEN]
    print(f"Rows exceeding {MAX_LEN} tokens (no trimming): {len(OVER)}")

    # ---------- ADAPTIVE (per-row H2 @ p75) RUN (parallel) ----------
    print("[INFO] Adaptive p75 pass (parallel)...")
    adapt_rows = []
    examples_written = 0

    with Pool(processes=workers) as pool:
        for out in tqdm(pool.imap_unordered(process_row_adapt, args_iter, chunksize=32), total=len(args_iter)):
            kg_text     = out["kg_text"]
            raw_prompt  = out["raw_prompt"]
            kg_prompt   = out["kg_prompt"]

            KG_tokens, RAW_tokens, WITHKG_tokens = count_tokens_batch(tok, [kg_text, raw_prompt, kg_prompt])

            adapt_rows.append({
                "row_idx": out["ridx"],
                "H1_lines": out["H1_lines"],
                "H2_lines": out["H2_lines"],
                "KG_tokens": KG_tokens,
                "RAW_tokens": RAW_tokens,
                "WITHKG_tokens": WITHKG_tokens,
                "H2_tau_used": out["H2_tau_used"],
            })

            if examples_written < 3:
                with open(os.path.join(OUT_EXAMPLES_DIR, f"row{out['ridx']}_adapt_kg_context.txt"), "w") as f:
                    f.write(kg_text)
                examples_written += 1

    ADAPT = pd.DataFrame(adapt_rows).sort_values("row_idx").reset_index(drop=True)
    ADAPT.to_csv(OUT_STATS_ADAPT, index=False)
    print(f"[OK] Wrote adaptive (H2@p75) stats -> {OUT_STATS_ADAPT}")

    print("Adaptive summary (H2@p75):")
    for c in ["H1_lines","H2_lines","KG_tokens","WITHKG_tokens"]:
        print(c, summarize(ADAPT, c))

if __name__ == "__main__":
    main()
