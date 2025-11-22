# In [1]
import os, sys, json, re, math, pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from transformers import AutoTokenizer

# allow importing your shared utils
PROJECT_DIR = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis"
sys.path.append(PROJECT_DIR)

from common_textgen import serialize_structured_readable, serialize_notes

# === Assets (adjust if needed) ===
KG_NODES_CSV      = f"{PROJECT_DIR}/KG/kg_output4/kg_nodes.csv"
KG_EDGES_CSV      = f"{PROJECT_DIR}/KG/kg_output4/kg_edges.csv"
KG_EDGES_CANON_CSV= f"{PROJECT_DIR}/KG/kg_output4/kg_edges_canon.csv"  # produced by your kg_rel_canon.py

ICD9_PROC_MAP = f"{PROJECT_DIR}/KG/kg_output4/code2cui_icd9_proc.pkl"
LOINC_MAP     = f"{PROJECT_DIR}/KG/kg_output4/code2cui_loinc.pkl"
ATC_MAP       = f"{PROJECT_DIR}/KG/kg_output4/code2cui_atc.pkl"

CODE2NAME_PKL    = f"{PROJECT_DIR}/gen/withKG/RAG/kg_recommender/code2name.pkl"        # primary titles (if built)
ICD9_PROFILES_JS = f"{PROJECT_DIR}/gen/withKG/RAG/kg_recommender/icd9_profiles.json"   # fallback titles (optional)

DATA_PKL    = f"{PROJECT_DIR}/dataset/final_data/test_df.pkl"   # or your file
BASE_MODEL  = f"{PROJECT_DIR}/models/Llama-3.1-8B-Instruct"  # tokenizer only for budgeting

# Prompt budgets (align with SLURM script you use)
MAX_LEN        = 4096
KG_HINT_BUDGET = 600   # tokens reserved for [KG CONTEXT]
N_MAX_TERMS    = 12    # lines parsed after [OUTPUT]

# For mining/scoring (show only the clinically meaningful buckets)
REL_BUCKET_WHITELIST = {"etiology","finding_site","morphology","pathology","proc_site","proc_method","measurement"}
MIN_EDGE_SCORE       = 0.4  # drop weaker/meta wiring

# caps per evidence source
H1_PER_SRC = 3
H2_PER_SRC = 50   # we will hard-cap globally by tokens later anyway


## 2) Small Helpers:
# In [2]
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

## 3) Load KG:
# In [2]
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


## 4) Evidence --> CUIs:
# In [4]
def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    """
    Returns:
      src2cuis: { "ATC:XXX": [...], "LNC:YYY":[...], "PROC:12.34":[...] }
      ev_union: set(CUIs)
    """
    src2cuis, ev = {}, set()

    # ATC (we're using the 'ndc' column in your dataset to hold ATC-like tokens)
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


## 5) Scored H1/H2 path miner (uses rela_canon/rela_score already in the canon CSV):
# In [5]
def mine_paths_scored(
    G: nx.DiGraph,
    node_name: Dict[str,str],
    cui_to_icd9: Dict[str, List[str]],
    ev_cuis: Set[str],
    rel_bucket_whitelist: Set[str] = None,
    min_edge_score: float = 0.2,
    h1_per_src: int = 3,
    h2_per_src: int = 50,
):
    """
    Scores:
      H1: score = rela_score + small bonus if nbr has ICD9 anchor
      H2: score = sum(rela_scores) + 1.5 * (#anchors on w)
      (meta & weak edges are already filtered by MIN_EDGE_SCORE)
    """
    H1, H2 = [], []
    for u in ev_cuis:
        if u not in G: continue

        # H1
        h1_cands=[]
        for v in G.successors(u):
            d = G[u][v]
            bucket = (d.get("rela_canon") or "").lower()
            escore = float(d.get("rela_score", 0.2))
            if escore < min_edge_score: continue
            if rel_bucket_whitelist and bucket not in rel_bucket_whitelist: continue
            base = escore + (0.25 if len(cui_to_icd9.get(v, []))>0 else 0.0)
            h1_cands.append({
                "src_cui": u, "nbr_cui": v,
                "src_name": node_name.get(u,""), "nbr_name": node_name.get(v,""),
                "rel": _norm(d.get("rel","")), "rela": _norm(d.get("rela","")),
                "rela_canon": bucket, "rela_score": escore, "score": base
            })
        h1_cands.sort(key=lambda z: z["score"], reverse=True)
        H1.extend(h1_cands[:h1_per_src])

        # H2
        h2_cands=[]
        for v in G.successors(u):
            d_uv = G[u][v]
            bucket_uv = (d_uv.get("rela_canon") or "").lower()
            esc_uv    = float(d_uv.get("rela_score", 0.2))
            if esc_uv < min_edge_score: continue
            if rel_bucket_whitelist and bucket_uv not in rel_bucket_whitelist: continue

            for w in G.successors(v):
                d_vw = G[v][w]
                bucket_vw = (d_vw.get("rela_canon") or "").lower()
                esc_vw    = float(d_vw.get("rela_score", 0.2))
                if esc_vw < min_edge_score: continue
                if rel_bucket_whitelist and bucket_vw not in rel_bucket_whitelist: continue

                anchors = cui_to_icd9.get(w, [])
                score = (esc_uv + esc_vw) + 1.5 * len(set(anchors))
                h2_cands.append({
                    "u": u, "v": v, "w": w,
                    "u_name": node_name.get(u,""),
                    "v_name": node_name.get(v,""),
                    "w_name": node_name.get(w,""),
                    "rel_uv": _norm(d_uv.get("rel","")),  "rela_uv": _norm(d_uv.get("rela","")),
                    "rel_vw": _norm(d_vw.get("rel","")),  "rela_vw": _norm(d_vw.get("rela","")),
                    "w_icd9": anchors,
                    "score": score,
                })
        h2_cands.sort(key=lambda z: z["score"], reverse=True)
        H2.extend(h2_cands[:h2_per_src])

    H1.sort(key=lambda z: z["score"], reverse=True)
    H2.sort(key=lambda z: z["score"], reverse=True)
    return {"H1": H1, "H2": H2}


## 6) Render KG context with relation labels & budgets:
# In [6]
def render_kg_context_paths(
    tok,
    paths: Dict[str, list],
    code2name: Dict[str, str],
    profiles: Dict[str, str],
    budget_tokens: int = 600,
    h2_first_ratio: float = 0.7
) -> str:
    def _rela_or_rel(rela: str, rel: str) -> str:
        rela = (rela or "").strip(); rel = (rel or "").strip()
        return rela if rela else (rel if rel else "")

    if budget_tokens <= 0: return ""
    h2_budget = int(budget_tokens * h2_first_ratio)
    h1_budget = budget_tokens - h2_budget

    # H2 first
    h2_lines = ["[KG CONTEXT — H2 PATHS]"]; seen_h2=set()
    for c in paths.get("H2", []):
        r1 = _rela_or_rel(c.get("rela_uv",""), c.get("rel_uv",""))
        r2 = _rela_or_rel(c.get("rela_vw",""), c.get("rel_vw",""))
        arrow1 = f" --{r1}--> " if r1 else " → "
        arrow2 = f" --{r2}--> " if r2 else " → "

        anchors=[]
        for code in c.get("w_icd9", []) or []:
            t = short_title_for_code(code, code2name, profiles)
            anchors.append(f"{code} — {t}" if t else f"{code}")
        a_str = " | ".join(anchors) if anchors else "-"

        u = c.get("u_name") or c.get("u") or ""
        v = c.get("v_name") or c.get("v") or ""
        w = c.get("w_name") or c.get("w") or ""
        line = f"- {u}{arrow1}{v}{arrow2}{w} [ICD-9: {a_str}]"

        if line in seen_h2: continue
        trial = "\n".join(h2_lines + [line])
        if count_tokens(tok, trial) <= h2_budget:
            h2_lines.append(line); seen_h2.add(line)
        else:
            break
    h2_block = "\n".join(h2_lines) if len(h2_lines) > 1 else ""

    # H1
    h1_lines = ["[KG CONTEXT — H1 PATHS]"]; seen_h1=set()
    for c in paths.get("H1", []):
        rel_lab = _rela_or_rel(c.get("rela",""), c.get("rel",""))
        arrow = f" --{rel_lab}--> " if rel_lab else " → "
        u = c.get("src_name") or c.get("src_cui") or ""
        v = c.get("nbr_name") or c.get("nbr_cui") or ""
        line = f"- {u}{arrow}{v}"
        if line in seen_h1: continue
        trial = "\n".join(h1_lines + [line])
        if count_tokens(tok, trial) <= h1_budget:
            h1_lines.append(line); seen_h1.add(line)
        else:
            break
    h1_block = "\n".join(h1_lines) if len(h1_lines) > 1 else ""

    combo = "\n".join([b for b in (h2_block, h1_block) if b])
    if count_tokens(tok, combo) > budget_tokens:
        combo = trim_to_token_budget(tok, combo, budget_tokens)
    return combo

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


## 7) Load assets (KG, maps, tokeniser, data):
# In [7]
G, node_name, node_sem, cui_to_icd9 = load_kg_dual(
    KG_NODES_CSV, KG_EDGES_CSV, KG_EDGES_CANON_CSV, min_edge_score=MIN_EDGE_SCORE
)
print(f"KG: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"CUIs with ICD9 anchors: {sum(1 for v in cui_to_icd9.values() if v)}")

with open(ICD9_PROC_MAP, "rb") as f: icd9_proc_map = pickle.load(f)
with open(LOINC_MAP, "rb") as f:     loinc_map     = pickle.load(f)
with open(ATC_MAP, "rb") as f:       atc_map       = pickle.load(f)

code2name = {}
if os.path.exists(CODE2NAME_PKL):
    with open(CODE2NAME_PKL, "rb") as f: code2name = pickle.load(f)
profiles = load_profiles_json(ICD9_PROFILES_JS)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

# dataset
try:
    df = pd.read_pickle(DATA_PKL)
except Exception:
    with open(DATA_PKL, "rb") as f: df = pickle.load(f)

print("Dataset shape:", df.shape)


## 8) Pick a row, mine paths, render context, build prompts & count:
# In [8]
i = 0  # change row index here to inspect different visits
row = df.iloc[i]

# budget accounting
tail = build_tail(N_MAX_TERMS)
header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
base_tokens = count_tokens(tok, header) + count_tokens(tok, tail)

# structured -> CUIs
src2cuis, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
print("Structured → CUIs (truncated):")
for k,v in list(src2cuis.items())[:8]:
    print(f"  {k}: {v[:8]}{' ...' if len(v)>8 else ''}")
print("Union evidence CUIs:", len(ev_cuis))

# score & mine top paths
paths = mine_paths_scored(
    G=G,
    node_name=node_name,
    cui_to_icd9=cui_to_icd9,
    ev_cuis=set(ev_cuis),
    rel_bucket_whitelist=REL_BUCKET_WHITELIST,
    min_edge_score=MIN_EDGE_SCORE,
    h1_per_src=H1_PER_SRC,
    h2_per_src=H2_PER_SRC,
)
print(f"H1 paths: {len(paths['H1'])}, H2 paths: {len(paths['H2'])}")
if paths["H2"]:
    print("Top H2:", {k: paths["H2"][0][k] for k in ('u_name','v_name','w_name','w_icd9','score')})

# render KG block
kg_text = render_kg_context_paths(
    tok=tok,
    paths=paths,
    code2name=code2name,
    profiles=profiles,
    budget_tokens=KG_HINT_BUDGET,
    h2_first_ratio=0.7
)
kg_tokens = count_tokens(tok, kg_text)
print("\n[KG CONTEXT tokens]:", kg_tokens)
print(kg_text[:1200])

# fit notes to remaining space
notes_room = MAX_LEN - (base_tokens + kg_tokens + 16)
notes_text = serialize_notes(row)
notes_trim = trim_to_token_budget(tok, notes_text, max(0, notes_room))

raw_prompt = "\n".join([p for p in (header, notes_trim, tail) if p])
kg_prompt  = "\n".join([p for p in (header, notes_trim, kg_text, tail) if p])

print("\nRAW prompt tokens:", count_tokens(tok, raw_prompt))
print("KG  prompt tokens:", count_tokens(tok, kg_prompt))

print("\n=== RAW PROMPT (head) ===\n", raw_prompt[:1500])
print("\n=== KG  PROMPT (head) ===\n", kg_prompt[:1500])


# # To see H1 paths:
# for i in paths['H1']:
#      print(f"{i['src_name']} -- {i['rela']} --> {i['nbr_name']} : rela score={i['rela_score']}")

# # To see H2 paths:
# for i in paths['H2']:
#      print(f"{i['u_name']} -- {i['rela_uv']} --> {i['v_name']} -- {i['rela_vw']} --> {i['w_name']} : rela score={i['score']} ==> ICD:{i['w_icd9']} ")

# # ICD sets from H2 paths:
# cd = set()
# for i in paths['H2']:
#      cd.update(i['w_icd9'])

############# Quick knobs to experiment without rerunning everything:
# In [9]
# def rebuild_with(budget=600, h2_ratio=0.7, min_edge_score=0.4,
#                  h1_per_src=3, h2_per_src=50):
#     paths2 = mine_paths_scored(
#         G=G,
#         node_name=node_name,
#         cui_to_icd9=cui_to_icd9,
#         ev_cuis=set(ev_cuis),
#         rel_bucket_whitelist=REL_BUCKET_WHITELIST,
#         min_edge_score=min_edge_score,
#         h1_per_src=h1_per_src,
#         h2_per_src=h2_per_src,
#     )
#     kg2 = render_kg_context_paths(tok, paths2, code2name, profiles, budget_tokens=budget, h2_first_ratio=h2_ratio)
#     notes_room2 = MAX_LEN - (base_tokens + count_tokens(tok, kg2) + 16)
#     notes_trim2 = trim_to_token_budget(tok, serialize_notes(row), max(0, notes_room2))
#     kg_prompt2  = "\n".join([p for p in (header, notes_trim2, kg2, tail) if p])
#     print(f"[budget={budget}, h2_ratio={h2_ratio}] KG tokens={count_tokens(tok, kg2)} | KG-prompt={count_tokens(tok, kg_prompt2)}")
#     print(kg2[:800])
#     return kg2, kg_prompt2

# # Example: tighter KG block
# _ = rebuild_with(budget=1000, h2_ratio=0.7)

# # Example: more aggressive H2 mining (will be trimmed by token budget anyway)
# _ = rebuild_with(budget=800, h2_ratio=0.7, h2_per_src=100)

# Mine all (kept filtered only by your REL_BUCKET_WHITELIST and MIN_EDGE_SCORE),
# do NOT enforce h1_per_src / h2_per_src caps here — we'll select later by thresholds.


## 10) Mine all scored H1/H2 (no per-src caps used for selection)
def mine_paths_scored_all(
    G: nx.DiGraph,
    node_name: Dict[str,str],
    cui_to_icd9: Dict[str, List[str]],
    ev_cuis: Set[str],
    rel_bucket_whitelist: Set[str],
    min_edge_score: float
):
    H1, H2 = [], []

    for u in ev_cuis:
        if u not in G:
            continue

        # H1: u -> v
        for v in G.successors(u):
            d = G[u][v]
            bucket = (d.get("rela_canon") or "").lower()
            escore = float(d.get("rela_score", 0.2))
            if escore < min_edge_score: 
                continue
            if rel_bucket_whitelist and bucket not in rel_bucket_whitelist:
                continue

            score = escore + (0.25 if len(cui_to_icd9.get(v, [])) > 0 else 0.0)
            H1.append({
                "src_cui": u, "nbr_cui": v,
                "src_name": node_name.get(u,""), "nbr_name": node_name.get(v,""),
                "rel": _norm(d.get("rel","")), "rela": _norm(d.get("rela","")),
                "rela_canon": bucket, "rela_score": escore, "score": score
            })

        # H2: u -> v -> w
        for v in G.successors(u):
            if v not in G: 
                continue
            d_uv = G[u][v]
            bucket_uv = (d_uv.get("rela_canon") or "").lower()
            esc_uv    = float(d_uv.get("rela_score", 0.2))
            if esc_uv < min_edge_score: 
                continue
            if rel_bucket_whitelist and bucket_uv not in rel_bucket_whitelist:
                continue

            for w in G.successors(v):
                d_vw = G[v][w]
                bucket_vw = (d_vw.get("rela_canon") or "").lower()
                esc_vw    = float(d_vw.get("rela_score", 0.2))
                if esc_vw < min_edge_score: 
                    continue
                if rel_bucket_whitelist and bucket_vw not in rel_bucket_whitelist:
                    continue

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
                    "score": score,
                    "s_uv": float(esc_uv), "s_vw": float(esc_vw),
                })

    # sort once globally
    H1.sort(key=lambda z: z["score"], reverse=True)
    H2.sort(key=lambda z: z["score"], reverse=True)
    return {"H1": H1, "H2": H2}

## 11) Thresholding + render without any token budget clamp
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


## 12) Global thresholds over the entire test set, build full prompts, and measure tokens
# Choose global thresholds here (no token budgets involved)
TAU_H1 = 1.0   # keep H1 if edge rela_score >= 1.0
TAU_H2 = 2.2   # keep H2 if (s_uv + s_vw + anchor_bonus) >= 2.2  (tweak freely)

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

rows_stats = []
for ridx in range(len(df)):
    row = df.iloc[ridx]
    # evidence → CUIs
    _, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

    # mine ALL (with your canonical buckets + MIN_EDGE_SCORE)
    paths_all = mine_paths_scored_all(
        G=G,
        node_name=node_name,
        cui_to_icd9=cui_to_icd9,
        ev_cuis=set(ev_cuis),
        rel_bucket_whitelist=REL_BUCKET_WHITELIST,
        min_edge_score=MIN_EDGE_SCORE
    )

    # threshold select (no per-src caps)
    paths_sel = select_paths_by_threshold(paths_all, TAU_H1, TAU_H2)

    # render full KG context (no token trimming)
    kg_text = render_kg_context_no_budget(paths_sel, code2name, profiles)

    # build full prompts (no trimming)
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
        "KG_tokens": count_tokens(tok, kg_text),
        "RAW_tokens": count_tokens(tok, raw_prompt),
        "WITHKG_tokens": count_tokens(tok, kg_prompt),
        "DELTA_tokens": count_tokens(tok, kg_prompt) - count_tokens(tok, raw_prompt),
    })

STATS = pd.DataFrame(rows_stats)
print("Collected stats on", STATS.shape[0], "rows")
STATS.head(3)

## 13) Summaries you care about (line counts + token impacts)
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

for col in ["H1_lines","H2_lines","KG_tokens","RAW_tokens","WITHKG_tokens","DELTA_tokens"]:
    print(col, summarize(STATS, col))

# rows that would exceed a typical 4096 context if you *didn't* trim anywhere
OVER = STATS[STATS["WITHKG_tokens"] > 4096]
print(f"\nRows exceeding 4096 tokens (no trimming): {len(OVER)}")
OVER.sort_values("WITHKG_tokens", ascending=False).head(10)

## 14) Optional: per-row adaptive thresholding from H2 score distribution
def propose_row_thresholds(H2_rows: List[dict]):
    if not H2_rows:
        return {"mean":0.0,"p50":0.0,"p75":0.0,"p90":0.0}
    s = np.array([h["score"] for h in H2_rows], dtype=float)
    return {"mean":float(s.mean()),"p50":float(np.quantile(s,0.5)),
            "p75":float(np.quantile(s,0.75)),"p90":float(np.quantile(s,0.9))}

ADAPT = []
for ridx in range(len(df)):
    row = df.iloc[ridx]
    _, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
    paths_all = mine_paths_scored_all(G, node_name, cui_to_icd9, set(ev_cuis),
                                      REL_BUCKET_WHITELIST, MIN_EDGE_SCORE)
    thr = propose_row_thresholds(paths_all["H2"])
    # example policy: fixed H1 >=1.0; H2 >= row p75
    paths_sel = select_paths_by_threshold(paths_all, tau_h1=1.0, tau_h2=thr["p75"])
    kg_text = render_kg_context_no_budget(paths_sel, code2name, profiles)

    header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
    notes  = serialize_notes(row)
    tail   = build_tail()

    raw_prompt = "\n".join([p for p in (header, notes, tail) if p])
    kg_prompt  = "\n".join([p for p in (header, notes, kg_text, tail) if p])

    ADAPT.append({
        "row_idx": ridx,
        "H1_lines": len(paths_sel["H1"]),
        "H2_lines": len(paths_sel["H2"]),
        "KG_tokens": count_tokens(tok, kg_text),
        "RAW_tokens": count_tokens(tok, raw_prompt),
        "WITHKG_tokens": count_tokens(tok, kg_prompt),
    })

ADAPT = pd.DataFrame(ADAPT)
print("Adaptive summary (H2@p75):")
for c in ["H1_lines","H2_lines","KG_tokens","WITHKG_tokens"]:
    print(c, summarize(ADAPT, c))


## 15) Quick preview helper for any row_idx under current thresholds
def preview_row(row_idx:int, tau_h1=TAU_H1, tau_h2=TAU_H2, max_lines=25):
    row = df.iloc[row_idx]
    _, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
    paths_all = mine_paths_scored_all(G, node_name, cui_to_icd9, set(ev_cuis),
                                      REL_BUCKET_WHITELIST, MIN_EDGE_SCORE)
    paths_sel = select_paths_by_threshold(paths_all, tau_h1, tau_h2)
    kg_text = render_kg_context_no_budget(paths_sel, code2name, profiles)
    header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
    notes  = serialize_notes(row)
    tail   = build_tail()

    raw_prompt = "\n".join([p for p in (header, notes, tail) if p])
    kg_prompt  = "\n".join([p for p in (header, notes, kg_text, tail) if p])

    print(f"[row {row_idx}] H1={len(paths_sel['H1'])} H2={len(paths_sel['H2'])} | KG_tokens={count_tokens(tok, kg_text)} | WITHKG={count_tokens(tok, kg_prompt)}")
    print("\n--- KG CONTEXT (head) ---")
    print("\n".join(kg_text.splitlines()[:max_lines]))

# example:
preview_row(0, tau_h1=1.0, tau_h2=2.2, max_lines=40)
