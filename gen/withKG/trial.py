# # -*- coding: utf-8 -*-
# """
# Token budgeting analysis for KG-augmented prompts (no model inference required).
# Adds per-tokenizer token counts for:
#   - full prompt without KG
#   - full prompt with KG
#   - NOTES segment only
#   - KG segment only

# Assumptions:
# - ATC meds are already stored in the dataframe under column 'ndc' (strings); we map them via code2cui_atc.pkl
# - We do not run the model; only tokenization analysis
# """

# import os, re, pickle, time
# from typing import List, Dict, Set, Tuple, Optional

# import numpy as np
# import pandas as pd

# try:
#     from IPython.display import display  # for notebooks
# except Exception:
#     def display(x):  # fallback for pure CLI
#         print(x)

# # ------------------------- CONFIG -------------------------
# TEST_PKL   = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl"

# KG_DIR     = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2"
# KG_PKL     = os.path.join(KG_DIR, "medical_knowledge_graph.pkl")
# ICD9_DX    = os.path.join(KG_DIR, "code2cui_icd9_dx.pkl")
# ICD9_PROC  = os.path.join(KG_DIR, "code2cui_icd9_proc.pkl")
# LNC_MAP    = os.path.join(KG_DIR, "code2cui_loinc.pkl")
# ATC_MAP    = os.path.join(KG_DIR, "code2cui_atc.pkl")  # ATC codes from 'ndc' column

# TOKENIZERS = {
#     "llama_3p1_8b": "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct",
#     "llama_3p2_1b": "meta-llama/Llama-3.2-1B-Instruct",
# }

# # None = use ALL rows; or set an int for quick sampling (e.g., 1000)
# BUDGET_SAMPLE_N: Optional[int] = None
# BUDGET_SEED = 13

# # KG expansion & rendering
# KG_HOP = 1
# MAX_KG_LINES = 64

# # Prompt budgets (for suggestions only; not enforced here)
# ASSISTANT_TOKENS_BUDGET = 128
# TARGET_INPUT_BUDGET = 3072

# SAVE_CSV = True
# CSV_OUT  = "token_budget_stats.csv"

# from transformers import AutoTokenizer

# # ------------------------------ helpers ------------------------------
# TEXT_COLS_SAFE = [
#     "Chief Complaint","History of Present Illness","Past Medical History",
#     "Family History","Physical Exam","Pertinent Results",
#     "Brief Hospital Course","Medications on Admission"
# ]

# def clean_text(x) -> str:
#     if x is None: return ""
#     s = str(x).replace("\x00"," ").replace("\r"," ")
#     s = re.sub(r"_+"," ", s)
#     return re.sub(r"\s+"," ", s).strip()

# def to_list(x) -> List[str]:
#     if x is None: return []
#     if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
#         try: it = list(x)
#         except Exception: it = [x]
#     else:
#         s = str(x).strip()
#         if not s or s.lower() in ("nan","none","null"): return []
#         if s.startswith("[") and s.endswith("]"):
#             try:
#                 import ast
#                 it = list(ast.literal_eval(s))
#             except Exception:
#                 it = re.split(r"[,\s]+", s)
#         else:
#             it = re.split(r"[,\s]+", s)
#     out=[]
#     for v in it:
#         if v is None: continue
#         if isinstance(v, float) and np.isnan(v): continue
#         sv = str(v).strip()
#         if sv: out.append(sv)
#     return out

# def _strip(s: str) -> str:
#     return re.sub(r"\s+","", str(s or "")).upper()

# def format_icd9_proc(code: str) -> str:
#     c = _strip(code).rstrip(".")
#     if not c: return ""
#     if c.startswith("PRO_"): c = c[4:]
#     if c[0].isdigit(): return c[:2]+"."+c[2:] if len(c)>2 and "." not in c else c
#     return c

# def serialize_structured(row: pd.Series) -> str:
#     # ATC meds come from 'ndc' column (already ATC codes)
#     atc  = to_list(row.get("ndc", []))
#     proc = to_list(row.get("pro_code", []))
#     labs = to_list(row.get("lab_test", []))
#     parts=[]
#     parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
#                  f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
#     if atc:  parts.append("[ATC] "  + " ".join(atc[:32]))
#     if proc: parts.append("[PROC] " + " ".join(proc[:32]))
#     if labs: parts.append("[LAB] "  + " ".join(labs[:64]))
#     return "\n".join(parts)

# def serialize_notes(row: pd.Series) -> str:
#     chunks=[]
#     for col in TEXT_COLS_SAFE:
#         if col in row:
#             t = clean_text(row[col])
#             if t: chunks.append(f"[{col.upper()}] {t}")
#     return "\n".join(chunks)

# def build_prompt_base_and_notes(row: pd.Series, subject_col="subject_id_x") -> Tuple[str, str]:
#     """Return (base_prompt_without_KG, notes_text_only)"""
#     s = [f"[VISIT] subject_id={row.get(subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
#     s.append(serialize_structured(row))
#     notes = serialize_notes(row)  # <--- capture notes block
#     if notes: s.append(notes)
#     s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
#     s.append("1. List only the ICD-9 codes separated by spaces")
#     s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
#     s.append("3. Include only codes directly supported by the clinical information")
#     s.append("4. Do not include any explanations or text besides the codes themselves")
#     s.append("[CODES]")
#     base = "\n".join([x for x in s if x])
#     return base, notes or ""

# # ------------------- KG evidence & context -------------------
# def evidence_cuis_from_row(row: pd.Series,
#                            icd9_proc_map: Dict[str, List[str]],
#                            loinc_map: Dict[str, List[str]],
#                            atc_map: Dict[str, List[str]]) -> Set[str]:
#     ev = set()
#     # Procedures
#     for c in to_list(row.get("pro_code", [])):
#         cc = format_icd9_proc(c)
#         ev.update(icd9_proc_map.get(cc, []))
#     # Labs (LOINC)
#     for c in to_list(row.get("lab_test", [])):
#         ev.update(loinc_map.get(_strip(c), []))
#     # Meds (ATC) coming from 'ndc' column
#     for c in to_list(row.get("ndc", [])):
#         ev.update(atc_map.get(_strip(c), []))
#     return ev

# def expand_cuis(G, seeds: Set[str], hop: int) -> Set[str]:
#     if hop <= 0 or not G or not seeds:
#         return set(seeds)
#     frontier = set(seeds)
#     visited  = set(seeds)
#     for _ in range(hop):
#         nxt=set()
#         for u in frontier:
#             if u in G:
#                 for v in G.successors(u):
#                     if v not in visited:
#                         visited.add(v); nxt.add(v)
#         frontier = nxt
#         if not frontier: break
#     return visited

# def render_kg_context(ev_cuis: Set[str], expanded_cuis: Set[str], kg_lines_cap=64) -> str:
#     ev_list  = sorted(ev_cuis)
#     exp_only = sorted(set(expanded_cuis) - set(ev_cuis))
#     lines=[]
#     if ev_list:
#         chunk = ev_list[:kg_lines_cap]
#         lines.append("[KG-EVIDENCE] " + " ".join(chunk))
#     if exp_only:
#         chunk = exp_only[:kg_lines_cap]
#         lines.append("[KG-EXPANDED] " + " ".join(chunk))
#     return "\n".join(lines)

# def build_prompt_with_kg(row: pd.Series,
#                          icd9_proc_map, loinc_map, atc_map,
#                          G, hop: int, kg_lines_cap: int,
#                          subject_col="subject_id_x") -> Tuple[str, str, dict]:
#     """
#     Returns:
#       - prompt_with_KG
#       - kg_only_text (for tokenizing KG segment alone)
#       - stats dict
#     """
#     base, notes = build_prompt_base_and_notes(row, subject_col)
#     ev   = evidence_cuis_from_row(row, icd9_proc_map, loinc_map, atc_map)
#     expanded = expand_cuis(G, ev, hop)
#     kg_ctx = render_kg_context(ev, expanded, kg_lines_cap=kg_lines_cap)
#     if kg_ctx:
#         prompt = base.replace("[CODES]", f"{kg_ctx}\n[CODES]")
#     else:
#         prompt = base
#     stats = {
#         "evidence_cuis": len(ev),
#         "expanded_cuis": len(expanded),
#         "kg_chars": len(kg_ctx),
#         "notes_chars": len(notes),
#     }
#     return prompt, kg_ctx, stats

# # ------------------- token counting & summaries -------------------
# def token_len(tok, text: str) -> int:
#     out = tok(text, add_special_tokens=False, return_length=True, truncation=False)
#     return int(out.get("length", [0])[0])

# def summarize_token_cols(df: pd.DataFrame, cols: List[str], label: str):
#     """Summarize only numeric token columns that match the given list."""
#     if not cols:
#         print(f"\n=== Token counts summary {label} ===\n(no token columns)")
#         return
#     desc = df[cols].describe(percentiles=[.5,.9,.95]).T
#     keep = [c for c in ["mean","50%","90%","95%","max"] if c in desc.columns]
#     print(f"\n=== Token counts summary {label} ===")
#     # Group columns by tokenizer nick
#     nicks = sorted({c.split("_noKG")[0].split("_withKG")[0].rsplit("_",1)[0] for c in cols})
#     for nick in nicks:
#         sub_cols = [c for c in cols if c.startswith(nick)]
#         print(f"\n[{nick}] {label}")
#         print(desc.loc[sub_cols, keep])

# # ------------------------------ main ------------------------------
# def main():
#     t0 = time.time()
#     print("[INFO] Config loaded.")
#     print("[INFO] Loading test dataframe…")
#     df = pd.read_pickle(TEST_PKL)
#     if isinstance(BUDGET_SAMPLE_N, int) and BUDGET_SAMPLE_N > 0 and BUDGET_SAMPLE_N < len(df):
#         rng = np.random.default_rng(BUDGET_SEED)
#         idx = rng.choice(len(df), size=BUDGET_SAMPLE_N, replace=False)
#         df = df.iloc[idx].reset_index(drop=True)
#     print(f"[INFO] Rows considered: {len(df)}")

#     print("[INFO] Loading KG + maps…")
#     import networkx as nx
#     with open(KG_PKL, "rb") as f:
#         G = pickle.load(f)
#         assert isinstance(G, nx.DiGraph)
#     icd9_dx_map   = pickle.load(open(ICD9_DX,   "rb"))
#     icd9_proc_map = pickle.load(open(ICD9_PROC, "rb"))
#     loinc_map     = pickle.load(open(LNC_MAP,   "rb"))
#     atc_map       = pickle.load(open(ATC_MAP,   "rb"))

#     print("[INFO] Loading tokenizer(s)…")
#     toks = {}
#     for nick, path in TOKENIZERS.items():
#         tok = AutoTokenizer.from_pretrained(path, use_fast=True)
#         if tok.pad_token_id is None:
#             tok.pad_token = tok.eos_token
#         tok.padding_side = "right"
#         toks[nick] = tok
#     print("Loaded tokenizers:", list(toks.keys()))

#     # Build prompts & gather stats
#     rows = []
#     for i, r in df.iterrows():
#         base, notes = build_prompt_base_and_notes(r)
#         withKG, kg_only, stat = build_prompt_with_kg(
#             r, icd9_proc_map, loinc_map, atc_map,
#             G=G, hop=KG_HOP, kg_lines_cap=MAX_KG_LINES
#         )
#         rows.append({
#             "idx": i,
#             "hadm_id": r.get("hadm_id", ""),
#             "evidence_cuis": stat["evidence_cuis"],
#             "expanded_cuis": stat["expanded_cuis"],
#             # lightweight analysis; we omit scanning for allowed_icd9 to keep this fast
#             "allowed_icd9": 0,
#             "notes_chars": stat["notes_chars"],
#             "kg_chars": stat["kg_chars"],
#             "_prompt_noKG": base,
#             "_prompt_withKG": withKG,
#             "_notes_only": notes,
#             "_kg_only": kg_only,
#         })

#     stats_df = pd.DataFrame(rows)

#     # Token counts per tokenizer (numeric columns only)
#     token_cols_all = []
#     for nick, tok in toks.items():
#         # Full prompts
#         stats_df[f"tokens_{nick}_noKG"]    = [token_len(tok, p) for p in stats_df["_prompt_noKG"]]
#         stats_df[f"tokens_{nick}_withKG"]  = [token_len(tok, p) for p in stats_df["_prompt_withKG"]]
#         # Segments: NOTES-only and KG-only
#         stats_df[f"tokens_{nick}_notes"]   = [token_len(tok, p) for p in stats_df["_notes_only"]]
#         stats_df[f"tokens_{nick}_kg"]      = [token_len(tok, p) for p in stats_df["_kg_only"]]

#         token_cols_all += [
#             f"tokens_{nick}_noKG",
#             f"tokens_{nick}_withKG",
#             f"tokens_{nick}_notes",
#             f"tokens_{nick}_kg",
#         ]

#     print("\nstats df (head):")
#     display(stats_df.head(10))

#     print("\n=== Summary: evidence sizes ===")
#     display(stats_df[["evidence_cuis","expanded_cuis","allowed_icd9"]].describe())

#     # Summaries (exclude prompt text cols)
#     token_cols_noKG    = [c for c in stats_df.columns if c.startswith("tokens_") and c.endswith("_noKG")]
#     token_cols_withKG  = [c for c in stats_df.columns if c.startswith("tokens_") and c.endswith("_withKG")]
#     token_cols_notes   = [c for c in stats_df.columns if c.startswith("tokens_") and c.endswith("_notes")]
#     token_cols_kg      = [c for c in stats_df.columns if c.startswith("tokens_") and c.endswith("_kg")]

#     summarize_token_cols(stats_df, token_cols_noKG, "(no KG)")
#     summarize_token_cols(stats_df, token_cols_withKG, "(with KG)")
#     summarize_token_cols(stats_df, token_cols_notes, "(NOTES only)")
#     summarize_token_cols(stats_df, token_cols_kg, "(KG only)")

#     # Over-budget rows per tokenizer (full prompt with KG)
#     for nick in toks.keys():
#         col = f"tokens_{nick}_withKG"
#         over = stats_df[stats_df[col] > TARGET_INPUT_BUDGET].copy()
#         print(f"\nRows exceeding {TARGET_INPUT_BUDGET} tokens for {nick}: {len(over)}")
#         if not over.empty:
#             print("Over head:")
#             display(over[["idx","hadm_id","evidence_cuis","expanded_cuis",
#                           "notes_chars","kg_chars",
#                           f"tokens_{nick}_noKG", f"tokens_{nick}_withKG",
#                           f"tokens_{nick}_notes", f"tokens_{nick}_kg"]].head(10))

#     # Suggested budgets (based on 95th percentiles)
#     print("\n=== Suggested budgets (heuristic) ===")
#     for nick in toks.keys():
#         with_col = f"tokens_{nick}_withKG"
#         no_col   = f"tokens_{nick}_noKG"
#         notes_col= f"tokens_{nick}_notes"
#         kg_col   = f"tokens_{nick}_kg"

#         p95_with   = stats_df[with_col].quantile(0.95)
#         p95_no     = stats_df[no_col].quantile(0.95)
#         p95_notes  = stats_df[notes_col].quantile(0.95)
#         p95_kg     = stats_df[kg_col].quantile(0.95)

#         suggested_prompt_budget = int(min(TARGET_INPUT_BUDGET - ASSISTANT_TOKENS_BUDGET, p95_with))
#         suggested_prompt_budget = max(512, suggested_prompt_budget)

#         # Split prompt budget between notes & KG based on their relative p95 sizes
#         total_p95_segments = max(1, p95_notes + p95_kg)
#         notes_budget = int(max(128, suggested_prompt_budget * (p95_notes / total_p95_segments)))
#         kg_budget    = int(max(64,  suggested_prompt_budget - notes_budget))

#         print(f"[{nick}] p95(noKG)={int(p95_no)} | p95(withKG)={int(p95_with)} | "
#               f"p95(NOTES)={int(p95_notes)} | p95(KG)={int(p95_kg)}\n"
#               f"  -> suggested prompt budget={suggested_prompt_budget} "
#               f"(NOTES≈{notes_budget}, KG≈{kg_budget}; assistant={ASSISTANT_TOKENS_BUDGET})")

#     if SAVE_CSV:
#         cols = [c for c in stats_df.columns if not c.startswith("_prompt")]
#         stats_df[cols].to_csv(CSV_OUT, index=False)
#         print(f"\n[INFO] Wrote snapshot CSV: {os.path.abspath(CSV_OUT)}")

#     print(f"\n[INFO] Token budgeting finished in {time.time()-t0:.1f}s")

# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""
Token budgeting + prompt inspection for KG-augmented codegen prompts.

What you get:
- Token counts (per tokenizer) for:
    * full prompt without KG
    * full prompt with KG
    * NOTES-only segment
    * KG-only segment ([KG_HINTS] block)
- Evidence stats (counts of CUIs)
- Over-budget row preview
- Pretty, notebook-parity [KG_HINTS] with neighbor triples & names and candidate ICD-9
- A few fully printed examples (NOTES + KG_HINTS + per-tokenizer token counts)

Assumptions:
- ATC meds are already stored in the dataframe under 'ndc' (strings)
- 'ndc' values are mapped via code2cui_atc.pkl
"""

import os, re, pickle, time
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

try:
    from IPython.display import display  # for notebooks
except Exception:
    def display(x):  # CLI fallback
        print(x)

# ------------------------- CONFIG -------------------------
TEST_PKL   = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl"

KG_DIR     = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2"
KG_PKL     = os.path.join(KG_DIR, "medical_knowledge_graph.pkl")
ICD9_DX    = os.path.join(KG_DIR, "code2cui_icd9_dx.pkl")
ICD9_PROC  = os.path.join(KG_DIR, "code2cui_icd9_proc.pkl")
LNC_MAP    = os.path.join(KG_DIR, "code2cui_loinc.pkl")
ATC_MAP    = os.path.join(KG_DIR, "code2cui_atc.pkl")  # ATC codes pulled from 'ndc' column

# Use all rows (None) or a random sample for quick iteration
BUDGET_SAMPLE_N: Optional[int] = None
BUDGET_SEED = 13

# KG knobs
KG_HOP = 1                   # 0/1/2
REL_WHITELIST: Set[str] = set()     # e.g. {"RO","RN"} (empty => allow all)
RELA_WHITELIST: Set[str] = set()    # e.g. {"isa","mapped_from"} (empty => allow all)
MAX_NEIGHBORS_SHOW = 24      # how many neighbor edges to render in [KG_HINTS]
MAX_CANDIDATES = 32          # how many ICD-9 candidate codes to render

# Tokenizers to compare
from transformers import AutoTokenizer
TOKENIZERS = {
    "llama_3p1_8b": "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct",
    "llama_3p2_1b": "meta-llama/Llama-3.2-1B-Instruct",
}

# Prompt budget targets (suggestion only)
ASSISTANT_TOKENS_BUDGET = 128
TARGET_INPUT_BUDGET = 3072

# How many full examples to print verbatim
PRINT_N_EXAMPLES = 2

SAVE_CSV = True
CSV_OUT  = "token_budget_stats.csv"

# ------------------------------ helpers ------------------------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x) -> str:
    if x is None: return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def to_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        try: it = list(x)
        except Exception: it = [x]
    else:
        s = str(x).strip()
        if not s or s.lower() in ("nan","none","null"): return []
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                it = list(ast.literal_eval(s))
            except Exception:
                it = re.split(r"[,\s]+", s)
        else:
            it = re.split(r"[,\s]+", s)
    out=[]
    for v in it:
        if v is None: continue
        if isinstance(v, float) and np.isnan(v): continue
        sv = str(v).strip()
        if sv: out.append(sv)
    return out

def _strip(s: str) -> str:
    return re.sub(r"\s+","", str(s or "")).upper()

def format_icd9(code: str) -> str:
    c = re.sub(r"\s+","", str(code)).upper().rstrip(".")
    if not c: return ""
    if c[0].isdigit(): return (c[:3]+"."+c[3:]) if (len(c)>3 and "." not in c) else c
    if c[0] == "V":   return (c[:3]+"."+c[3:]) if (len(c)>3 and "." not in c) else c
    if c[0] == "E":   return (c[:4]+"."+c[4:]) if (len(c)>4 and "." not in c) else c
    return c

def format_icd9_proc(code: str) -> str:
    c = _strip(code).rstrip(".")
    if not c: return ""
    if c.startswith("PRO_"): c = c[4:]
    if c[0].isdigit(): return c[:2]+"."+c[2:] if len(c)>2 and "." not in c else c
    return c

def serialize_structured(row: pd.Series) -> str:
    # ATC meds come from 'ndc' (already ATC codes)
    atc  = to_list(row.get("ndc", []))
    proc = to_list(row.get("pro_code", []))
    labs = to_list(row.get("lab_test", []))
    parts=[]
    parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
                 f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
    if atc:  parts.append("[ATC] "  + " ".join(atc[:32]))
    if proc: parts.append("[PROC] " + " ".join(proc[:32]))
    if labs: parts.append("[LAB] "  + " ".join(labs[:64]))
    return "\n".join(parts)

def serialize_notes(row: pd.Series) -> str:
    chunks=[]
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t: chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)

def build_prompt_base_and_notes(row: pd.Series, subject_col="subject_id_x") -> Tuple[str, str]:
    """Return (base_prompt_without_KG, notes_text_only)"""
    s = [f"[VISIT] subject_id={row.get(subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
    s.append(serialize_structured(row))
    notes = serialize_notes(row)  # ← capture NOTES block
    if notes: s.append(notes)
    s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
    s.append("1. List only the ICD-9 codes separated by spaces")
    s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
    s.append("3. Include only codes directly supported by the clinical information")
    s.append("4. Do not include any explanations or text besides the codes themselves")
    s.append("[CODES]")
    base = "\n".join([x for x in s if x])
    return base, notes or ""

# ------------------- KG evidence (notebook-parity) -------------------
def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    """
    Returns:
      src2cuis: { "ATC:XXX": [...], "LNC:YYY":[...], "PROC:12.34":[...] }
      ev_union: set(CUIs)
    """
    src2cuis = {}
    ev = set()

    # ATC via 'ndc'
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        cuis = atc_map.get(key, [])
        if cuis:
            src2cuis[f"ATC:{key}"] = cuis
            ev.update(cuis)

    # LOINC
    for c in to_list(row.get("lab_test", [])):
        key = _strip(c)
        cuis = loinc_map.get(key, [])
        if cuis:
            src2cuis[f"LNC:{key}"] = cuis
            ev.update(cuis)

    # ICD-9 PROC
    for c in to_list(row.get("pro_code", [])):
        cc = format_icd9_proc(c)
        cuis = icd9_proc_map.get(cc, [])
        if cuis:
            src2cuis[f"PROC:{cc}"] = cuis
            ev.update(cuis)

    return src2cuis, ev

def expand_k_hops(G: nx.DiGraph,
                  sources: Set[str],
                  k: int,
                  rel_whitelist: Set[str] = None,
                  rela_whitelist: Set[str] = None,
                  max_neighbors_total: int = 200) -> Tuple[Set[str], List[Tuple[str,str,str]]]:
    """Return (neighbors_set, edges_list(u, rel|rela, v))."""
    if k <= 0 or not sources:
        return set(), []
    seen = set(sources)
    frontier = set(sources)
    all_new = set()
    edges = []
    for _ in range(k):
        next_frontier = set()
        for u in frontier:
            if u not in G: continue
            for v in G.successors(u):
                d = G[u][v]
                rel  = (d.get('rel') or '').strip()
                rela = (d.get('rela') or '').strip()
                if rel_whitelist  and rel  not in rel_whitelist:  continue
                if rela_whitelist and rela not in rela_whitelist: continue
                if v in seen: continue
                if len(edges) < max_neighbors_total:
                    edges.append((u, (rela if rela else rel), v))
                next_frontier.add(v)
        next_frontier -= seen
        seen |= next_frontier
        all_new |= next_frontier
        if not next_frontier:
            break
    return all_new, edges

def map_cuis_to_icd9_dx(cuis: Set[str],
                        icd9_dx_map: Dict[str, List[str]],
                        max_codes: int = 64) -> List[str]:
    out=[]
    S=set(cuis)
    for code, cset in icd9_dx_map.items():
        if S.intersection(cset):
            out.append(code)
            if len(out) >= max_codes: break
    # normalize ICD-9 formatting
    out = [format_icd9(c) for c in out]
    out = [c for c in out if c]
    return sorted(set(out))[:max_codes]

def build_kg_hints_text(row: pd.Series,
                        KG: nx.DiGraph,
                        icd9_dx_map: Dict[str, List[str]],
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]],
                        hop: int,
                        rel_whitelist: Set[str] = None,
                        rela_whitelist: Set[str] = None,
                        max_neighbors_show: int = 24,
                        max_candidates: int = 32) -> Tuple[str, Set[str], Dict[str,int]]:
    """
    Notebook-style [KG_HINTS] text with:
      - Evidence CUIs per source,
      - Neighbor triples with names,
      - Candidate ICD-9 list from CUIs.
    Returns (hints_text, candidate_codes_set, stats_dict).
    """
    src2cui, ev_union = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

    lines = ["[KG_HINTS]"]

    # Evidence CUIs
    if src2cui:
        lines.append("Evidence CUIs linked from visit data:")
        for src, cu in list(src2cui.items())[:32]:
            lines.append(f"- {src}: {', '.join(sorted(set(cu))[:8])}")
    else:
        lines.append("Evidence CUIs linked from visit data: (none)")

    # Neighbors
    neighbor_set=set()
    edges_printed=0
    if hop >= 1 and ev_union:
        neigh_set, edge_tuples = expand_k_hops(
            KG, ev_union, k=hop,
            rel_whitelist=rel_whitelist, rela_whitelist=rela_whitelist,
            max_neighbors_total=max_neighbors_show*4
        )
        neighbor_set = neigh_set
        if edge_tuples:
            lines.append(f"Neighbors within {hop} hop(s):")
            for (u, reltxt, v) in edge_tuples[:max_neighbors_show]:
                nmu = KG.nodes[u].get("name","Unknown") if u in KG else "Unknown"
                nmv = KG.nodes[v].get("name","Unknown") if v in KG else "Unknown"
                lines.append(f"- {u} [{nmu}] --{reltxt}--> {v} [{nmv}]")
                edges_printed += 1
        else:
            lines.append(f"Neighbors within {hop} hop(s): (none)")

    # Candidate ICD-9
    bag = set(ev_union) | neighbor_set
    candidates = map_cuis_to_icd9_dx(bag, icd9_dx_map, max_codes=max_candidates)
    if candidates:
        lines.append("Candidate ICD-9-CM diagnosis codes suggested by KG (optional):")
        lines.append("  " + " ".join(candidates))
    else:
        lines.append("Candidate ICD-9-CM diagnosis codes suggested by KG: (none)")

    hints = "\n".join(lines)
    stats = {
        "evidence_cuis": len(ev_union),
        "expanded_cuis": len(bag),
        "neighbor_edges_printed": edges_printed,
        "kg_chars": len(hints),
    }
    return hints, set(candidates), stats

def build_prompt_with_kg(row: pd.Series,
                         KG, icd9_dx_map, icd9_proc_map, loinc_map, atc_map,
                         hop: int, max_neighbors_show: int, max_candidates: int,
                         rel_w: Optional[Set[str]], rela_w: Optional[Set[str]],
                         subject_col="subject_id_x") -> Tuple[str, str, dict]:
    """
    Returns:
      - prompt_with_KG (base + KG_HINTS inserted before [CODES])
      - kg_only_text   (just the [KG_HINTS] block)
      - stats dict     (evidence/expanded counts, chars, etc.)
    """
    base, notes = build_prompt_base_and_notes(row, subject_col)
    hints, cand, s = build_kg_hints_text(
        row, KG, icd9_dx_map, icd9_proc_map, loinc_map, atc_map,
        hop=hop, rel_whitelist=rel_w, rela_whitelist=rela_w,
        max_neighbors_show=max_neighbors_show, max_candidates=max_candidates
    )
    if hints:
        prompt = base.replace("[CODES]", f"{hints}\n[CODES]")
    else:
        prompt = base
    s["notes_chars"] = len(notes)
    return prompt, hints, s

# ------------------- token counting & summaries -------------------
def token_len(tok, text: str) -> int:
    out = tok(text, add_special_tokens=False, return_length=True, truncation=False)
    return int(out.get("length", [0])[0])

def summarize_token_cols(df: pd.DataFrame, cols: List[str], label: str):
    if not cols:
        print(f"\n=== Token counts summary {label} ===\n(no token columns)")
        return
    desc = df[cols].describe(percentiles=[.5,.9,.95]).T
    keep = [c for c in ["mean","50%","90%","95%","max"] if c in desc.columns]
    print(f"\n=== Token counts summary {label} ===")
    # Group columns by tokenizer nick
    nicks = sorted({c.split("_noKG")[0].split("_withKG")[0].split("_notes")[0].split("_kg")[0]
                    for c in cols})
    for nick in nicks:
        sub_cols = [c for c in cols if c.startswith(nick)]
        print(f"\n[{nick}] {label}")
        print(desc.loc[sub_cols, keep])

# ------------------------------ main ------------------------------
def main():
    t0 = time.time()
    print("[INFO] Config loaded.")
    print("[INFO] Loading test dataframe…")
    df = pd.read_pickle(TEST_PKL)
    if isinstance(BUDGET_SAMPLE_N, int) and BUDGET_SAMPLE_N > 0 and BUDGET_SAMPLE_N < len(df):
        rng = np.random.default_rng(BUDGET_SEED)
        idx = rng.choice(len(df), size=BUDGET_SAMPLE_N, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    print(f"[INFO] Rows considered: {len(df)}")

    print("[INFO] Loading KG + maps…")
    with open(KG_PKL, "rb") as f:
        G = pickle.load(f)
        assert isinstance(G, nx.DiGraph)
    icd9_dx_map   = pickle.load(open(ICD9_DX,   "rb"))
    icd9_proc_map = pickle.load(open(ICD9_PROC, "rb"))
    loinc_map     = pickle.load(open(LNC_MAP,   "rb"))
    atc_map       = pickle.load(open(ATC_MAP,   "rb"))

    rel_w  = REL_WHITELIST  or None
    rela_w = RELA_WHITELIST or None

    print("[INFO] Loading tokenizer(s)…")
    toks = {}
    for nick, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        toks[nick] = tok
    print("Loaded tokenizers:", list(toks.keys()))

    # Build prompts & gather stats
    rows = []
    notes_texts = []   # keep for example printing
    kg_texts = []      # keep for example printing
    base_prompts = []
    withKG_prompts = []

    for i, r in df.iterrows():
        base, notes = build_prompt_base_and_notes(r)
        withKG, kg_only, stat = build_prompt_with_kg(
            r, G, icd9_dx_map, icd9_proc_map, loinc_map, atc_map,
            hop=KG_HOP, max_neighbors_show=MAX_NEIGHBORS_SHOW, max_candidates=MAX_CANDIDATES,
            rel_w=rel_w, rela_w=rela_w
        )
        rows.append({
            "idx": i,
            "hadm_id": r.get("hadm_id", ""),
            "evidence_cuis": stat["evidence_cuis"],
            "expanded_cuis": stat["expanded_cuis"],
            "neighbor_edges_printed": stat["neighbor_edges_printed"],
            "notes_chars": stat["notes_chars"],
            "kg_chars": stat["kg_chars"],
        })
        notes_texts.append(notes)
        kg_texts.append(kg_only)
        base_prompts.append(base)
        withKG_prompts.append(withKG)

    stats_df = pd.DataFrame(rows)

    # Token counts per tokenizer
    token_cols_all = []
    for nick, tok in toks.items():
        stats_df[f"{nick}_noKG"]    = [token_len(tok, p) for p in base_prompts]
        stats_df[f"{nick}_withKG"]  = [token_len(tok, p) for p in withKG_prompts]
        stats_df[f"{nick}_notes"]   = [token_len(tok, p) for p in notes_texts]
        stats_df[f"{nick}_kg"]      = [token_len(tok, p) for p in kg_texts]
        token_cols_all += [f"{nick}_noKG", f"{nick}_withKG", f"{nick}_notes", f"{nick}_kg"]

    print("\nstats df (head):")
    display(stats_df.head(10))

    print("\n=== Summary: evidence sizes ===")
    display(stats_df[["evidence_cuis","expanded_cuis","neighbor_edges_printed"]].describe())

    # Summaries
    summarize_token_cols(stats_df, [c for c in token_cols_all if c.endswith("_noKG")], "(no KG)")
    summarize_token_cols(stats_df, [c for c in token_cols_all if c.endswith("_withKG")], "(with KG)")
    summarize_token_cols(stats_df, [c for c in token_cols_all if c.endswith("_notes")], "(NOTES only)")
    summarize_token_cols(stats_df, [c for c in token_cols_all if c.endswith("_kg")], "(KG only)")

    # Over-budget rows per tokenizer (full prompt with KG)
    for nick in toks.keys():
        col = f"{nick}_withKG"
        over = stats_df[stats_df[col] > TARGET_INPUT_BUDGET].copy()
        print(f"\nRows exceeding {TARGET_INPUT_BUDGET} tokens for {nick}: {len(over)}")
        if not over.empty:
            head_cols = ["idx","hadm_id","evidence_cuis","expanded_cuis","neighbor_edges_printed",
                         "notes_chars","kg_chars", f"{nick}_noKG", f"{nick}_withKG", f"{nick}_notes", f"{nick}_kg"]
            print("Over head:")
            display(over[head_cols].head(10))

    # Suggested budgets (based on 95th percentiles)
    print("\n=== Suggested budgets (heuristic) ===")
    for nick in toks.keys():
        with_col = f"{nick}_withKG"
        no_col   = f"{nick}_noKG"
        notes_col= f"{nick}_notes"
        kg_col   = f"{nick}_kg"

        p95_with   = stats_df[with_col].quantile(0.95)
        p95_no     = stats_df[no_col].quantile(0.95)
        p95_notes  = stats_df[notes_col].quantile(0.95)
        p95_kg     = stats_df[kg_col].quantile(0.95)

        suggested_prompt_budget = int(min(TARGET_INPUT_BUDGET - ASSISTANT_TOKENS_BUDGET, p95_with))
        suggested_prompt_budget = max(512, suggested_prompt_budget)

        total_p95_segments = max(1, p95_notes + p95_kg)
        notes_budget = int(max(128, suggested_prompt_budget * (p95_notes / total_p95_segments)))
        kg_budget    = int(max(64,  suggested_prompt_budget - notes_budget))

        print(f"[{nick}] p95(noKG)={int(p95_no)} | p95(withKG)={int(p95_with)} | "
              f"p95(NOTES)={int(p95_notes)} | p95(KG)={int(p95_kg)}\n"
              f"  -> suggested prompt budget={suggested_prompt_budget} "
              f"(NOTES≈{notes_budget}, KG≈{kg_budget}; assistant={ASSISTANT_TOKENS_BUDGET})")

    # -------- Pretty print a couple examples (NOTES + KG) --------
    N = min(PRINT_N_EXAMPLES, len(df))
    if N > 0:
        print(f"\n=== Full examples (N={N}) ===")
        for i in range(N):
            print("-"*120)
            print(f"[ROW {i}] hadm_id={df.iloc[i].get('hadm_id','?')}")
            print("\n[NOTES]:")
            print(notes_texts[i] if notes_texts[i] else "(none)")
            print("\n[KG_HINTS]:")
            print(kg_texts[i] if kg_texts[i] else "(none)")
            # token counts per tokenizer for this row
            for nick in toks.keys():
                base_n  = stats_df.iloc[i][f"{nick}_noKG"]
                with_n  = stats_df.iloc[i][f"{nick}_withKG"]
                notes_n = stats_df.iloc[i][f"{nick}_notes"]
                kg_n    = stats_df.iloc[i][f"{nick}_kg"]
                print(f"\n[{nick}] tokens -> base(noKG)={base_n} | withKG={with_n} | notes={notes_n} | kg={kg_n}")
            print("-"*120)

    # Save CSV snapshot (no giant text cols)
    if SAVE_CSV:
        out_df = stats_df.copy()
        out_df.to_csv(CSV_OUT, index=False)
        print(f"\n[INFO] Wrote snapshot CSV: {os.path.abspath(CSV_OUT)}")

    print(f"\n[INFO] Token budgeting finished in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
