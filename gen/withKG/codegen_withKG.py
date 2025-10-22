# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Inference for 'codegen' baseline with KG-aware constraints.

# What this script does (brief):
# 1) Loads a fine-tuned codegen model (LoRA adapter or merged model).
# 2) Builds prompts per visit (structured + notes → "[CODES]" target).
# 3) Generates raw ICD-9 codes with the model.
# 4) Builds a visit-specific KG prior from procedures/labs/(ATC meds) → CUIs (0/1-hop).
# 5) Uses that prior to HARD-FILTER (or SOFT-RERANK) the model’s codes.
# 6) Computes evaluation metrics (micro/macro/samples F1; optional hierarchical parent recall).

# Inputs you likely have:
# - Test dataframe pickle with columns like: subject_id_x, hadm_id, icd_code (gold), pro_code, lab_test, ndc|atc, and some note columns.
# - KG files produced by your builder.
# """

# import os, re, json, math, time, argparse, pickle
# from typing import List, Dict, Set, Tuple
# from collections import defaultdict

# import numpy as np
# import pandas as pd

# import torch
# from sklearn.metrics import f1_score, precision_score, recall_score

# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.utils import logging as hf_logging
# from peft import PeftModel, LoraConfig, get_peft_model

# import networkx as nx

# # ---------------- Env & logging ----------------
# os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# hf_logging.set_verbosity_error()

# # ------------------------------- ICD-9 formatting --------------------------------
# def format_icd9(code: str) -> str:
#     code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
#     if not code: return ""
#     if code[0].isdigit():
#         return (code[:3]+"."+code[3:]) if (len(code)>3 and "." not in code) else code
#     if code[0] == "V":
#         return (code[:3]+"."+code[3:]) if (len(code)>3 and "." not in code) else code
#     if code[0] == "E":
#         return (code[:4]+"."+code[4:]) if (len(code)>4 and "." not in code) else code
#     return code

# def is_valid_icd9(code: str) -> bool:
#     if not code: return False
#     if code[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
#     if code.startswith('V'): return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
#     if code.startswith('E'): return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
#     return False

# def get_icd9_parent(code: str) -> str:
#     if not code: return code
#     if code[0].isdigit(): return code.split('.')[0][:3]
#     if code[0]=='V':     return code.split('.')[0][:3]
#     if code[0]=='E':     return code.split('.')[0][:4]
#     return code


# # ------------------------------- Prompt building ---------------------------------
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

# def serialize_structured(row: pd.Series) -> str:
#     ndc  = to_list(row.get("ndc", []))
#     proc = to_list(row.get("pro_code", []))
#     labs = to_list(row.get("lab_test", []))
#     parts=[]
#     parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
#                  f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
#     if ndc:  parts.append("[NDC] "  + " ".join(ndc[:32]))
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

# def build_input_text(row: pd.Series, use_structured=True, use_notes=True,
#                      subject_col="subject_id_x") -> str:
#     s = [f"[VISIT] subject_id={row.get(subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
#     if use_structured: s.append(serialize_structured(row))
#     if use_notes:
#         t = serialize_notes(row)
#         if t: s.append(t)
#     s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
#     s.append("1. List only the ICD-9 codes separated by spaces")
#     s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
#     s.append("3. Include only codes directly supported by the clinical information")
#     s.append("4. Do not include any explanations or text besides the codes themselves")
#     s.append("[CODES]")
#     return "\n".join([x for x in s if x])


# # ------------------------------- Model loading -----------------------------------
# def load_model_and_tokenizer(base_model: str,
#                              adapter_dir: str = None,
#                              merged_model_dir: str = None):
#     """
#     If merged_model_dir is provided: load that directly.
#     Else: load base model + attach LoRA adapter from adapter_dir.
#     """
#     if torch.cuda.is_available():
#         use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
#         dtype = torch.bfloat16 if use_bf16 else torch.float16
#     else:
#         dtype = torch.float32

#     if merged_model_dir:
#         tok = AutoTokenizer.from_pretrained(merged_model_dir, use_fast=True)
#         if tok.pad_token is None: tok.pad_token = tok.eos_token
#         tok.padding_side = "right"
#         model = AutoModelForCausalLM.from_pretrained(
#             merged_model_dir,
#             torch_dtype=dtype,
#             low_cpu_mem_usage=True
#         )
#     else:
#         tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
#         if tok.pad_token is None: tok.pad_token = tok.eos_token
#         tok.padding_side = "right"
#         base = AutoModelForCausalLM.from_pretrained(
#             base_model,
#             torch_dtype=dtype,
#             low_cpu_mem_usage=True
#         )
#         if not adapter_dir:
#             raise ValueError("Provide --adapter_dir (LoRA) or --merged_model_dir.")
#         model = PeftModel.from_pretrained(base, adapter_dir)

#     model.config.pad_token_id = tok.pad_token_id
#     model.config.use_cache = False
#     return model, tok


# # ------------------------------- KG prior ----------------------------------------
# def visit_evidence_cuis(row: pd.Series,
#                         icd9_proc_map: Dict[str, List[str]],
#                         loinc_map: Dict[str, List[str]],
#                         atc_map: Dict[str, List[str]] = None) -> Set[str]:
#     """
#     Map evidence codes in the visit to CUIs (0-hop seed).
#     - Procedures: 'pro_code' (ICD-9-PROC)
#     - Labs: 'lab_test'   (LOINC)
#     - Meds: 'atc'        (ATC)  [optional column]
#     """
#     ev = set()

#     for c in to_list(row.get("pro_code", [])):
#         cc = format_icd9(c)  # proc formatter not strictly needed for keys if already normalized earlier
#         ev.update(icd9_proc_map.get(cc, []))

#     for c in to_list(row.get("lab_test", [])):
#         ev.update(loinc_map.get(str(c).strip().upper(), []))

#     # Only if you actually have ATC column (not NDC)
#     if atc_map is not None and "atc" in row.index:
#         for c in to_list(row.get("atc", [])):
#             ev.update(atc_map.get(str(c).strip().upper(), []))

#     return ev


# def allowed_dx_from_evidence(ev_cuis: Set[str],
#                              icd9_dx_map: Dict[str, List[str]],
#                              G: nx.DiGraph = None,
#                              hop: int = 0,
#                              rel_whitelist: Set[str] = None,
#                              rela_whitelist: Set[str] = None) -> Set[str]:
#     """
#     From evidence CUIs, optionally add 1-hop neighbors (filtered by REL/RELA),
#     then return the set of ICD-9 diagnosis codes whose CUIs intersect this set.
#     """
#     if not ev_cuis:
#         return set()

#     S = set(ev_cuis)
#     if hop >= 1 and G is not None:
#         exp = set()
#         for u in S:
#             if u not in G: continue
#             for v in G.successors(u):
#                 data = G[u][v]
#                 rel  = (data.get("rel") or "").strip()
#                 rela = (data.get("rela") or "").strip()
#                 ok_rel  = (not rel_whitelist)  or (rel in rel_whitelist)
#                 ok_rela = (not rela_whitelist) or (rela in rela_whitelist)
#                 if ok_rel and ok_rela:
#                     exp.add(v)
#         S |= exp

#     allowed = set()
#     for code, cuis in icd9_dx_map.items():
#         if S.intersection(cuis):
#             allowed.add(code)
#     return allowed


# # ------------------------------- Generation --------------------------------------
# @torch.no_grad()
# def generate_codes(model, tok, prompts: List[str],
#                    labels_vocab: Set[str] = None,
#                    max_len=3072, max_new=96, batch_size=16) -> List[List[str]]:
#     """
#     Generate raw ICD-9 codes per prompt, normalize + (optionally) keep only in labels_vocab.
#     """
#     model.eval()
#     device = next(model.parameters()).device
#     preds = []

#     for i in range(0, len(prompts), batch_size):
#         batch = prompts[i:i+batch_size]
#         inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.amp.autocast('cuda', enabled=(device.type=='cuda' and torch.cuda.get_device_capability(0)[0] >= 8)):
#             out = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new,
#                 do_sample=False,
#                 num_beams=1,
#                 no_repeat_ngram_size=2,
#                 eos_token_id=tok.eos_token_id,
#                 pad_token_id=tok.pad_token_id,
#                 return_dict_in_generate=True
#             )

#         seq = out.sequences
#         gen_only = seq[:, inputs["input_ids"].shape[1]:]
#         texts = tok.batch_decode(gen_only, skip_special_tokens=True)

#         for t in texts:
#             tokens = re.split(r"[^A-Za-z0-9\.]+", t)
#             cand = [format_icd9(z) for z in tokens if z]
#             keep=[]
#             seen=set()
#             for c in cand:
#                 if not is_valid_icd9(c): continue
#                 if labels_vocab and (c not in labels_vocab): continue
#                 if c in seen: continue
#                 seen.add(c); keep.append(c)
#             preds.append(keep)

#         if device.type == "cuda" and (i//batch_size) % 10 == 0:
#             torch.cuda.empty_cache()
#     return preds


# # ------------------------------- Evaluation --------------------------------------
# def codes_to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
#     idx = {c:i for i,c in enumerate(label_vocab)}
#     Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
#     for i, lst in enumerate(code_lists):
#         for c in lst:
#             j = idx.get(c)
#             if j is not None: Y[i, j] = 1
#     return Y

# def eval_sets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     return {
#         "micro_f1":   float(f1_score(y_true, y_pred, average="micro",   zero_division=0)),
#         "macro_f1":   float(f1_score(y_true, y_pred, average="macro",   zero_division=0)),
#         "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
#         "micro_precision":   float(precision_score(y_true, y_pred, average="micro",   zero_division=0)),
#         "macro_precision":   float(precision_score(y_true, y_pred, average="macro",   zero_division=0)),
#         "samples_precision": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
#         "micro_recall":      float(recall_score(y_true, y_pred, average="micro",      zero_division=0)),
#         "macro_recall":      float(recall_score(y_true, y_pred, average="macro",      zero_division=0)),
#         "samples_recall":    float(recall_score(y_true, y_pred, average="samples",    zero_division=0)),
#     }

# def hierarchical_parent_recall(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> float:
#     code_to_parent = {code: get_icd9_parent(code) for code in label_vocab}
#     n = y_true.shape[0]
#     hits = 0
#     denom = 0
#     for i in range(n):
#         pred_par = {code_to_parent[label_vocab[j]] for j in np.where(y_pred[i])[0]}
#         true_par = {code_to_parent[label_vocab[j]] for j in np.where(y_true[i])[0]}
#         hits += len(pred_par & true_par)
#         denom += len(true_par)
#     return float(hits/denom) if denom>0 else 0.0


# # ------------------------------- Main --------------------------------------------
# def main():
#     ap = argparse.ArgumentParser(description="KG-aware inference for codegen baseline")
#     # Data
#     ap.add_argument("--test_pickle", required=True, help="Pickle DataFrame with held-out visits")
#     ap.add_argument("--subject_col", default="subject_id_x")
#     ap.add_argument("--label_col", default="icd_code")

#     # Model
#     ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
#     ap.add_argument("--adapter_dir", default=None, help="LoRA adapter directory (if using adapters)")
#     ap.add_argument("--merged_model_dir", default=None, help="Merged model directory (optional alternative)")

#     # KG + maps
#     ap.add_argument("--kg_pkl", required=True, help="medical_knowledge_graph.pkl")
#     ap.add_argument("--icd9_dx_map_pkl",   required=True, help="code2cui_icd9_dx.pkl")
#     ap.add_argument("--icd9_proc_map_pkl", required=True, help="code2cui_icd9_proc.pkl")
#     ap.add_argument("--loinc_map_pkl",     required=True, help="code2cui_loinc.pkl")
#     ap.add_argument("--atc_map_pkl",       default=None,  help="code2cui_atc.pkl (optional if dataset has ATC)")

#     # KG expansion & policy
#     ap.add_argument("--kg_hop", type=int, default=0, choices=[0,1], help="0=use evidence CUIs only; 1=add 1-hop neighbors")
#     ap.add_argument("--rel_whitelist", default="", help="Comma-separated REL values (empty=all). Example: RO,RN")
#     ap.add_argument("--rela_whitelist", default="", help="Comma-separated RELA values (empty=all)")
#     ap.add_argument("--kg_strategy", choices=["hard_filter","soft_rerank"], default="hard_filter")
#     ap.add_argument("--fallback_k", type=int, default=5, help="If hard_filter empties, keep top-K unfiltered")

#     # Generation
#     ap.add_argument("--max_len", type=int, default=3072)
#     ap.add_argument("--gen_max_new", type=int, default=96)
#     ap.add_argument("--batch_size", type=int, default=16)

#     # Metrics
#     ap.add_argument("--use_complete_label_space", type=int, default=0,
#                     help="0: label space = unique gold codes in test; 1: try icd9.pkl (not required here)")

#     args = ap.parse_args()

#     # ---------------- Load data ----------------
#     df = pd.read_pickle(args.test_pickle)
#     assert args.label_col in df.columns

#     # Build prompts
#     df["input_text"] = df.apply(lambda r: build_input_text(r, True, True, args.subject_col), axis=1)
#     prompts = df["input_text"].astype(str).tolist()

#     # Build gold lists for evaluation
#     gold = []
#     for codes in df[args.label_col].tolist():
#         lst = to_list(codes)
#         lst = [format_icd9(c) for c in lst if c]
#         lst = [c for c in lst if is_valid_icd9(c)]
#         gold.append(sorted(set(lst)))

#     # Label space for metrics
#     if args.use_complete_label_space:
#         # Optional: load a full icd9.pkl if you want; otherwise we stick to test gold space
#         # Here we keep it simple: use test gold space
#         pass
#     label_vocab = sorted({c for lst in gold for c in lst})  # FULL=test-space
#     label_set   = set(label_vocab)

#     # ---------------- Load model ----------------
#     model, tok = load_model_and_tokenizer(args.base_model, args.adapter_dir, args.merged_model_dir)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # ---------------- Load KG + maps ----------------
#     with open(args.kg_pkl, "rb") as f:
#         KG: nx.DiGraph = pickle.load(f)
#     icd9_dx_map   = pickle.load(open(args.icd9_dx_map_pkl, "rb"))     # dict: code -> [CUIs]
#     icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))   # dict: code -> [CUIs]
#     loinc_map     = pickle.load(open(args.loinc_map_pkl, "rb"))       # dict: code -> [CUIs]
#     atc_map       = pickle.load(open(args.atc_map_pkl, "rb")) if args.atc_map_pkl else None

#     rel_w  = {s.strip() for s in args.rel_whitelist.split(",")  if s.strip()}
#     rela_w = {s.strip() for s in args.rela_whitelist.split(",") if s.strip()}

#     # ---------------- Raw generation ----------------
#     raw_preds = generate_codes(model, tok, prompts, labels_vocab=label_set,
#                                max_len=args.max_len, max_new=args.gen_max_new, batch_size=args.batch_size)

#     # ---------------- KG post-processing ----------------
#     final_preds = []
#     t0 = time.time()
#     for i, pred in enumerate(raw_preds):
#         row = df.iloc[i]
#         ev = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
#         allowed = allowed_dx_from_evidence(ev, icd9_dx_map, G=KG, hop=args.kg_hop,
#                                            rel_whitelist=rel_w if rel_w else None,
#                                            rela_whitelist=rela_w if rela_w else None)

#         if args.kg_strategy == "hard_filter":
#             kept = [c for c in pred if (c in allowed)]
#             if not kept:
#                 # graceful fallback
#                 kept = pred[:max(0, args.fallback_k)]
#             final_preds.append(kept)
#         else:  # soft_rerank
#             scored = [(c, 1 if c in allowed else 0) for c in pred]
#             scored.sort(key=lambda x: (x[1]==0,))  # allowed first, stable within groups
#             final_preds.append([c for c,_ in scored])

#     post_ms = (time.time() - t0) * 1000.0

#     # ---------------- Metrics ----------------
#     Y_true = codes_to_multihot(gold, label_vocab)
#     Y_raw  = codes_to_multihot(raw_preds, label_vocab)
#     Y_fin  = codes_to_multihot(final_preds, label_vocab)

#     m_raw = eval_sets(Y_true, Y_raw)
#     m_fin = eval_sets(Y_true, Y_fin)
#     m_fin["hierarchical_parent_recall"] = hierarchical_parent_recall(Y_true, Y_fin, label_vocab)

#     print("\n=== Metrics (TEST-space label vocab) ===")
#     print("Raw model:")
#     for k,v in m_raw.items():
#         print(f"  {k}: {v:.4f}")
#     print("\nKG-postprocessed:")
#     for k,v in m_fin.items():
#         if isinstance(v, float):
#             print(f"  {k}: {v:.4f}")
#         else:
#             print(f"  {k}: {v}")

#     print(f"\nKG post-processing time: {post_ms:.1f} ms total ({post_ms/len(df):.2f} ms/sample)")
#     # Show a few examples
#     n_show = min(5, len(df))
#     if n_show > 0:
#         print("\n=== Sample predictions (after KG) ===")
#         for i in range(n_show):
#             print("-"*80)
#             print(f"Example {i}:")
#             print("GOLD:", " ".join(gold[i]) if gold[i] else "(none)")
#             print("RAW :", " ".join(raw_preds[i]) if raw_preds[i] else "(none)")
#             print("KG  :", " ".join(final_preds[i]) if final_preds[i] else "(none)")


# if __name__ == "__main__":
#     main()


# codegen_infer_with_KG.py
# Inference-only pipeline for generative ICD-9 prediction with KG soft re-rank
# - Loads finetuned LoRA adapter
# - Generates codes (RAW)
# - Applies KG soft re-rank (no hard pruning) with parent-bridge and 1-hop expansion
# - Reports IDENTICAL metric pack for RAW and KG (including parent-level micro P/R/F1)


import os, re, json, time, argparse, pickle, datetime
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers.utils import logging as hf_logging

import networkx as nx

# # ---------------- Env & logging ----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

# ---------------- ICD9 helpers ----------------
def format_icd9(code: str) -> str:
    code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
    if not code: return ""
    if code[0].isdigit():
        return code[:3]+"."+code[3:] if len(code)>3 and "." not in code else code
    if code[0] == "V":
        return code[:3]+"."+code[3:] if len(code)>3 and "." not in code else code
    if code[0] == "E":
        return code[:4]+"."+code[4:] if len(code)>4 and "." not in code else code
    return code

def is_valid_icd9(code: str) -> bool:
    if not code: return False
    if code[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    if code.startswith('V'): return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    if code.startswith('E'): return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    return False

def get_icd9_parent(code: str) -> str:
    if not code: return code
    if code[0].isdigit(): return code.split('.')[0][:3]
    if code.startswith('V'):
        base = code.split('.')[0]; return base[:3]
    if code.startswith('E'):
        base = code.split('.')[0]; return base[:4] if len(base) >= 4 else base
    return code

# --------------- prompt builders ---------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x):
    if pd.isna(x): return ""
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

def serialize_structured(row: pd.Series) -> str:
    parts=[]
    parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
                 f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
    ndc  = to_list(row.get("ndc", []))
    proc = to_list(row.get("pro_code", []))
    labs = to_list(row.get("lab_test", []))
    atc  = to_list(row.get("atc", row.get("ATC", [])))
    if ndc:  parts.append("[NDC] "  + " ".join(ndc[:32]))
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

def build_input_text(row: pd.Series, subject_col="subject_id_x",
                     use_structured=True, use_notes=True) -> str:
    s = [f"[VISIT] subject_id={row.get(subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
    if use_structured: s.append(serialize_structured(row))
    if use_notes:
        t = serialize_notes(row)
        if t: s.append(t)
    s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
    s.append("1. List only the ICD-9 codes separated by spaces")
    s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
    s.append("3. Include only codes directly supported by the clinical information")
    s.append("4. Do not include any explanations or text besides the codes themselves")
    s.append("[CODES]")
    return "\n".join([x for x in s if x])

# --------------- label spaces & metrics ---------------
def load_full_label_space(icd9_pkl: str) -> List[str]:
    df = pd.read_pickle(icd9_pkl)
    codes = [format_icd9(str(x)) for x in df['icd_code'].astype(str).tolist()]
    return sorted({c for c in codes if is_valid_icd9(c)})

def load_dataset_label_space(data_pkl: str, label_col: str) -> List[str]:
    df = pd.read_pickle(data_pkl)
    uni=set()
    for raw in df[label_col].tolist():
        lst = to_list(raw)
        for c in lst:
            cc = format_icd9(c)
            if is_valid_icd9(cc):
                uni.add(cc)
    return sorted(uni)

def load_top_codes_csv(path: str) -> List[str]:
    """
    Expected schema:
      ICD9_Code,Frequency,Percentage,Description
    """
    if not path or not os.path.exists(path): return []
    df = pd.read_csv(path)
    if "ICD9_Code" not in df.columns:
        raise ValueError("Top-50 CSV must contain column 'ICD9_Code'")
    codes = [format_icd9(str(x)) for x in df["ICD9_Code"].astype(str).tolist()]
    return [c for c in codes if is_valid_icd9(c)]

def load_bottom_codes_csv(path: str) -> List[str]:
    """
    Expected schema:
      Code,Frequency
    """
    if not path or not os.path.exists(path): return []
    df = pd.read_csv(path)
    if "Code" not in df.columns:
        raise ValueError("Bottom-50 CSV must contain column 'Code'")
    codes = [format_icd9(str(x)) for x in df["Code"].astype(str).tolist()]
    return [c for c in codes if is_valid_icd9(c)]

def restrict_to(lists: List[List[str]], allowed: Set[str]) -> List[List[str]]:
    return [[c for c in lst if c in allowed] for lst in lists]

def multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(label_vocab)}
    Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
    for i,lst in enumerate(code_lists):
        for c in lst:
            j = idx.get(c)
            if j is not None: Y[i,j]=1
    return Y

def metrics_full(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str,float]:
    return {
        "micro_f1":         float(f1_score(y_true, y_pred, average="micro",   zero_division=0)),
        "macro_f1":         float(f1_score(y_true, y_pred, average="macro",   zero_division=0)),
        "samples_f1":       float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision":  float(precision_score(y_true, y_pred, average="micro",   zero_division=0)),
        "macro_precision":  float(precision_score(y_true, y_pred, average="macro",   zero_division=0)),
        "samples_precision":float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_recall":     float(recall_score(y_true, y_pred, average="micro",      zero_division=0)),
        "macro_recall":     float(recall_score(y_true, y_pred, average="macro",      zero_division=0)),
        "samples_recall":   float(recall_score(y_true, y_pred, average="samples",    zero_division=0)),
    }

def parent_lists(code_lists: List[List[str]]) -> List[List[str]]:
    return [[get_icd9_parent(c) for c in lst] for lst in code_lists]

def metrics_parent_micro(gold_lists: List[List[str]], pred_lists: List[List[str]]) -> Dict[str,float]:
    # micro P/R/F1 on parent space induced by gold labels (same pack for RAW & KG)
    labels = sorted({p for lst in gold_lists for p in lst})
    idx = {c:i for i,c in enumerate(labels)}
    def mh(lists):
        Y = np.zeros((len(lists), len(labels)), dtype=np.int32)
        for i,lst in enumerate(lists):
            for c in set(lst):
                j = idx.get(c)
                if j is not None: Y[i,j]=1
        return Y
    Yt, Yp = mh(gold_lists), mh(pred_lists)
    return {
        "parent_micro_precision": float(precision_score(Yt, Yp, average="micro", zero_division=0)),
        "parent_micro_recall":    float(recall_score(Yt, Yp,    average="micro", zero_division=0)),
        "parent_micro_f1":        float(f1_score(Yt, Yp,        average="micro", zero_division=0)),
    }

# --------------- model & generation ---------------
def load_finetuned(base_model: str, adapter_dir: str):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok

@torch.no_grad()
def generate_codes(model, tok, prompts: List[str], labels_vocab: List[str],
                   max_new=96, batch_size=16, max_len=3072) -> List[List[str]]:
    device = next(model.parameters()).device
    allowed = set(labels_vocab)
    preds=[]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda' and torch.cuda.get_device_capability(0)[0]>=8)):
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
                no_repeat_ngram_size=2, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True
            )
        seq = out.sequences
        gen_only = seq[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        for t in texts:
            tokens = re.split(r"[^A-Za-z0-9\.]+", t)
            cand = [format_icd9(z) for z in tokens if z]
            seen, keep=set(), []
            for c in cand:
                if c in allowed and is_valid_icd9(c) and c not in seen:
                    seen.add(c); keep.append(c)
            preds.append(keep)
        if device.type=='cuda' and (i//batch_size)%20==0:
            torch.cuda.empty_cache()
    return preds

# --------------- KG post-processing ---------------
def parent_bridge_allowed(allowed_codes: Set[str], candidate: str) -> bool:
    if candidate in allowed_codes: return True
    p = get_icd9_parent(candidate)
    if p in allowed_codes: return True
    for a in allowed_codes:
        if get_icd9_parent(a) == p:
            return True
    return False

def kg_score_codes(pred_codes: List[str],
                   allowed_codes: Set[str],
                   alpha_allowed: float = 1.0,
                   beta_parent_bridge: float = 0.5) -> List[Tuple[str, float]]:
    out=[]
    for c in pred_codes:
        s=0.0
        if c in allowed_codes: s += alpha_allowed
        elif parent_bridge_allowed(allowed_codes, c): s += beta_parent_bridge
        out.append((c, s))
    out.sort(key=lambda x: (-x[1],))
    return out

# def visit_evidence_cuis(row: pd.Series,
#                         icd9_proc_map: Dict[str, List[str]],
#                         loinc_map: Dict[str, List[str]],
#                         atc_map: Dict[str, List[str]],
#                         ndc_map: Dict[str, List[str]] = None) -> Set[str]:
#     ev=set()
#     for c in to_list(row.get("pro_code", [])):
#         c = str(c).strip().upper()
#         if c and c[0].isdigit() and "." not in c and len(c)>2:
#             c = c[:2]+"."+c[2:]
#         ev.update(icd9_proc_map.get(c, []))
#     for c in to_list(row.get("lab_test", [])):
#         ev.update(loinc_map.get(str(c).strip().upper(), []))
#     for n in to_list(row.get("ndc", [])):
#         if ndc_map:
#             ev.update(ndc_map.get(str(n).strip().upper(), []))
#     for a in to_list(row.get("atc", row.get("ATC", []))):
#         ev.update(atc_map.get(str(a).strip().upper(), []))
#     return ev

def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]] = None) -> Set[str]:
    """
    Build visit evidence CUIs from:
      - ICD-9 procedures in row['pro_code'] via icd9_proc_map
      - LOINC lab tests in row['lab_test'] via loinc_map
      - ATC medication codes:
          * use row['ndc'] which in THIS DATASET already contains ATC codes
            (upper-cased string keys in atc_map)
          * if 'ndc' missing, fallback to row['atc'] if present
    """
    ev = set()

    # Procedures (ICD-9 PROC -> CUIs)
    for c in to_list(row.get("pro_code", [])):
        cc = format_icd9(c)
        ev.update(icd9_proc_map.get(cc, []))

    # Labs (LOINC -> CUIs)
    for c in to_list(row.get("lab_test", [])):
        key = str(c).strip().upper()
        ev.update(loinc_map.get(key, []))

    # Meds (ATC -> CUIs) — your dataset puts ATC codes in 'ndc'
    if atc_map is not None:
        # primary: 'ndc' column holds ATC codes
        if "ndc" in row.index:
            for c in to_list(row.get("ndc", [])):
                key = str(c).strip().upper()
                ev.update(atc_map.get(key, []))
        # fallback: if a dedicated 'atc' column exists, use it too
        elif "atc" in row.index:
            for c in to_list(row.get("atc", [])):
                key = str(c).strip().upper()
                ev.update(atc_map.get(key, []))

    return ev

def allowed_dx_from_evidence(ev_cuis: Set[str],
                             icd9_dx_map: Dict[str, List[str]],
                             G=None, hop:int=0,
                             rel_whitelist: Set[str]=None,
                             rela_whitelist: Set[str]=None) -> Set[str]:
    if not ev_cuis: return set()
    bag = set(ev_cuis)
    if G is not None and hop >= 1:
        nxt=set()
        for u in ev_cuis:
            if u in G:
                for v in G.successors(u):
                    d = G[u][v]
                    rel = d.get('rel','') or ''
                    rela = d.get('rela','') or ''
                    if (rel_whitelist and rel not in rel_whitelist): continue
                    if (rela_whitelist and rela not in rela_whitelist): continue
                    nxt.add(v)
        bag |= nxt
    allowed=set()
    for code, cuis in icd9_dx_map.items():
        if bag.intersection(cuis):
            allowed.add(code)
    return allowed

# --------------- evaluation wrappers ---------------
def compute_metrics_pack(gold_lists: List[List[str]],
                         pred_lists: List[List[str]],
                         label_vocab: List[str]) -> Dict[str, float]:
    Yt = multihot(gold_lists, label_vocab)
    Yp = multihot(pred_lists, label_vocab)
    M  = metrics_full(Yt, Yp)
    M.update(metrics_parent_micro(parent_lists(gold_lists), parent_lists(pred_lists)))
    return M

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    # data (TEST ONLY + merged for dataset label space)
    ap.add_argument("--test_pickle", required=True)
    ap.add_argument("--data_pickle", required=True, help="merged_icd9.pkl for dataset label space")
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--icd9_pickle", required=True, help="Complete ICD-9 list to define FULL label space")

    # code buckets
    ap.add_argument("--top_codes_csv", required=True)  # ICD9_Code column
    ap.add_argument("--bot_codes_csv", required=True)  # Code column

    # model
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)

    # gen
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)

    # KG assets (explicit)
    ap.add_argument("--kg_pkl", required=True)
    ap.add_argument("--icd9_dx_map_pkl", required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl", required=True)
    ap.add_argument("--atc_map_pkl", required=True)
    ap.add_argument("--ndc_map_pkl", default=None)

    # KG behavior
    ap.add_argument("--kg_strategy", choices=["soft_rerank","hard_filter"], default="soft_rerank")
    ap.add_argument("--kg_hop", type=int, default=0)
    ap.add_argument("--rel_whitelist", default="")
    ap.add_argument("--rela_whitelist", default="")
    ap.add_argument("--fallback_k", type=int, default=20)

    # Examples printing
    ap.add_argument("--show_examples", type=int, default=5, help="Number of examples to print (0 to skip)")
    ap.add_argument("--examples_seed", type=int, default=0, help="Seed for random example selection; -1 = first N")

    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # load test set
    test_df = pd.read_pickle(args.test_pickle)
    # prompts
    test_df["input_text"] = test_df.apply(
        lambda r: build_input_text(r, args.subject_col,
                                   use_structured=bool(args.use_structured),
                                   use_notes=bool(args.use_notes)), axis=1)
    # gold codes
    def extract_gold(df_):
        out=[]
        for raw in df_[args.label_col].tolist():
            lst = to_list(raw)
            lst = [format_icd9(c) for c in lst if c]
            lst = [c for c in lst if is_valid_icd9(c)]
            out.append(sorted(set(lst)))
        return out
    gold = extract_gold(test_df)

    # label spaces
    labels_full     = load_full_label_space(args.icd9_pickle)
    labels_dataset  = load_dataset_label_space(args.data_pickle, args.label_col)
    top_codes       = load_top_codes_csv(args.top_codes_csv)
    bot_codes       = load_bottom_codes_csv(args.bot_codes_csv)

    print(f"[LABELS] FULL={len(labels_full)} | DATASET={len(labels_dataset)} | TOP50={len(top_codes)} | BOT50={len(bot_codes)}")

    # model
    model, tok = load_finetuned(args.base_model, args.adapter_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # RAW generation on FULL space
    print("\n=== RAW generation ===")
    t_raw = time.time()
    prompts = test_df["input_text"].astype(str).tolist()
    raw_preds = generate_codes(model, tok, prompts, labels_full,
                               max_new=args.gen_max_new, batch_size=args.batch_size, max_len=args.max_len)
    t_raw = time.time() - t_raw
    print(f"[RAW] time: {t_raw:.1f}s ({t_raw/len(test_df):.3f}s/sample)")

    # KG assets
    print("\n=== Loading KG assets ===")
    KG = pickle.load(open(args.kg_pkl, "rb"))
    icd9_dx_map   = pickle.load(open(args.icd9_dx_map_pkl, "rb"))
    icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
    loinc_map     = pickle.load(open(args.loinc_map_pkl, "rb"))
    atc_map       = pickle.load(open(args.atc_map_pkl, "rb"))
    ndc_map       = pickle.load(open(args.ndc_map_pkl, "rb")) if args.ndc_map_pkl else None

    rel_w  = set([s.strip() for s in args.rel_whitelist.split(",") if s.strip()])
    rela_w = set([s.strip() for s in args.rela_whitelist.split(",") if s.strip()])

    # KG post-processing
    print("\n=== KG post-processing ===")
    final_preds=[]
    t_kg = time.time()
    for i, pred in enumerate(raw_preds):
        row = test_df.iloc[i]
        ev = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map, ndc_map)
        allowed = allowed_dx_from_evidence(ev, icd9_dx_map, G=KG, hop=args.kg_hop,
                                           rel_whitelist=rel_w if rel_w else None,
                                           rela_whitelist=rela_w if rela_w else None)
        if args.kg_strategy == "hard_filter":
            kept = [c for c in pred if (c in allowed or parent_bridge_allowed(allowed, c))]
            if not kept: kept = pred[:max(0, args.fallback_k)]
            final = kept
        else:
            scored = kg_score_codes(pred, allowed, alpha_allowed=1.0, beta_parent_bridge=0.5)
            final = [c for c,_ in scored]
        final_preds.append(final)
    t_kg = time.time() - t_kg
    print(f"[KG] time: {t_kg*1000:.1f} ms total ({t_kg/len(test_df)*1000:.2f} ms/sample)")

    # --------- EVALUATIONS (identical metric pack for RAW and KG) ---------
    results = {}

    # A) FULL label space
    res_full = {}
    res_full["RAW"] = compute_metrics_pack(gold, raw_preds, labels_full)
    res_full["KG"]  = compute_metrics_pack(gold, final_preds, labels_full)
    results["FULL"] = res_full

    # B) DATASET label space
    allowed_ds = set(labels_dataset)
    gold_ds = restrict_to(gold, allowed_ds)
    raw_ds  = restrict_to(raw_preds, allowed_ds)
    kg_ds   = restrict_to(final_preds, allowed_ds)
    res_ds = {}
    res_ds["RAW"] = compute_metrics_pack(gold_ds, raw_ds, labels_dataset)
    res_ds["KG"]  = compute_metrics_pack(gold_ds, kg_ds,  labels_dataset)
    results["DATASET"] = res_ds

    # C) TOP-50 codes
    allowed_top = set(top_codes)
    gold_top = restrict_to(gold, allowed_top)
    raw_top  = restrict_to(raw_preds, allowed_top)
    kg_top   = restrict_to(final_preds, allowed_top)
    res_top = {}
    if len(allowed_top)>0:
        res_top["RAW"] = compute_metrics_pack(gold_top, raw_top, top_codes)
        res_top["KG"]  = compute_metrics_pack(gold_top, kg_top,  top_codes)
    else:
        res_top["RAW"] = {}
        res_top["KG"]  = {}
    results["TOP50"] = res_top

    # D) BOTTOM-50 codes
    allowed_bot = set(bot_codes)
    gold_bot = restrict_to(gold, allowed_bot)
    raw_bot  = restrict_to(raw_preds, allowed_bot)
    kg_bot   = restrict_to(final_preds, allowed_bot)
    res_bot = {}
    if len(allowed_bot)>0:
        res_bot["RAW"] = compute_metrics_pack(gold_bot, raw_bot, bot_codes)
        res_bot["KG"]  = compute_metrics_pack(gold_bot, kg_bot,  bot_codes)
    else:
        res_bot["RAW"] = {}
        res_bot["KG"]  = {}
    results["BOT50"] = res_bot

    # Pretty print summary
    def show_block(name, blk):
        print(f"\n=== {name} ===")
        for who in ("RAW","KG"):
            print(f"{who}:")
            for k,v in blk.get(who,{}).items():
                print(f"  {k}: {v:.4f}" if isinstance(v,(int,float)) else f"  {k}: {v}")

    show_block("FULL", results["FULL"])
    show_block("DATASET", results["DATASET"])
    show_block("TOP50", results["TOP50"])
    show_block("BOT50", results["BOT50"])

    # ---------------- Show N examples (random by seed, or first N) ----------------
    n_show = max(0, int(args.show_examples))
    if n_show > 0 and len(df) > 0:
        print(f"\n=== Sample predictions (n={n_show}) ===")
        if args.examples_seed >= 0:
            rng = np.random.default_rng(args.examples_seed)
            idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
        else:
            idxs = np.arange(min(n_show, len(df)))
        for k, i in enumerate(idxs):
            print("-"*80)
            print(f"Example {k} (row {i}):")
            print("GOLD:", " ".join(gold[i]) if gold[i] else "(none)")
            print("RAW :", " ".join(raw_preds[i]) if raw_preds[i] else "(none)")
            print("KG  :", " ".join(final_preds[i]) if final_preds[i] else "(none)")

    # Save
    out_dir = "runs_codegen_infer"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, f"metrics_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, f"preds_raw_{ts}.json"), "w") as f:
        json.dump(raw_preds, f)
    with open(os.path.join(out_dir, f"preds_kg_{ts}.json"), "w") as f:
        json.dump(final_preds, f)
    print(f"\n[INFO] Saved outputs to {out_dir}/")

if __name__ == "__main__":
    main()

