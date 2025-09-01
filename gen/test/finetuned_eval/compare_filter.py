# #!/usr/bin/env python3
# """
# Compare vocab-filtered vs unfiltered decoding across the TEST split.

# - Rebuilds prompts exactly like training (structured + notes + [TASK] + [CODES]).
# - Loads MERGED model if present; else loads base + adapter_best.
# - Generates for all TEST rows (or --limit N), GREEDY decoding.
# - Produces:
#   * Summary of where the filter changes predictions.
#   * Standard metrics (mapping to label space), WITH vs WITHOUT (label-only).
#   * Strict metrics that penalize non-vocab tokens as FP (to show filter benefit).
#   * CSV of per-row diffs including removed non-vocab tokens.

# Usage:
#   python compare_filter_full.py \
#     --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#     --data_pickle mergeddf.pkl \
#     --limit 2000 \
#     --batch_size 4
# """

# import os, re, json, pickle, argparse, collections
# from typing import List, Dict, Any, Tuple

# import numpy as np
# import pandas as pd

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, precision_score, recall_score

# from transformers.utils import logging as hf_logging
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import yaml

# # ============== Env / logging / reproducibility ==============
# os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# hf_logging.set_verbosity_error()

# # --------------------------- Config mirroring ---------------------------
# SUBJECT_COL = "subject_id_x"
# LABEL_COL   = "icd_code"

# TEXT_COLS_SAFE = [
#     "Chief Complaint","History of Present Illness","Past Medical History",
#     "Family History","Physical Exam","Pertinent Results",
#     "Brief Hospital Course","Medications on Admission"
# ]

# DEFAULT_MAX_LEN     = 3072
# DEFAULT_GEN_MAX_NEW = 96

# # --------------------------- Normalization -----------------------------

# def normalize_code(s: str, remove_internal_dots: bool = True) -> str:
#     if s is None:
#         return ""
#     s = str(s).upper().strip()
#     s = re.sub(r"[^A-Z0-9\.]+", "", s)  # keep A-Z, 0-9, and dots
#     s = s.rstrip(".")
#     if remove_internal_dots:
#         s = s.replace(".", "")
#     return s


# def build_vocab_index(labels_vocab: List[str], remove_internal_dots: bool = True) -> Dict[str, str]:
#     idx = {}
#     for lab in labels_vocab:
#         key = normalize_code(lab, remove_internal_dots=remove_internal_dots)
#         idx.setdefault(key, lab)
#     return idx

# # --------------------------- Serialization -----------------------------

# def clean_text(x: Any) -> str:
#     if pd.isna(x):
#         return ""
#     s = str(x).replace("\x00"," ").replace("\r"," ")
#     s = re.sub(r"_+"," ", s)
#     return re.sub(r"\s+"," ", s).strip()


# def to_list(x) -> List[str]:
#     if isinstance(x, list):
#         return [str(v) for v in x]
#     if pd.isna(x):
#         return []
#     s = str(x).strip()
#     if s.startswith("[") and s.endswith("]"):
#         try:
#             import ast
#             v = ast.literal_eval(s)
#             if isinstance(v, list):
#                 return [str(z) for z in v]
#         except Exception:
#             pass
#     return [t for t in re.split(r"[,\s]+", s) if t]


# def serialize_structured(row: pd.Series) -> str:
#     parts = []
#     parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
#                  f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
#     ndc  = to_list(row.get("ndc", []))
#     proc = to_list(row.get("pro_code", []))
#     labs = to_list(row.get("lab_test", []))
#     if ndc:  parts.append("[NDC] "  + " ".join(ndc[:32]))
#     if proc: parts.append("[PROC] " + " ".join(proc[:32]))
#     if labs: parts.append("[LAB] "  + " ".join(labs[:64]))
#     return "\n".join(parts)


# def serialize_notes(row: pd.Series) -> str:
#     chunks = []
#     for col in TEXT_COLS_SAFE:
#         if col in row:
#             t = clean_text(row[col])
#             if t:
#                 chunks.append(f"[{col.upper()}] {t}")
#     return "\n".join(chunks)


# def build_input_text(row: pd.Series, use_structured: bool=True, use_notes: bool=True) -> str:
#     s = [f"[VISIT] subject_id={row.get(SUBJECT_COL,'?')} hadm_id={row.get('hadm_id','?')}"]
#     if use_structured:
#         s.append(serialize_structured(row))
#     if use_notes:
#         t = serialize_notes(row)
#         if t:
#             s.append(t)
#     s.append("[TASK] Predict ICD diagnosis codes (space-separated). Output ONLY the codes, separated by single spaces.")
#     s.append("[CODES]")
#     return "\n".join([x for x in s if x])

# # --------------------------- Split -------------------------------------

# def subject_splits(df: pd.DataFrame, subject_col: str, test_size=0.10, val_size=0.10, seed=42):
#     subs = df[subject_col].dropna().unique()
#     from sklearn.model_selection import train_test_split
#     train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
#     train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
#     tr = df[df[subject_col].isin(train_subs)].copy()
#     va = df[df[subject_col].isin(val_subs)].copy()
#     te = df[df[subject_col].isin(test_subs)].copy()
#     return tr, va, te

# # --------------------------- Generation --------------------------------

# def extract_candidates(decoded_text: str) -> List[str]:
#     tail = decoded_text.split("[CODES]")[-1]
#     tokens = re.split(r"[^A-Za-z0-9\.]+", tail)
#     return [t for t in tokens if t]

# @torch.no_grad()
# def generate_batch(model, tok, prompts: List[str], max_len: int, gen_max_new: int) -> List[str]:
#     model.eval()
#     inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
#     out = model.generate(
#         **inputs,
#         max_new_tokens=gen_max_new,
#         do_sample=False,
#         num_beams=1,
#         no_repeat_ngram_size=2,
#         eos_token_id=tok.eos_token_id,
#         pad_token_id=tok.pad_token_id,
#     )
#     texts = tok.batch_decode(out, skip_special_tokens=True)
#     return texts

# # WITH and WITHOUT vocab filter

# def predict_with_filter(decoded_text: str, labels_vocab: List[str], remove_internal_dots: bool=True) -> List[str]:
#     cand = extract_candidates(decoded_text)
#     idx = build_vocab_index(labels_vocab, remove_internal_dots=remove_internal_dots)
#     seen, keep = set(), []
#     for z in cand:
#         key = normalize_code(z, remove_internal_dots=remove_internal_dots)
#         if not key:
#             continue
#         canon = idx.get(key)
#         if canon and canon not in seen:
#             seen.add(canon)
#             keep.append(canon)
#     return keep


# def predict_without_filter(decoded_text: str, remove_internal_dots: bool=True) -> List[str]:
#     cand = extract_candidates(decoded_text)
#     seen, keep = set(), []
#     for z in cand:
#         nz = normalize_code(z, remove_internal_dots=remove_internal_dots)
#         if nz and nz not in seen:
#             seen.add(nz)
#             keep.append(nz)
#     return keep

# # Map token list to label-only list (drop tokens not in vocab)

# def keep_only_vocab(tokens: List[str], labels_vocab: List[str], remove_internal_dots: bool=True) -> List[str]:
#     idx = build_vocab_index(labels_vocab, remove_internal_dots=remove_internal_dots)
#     seen, keep = set(), []
#     for t in tokens:
#         key = normalize_code(t, remove_internal_dots=remove_internal_dots)
#         canon = idx.get(key)
#         if canon and canon not in seen:
#             seen.add(canon)
#             keep.append(canon)
#     return keep

# # --------------------------- Metrics -----------------------------------

# def codes_to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
#     idx = {c:i for i,c in enumerate(label_vocab)}
#     Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
#     for i, lst in enumerate(code_lists):
#         for c in lst:
#             j = idx.get(c)
#             if j is not None:
#                 Y[i, j] = 1
#     return Y


# def eval_sets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     return {
#         "micro_f1":   f1_score(y_true, y_pred, average="micro",   zero_division=0),
#         "macro_f1":   f1_score(y_true, y_pred, average="macro",   zero_division=0),
#         "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
#         "micro_precision":   precision_score(y_true, y_pred, average="micro",   zero_division=0),
#         "macro_precision":   precision_score(y_true, y_pred, average="macro",   zero_division=0),
#         "samples_precision": precision_score(y_true, y_pred, average="samples", zero_division=0),
#         "micro_recall":      recall_score(y_true, y_pred, average="micro",   zero_division=0),
#         "macro_recall":      recall_score(y_true, y_pred, average="macro",   zero_division=0),
#         "samples_recall":    recall_score(y_true, y_pred, average="samples", zero_division=0),
#     }

# # Strict micro metrics (penalize non-vocab tokens as FP)

# def strict_micro_metrics(golds: List[List[str]], preds: List[List[str]], labels_vocab: List[str], remove_internal_dots: bool=True) -> Dict[str, float]:
#     # Canonicalize gold to vocab space
#     idx = build_vocab_index(labels_vocab, remove_internal_dots=remove_internal_dots)
#     tp = fp = fn = 0
#     for g, p in zip(golds, preds):
#         gset = set()
#         for x in g:
#             k = normalize_code(x, remove_internal_dots=remove_internal_dots)
#             c = idx.get(k)
#             if c:
#                 gset.add(c)
#         # For strict case, predictions list is already what we want (either filtered labels or normalized raw tokens)
#         pset = set(p)
#         # Map any label-like tokens in pset to canonical to maximize fair matching
#         pset_canon = set()
#         for x in pset:
#             k = normalize_code(x, remove_internal_dots=remove_internal_dots)
#             c = idx.get(k, x)  # if not a known label, keep as-is so it counts as FP
#             pset_canon.add(c)
#         tp += len(gset & pset_canon)
#         fp += len(pset_canon - gset)
#         fn += len(gset - pset_canon)
#     prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
#     return {"micro_precision":prec, "micro_recall":rec, "micro_f1":f1, "tp":tp, "fp":fp, "fn":fn}

# # --------------------------- Model loading -----------------------------

# def detect_dtype():
#     if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
#         try:
#             torch.backends.cuda.matmul.allow_tf32 = True
#             torch.set_float32_matmul_precision("high")
#         except Exception:
#             pass
#         return torch.bfloat16
#     elif torch.cuda.is_available():
#         return torch.float16
#     else:
#         return torch.float32


# def load_tokenizer(run_dir: str):
#     tok_dir = os.path.join(run_dir, "tokenizer")
#     if not os.path.isdir(tok_dir):
#         raise FileNotFoundError(f"Tokenizer dir not found: {tok_dir}")
#     tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token
#     tok.padding_side = "right"
#     return tok


# def load_model(run_dir: str, base_model_name_from_config: str | None, torch_dtype):
#     merged_dir = os.path.join(run_dir, "model_merged")
#     if os.path.isdir(merged_dir):
#         print(f"[INFO] Loading MERGED model from: {merged_dir}")
#         model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch_dtype, device_map="auto")
#         model.config.pad_token_id = model.config.eos_token_id
#         model.config.use_cache = False
#         return model

#     # Fall back to adapter
#     adapter_dir = os.path.join(run_dir, "adapter_best")
#     if not os.path.isdir(adapter_dir):
#         raise FileNotFoundError("Neither merged model nor adapter_best found in run_dir.")

#     if not base_model_name_from_config:
#         cfg_path = os.path.join(run_dir, "config.json")
#         with open(cfg_path, "r") as f:
#             cfg = json.load(f)
#         base_model_name_from_config = cfg.get("model")

#     print(f"[INFO] Loading BASE model: {base_model_name_from_config}")
#     base = AutoModelForCausalLM.from_pretrained(base_model_name_from_config, torch_dtype=torch_dtype, device_map="auto")
#     base.config.pad_token_id = base.config.eos_token_id
#     base.config.use_cache = False
#     print(f"[INFO] Attaching LoRA adapter from: {adapter_dir}")
#     model = PeftModel.from_pretrained(base, adapter_dir)
#     return model

# # --------------------------- Main -------------------------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run_dir", required=True)
#     ap.add_argument("--data_pickle", default="mergeddf.pkl")
#     ap.add_argument("--limit", type=int, default=None, help="Evaluate only first N test rows")
#     ap.add_argument("--batch_size", type=int, default=4)
#     ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
#     ap.add_argument("--gen_max_new", type=int, default=DEFAULT_GEN_MAX_NEW)
#     ap.add_argument("--remove_internal_dots", type=int, default=1)
#     ap.add_argument("--no_structured", action="store_true")
#     ap.add_argument("--no_notes", action="store_true")
#     ap.add_argument("--out_csv", default=None)
#     args = ap.parse_args()

#     remove_dots = bool(args.remove_internal_dots)

#     # Load labels
#     with open(os.path.join(args.run_dir, "label_space.json"), "r") as f:
#         labels_vocab = json.load(f)["labels"]

#     # Load data & split
#     df = pickle.load(open(args.data_pickle, "rb"))
#     assert LABEL_COL in df.columns and SUBJECT_COL in df.columns
#     _, _, test_df = subject_splits(df, SUBJECT_COL, test_size=0.10, val_size=0.10, seed=42)

#     # Build prompts
#     use_structured = not args.no_structured
#     use_notes      = not args.no_notes
#     test_df = test_df.copy()
#     test_df["input_text"] = test_df.apply(lambda r: build_input_text(r, use_structured, use_notes), axis=1)

#     if args.limit is not None:
#         test_df = test_df.iloc[:args.limit].copy()

#     # Canonical GOLD for metrics
#     idx_map = build_vocab_index(labels_vocab, remove_internal_dots=remove_dots)
#     gold_lists = []
#     for _, row in test_df.iterrows():
#         row_gold = []
#         for g in row[LABEL_COL]:
#             kg = normalize_code(g, remove_internal_dots=remove_dots)
#             cg = idx_map.get(kg, kg)
#             if cg:
#                 row_gold.append(cg)
#         gold_lists.append(sorted(set(row_gold)))

#     # Load tokenizer & model
#     tok = load_tokenizer(args.run_dir)
#     model = load_model(args.run_dir, base_model_name_from_config=None, torch_dtype=detect_dtype())

#     # Generate in batches
#     decoded_texts = []
#     prompts = test_df["input_text"].astype(str).tolist()
#     for i in range(0, len(prompts), args.batch_size):
#         batch = prompts[i:i+args.batch_size]
#         decoded_texts.extend(generate_batch(model, tok, batch, args.max_len, args.gen_max_new))

#     # Build predictions
#     preds_with = []
#     preds_wo_norm = []
#     preds_wo_labelonly = []
#     removed_counts = []
#     removed_tokens_all = []

#     for text in decoded_texts:
#         pw = predict_with_filter(text, labels_vocab, remove_internal_dots=remove_dots)
#         pwn = predict_without_filter(text, remove_internal_dots=remove_dots)
#         pwo = keep_only_vocab(pwn, labels_vocab, remove_internal_dots=remove_dots)

#         preds_with.append(pw)
#         preds_wo_norm.append(pwn)
#         preds_wo_labelonly.append(pwo)

#         # tokens that would be removed by the filter (non-vocab)
#         idx = build_vocab_index(labels_vocab, remove_internal_dots=remove_dots)
#         removed = []
#         for t in pwn:
#             key = normalize_code(t, remove_internal_dots=remove_dots)
#             if key not in idx:
#                 removed.append(t)
#         removed_counts.append(len(removed))
#         removed_tokens_all.extend(removed)

#     # Summaries of differences
#     label_diffs = [set(a) != set(b) for a,b in zip(preds_with, preds_wo_labelonly)]
#     any_removed = [c > 0 for c in removed_counts]

#     total = len(test_df)
#     n_label_diff = sum(label_diffs)
#     n_any_removed = sum(any_removed)

#     print("\n===== FILTER IMPACT SUMMARY =====")
#     print(f"Total evaluated samples: {total}")
#     print(f"Samples with ANY non-vocab tokens removed by filter: {n_any_removed} ({n_any_removed/total:.1%})")
#     print(f"Samples where label predictions CHANGE because of filter: {n_label_diff} ({n_label_diff/total:.1%})")

#     # Top removed tokens
#     cnt = collections.Counter(removed_tokens_all)
#     if cnt:
#         print("\nTop 20 removed (non-vocab) tokens:")
#         for tok_, c in cnt.most_common(20):
#             print(f"  {tok_:<16}  {c}")

#     # STANDARD metrics (mapped to label space)
#     Y_true = codes_to_multihot(gold_lists, labels_vocab)
#     Y_pred_with = codes_to_multihot(preds_with, labels_vocab)
#     Y_pred_wo_labelonly = codes_to_multihot(preds_wo_labelonly, labels_vocab)

#     std_with = eval_sets(Y_true, Y_pred_with)
#     std_wo   = eval_sets(Y_true, Y_pred_wo_labelonly)

#     print("\n===== STANDARD METRICS (label-space mapping) =====")
#     for k in ["micro_precision","micro_recall","micro_f1","macro_f1","samples_f1"]:
#         print(f"{k:>18}: WITH={std_with[k]:.4f} | WO={std_wo[k]:.4f} | Δ={std_with[k]-std_wo[k]:+.4f}")

#     # STRICT metrics (penalize non-vocab as FP)
#     strict_with = strict_micro_metrics(gold_lists, preds_with, labels_vocab, remove_internal_dots=remove_dots)
#     strict_wo   = strict_micro_metrics(gold_lists, preds_wo_norm, labels_vocab, remove_internal_dots=remove_dots)

#     print("\n===== STRICT MICRO (non-vocab count as FP) =====")
#     for k in ["micro_precision","micro_recall","micro_f1","tp","fp","fn"]:
#         print(f"{k:>14}: WITH={strict_with[k]} | WO={strict_wo[k]} | Δ={strict_with[k]-strict_wo[k]:+}")

#     # Write per-row CSV of differences
#     if args.out_csv is None:
#         args.out_csv = os.path.join(args.run_dir, "compare_filter_diffs.csv")

#     out_rows = []
#     for i, (_, row) in enumerate(test_df.iterrows()):
#         out_rows.append({
#             "idx": i,
#             "subject_id": row.get(SUBJECT_COL),
#             "hadm_id": row.get("hadm_id"),
#             "gold": " ".join(gold_lists[i]),
#             "pred_with": " ".join(preds_with[i]),
#             "pred_without": " ".join(preds_wo_norm[i]),
#             "removed_non_vocab_count": removed_counts[i],
#             "label_diff": label_diffs[i],
#         })
#     pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
#     print(f"\n[WROTE] Per-row diff CSV → {args.out_csv}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Compare RAW (decoder-emitted) code-like tokens vs PROCESSED (normalized + canonicalized) codes.

Goal: quantify how often normalization changes codes (case/dots), drops tokens that
aren't in label space, or merges duplicates into a canonical code.

What this script does:
  1) Loads tokenizer + model from a run_dir (prefers merged; falls back to base+adapter).
  2) Rebuilds prompts exactly like training and decodes GREEDY on TEST split (or --limit).
  3) For each sample, extracts the FIRST LINE after "[CODES]" and parses *code-like* tokens only.
  4) Produces, per row:
        - raw_codes              (detected code-like tokens, minimally cleaned)
        - normalized_codes       (normalize_code on raw_codes)
        - canonical_vocab_codes  (normalized mapped to label-space canonical form, dedup)
     and counts differences.
  5) Prints an overall summary and writes a CSV of per-row results.
  6) Optionally writes a text file with a few example rows where normalization changes things.

Usage:
  python compare_normalization_effect.py \
    --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
    --data_pickle mergeddf.pkl \
    --limit 1000 \
    --batch_size 4 \
    --examples_out examples_norm_effect.txt \
    --examples_k 5

Notes:
- Default normalization removes internal dots and uppercases (ICD-9 style in your runs).
- "Code-like" detection uses a conservative ICD-9 pattern (numeric, V, E ranges) and trims trailing punctuation.
"""

import os, re, json, pickle, argparse, collections
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.model_selection import train_test_split

from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml

# ============== Env / logging / reproducibility ==============
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

# --------------------------- Columns & Prompt ---------------------------
SUBJECT_COL = "subject_id_x"
LABEL_COL   = "icd_code"

TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

DEFAULT_MAX_LEN     = 3072
DEFAULT_GEN_MAX_NEW = 96

# --------------------------- Normalization -----------------------------

def normalize_code(s: str, remove_internal_dots: bool = True, strip_leading_zeros: bool = True) -> str:
    """Canonicalize a code token.
    - Uppercase
    - Keep only A–Z, 0–9, and dots; strip sentence-ending dot
    - Optionally remove internal dots
    - Optionally strip **leading zeros** at the very start (no V/E special-case)
    """
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9\.]+", "", s)
    s = s.rstrip(".")
    if remove_internal_dots:
        s = s.replace(".", "")
    if strip_leading_zeros:
        s = re.sub(r"^0+", "", s)
    return s


def build_vocab_index(labels_vocab: List[str], remove_internal_dots: bool = True, strip_leading_zeros: bool = True) -> Dict[str, str]:
    idx = {}
    for lab in labels_vocab:
        key = normalize_code(lab, remove_internal_dots=remove_internal_dots, strip_leading_zeros=strip_leading_zeros)
        idx.setdefault(key, lab)
    return idx

# --------------------------- Code-like detection -----------------------
# Conservative ICD-9 patterns (numeric, V, E). We do **not** normalize V/E beyond uppercase.
ICD9_NUM   = re.compile(r"^\d{3}(?:\.\d{1,2})?$|^\d{4,5}$")
ICD9_VCODE = re.compile(r"^V\d{2,3}(?:\.\d{1,2})?$")
ICD9_ECODE = re.compile(r"^E\d{3,4}(?:\.\d{1,2})?$")

def is_code_like(tok: str) -> bool:
    if not tok:
        return False
    t = tok.strip().strip(',;:)]}(').upper()
    return bool(ICD9_NUM.match(t) or ICD9_VCODE.match(t) or ICD9_ECODE.match(t))

# --------------------------- Prompt build ------------------------------

def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()


def to_list(x) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(z) for z in v]
        except Exception:
            pass
    return [t for t in re.split(r"[,\s]+", s) if t]


def serialize_structured(row: pd.Series) -> str:
    parts = []
    parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
                 f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
    ndc  = to_list(row.get("ndc", []))
    proc = to_list(row.get("pro_code", []))
    labs = to_list(row.get("lab_test", []))
    if ndc:  parts.append("[NDC] "  + " ".join(ndc[:32]))
    if proc: parts.append("[PROC] " + " ".join(proc[:32]))
    if labs: parts.append("[LAB] "  + " ".join(labs[:64]))
    return "\n".join(parts)


def serialize_notes(row: pd.Series) -> str:
    chunks = []
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t:
                chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)


def build_input_text(row: pd.Series, use_structured: bool=True, use_notes: bool=True) -> str:
    s = [f"[VISIT] subject_id={row.get(SUBJECT_COL,'?')} hadm_id={row.get('hadm_id','?')}"]
    if use_structured:
        s.append(serialize_structured(row))
    if use_notes:
        t = serialize_notes(row)
        if t:
            s.append(t)
    s.append("[TASK] Predict ICD diagnosis codes (space-separated). Output ONLY the codes, separated by single spaces.")
    s.append("[CODES]")
    return "\n".join([x for x in s if x])

# --------------------------- Split & Decode ----------------------------

def subject_splits(df: pd.DataFrame, subject_col: str, test_size=0.10, val_size=0.10, seed=42):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    return tr, va, te


def first_line_after_codes(decoded_text: str) -> str:
    tail = decoded_text.split("[CODES]", 1)[-1]
    tail = tail.lstrip()
    return tail.splitlines()[0] if tail else ""

@torch.no_grad()
def generate_batch(model, tok, prompts: List[str], max_len: int, gen_max_new: int) -> List[str]:
    model.eval()
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=gen_max_new,
        do_sample=False,
        num_beams=1,
        no_repeat_ngram_size=2,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    return texts

# --------------------------- Model loading -----------------------------

def detect_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    else:
        return torch.float32


def load_tokenizer(run_dir: str):
    tok_dir = os.path.join(run_dir, "tokenizer")
    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(f"Tokenizer dir not found: {tok_dir}")
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(run_dir: str, base_model_name_from_config: str | None, torch_dtype):
    merged_dir = os.path.join(run_dir, "model_merged")
    if os.path.isdir(merged_dir):
        print(f"[INFO] Loading MERGED model from: {merged_dir}")
        model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch_dtype, device_map="auto")
        model.config.pad_token_id = model.config.eos_token_id
        model.config.use_cache = False
        return model

    adapter_dir = os.path.join(run_dir, "adapter_best")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError("Neither merged model nor adapter_best found in run_dir.")

    if not base_model_name_from_config:
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        base_model_name_from_config = cfg.get("model")

    print(f"[INFO] Loading BASE model: {base_model_name_from_config}")
    base = AutoModelForCausalLM.from_pretrained(base_model_name_from_config, torch_dtype=torch_dtype, device_map="auto")
    base.config.pad_token_id = base.config.eos_token_id
    base.config.use_cache = False
    print(f"[INFO] Attaching LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    return model

# --------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_pickle", default="mergeddf.pkl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--gen_max_new", type=int, default=DEFAULT_GEN_MAX_NEW)
    ap.add_argument("--remove_internal_dots", type=int, default=1)
    ap.add_argument("--strip_leading_zeros", type=int, default=1)
    ap.add_argument("--no_structured", action="store_true")
    ap.add_argument("--no_notes", action="store_true")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--examples_out", default=None)
    ap.add_argument("--examples_k", type=int, default=6)
    args = ap.parse_args()

    remove_dots = bool(args.remove_internal_dots)
    strip_zeros = bool(args.strip_leading_zeros)

    # Label space
    with open(os.path.join(args.run_dir, "label_space.json"), "r") as f:
        labels_vocab = json.load(f)["labels"]
    idx_map = build_vocab_index(labels_vocab, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros)

    # Data + split
    df = pickle.load(open(args.data_pickle, "rb"))
    assert LABEL_COL in df.columns and SUBJECT_COL in df.columns
    _, _, test_df = subject_splits(df, SUBJECT_COL, test_size=0.10, val_size=0.10, seed=42)

    # Prompts
    use_structured = not args.no_structured
    use_notes      = not args.no_notes
    test_df = test_df.copy()
    test_df["input_text"] = test_df.apply(lambda r: build_input_text(r, use_structured, use_notes), axis=1)

    if args.limit is not None:
        test_df = test_df.iloc[:args.limit].copy()

    # Canonical gold (for reference)
    gold_lists = []
    for _, row in test_df.iterrows():
        row_gold = []
        for g in row[LABEL_COL]:
            kg = normalize_code(g, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros)
            cg = idx_map.get(kg, kg)
            if cg:
                row_gold.append(cg)
        gold_lists.append(sorted(set(row_gold)))

    # Load model
    tok = load_tokenizer(args.run_dir)
    model = load_model(args.run_dir, base_model_name_from_config=None, torch_dtype=detect_dtype())

    # Decode
    prompts = test_df["input_text"].astype(str).tolist()
    decoded = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        decoded.extend(generate_batch(model, tok, batch, args.max_len, args.gen_max_new))

    # Process rows
    rows = []
    total_raw = total_changed = total_dropped = total_merged = 0
    samp_any_change = samp_any_drop = samp_any_merge = 0

    for i, (_, row) in enumerate(test_df.iterrows()):
        line = first_line_after_codes(decoded[i])
        raw_tokens = [t.strip().strip(',;:)]}(') for t in line.split() if t.strip()]
        raw_codes = [t for t in raw_tokens if is_code_like(t)]

        norm_codes = [normalize_code(t, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros) for t in raw_codes]

        seen = set(); canon = []
        dropped = 0
        for z in norm_codes:
            c = idx_map.get(z)
            if c:
                if c not in seen:
                    seen.add(c); canon.append(c)
            else:
                dropped += 1

        changed = sum(1 for r, n in zip(raw_codes, norm_codes) if n != r.upper().strip().rstrip('.'))
        mapped_cnt = sum(1 for z in norm_codes if z in idx_map)
        merged = max(0, mapped_cnt - len(canon))

        total_raw += len(raw_codes)
        total_changed += changed
        total_dropped += dropped
        total_merged += merged

        samp_any_change += int(changed > 0)
        samp_any_drop   += int(dropped > 0)
        samp_any_merge  += int(merged > 0)

        rows.append({
            "idx": i,
            "subject_id": row.get(SUBJECT_COL),
            "hadm_id": row.get("hadm_id"),
            "gold": " ".join(gold_lists[i]),
            "raw_codes": " ".join(raw_codes),
            "normalized_codes": " ".join(norm_codes),
            "canonical_vocab_codes": " ".join(canon),
            "n_raw_codes": len(raw_codes),
            "n_norm_changed_tokens": changed,
            "n_dropped_by_vocab": dropped,
            "n_merged_duplicates": merged,
            "any_change": changed > 0,
            "any_drop": dropped > 0,
            "any_merge": merged > 0,
        })

    df_out = pd.DataFrame(rows)

    # Summary
    N = len(test_df)
    print("\n===== NORMALIZATION (LEADING ZERO) — SUMMARY =====")
    print(f"Total evaluated samples: {N}")
    print(f"Samples with ANY normalization change: {samp_any_change} ({samp_any_change/max(1,N):.1%})")
    print(f"Samples with ANY dropped-by-vocab token: {samp_any_drop} ({samp_any_drop/max(1,N):.1%})")
    print(f"Samples with ANY merged duplicates: {samp_any_merge} ({samp_any_merge/max(1,N):.1%})")
    print(f"\nTotal code-like tokens: {total_raw}")
    print(f"Tokens changed by normalization: {total_changed} ({total_changed/max(1,total_raw):.1%})")
    print(f"Tokens dropped by vocab: {total_dropped}")
    print(f"Tokens merged (duplicates collapsed): {total_merged}")

    # Write CSV
    out_csv = args.out_csv or os.path.join(args.run_dir, "compare_normalization_effect_leading_zero.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[WROTE] per-row CSV → {out_csv}")

    # Examples
    if args.examples_out:
        df_ex = df_out.copy()
        df_ex["score"] = df_ex["n_norm_changed_tokens"] + df_ex["n_dropped_by_vocab"] + df_ex["n_merged_duplicates"]
        df_ex = df_ex.sort_values(["score","n_norm_changed_tokens","n_dropped_by_vocab","n_merged_duplicates"], ascending=False).head(args.examples_k)
        lines = []
        for _, r in df_ex.iterrows():
            lines.append("="*80)
            lines.append(f"idx={int(r['idx'])} | subject_id={r['subject_id']} | hadm_id={r['hadm_id']}")
            lines.append(f"GOLD:                 {r['gold']}")
            lines.append(f"RAW codes:            {r['raw_codes']}")
            lines.append(f"NORMALIZED codes:     {r['normalized_codes']}")
            lines.append(f"CANONICAL (vocab):    {r['canonical_vocab_codes']}")
            lines.append(f"changed={int(r['n_norm_changed_tokens'])} dropped={int(r['n_dropped_by_vocab'])} merged={int(r['n_merged_duplicates'])}")
        text = "\n".join(lines)
        with open(args.examples_out, 'w') as f:
            f.write(text + "\n")
        print(f"[WROTE] examples → {args.examples_out}")

if __name__ == "__main__":
    main()

