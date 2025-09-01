#!/usr/bin/env python3
"""
Single-sample compare: WITH vocab filter vs WITHOUT filter (normalized) vs RAW tokens (no normalization).

Adds a third view that prints the *raw tokens* the decoder emitted after [CODES]
without any normalization or vocab mapping. This is for visual inspection only;
metrics are still computed on canonicalized predictions.

Usage examples:
  python single_sample_compare_raw_tokens.py \
      --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
      --data_pickle mergeddf.pkl \
      --row_index 3410

  python single_sample_compare_raw_tokens.py \
      --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
      --data_pickle mergeddf.pkl \
      --hadm_id 28292310
"""

import os, re, json, pickle, argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.model_selection import train_test_split

# --------------------------- Config ---------------------------
SUBJECT_COL = "subject_id_x"
LABEL_COL   = "icd_code"

TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

DEFAULT_MAX_LEN    = 3072
DEFAULT_GEN_MAX_NEW= 96

# --------------------------- Normalization -----------------------------

# def normalize_code(s: str, remove_internal_dots: bool = True) -> str:
#     """Canonicalize a predicted or gold code.
#     - Uppercase
#     - Trim
#     - Remove trailing sentence dot
#     - Optionally remove internal dots for canonical matching
#     """
#     if s is None:
#         return ""
#     s = str(s).upper().strip()
#     s = re.sub(r"[^A-Z0-9\.]+", "", s)  # keep alnum + dots
#     s = s.rstrip(".")
#     if remove_internal_dots:
#         s = s.replace(".", "")
#     return s

def normalize_code(s: str, remove_internal_dots: bool = True, strip_leading_zeros: bool = True) -> str:
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

# --------------------------- Serialization ---------------------------

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

# --------------------------- Split ---------------------------

def subject_splits(df: pd.DataFrame, subject_col: str, test_size=0.10, val_size=0.10, seed=42):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    return tr, va, te

# --------------------------- Generation ---------------------------

def extract_candidates(decoded_text: str) -> List[str]:
    tail = decoded_text.split("[CODES]", 1)[-1]
    tokens = re.split(r"[^A-Za-z0-9\.]+", tail)
    return [t for t in tokens if t]

@torch.no_grad()
def generate_once(model, tok, prompt: str, max_len: int, gen_max_new: int) -> str:
    model.eval()
    inputs = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=gen_max_new,
        do_sample=False,
        num_beams=1,
        no_repeat_ngram_size=2,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]

# --------------------------- Predict variants ---------------------------

def predict_codes_from_text(
    decoded_text: str,
    labels_vocab: List[str],
    remove_internal_dots: bool = True,
    strip_leading_zeros: bool = True,
    use_vocab_filter: bool = True,
) -> List[str]:
    cand = extract_candidates(decoded_text)

    if not use_vocab_filter:
        # Just normalize and dedup (no vocab check)
        seen, keep = set(), []
        for z in cand:
            nz = normalize_code(z, remove_internal_dots=remove_internal_dots, strip_leading_zeros=strip_leading_zeros)
            if nz and nz not in seen:
                seen.add(nz); keep.append(nz)
        return keep

    # With filter: map normalized token to canonical vocab form
    idx = build_vocab_index(labels_vocab, remove_internal_dots=remove_internal_dots, strip_leading_zeros=strip_leading_zeros)
    seen, keep = set(), []
    for z in cand:
        key = normalize_code(z, remove_internal_dots=remove_internal_dots, strip_leading_zeros=strip_leading_zeros)
        if not key:
            continue
        canon = idx.get(key)
        if canon and canon not in seen:
            seen.add(canon); keep.append(canon)
    return keep


def extract_raw_tokens_no_norm(decoded_text: str, dedup: bool = True) -> List[str]:
    toks = extract_candidates(decoded_text)
    if not dedup:
        return toks
    seen, keep = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); keep.append(t)
    return keep

# --------------------------- Metrics (per-sample) ---------------------------

def per_sample_prf1(gold: List[str], pred: List[str], labels_vocab: List[str], remove_internal_dots: bool = True, strip_leading_zeros: bool = True) -> Tuple[float,float,float,List[str],List[str]]:
    idx = build_vocab_index(labels_vocab, remove_internal_dots=remove_internal_dots, strip_leading_zeros=strip_leading_zeros)
    gold_canon = []
    for g in gold:
        kg = normalize_code(g, remove_internal_dots=remove_internal_dots, strip_leading_zeros=strip_leading_zeros)
        cg = idx.get(kg)
        if cg:
            gold_canon.append(cg)
    pred_canon = []
    for p in pred:
        kp = normalize_code(p, remove_internal_dots=remove_internal_dots, strip_leading_zeros=strip_leading_zeros)
        cp = idx.get(kp, p)
        pred_canon.append(cp)
    gset, pset = set(gold_canon), set(pred_canon)
    tp = len(gset & pset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1, sorted(list(gset - pset)), sorted(list(pset - gset))

# --------------------------- Loader ---------------------------

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

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_pickle", default="mergeddf.pkl")
    ap.add_argument("--row_index", type=int, default=None)
    ap.add_argument("--hadm_id", type=str, default=None)
    ap.add_argument("--subject_id", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--gen_max_new", type=int, default=DEFAULT_GEN_MAX_NEW)
    ap.add_argument("--remove_internal_dots", type=int, default=1)
    ap.add_argument("--strip_leading_zeros", type=int, default=1)
    ap.add_argument("--no_structured", action="store_true")
    ap.add_argument("--no_notes", action="store_true")
    args = ap.parse_args()

    remove_dots = bool(args.remove_internal_dots)
    strip_zeros = bool(args.strip_leading_zeros)

    # Load data & split
    df = pickle.load(open(args.data_pickle, "rb"))
    assert LABEL_COL in df.columns and SUBJECT_COL in df.columns
    _, _, test_df = subject_splits(df, SUBJECT_COL, test_size=0.10, val_size=0.10, seed=42)

    # Build prompt
    use_structured = not args.no_structured
    use_notes      = not args.no_notes
    test_df = test_df.copy()
    test_df["input_text"] = test_df.apply(lambda r: build_input_text(r, use_structured, use_notes), axis=1)

    # Select one row
    row = None; chosen_by = None
    if args.hadm_id is not None:
        m = test_df[test_df.get("hadm_id").astype(str) == str(args.hadm_id)]
        if len(m) > 0:
            row = m.iloc[0]; chosen_by = f"hadm_id={args.hadm_id}"
    if row is None and args.subject_id is not None:
        m = test_df[test_df[SUBJECT_COL].astype(str) == str(args.subject_id)]
        if len(m) > 0:
            row = m.iloc[0]; chosen_by = f"subject_id={args.subject_id}"
    if row is None and args.row_index is not None and 0 <= args.row_index < len(test_df):
        row = test_df.iloc[args.row_index]; chosen_by = f"row_index={args.row_index} (in test_df)"
    if row is None:
        row = test_df.iloc[0]; chosen_by = "row_index=0 (default)"

    # Labels
    with open(os.path.join(args.run_dir, "label_space.json"), "r") as f:
        labels_vocab = json.load(f)["labels"]
    idx_map = build_vocab_index(labels_vocab, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros)

    # Canonical GOLD (for display & metrics)
    gold_canon = []
    for g in row[LABEL_COL]:
        kg = normalize_code(g, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros)
        cg = idx_map.get(kg, kg)
        gold_canon.append(cg)
    gold_canon = sorted(set(gold_canon))

    # Load model and decode
    tok = load_tokenizer(args.run_dir)
    model = load_model(args.run_dir, base_model_name_from_config=None, torch_dtype=detect_dtype())
    decoded = generate_once(model, tok, row["input_text"], max_len=args.max_len, gen_max_new=args.gen_max_new)

    # Predictions
    pred_with = predict_codes_from_text(decoded, labels_vocab, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros, use_vocab_filter=True)
    pred_wo   = predict_codes_from_text(decoded, labels_vocab, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros, use_vocab_filter=False)
    raw_no_norm = extract_raw_tokens_no_norm(decoded, dedup=True)

    # Metrics (for the two normalized variants)
    p_with, r_with, f1_with, fn_with, fp_with = per_sample_prf1(gold_canon, pred_with, labels_vocab, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros)
    p_wo, r_wo, f1_wo, fn_wo, fp_wo = per_sample_prf1(gold_canon, pred_wo, labels_vocab, remove_internal_dots=remove_dots, strip_leading_zeros=strip_zeros)

    # Print
    print("\n" + "="*80)
    print("Single-sample generation (", chosen_by, ")")
    print(f"subject_id={row.get(SUBJECT_COL)} | hadm_id={row.get('hadm_id')}")

    print(f"\nDecoded →\n{decoded}")

    print("\n--- GOLD (canonical) ---")
    print(" ".join(gold_canon) if gold_canon else "(none)")

    print("\n--- PRED (WITH vocab filter) ---")
    print(" ".join(pred_with) if pred_with else "(none)")
    print(f"P={p_with:.4f} R={r_with:.4f} F1={f1_with:.4f}")
    print(f"FN ({len(fn_with)}): ", " ".join(fn_with) if fn_with else "(none)")
    print(f"FP ({len(fp_with)}): ", " ".join(fp_with) if fp_with else "(none)")

    print("\n--- PRED (WITHOUT vocab filter; normalized tokens) ---")
    print(" ".join(pred_wo) if pred_wo else "(none)")
    print(f"P={p_wo:.4f} R={r_wo:.4f} F1={f1_wo:.4f}")
    print(f"FN ({len(fn_wo)}): ", " ".join(fn_wo) if fn_wo else "(none)")
    print(f"FP ({len(fp_wo)}): ", " ".join(fp_wo) if fp_wo else "(none)")

    print("\n--- PRED (RAW tokens; NO normalization, NO vocab filter) ---")
    print(" ".join(raw_no_norm) if raw_no_norm else "(none)")
    print("(metrics not computed on raw tokens)")

    # Prompt tail for context
    tail_lines = str(row["input_text"]).splitlines()[-8:]
    print("\n--- Prompt tail ---")
    for line in tail_lines:
        line = line.strip()
        if line.startswith("[CODES]"):
            print("[CODES]  <-- target starts after this marker during training")
        else:
            print(line[:180])

if __name__ == "__main__":
    main()
