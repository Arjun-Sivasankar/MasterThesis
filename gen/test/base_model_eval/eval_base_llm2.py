# -*- coding: utf-8 -*-
"""
Fast BASE-model evaluation for generative ICD code prediction (no finetuning, no LoRA).

Why this is faster
- Single-pass generation -> parse twice (filtered & unfiltered) = ~1× time, not 2×
- Left padding for generation (more efficient with uneven prompts)
- Optional early stop on double newline to avoid rambling
- Higher default generation batch size

Defaults
- BOTH modalities ON (structured + notes)
- Filter mode BOTH (but still single-pass generation)

Examples
--------
# Full eval, both metrics, both modalities:
python eval_base_llama_fast.py --data_pickle mergeddf.pkl --model meta-llama/Llama-3.2-1B-Instruct

# Faster: raise batch size (if VRAM allows), still single-pass:
python eval_base_llama_fast.py --gen_batch_size 8

# Filtered metrics only (same single-pass gen, slightly less postproc):
python eval_base_llama_fast.py --filter_mode filtered
"""

import os, re, json, time, argparse, datetime, logging, pickle, random
from typing import List, Any, Dict, Optional
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from transformers import StoppingCriteria, StoppingCriteriaList

# ----------------- Quiet & deterministic -----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------- Columns / text sections -----------------
SUBJECT_COL = "subject_id_x"
LABEL_COL   = "icd_code"
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

# ----------------- Helpers -----------------
def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def make_run_dir(base="runs_base_eval_fast", run_name=None):
    tag = run_name or f"{now_tag()}_base_eval_fast"
    path = os.path.join(base, tag)
    os.makedirs(path, exist_ok=False)
    return path

def save_json(path: str, obj: dict):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def clean_text(x: Any) -> str:
    if isinstance(x, (list, tuple, set, dict, np.ndarray, pd.Series)): return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def to_list(x) -> List[str]:
    """Robust label coercion -> list[str], safe for arrays/lists/strings/NaN."""
    def _norm(z):
        s = str(z); s = re.sub(r"\s+","", s.upper())
        return s[:-1] if s.endswith(".") else s
    if isinstance(x, (list, tuple, set)):
        return [_norm(v) for v in x if str(v).strip()]
    if isinstance(x, np.ndarray):
        return [_norm(v) for v in x.reshape(-1).tolist() if str(v).strip()]
    if isinstance(x, pd.Series):
        return [_norm(v) for v in x.tolist() if str(v).strip()]
    try:
        if pd.isna(x): return []
    except Exception:
        pass
    if isinstance(x, str):
        s = x.strip()
        if not s: return []
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
                    if isinstance(v, np.ndarray): v = v.tolist()
                    if isinstance(v, pd.Series):  v = v.tolist()
                    return [_norm(z) for z in v if str(z).strip()]
            except Exception:
                pass
        return [_norm(t) for t in re.split(r"[,\s]+", s) if t]
    return [_norm(x)]

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

def serialize_notes(row: pd.Series, text_cols: List[str]) -> str:
    chunks=[]
    for col in text_cols:
        if col in row:
            t = clean_text(row[col])
            if t: chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)

def build_input_text(row: pd.Series, use_structured=True, use_notes=True, text_cols=TEXT_COLS_SAFE) -> str:
    s = [f"[VISIT] subject_id={row.get(SUBJECT_COL,'?')} hadm_id={row.get('hadm_id','?')}"]
    if use_structured: s.append(serialize_structured(row))
    if use_notes:
        t = serialize_notes(row, text_cols)
        if t: s.append(t)
    s.append("[TASK] Predict ICD diagnosis codes (space-separated). Output ONLY the codes, separated by single spaces.")
    s.append("[CODES]")
    return "\n".join([x for x in s if x])

def subject_splits(df: pd.DataFrame, subject_col=SUBJECT_COL,
                   test_size=0.10, val_size=0.10, seed=42):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    logging.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

def lock_label_space(full_df: pd.DataFrame) -> MultiLabelBinarizer:
    all_codes = sorted({str(code) for codes in full_df[LABEL_COL] for code in codes})
    mlb = MultiLabelBinarizer(classes=all_codes); mlb.fit([all_codes])
    logging.info(f"Total unique ICD codes: {len(all_codes)}")
    return mlb

def normalize_code(c: str) -> str:
    c = c.strip().upper()
    c = re.sub(r"\s+","", c)
    return c[:-1] if c.endswith(".") else c

# ----------------- Optimized generation -----------------
class DoubleNewlineStop(StoppingCriteria):
    def __init__(self, tok, max_new_tokens: int, lookback: int = 96):
        self.tok = tok; self.max_new = max_new_tokens; self.lookback = lookback
    def __call__(self, input_ids, scores, **kwargs):
        last = input_ids[0].tolist()[-min(self.lookback, len(input_ids[0])):]
        tail = self.tok.decode(last, skip_special_tokens=True)
        return "\n\n" in tail or len(last) >= self.max_new

@torch.inference_mode()
def generate_texts(model, tok, prompts: List[str],
                   max_new=96, batch_size=8, max_len=3072,
                   stop_on_double_newline=True) -> List[str]:
    texts = []
    old_side = tok.padding_side
    tok.padding_side = "left"  # faster for varied prompt lengths
    stops = StoppingCriteriaList([DoubleNewlineStop(tok, max_new)]) if stop_on_double_newline else None
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False, num_beams=1,
            no_repeat_ngram_size=2,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            use_cache=True, stopping_criteria=stops
        )
        texts.extend(tok.batch_decode(out, skip_special_tokens=True))
    tok.padding_side = old_side
    return texts

def parse_codes(text: str, labels_vocab: Optional[List[str]] = None) -> List[str]:
    tail = text.split("[CODES]")[-1]
    tokens = re.split(r"[^A-Za-z0-9\.]+", tail)
    cand = [normalize_code(z) for z in tokens if z]
    allowed = set(labels_vocab) if labels_vocab is not None else None
    seen, keep = set(), []
    for c in cand:
        if (allowed is None or c in allowed) and c not in seen:
            seen.add(c); keep.append(c)
    return keep

def codes_to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(label_vocab)}
    Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
    for i, lst in enumerate(code_lists):
        for c in lst:
            j = idx.get(c)
            if j is not None: Y[i, j] = 1
    return Y

def eval_sets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1":   f1_score(y_true, y_pred, average="micro",   zero_division=0),
        "macro_f1":   f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "micro_precision":   precision_score(y_true, y_pred, average="micro",   zero_division=0),
        "macro_precision":   precision_score(y_true, y_pred, average="macro",   zero_division=0),
        "samples_precision": precision_score(y_true, y_pred, average="samples", zero_division=0),
        "micro_recall":      recall_score(y_true, y_pred, average="micro",      zero_division=0),
        "macro_recall":      recall_score(y_true, y_pred, average="macro",      zero_division=0),
        "samples_recall":    recall_score(y_true, y_pred, average="samples",    zero_division=0),
    }

def show_examples(df: pd.DataFrame, preds: List[List[str]], n=5, seed=42):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n, len(df)), replace=False)
    for i in idxs:
        row = df.iloc[i]
        gold = sorted({normalize_code(c) for c in row[LABEL_COL]})
        pred = preds[i]
        print("\n" + "="*80)
        if show_prompts and "input_text" in row:
            print("\n----- PROMPT (FULL) -----")
            print(row["input_text"])  # Print full prompt without truncation
            print("----- END PROMPT -----\n")
            
        print(f"idx={i} subject_id={row.get(SUBJECT_COL)} hadm_id={row.get('hadm_id')}")
        print("- GOLD:", " ".join(gold) if gold else "(none)")
        print("- PRED:", " ".join(pred) if pred else "(none)")

# ----------------- Load model/tokenizer -----------------
def load_base(model_name: str):
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # we switch to left only during generation
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = True
    return model, tok

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pickle", type=str, default="mergeddf.pkl")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--run_name", type=str, default=None)

    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--gen_batch_size", type=int, default=16)
    ap.add_argument("--stop_on_double_newline", action="store_true")
    ap.add_argument("--no_stop_on_double_newline", dest="stop_on_double_newline", action="store_false")
    ap.set_defaults(stop_on_double_newline=True)

    # Modalities: ON by default
    group_struct = ap.add_mutually_exclusive_group()
    group_struct.add_argument("--structured", dest="use_structured", action="store_true")
    group_struct.add_argument("--no-structured", dest="use_structured", action="store_false")
    group_notes = ap.add_mutually_exclusive_group()
    group_notes.add_argument("--notes", dest="use_notes", action="store_true")
    group_notes.add_argument("--no-notes", dest="use_notes", action="store_false")
    ap.set_defaults(use_structured=True, use_notes=True)

    ap.add_argument("--filter_mode", choices=["filtered","unfiltered","both"], default="both")
    ap.add_argument("--subset", type=int, default=None, help="limit #test samples for quick runs")
    ap.add_argument("--save_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true", help="try torch.compile for small extra speed")
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"CUDA: {torch.cuda.is_available()} | Model: {args.model}")
    print(f"Config -> MAX_LEN:{args.max_len} GEN_MAX_NEW:{args.gen_max_new} GEN_BS:{args.gen_batch_size} "
          f"STRUCTURED:{args.use_structured} NOTES:{args.use_notes} FILTER_MODE:{args.filter_mode}")

    # Data
    df = pickle.load(open(args.data_pickle, "rb"))
    assert LABEL_COL in df.columns and SUBJECT_COL in df.columns
    df[LABEL_COL] = df[LABEL_COL].apply(to_list)
    df["input_text"] = df.apply(lambda r: build_input_text(
        r, use_structured=args.use_structured, use_notes=args.use_notes), axis=1)
    df = df[df["input_text"].str.len() > 0].reset_index(drop=True)

    train_df, val_df, test_df = subject_splits(df, subject_col=SUBJECT_COL, seed=args.seed)
    mlb = lock_label_space(df)
    labels_vocab = mlb.classes_.tolist()
    y_test = mlb.transform(test_df[LABEL_COL].tolist())
    if args.subset is not None:
        test_df = test_df.iloc[:min(args.subset, len(test_df))].copy()
        y_test = y_test[:len(test_df)]

    # Run dir
    run_dir = make_run_dir(run_name=args.run_name)
    print("Run dir:", run_dir)
    save_json(os.path.join(run_dir, "config.json"), {
        **vars(args), "labels": len(labels_vocab), "test_size": len(test_df)
    })

    # Model
    model, tok = load_base(args.model)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}); continuing without it.")

    # --------- Single-pass generation ---------
    test_prompts = test_df["input_text"].astype(str).tolist()
    t0 = time.perf_counter()
    texts = generate_texts(
        model, tok, test_prompts,
        max_new=args.gen_max_new, batch_size=args.gen_batch_size,
        max_len=args.max_len, stop_on_double_newline=args.stop_on_double_newline
    )
    gen_secs = time.perf_counter() - t0
    print(f"[TIME] generation (single pass): {gen_secs/60:.2f} min for {len(test_prompts)} samples")

    results = {}

    # Parse/score: filtered
    if args.filter_mode in ("filtered","both"):
        preds_f = [parse_codes(t, labels_vocab) for t in texts]
        Yf = codes_to_multihot(preds_f, labels_vocab)
        m_f = eval_sets(y_test, Yf); m_f["gen_seconds"] = gen_secs
        results["filtered"] = m_f
        save_json(os.path.join(run_dir, "metrics_filtered.json"), m_f)
        print("\n=== BASE metrics (STRICT VOCAB FILTER) ===")
        print(json.dumps(m_f, indent=2))
        print("\n--- Sample predictions (filtered) ---")
        show_examples(test_df, preds_f, n=args.save_samples, seed=args.seed)

    # Parse/score: unfiltered
    if args.filter_mode in ("unfiltered","both"):
        preds_u = [parse_codes(t, None) for t in texts]
        Yu = codes_to_multihot(preds_u, labels_vocab)  # off-vocab codes drop out naturally
        m_u = eval_sets(y_test, Yu); m_u["gen_seconds"] = gen_secs
        results["unfiltered"] = m_u
        save_json(os.path.join(run_dir, "metrics_unfiltered.json"), m_u)
        print("\n=== BASE metrics (NO FILTER) ===")
        print(json.dumps(m_u, indent=2))
        print("\n--- Sample predictions (unfiltered) ---")
        show_examples(test_df, preds_u, n=args.save_samples, seed=args.seed)

    save_json(os.path.join(run_dir, "metrics_summary.json"), results)
    print(f"\nDone. Results saved under: {run_dir}")

if __name__ == "__main__":
    main()
