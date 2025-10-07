# -*- coding: utf-8 -*-
"""
Generative ICD-9 code prediction with LoRA (DDP-safe, epoch timing, optimized generation)

- Rank detection works BEFORE DDP init (uses LOCAL_RANK/RANK env)
- Enhanced timing logs with timestamps at appropriate points
- Optimized test generation with configurable batch size
- Proper device placement and memory management
- Mixed precision inference for generation
- Top-K ICD-9 code filtering capability
"""
import os
import re
import time
import json
import atexit
import random
import logging
import inspect
import datetime
import argparse
from typing import List, Any, Dict, Tuple, Set
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
import torch.distributed

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, EarlyStoppingCallback, Trainer, TrainerCallback
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model

# ---------------- Env & logging ----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# Enable TF32 on Ampere+ (A100/H100, etc.)
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

# ----- DDP helpers (handle rank BEFORE dist.init) -----
def dist_is_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _env_rank():
    for k in ("LOCAL_RANK", "RANK"):
        v = os.environ.get(k)
        if v is not None:
            return int(v)
    return 0

def get_rank():
    return torch.distributed.get_rank() if dist_is_initialized() else _env_rank()

def is_main_process():
    return get_rank() == 0

def barrier():
    if dist_is_initialized():
        try: torch.distributed.barrier()
        except Exception: pass

def rank0_print(*a, **k):
    if is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}]", *a, **k)

# Quiet non-main logs asap
if not is_main_process():
    logging.getLogger().setLevel(logging.ERROR)

def _cleanup_dist():
    try:
        if dist_is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
atexit.register(_cleanup_dist)

# ---------------- Args ----------------
def get_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", default=None, help="If provided, will subject-split into train/val/test.")
    ap.add_argument("--train_pickle", default=None)
    ap.add_argument("--val_pickle", default=None)
    ap.add_argument("--test_pickle", default=None)
    ap.add_argument("--icd9_pickle", default="MasterThesis/dataset/codes/icd9.pkl", help="Path to complete ICD-9 code list")
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--use_complete_icd9", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=0, 
                    help="If >0, only use top-k most frequent ICD-9 codes for training (0=use all)")

    # model/prompt
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--tgt_reserve_tok", type=int, default=128)
    ap.add_argument("--gen_max_new", type=int, default=96)

    # training
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--early_stop", type=int, default=1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--ddp_find_unused", type=int, default=0)

    # size & seed
    ap.add_argument("--train_size", type=int, default=-1, help="-1=all")
    ap.add_argument("--seed", type=int, default=42)

    # run dirs
    ap.add_argument("--run_root", default="runs_gen/diffsize")
    ap.add_argument("--run_name", default=None)

    # eval reporting
    ap.add_argument("--eval_sample_size", type=int, default=100, help="Subset of VAL for gen-eval during training")
    ap.add_argument("--test_examples", type=int, default=5, help="Number of test examples to show")
    ap.add_argument("--test_batch_size", type=int, default=16, help="Batch size for final test generation")

    # misc
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--merge_after", type=int, default=0)
    return ap.parse_args()

# ---------------- ICD-9 Code Handling ----------------
def format_icd9_properly(code: str) -> str:
    code = str(code).strip().upper()
    code = re.sub(r"\s+", "", code)
    if code.endswith("."):
        code = code[:-1]
    if not code:
        return code
    # Numeric head
    if code[0].isdigit():
        if '.' not in code and len(code) > 3:
            code = f"{code[:3]}.{code[3:]}"
        return code
    # V/E head
    if code[0] in ('V', 'E'):
        if '.' not in code and len(code) > 3:
            code = f"{code[:3]}.{code[3:]}"
        return code
    return code

def is_valid_icd9(code: str) -> bool:
    if not code:
        return False
    return code[0].isdigit() or code.startswith('V') or code.startswith('E')

def get_icd9_parent(code: str) -> str:
    if not code or len(code) < 3:
        return code
    head = code[0]
    if head.isdigit() or head in ('V', 'E'):
        return code.split('.')[0] if '.' in code else code[:3]
    return code

# ---------------- Visualization functions for Top-K ----------------
# ---------------- Visualization functions for Top-K ----------------
def plot_code_distribution(code_counter, top_k=None, save_path="./top-k/figs"):
    """Plot frequency, histogram, and cumulative coverage of ICD-9 code counts."""
    if not is_main_process():
        return

    Path(save_path).mkdir(parents=True, exist_ok=True)

    if not code_counter:
        return

    sorted_codes = sorted(code_counter.items(), key=lambda x: x[1], reverse=True)
    codes, counts = zip(*sorted_codes)

    # 1) Rank-frequency (log y)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(counts)), counts)
    plt.xlabel('Code Rank (by frequency)')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.title('ICD-9 Code Frequency Distribution')
    if top_k and top_k < len(counts):
        plt.axvline(x=top_k, linestyle='--')
        plt.text(top_k, counts[0], f'Top-{top_k} cutoff', ha='right', va='top')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/code_frequency_distribution.png", dpi=300)
    plt.close()

    # 2) Histogram of counts
    plt.figure(figsize=(12, 6))
    plt.hist(counts, bins=50)
    plt.xlabel('Frequency')
    plt.ylabel('Number of Codes')
    plt.title('Histogram of ICD-9 Code Frequencies')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/code_frequency_histogram.png", dpi=300)
    plt.close()

    # 3) Cumulative coverage (CDF)
    total_occurrences = sum(counts)
    cumulative = np.cumsum(counts) / max(1, total_occurrences)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(cumulative)), cumulative)
    plt.xlabel('Number of Top Codes')
    plt.ylabel('Cumulative Coverage')
    plt.title('Cumulative Coverage of ICD-9 Codes')
    for coverage in [0.5, 0.75, 0.9, 0.95, 0.99]:
        idx = int(np.searchsorted(cumulative, coverage))
        if 0 <= idx < len(cumulative):
            plt.plot(idx, cumulative[idx], 'o')
            plt.text(idx, cumulative[idx], f'Top-{idx} ({coverage:.0%})', ha='left', va='bottom')
    if top_k and top_k < len(counts):
        cov_k = cumulative[top_k - 1] if top_k > 0 else 0
        plt.axvline(x=top_k, linestyle='--')
        plt.text(top_k, cov_k, f'Top-{top_k} ({cov_k:.1%})', ha='right', va='bottom')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/code_cumulative_coverage.png", dpi=300)
    plt.close()

    # Save top-k list
    if top_k and top_k > 0:
        with open(f"{save_path}/top_{top_k}_codes.txt", 'w') as f:
            f.write(f"Top-{top_k} ICD-9 codes by frequency:\n\n")
            f.write("Rank\tCode\tFrequency\n")
            f.write("-" * 30 + "\n")
            for i, (code, count) in enumerate(sorted_codes[:top_k], start=1):
                f.write(f"{i}\t{code}\t{count}\n")

# ---------------- Prompting helpers ----------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x: Any) -> str:
    if pd.isna(x): return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def _is_scalar_nan(v) -> bool:
    """Return True only for scalar NaN/NA; never for arrays/containers."""
    try:
        # Containers are never treated as NaN here
        if isinstance(v, (list, tuple, set, dict, np.ndarray, pd.Series)):
            return False
        return pd.isna(v)
    except Exception:
        return False

def to_list(x) -> List[str]:
    """
    Convert many possible label representations into a list[str].
    Handles: list/tuple/set, numpy arrays, pandas Series, Python-list-in-string,
    comma/space separated strings, and ignores scalar NaN/None.
    """
    if x is None or _is_scalar_nan(x):
        return []

    # Already a Python container
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x if not _is_scalar_nan(v)]

    # Numpy / pandas containers
    if isinstance(x, (np.ndarray, pd.Series)):
        try:
            seq = x.tolist()
        except Exception:
            seq = list(x)
        return [str(v) for v in seq if not _is_scalar_nan(v)]

    # String-ish inputs
    s = str(x).strip()
    if not s:
        return []

    # Try to parse a Python list literal
    if s.startswith('[') and s.endswith(']'):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(z) for z in v if not _is_scalar_nan(z)]
            if isinstance(v, (np.ndarray, pd.Series)):
                vv = v.tolist() if hasattr(v, "tolist") else list(v)
                return [str(z) for z in vv if not _is_scalar_nan(z)]
        except Exception:
            pass

    # Fallback: split by comma/whitespace
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
    chunks=[]
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t: chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)

def build_input_text(row: pd.Series, use_structured=True, use_notes=True,
                     subject_col="subject_id_x") -> str:
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

# ---------------- Splits & labels ----------------
def subject_splits(df: pd.DataFrame, subject_col: str,
                   test_size=0.10, val_size=0.10, seed=42):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    log.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

def lock_label_space(frames: List[pd.DataFrame], label_col: str,
                     icd9_pkl_path: str = None, use_complete: bool = False,
                     top_k: int = 0) -> MultiLabelBinarizer:
    """
    Build the label space from TRAINING data only, with optional top-k filtering by frequency.
    If use_complete is True and icd9_pkl_path is provided, restrict the complete code list
    to the (top-k) training codes to lock the space.
    """
    log = logging.getLogger(__name__)

    # Extract TRAIN (only TRAIN is used for frequency)
    train_df = frames[0] if len(frames) > 0 else None

    # ---- Count frequencies from TRAIN set only ----
    train_unique_codes: Set[str] = set()
    code_counter: Counter = Counter()

    if train_df is not None and label_col in train_df.columns and len(train_df) > 0:
        if is_main_process():
            sample = train_df[label_col].iloc[0]
            log.info(f"[lock_label_space] Sample label type: {type(sample)}")
            log.info(f"[lock_label_space] Sample label content: {sample}")

        for labels in train_df[label_col].dropna():
            items = to_list(labels)
            for c in items:
                norm = format_icd9_properly(c)
                if is_valid_icd9(norm):
                    code_counter[norm] += 1

        train_unique_codes = set(code_counter.keys())
        log.info(f"[lock_label_space] Found {len(train_unique_codes)} unique valid ICD-9 codes in TRAIN set")
    else:
        # Fallback: derive from all frames (train/val/test) if train missing
        for fr in [f for f in frames if f is not None and label_col in f.columns]:
            for labels in fr[label_col].dropna():
                items = to_list(labels)
                for c in items:
                    norm = format_icd9_properly(c)
                    if is_valid_icd9(norm):
                        code_counter[norm] += 1
        train_unique_codes = set(code_counter.keys())
        if train_unique_codes:
            log.warning("[lock_label_space] Training dataframe missing/invalid; "
                        "label space derived from all frames.")
        log.info(f"[lock_label_space] Found {len(train_unique_codes)} unique valid ICD-9 codes across ALL frames")

    # ---- Apply top-k if requested ----
    if top_k and top_k > 0:
        nuniq = len(train_unique_codes)
        if nuniq == 0:
            log.warning("[lock_label_space] Top-k requested but no training codes found; label space will be empty.")
        elif top_k < nuniq:
            sorted_codes = code_counter.most_common()
            top_k_codes = {code for code, _ in sorted_codes[:top_k]}
            filtered_count = nuniq - len(top_k_codes)
            train_unique_codes = top_k_codes
            log.info(f"[lock_label_space] Using only top-{top_k} most frequent TRAIN codes "
                     f"(filtered {filtered_count} rarer codes)")
            if is_main_process():
                min_freq = sorted_codes[top_k - 1][1]
                rank0_print(f"Minimum frequency threshold at top-{top_k}: {min_freq} occurrences")
                rank0_print("Top 10 most common codes:")
                for i, (code, count) in enumerate(sorted_codes[:min(10, len(sorted_codes))], start=1):
                    rank0_print(f"  #{i}: {code} - {count} occurrences")
                if top_k < len(sorted_codes):
                    rank0_print(f"Codes around the cutoff (top-{top_k}):")
                    lo = max(0, top_k - 3); hi = min(len(sorted_codes), top_k + 2)
                    for i, (code, count) in enumerate(sorted_codes[lo:hi], start=lo + 1):
                        in_out = "included" if code in top_k_codes else "excluded"
                        rank0_print(f"  #{i}: {code} - {count} ({in_out})")
                try:
                    plot_code_distribution(code_counter, top_k=top_k, save_path="./top-k/figs")
                except Exception as e:
                    rank0_print(f"Error generating visualization: {e}")
        else:
            log.info(f"[lock_label_space] top_k={top_k} >= unique TRAIN codes ({nuniq}); using all TRAIN codes.")

    # ---- If NOT using complete ICD-9 list, fit on training codes directly ----
    if not use_complete or not icd9_pkl_path:
        all_codes = sorted(train_unique_codes)
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        log.info(f"[lock_label_space] Using {len(all_codes)} classes from "
                 f"{'top-k of ' if top_k > 0 else ''}TRAINING data only")
        return mlb

    # ---- Use the complete ICD-9 list, optionally restricted to (top-k) training codes ----
    try:
        icd9_df = pd.read_pickle(icd9_pkl_path)
        complete_codes = icd9_df['icd_code'].astype(str).tolist()
        complete_codes = [format_icd9_properly(c) for c in complete_codes]
        complete_codes = sorted({c for c in complete_codes if is_valid_icd9(c)})

        if top_k and top_k > 0:
            complete_codes = [c for c in complete_codes if c in train_unique_codes]
            log.info(f"[lock_label_space] Filtered complete ICD-9 list to top-{top_k} TRAIN codes")

        log.info(f"[lock_label_space] Loaded {len(complete_codes)} "
                 f"{'top-k ' if top_k > 0 else ''}complete ICD-9 codes from {icd9_pkl_path}")

        mlb = MultiLabelBinarizer(classes=complete_codes)
        mlb.fit([complete_codes])

        train_set = set(train_unique_codes)
        comp_set = set(complete_codes)
        in_complete = len(train_set & comp_set)
        not_in_complete = len(train_set - comp_set)
        log.info(f"[lock_label_space] Training coverage vs complete: in={in_complete}, missing={not_in_complete}")
        if not_in_complete > 0:
            log.warning("[lock_label_space] Some training codes not found in the complete ICD-9 set.")
        return mlb

    except Exception as e:
        log = logging.getLogger(__name__)
        log.error(f"[lock_label_space] Error loading complete ICD-9 codes: {e}")
        log.warning("[lock_label_space] Falling back to training-data-only label space")
        all_codes = sorted(train_unique_codes)
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        return mlb

def y_multi_hot(mlb: MultiLabelBinarizer, lists):
    formatted_lists = []
    for row in lists:
        if not row:
            formatted_lists.append([])
        else:
            formatted_lists.append([format_icd9_properly(str(c)) for c in row if is_valid_icd9(format_icd9_properly(str(c)))])
    return mlb.transform(formatted_lists)

# ---------------- Subject-safe subsetting ----------------
def nested_subject_sample(train_df, target_n, subject_col="subject_id_x", seed=13):
    if target_n is None or target_n < 0 or target_n >= len(train_df):
        return train_df
    
    rng = np.random.default_rng(seed)
    subjects = train_df[subject_col].drop_duplicates().tolist()
    rng.shuffle(subjects)
    chosen, count = [], 0
    for s in subjects:
        n_to_add = len(train_df[train_df[subject_col] == s])
        if count + n_to_add <= target_n:
            chosen.append(s)
            count += n_to_add
        else:
            break
    
    sub = train_df[train_df[subject_col].isin(chosen)].copy()
    log.info(f"[subset] requested={target_n} got={len(sub)} unique_subjects={sub[subject_col].nunique()}")
    return sub

# ---------------- Dataset (pre-tokenized) ----------------
class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int, label_col: str):
        self.name = rows.attrs.get('name', 'dataset')
        self.rows = rows
        self.tok = tok
        self.max_len = max_len
        self.tgt_reserve = tgt_reserve
        self.label_col = label_col
        
        # Pre-tokenize data for efficiency
        self.inputs = []
        for _, row in rows.iterrows():
            input_text = build_input_text(row)
            if input_text:
                # Truncate text to leave room for target tokens
                enc = tok(input_text, return_tensors="pt", truncation=True, max_length=max_len - tgt_reserve)
                self.inputs.append({
                    'input_ids': enc.input_ids[0],
                    'attention_mask': enc.attention_mask[0],
                    'labels': row.get(label_col, []),
                    'text': input_text
                })
        log.info(f"[{self.name}] rows with input_text: {len(self.inputs)}")
                
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

# ---- Hand-rolled collator (no tok.pad -> no warning) ----
def pad_collate(features, tok):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {
        "input_ids": torch.zeros(len(features), max_len).long(),
        "attention_mask": torch.zeros(len(features), max_len).long(),
        "labels": [f["labels"] for f in features],
        "text": [f["text"] for f in features]
    }
    
    for i, f in enumerate(features):
        batch["input_ids"][i, :len(f["input_ids"])] = f["input_ids"]
        batch["attention_mask"][i, :len(f["attention_mask"])] = f["attention_mask"]
    
    return batch

# ---------------- Model ----------------
def unwrap_model(m):
    if hasattr(m, 'module'):
        return m.module
    return m

def load_lm_and_tokenizer(model_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map=device_map,
        load_in_8bit=False,
    )

    # Set up LoRA config
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create PEFT model
    model = get_peft_model(model, config)
    
    rank0_print(f"CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    trainable_params, all_params = 0, 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    rank0_print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
    
    return model, tokenizer

# ---------------- Generation & metrics ----------------
@torch.no_grad()
def generate_codes(model, input_text: str, tokenizer, 
                  max_new_tokens=64, temperature=0.01, 
                  top_p=0.9, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Generate ICD-9 codes for a given input text"""
    model_dtype = next(model.parameters()).dtype
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.amp.autocast(device_type=device, dtype=model_dtype):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Get just the newly generated tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Extract the codes from the result
    codes = []
    for code in re.split(r'[\s,;]+', result):
        code = format_icd9_properly(code)
        if code and is_valid_icd9(code):
            codes.append(code)
    
    return codes

def codes_to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
    """Convert a list of code lists to a multi-hot encoded array"""
    if not code_lists:
        return np.zeros((0, len(label_vocab)))
    
    vocab_set = set(label_vocab)
    result = np.zeros((len(code_lists), len(label_vocab)))
    
    for i, codes in enumerate(code_lists):
        for code in codes:
            norm_code = format_icd9_properly(code)
            if norm_code in vocab_set:
                idx = label_vocab.index(norm_code)
                result[i, idx] = 1
    
    return result

def eval_sets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for multi-label classification"""
    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
        return {"precision": 0, "recall": 0, "f1": 0}
    
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    samples_f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "micro_f1": float(f1),
        "samples_f1": float(samples_f1)
    }

def hierarchical_eval(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> Dict[str, float]:
    """Evaluate hierarchical recall - did we get the parent code right?"""
    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
        return {"parent_recall": 0}
    
    # Map each code to its parent
    parent_map = {code: get_icd9_parent(code) for code in label_vocab}
    
    # For each true code, check if any prediction has the same parent
    parent_hits = 0
    parent_total = 0
    
    for i in range(y_true.shape[0]):
        true_indices = np.where(y_true[i] == 1)[0]
        pred_indices = np.where(y_pred[i] == 1)[0]
        
        true_parents = {parent_map[label_vocab[idx]] for idx in true_indices}
        pred_parents = {parent_map[label_vocab[idx]] for idx in pred_indices}
        
        parent_hits += len(true_parents & pred_parents)
        parent_total += len(true_parents)
    
    parent_recall = parent_hits / parent_total if parent_total > 0 else 0
    return {"parent_recall": float(parent_recall)}

# ---------------- Custom training callbacks ----------------
class DetailedEvalCallback(TrainerCallback):
    """Custom callback to evaluate generation performance on a sample of the validation set"""
    def __init__(self, eval_dataset, tok, batch_size=4, 
                 sample_size=100, label_vocab=None, 
                 max_new_tokens=64, temperature=0.01,
                 output_dir=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tok
        self.batch_size = batch_size
        self.sample_size = min(sample_size, len(eval_dataset))
        self.label_vocab = label_vocab
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.best_f1 = 0
        self.output_dir = output_dir
        
        # Create a fixed sample for consistency
        rng = np.random.RandomState(42)
        indices = rng.choice(len(eval_dataset), self.sample_size, replace=False)
        self.sample_indices = indices
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        if not is_main_process():
            return
        
        epoch = state.epoch
        rank0_print(f"[GenEval] Starting generation evaluation for epoch {epoch:.1f}...")
        
        # Generate predictions in batches
        true_codes = []
        pred_codes = []
        
        model_unwrapped = unwrap_model(model)
        model_unwrapped.eval()
        
        start_time = time.time()
        rank0_print(f"Generating predictions for {self.sample_size} samples in batches of {self.batch_size}...")
        
        batch_times = []
        
        for i in range(0, self.sample_size, self.batch_size):
            batch_start = time.time()
            batch_indices = self.sample_indices[i:i+self.batch_size]
            batch_samples = [self.eval_dataset[idx] for idx in batch_indices]
            
            for sample in batch_samples:
                # Generate codes
                input_text = sample['text']
                true = sample['labels']
                pred = generate_codes(
                    model_unwrapped, 
                    input_text, 
                    self.tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                
                true_codes.append(true)
                pred_codes.append(pred)
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # Calculate and print progress with timing info
            samples_so_far = min(i + self.batch_size, self.sample_size)
            percent_done = 100.0 * samples_so_far / self.sample_size
            avg_batch_time = sum(batch_times) / len(batch_times)
            batches_left = (self.sample_size - samples_so_far) / self.batch_size
            time_left = avg_batch_time * batches_left
            
            rank0_print(
                f"Generated {samples_so_far}/{self.sample_size} samples ({percent_done:.1f}%) - "
                f"Batch time: {batch_time:.2f}s - "
                f"Est. remaining: {time_left:.2f}s"
            )
        
        end_time = time.time()
        rank0_print(f"Generation complete in {end_time - start_time:.2f}s ({(end_time - start_time)/self.sample_size:.3f}s per sample)")
        
        # Evaluate the predictions
        y_true = codes_to_multihot(true_codes, self.label_vocab)
        y_pred = codes_to_multihot(pred_codes, self.label_vocab)
        
        metrics = eval_sets(y_true, y_pred)
        hier_metrics = hierarchical_eval(y_true, y_pred, self.label_vocab)
        
        # Combine metrics
        combined_metrics = {**metrics, **hier_metrics}
        
        # Check if this is the best model so far
        if combined_metrics['micro_f1'] > self.best_f1:
            self.best_f1 = combined_metrics['micro_f1']
            rank0_print(f"[GenEval] New best micro_f1: {self.best_f1:.4f}")
            
            # Save the best metrics
            if self.output_dir:
                with open(os.path.join(self.output_dir, "best_metrics.json"), "w") as f:
                    json.dump({
                        "epoch": epoch,
                        **combined_metrics,
                        "generation_sample_size": self.sample_size
                    }, f, indent=2)
        
        rank0_print(
            f"[GenEval] epoch: {epoch:.1f}, "
            f"micro_f1: {combined_metrics['micro_f1']:.4f}, "
            f"samples_f1: {combined_metrics['samples_f1']:.4f}, "
            f"parent_recall: {combined_metrics['parent_recall']:.4f}, "
            f"gen_eval_time: {end_time - start_time:.1f}s"
        )
        
        return combined_metrics

# ---------------- Run helpers ----------------
def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def make_run_dir(base, tag):
    path = os.path.join(base, tag)
    os.makedirs(path, exist_ok=True)
    return path

def save_json(path: str, obj: dict):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def show_test_predictions(df: pd.DataFrame, preds: List[List[str]],
                          label_col: str, label_vocab: List[str],
                          n_show: int = 5, seed: int = 0):
    if len(preds) == 0:
        return
    
    # Sample n_show predictions
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(preds), min(n_show, len(preds)), replace=False)
    
    rank0_print(f"\n===== {len(indices)} Example Predictions =====")
    
    for i, idx in enumerate(indices):
        # Get sample from the dataframe
        row = df.iloc[idx]
        
        # Get true and predicted codes
        true_codes = sorted(row.get(label_col, []))
        pred_codes = sorted(preds[idx])
        
        # Convert to multi-hot for precision/recall calculation
        true_hot = codes_to_multihot([true_codes], label_vocab)[0]
        pred_hot = codes_to_multihot([pred_codes], label_vocab)[0]
        
        # Calculate precision, recall, F1
        true_positive = sum((true_hot == 1) & (pred_hot == 1))
        precision = true_positive / max(1, sum(pred_hot))
        recall = true_positive / max(1, sum(true_hot))
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        
        # Print sample info
        rank0_print(f"\nExample {i+1}: hadm_id={row.get('hadm_id', 'N/A')}")
        rank0_print(f"  True codes ({len(true_codes)}): {' '.join(true_codes)}")
        rank0_print(f"  Pred codes ({len(pred_codes)}): {' '.join(pred_codes)}")
        rank0_print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Show which codes were correctly and incorrectly predicted
        correct = set(true_codes) & set(pred_codes)
        missed = set(true_codes) - set(pred_codes)
        extra = set(pred_codes) - set(true_codes)
        
        if correct:
            rank0_print(f"  Correct: {' '.join(sorted(correct))}")
        if missed:
            rank0_print(f"  Missed:  {' '.join(sorted(missed))}")
        if extra:
            rank0_print(f"  Extra:   {' '.join(sorted(extra))}")

# ---- TrainingArguments builder (prefers eval_strategy) ----
def make_training_args(args, RUN_DIR):
    tr_args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        ddp_find_unused_parameters=bool(args.ddp_find_unused),
        remove_unused_columns=False,
        report_to="none",  # Disable wandb
        log_level="error",
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        dataloader_drop_last=False,
    )
    return tr_args

# ---------------- Main ----------------
def main():
    # Parse arguments
    args = get_args()
    set_seed(args.seed)
    
    # Load data
    if args.data_pickle:
        # Subject split a single data file
        df = pd.read_pickle(args.data_pickle)
        train_df, val_df, test_df = subject_splits(df, args.subject_col, seed=args.seed)
    else:
        # Use provided train/val/test splits
        train_df = pd.read_pickle(args.train_pickle) if args.train_pickle else None
        val_df = pd.read_pickle(args.val_pickle) if args.val_pickle else None
        test_df = pd.read_pickle(args.test_pickle) if args.test_pickle else None
    
    # Apply size limit if specified
    if args.train_size > 0 and train_df is not None:
        train_df = nested_subject_sample(train_df, args.train_size, args.subject_col)
    
    # Load model and tokenizer
    model, tok = load_lm_and_tokenizer(args.llama_model)
    
    # Label space
    mlb = lock_label_space(
        [train_df, val_df, test_df], 
        args.label_col,
        icd9_pkl_path=args.icd9_pickle, 
        use_complete=bool(args.use_complete_icd9),
        top_k=args.top_k
    )
    
    labels_vocab = mlb.classes_.tolist()
    
    # Create datasets
    train_dataset = GenCodesDataset(
        train_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col
    ) if train_df is not None else None
    
    val_dataset = GenCodesDataset(
        val_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col
    ) if val_df is not None else None
    
    # Setup run directory
    size_str = f"N{args.train_size}" if args.train_size > 0 else "full"
    top_k_str = f"_top{args.top_k}" if args.top_k > 0 else ""
    tag = args.run_name or f"{now_tag()}_{size_str}_icd9{top_k_str}{'_complete' if args.use_complete_icd9 else ''}"
    
    RUN_DIR = make_run_dir(args.run_root, tag)
    rank0_print(f"Run dir: {RUN_DIR}")
    
    # Save config
    if is_main_process():
        save_json(os.path.join(RUN_DIR, "config.json"), {
            "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate,
            "epochs": args.epochs, "gen_max_new": args.gen_max_new, "tgt_reserve_tok": args.tgt_reserve_tok,
            "seed": args.seed, "train_rows": len(train_df) if train_df is not None else 0,
            "icd9_pickle": args.icd9_pickle, "use_complete_icd9": bool(args.use_complete_icd9),
            "total_label_space": len(labels_vocab),
            "test_batch_size": args.test_batch_size,
            "top_k": args.top_k
        })
    
    # Custom collator and data handlers
    data_collator = lambda features: pad_collate(features, tok)
    
    # Setup training arguments
    training_args = make_training_args(args, RUN_DIR)
    
    # Setup callbacks
    callbacks = []
    
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    
    gen_eval_callback = DetailedEvalCallback(
        val_dataset, tok,
        sample_size=args.eval_sample_size,
        batch_size=args.test_batch_size,
        label_vocab=labels_vocab,
        max_new_tokens=args.gen_max_new,
        output_dir=RUN_DIR
    )
    callbacks.append(gen_eval_callback)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Compile if requested (requires PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        try:
            unwrap_model(model).model = torch.compile(unwrap_model(model).model)
            rank0_print("Model compiled successfully!")
        except Exception as e:
            rank0_print(f"Model compilation failed: {e}")
    
    # Train
    rank0_print("Starting training…")
    trainer.train()
    
    # Generate test predictions if test data is available
    if test_df is not None:
        test_dataset = GenCodesDataset(
            test_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col
        )
        
        rank0_print(f"Generating predictions for {len(test_dataset)} test samples in batches of {args.test_batch_size}...")
        
        # Load best model
        best_checkpoint = os.path.join(RUN_DIR, "checkpoints")
        try:
            model = trainer.model  # Use the current model (should be best if load_best_model_at_end=True)
            model_unwrapped = unwrap_model(model)
            model_unwrapped.eval()
        except Exception as e:
            rank0_print(f"Error loading best model: {e}")
        
        # Generate predictions
        test_preds = []
        test_true = []
        batch_times = []
        start_time = time.time()
        
        for i in range(0, len(test_dataset), args.test_batch_size):
            batch_start = time.time()
            batch_end = min(i + args.test_batch_size, len(test_dataset))
            
            for j in range(i, batch_end):
                sample = test_dataset[j]
                input_text = sample['text']
                true_codes = sample['labels']
                
                pred_codes = generate_codes(
                    model_unwrapped,
                    input_text,
                    tok,
                    max_new_tokens=args.gen_max_new,
                    temperature=0.01
                )
                
                test_preds.append(pred_codes)
                test_true.append(true_codes)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress report
            percent_done = 100.0 * batch_end / len(test_dataset)
            avg_batch_time = sum(batch_times) / len(batch_times)
            batches_left = (len(test_dataset) - batch_end) / args.test_batch_size
            time_left = avg_batch_time * batches_left
            
            rank0_print(
                f"Test: {batch_end}/{len(test_dataset)} ({percent_done:.1f}%) - "
                f"Batch: {batch_time:.2f}s - ETA: {time_left:.2f}s"
            )
        
        total_time = time.time() - start_time
        rank0_print(f"Test generation complete in {total_time:.2f}s ({total_time/len(test_dataset):.3f}s per sample)")
        
        # Calculate metrics
        if test_true and test_preds:
            y_true = codes_to_multihot(test_true, labels_vocab)
            y_pred = codes_to_multihot(test_preds, labels_vocab)
            
            metrics = eval_sets(y_true, y_pred)
            hier_metrics = hierarchical_eval(y_true, y_pred, labels_vocab)
            combined_metrics = {**metrics, **hier_metrics}
            
            rank0_print(f"Test metrics: {json.dumps(combined_metrics, indent=2)}")
            
            # Save metrics and predictions
            if is_main_process():
                save_json(os.path.join(RUN_DIR, "test_metrics.json"), combined_metrics)
                
                # Show example predictions
                show_test_predictions(test_df, test_preds, args.label_col, labels_vocab, n_show=args.test_examples)
    
    # Clean up
    if args.merge_after and is_main_process():
        try:
            rank0_print("Merging LoRA weights...")
            from peft import PeftModel
            unwrapped = unwrap_model(model)
            merged_model = unwrapped.merge_and_unload()
            merged_model_path = os.path.join(RUN_DIR, "merged_model")
            merged_model.save_pretrained(merged_model_path)
            tok.save_pretrained(merged_model_path)
            rank0_print(f"Merged model saved to {merged_model_path}")
        except Exception as e:
            rank0_print(f"Error merging model: {e}")

if __name__ == "__main__":
    main()