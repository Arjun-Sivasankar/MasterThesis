# -*- coding: utf-8 -*-
"""
Hyperparameter optimization for LLama model finetuning with DDP support
- Space-optimized for HPC environments
- Automatic cleanup of less promising trials
- LoRA adapter-only saving
"""

import os
import re
import json
import random
import logging
import pickle
import datetime
import time
import atexit
import math
import argparse
import shutil
import tarfile
import subprocess
from typing import List, Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial, TrialState

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, EarlyStoppingCallback, Trainer, TrainerCallback
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- Env & logging ----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
            try: return int(v)
            except: pass
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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try: torch.distributed.barrier()
            except Exception: pass
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

    # model/prompt
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--tgt_reserve_tok", type=int, default=128)
    ap.add_argument("--gen_max_new", type=int, default=96)

    # HPO specific arguments
    ap.add_argument("--n_trials", type=int, default=20, help="Number of HPO trials to run")
    ap.add_argument("--metric", type=str, default="micro_f1", 
                    choices=["micro_f1", "macro_f1", "samples_f1"],
                    help="Metric to optimize for")
    ap.add_argument("--output_dir", type=str, default="runs_hpo/default", 
                    help="Directory to save HPO results")
    ap.add_argument("--pruning", type=int, default=1,
                    help="Enable trial pruning (1) or not (0)")
    ap.add_argument("--study_name", type=str, default="llama_hpo",
                    help="Name of the optimization study")
    ap.add_argument("--storage", type=str, default="",
                    help="Optuna storage string (empty for in-memory)")
    ap.add_argument("--compress_trials", type=int, default=1, 
                    help="Compress older trials to save space (1) or not (0)")
    ap.add_argument("--keep_trials", type=int, default=3,
                    help="Number of best trials to keep uncompressed")
                    
    # Space saving options
    ap.add_argument("--save_full_model", type=int, default=0, 
                    help="Save full model (1) or only adapters (0)")
    ap.add_argument("--auto_cleanup_threshold", type=float, default=0.85,
                    help="Clean up trials below this fraction of best score")

    # training
    ap.add_argument("--train_size", type=int, default=-1, help="-1=all")
    ap.add_argument("--eval_sample_size", type=int, default=100, help="Subset of VAL for gen-eval during training")
    ap.add_argument("--test_batch_size", type=int, default=8, help="Batch size for final test generation")
    
    # seed & misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    return ap.parse_args()

# ---------------- ICD-9 Code Handling ----------------
def format_icd9_properly(code: str) -> str:
    code = code.strip().upper()
    code = re.sub(r"\s+", "", code)
    if code.endswith("."): code = code[:-1]
    if code and code[0].isdigit():
        if '.' not in code and len(code) > 3:
            return code[:3] + '.' + code[3:]
    elif code and len(code) > 1:
        if code[0] in ('V', 'E') and '.' not in code and len(code) > 3:
            return code[:3] + '.' + code[3:]
    return code

def is_valid_icd9(code: str) -> bool:
    if not code: return False
    if code[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    if code.startswith('V'): return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    if code.startswith('E'): return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    return False

def normalize_code(c: str) -> str:
    return format_icd9_properly(c)

def get_icd9_parent(code: str) -> str:
    if not code or len(code) < 3: return code
    if code[0].isdigit(): return code.split('.')[0][:3]
    if code.startswith('V'):
        base = code.split('.')[0]; return base[:3]
    if code.startswith('E'):
        base = code.split('.')[0]; return base[:4] if len(base) >= 4 else base
    return code

# ---------------- Disk Usage Utilities ----------------
def get_dir_size(path):
    """Get directory size in bytes"""
    if not os.path.exists(path):
        return 0
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):  # Skip symbolic links
                    total_size += os.path.getsize(fp)
        return total_size
    except Exception as e:
        rank0_print(f"Error getting directory size: {e}")
        return 0

def format_size(bytes):
    """Format size in bytes to human readable format"""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024**2:
        return f"{bytes/1024:.2f} KB"
    elif bytes < 1024**3:
        return f"{bytes/1024**2:.2f} MB"
    else:
        return f"{bytes/1024**3:.2f} GB"

def compress_trial_dir(trial_dir):
    """Compress a trial directory into a tar.gz file and remove the original"""
    if not os.path.exists(trial_dir):
        return False
    
    try:
        archive_path = f"{trial_dir}.tar.gz"
        rank0_print(f"Compressing trial directory: {trial_dir} -> {archive_path}")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(trial_dir, arcname=os.path.basename(trial_dir))
            
        # Check if archive was created successfully
        if os.path.exists(archive_path):
            # Get sizes for logging
            original_size = get_dir_size(trial_dir)
            compressed_size = os.path.getsize(archive_path)
            savings = 100 * (1 - compressed_size / original_size) if original_size > 0 else 0
            
            rank0_print(f"Compression complete: {format_size(original_size)} -> "
                        f"{format_size(compressed_size)} ({savings:.1f}% saved)")
            
            # Remove original directory
            shutil.rmtree(trial_dir)
            return True
        return False
    except Exception as e:
        rank0_print(f"Error compressing trial directory: {e}")
        return False

def cleanup_trial_artifacts(trial_dir, keep_best=True):
    """Clean up artifacts to save space"""
    if not os.path.exists(trial_dir):
        return
    
    try:
        # Always remove checkpoints directory (only intermediate steps)
        checkpoints_dir = os.path.join(trial_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            size_before = get_dir_size(checkpoints_dir)
            shutil.rmtree(checkpoints_dir)
            rank0_print(f"Removed checkpoints directory, freed {format_size(size_before)}")
        
        if not keep_best:
            # Remove model weights (but keep metrics and parameters)
            model_dirs = [
                os.path.join(trial_dir, d) 
                for d in ["model", "adapter_only"]
                if os.path.exists(os.path.join(trial_dir, d))
            ]
            
            for mdir in model_dirs:
                if os.path.exists(mdir):
                    size_before = get_dir_size(mdir)
                    shutil.rmtree(mdir)
                    rank0_print(f"Removed model directory {mdir}, freed {format_size(size_before)}")
    except Exception as e:
        rank0_print(f"Error during cleanup: {e}")

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

def to_list(x) -> List[str]:
    if isinstance(x, list): return [str(v) for v in x]
    if pd.isna(x): return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast; v = ast.literal_eval(s)
            if isinstance(v, list): return [str(z) for z in v]
        except Exception: pass
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
                     icd9_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
    train_codes = set()
    for fr in frames:
        for codes in fr[label_col]:
            train_codes.update(format_icd9_properly(str(c)) for c in codes)
    train_codes = {c for c in train_codes if is_valid_icd9(c)}
    log.info(f"Found {len(train_codes)} unique valid ICD codes in training data")

    if not use_complete or not icd9_pkl_path:
        all_codes = sorted(train_codes)
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        log.info(f"Using {len(all_codes)} codes from training data only")
        return mlb

    try:
        icd9_df = pd.read_pickle(icd9_pkl_path)
        complete_codes = sorted(icd9_df['icd_code'].astype(str).tolist())
        complete_codes = [format_icd9_properly(code) for code in complete_codes]
        complete_codes = [code for code in complete_codes if is_valid_icd9(code)]
        log.info(f"Loaded {len(complete_codes)} complete ICD-9 codes from {icd9_pkl_path}")
        mlb = MultiLabelBinarizer(classes=complete_codes)
        mlb.fit([complete_codes])

        codes_in_complete = sum(1 for c in train_codes if c in set(complete_codes))
        codes_not_in_complete = len(train_codes) - codes_in_complete
        log.info(f"Training data coverage: in={codes_in_complete}, missing={codes_not_in_complete}")
        if codes_not_in_complete > 0:
            log.warning("Some training codes not found in complete ICD-9 set.")
        return mlb

    except Exception as e:
        log.error(f"Error loading complete ICD-9 codes: {e}")
        log.warning("Falling back to training-data-only label space")
        all_codes = sorted(train_codes)
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        return mlb

def y_multi_hot(mlb: MultiLabelBinarizer, lists):
    formatted_lists = []
    for row in lists:
        formatted_row = [format_icd9_properly(str(c)) for c in row]
        formatted_row = [c for c in formatted_row if is_valid_icd9(c)]
        formatted_lists.append(formatted_row)
    return mlb.transform(formatted_lists)

# ---------------- Subject-safe subsetting ----------------
def nested_subject_sample(train_df, target_n, subject_col="subject_id_x", seed=13):
    if target_n is None or target_n < 0 or target_n >= len(train_df):
        return train_df.copy()
    rng = np.random.default_rng(seed)
    subjects = train_df[subject_col].drop_duplicates().tolist()
    rng.shuffle(subjects)
    chosen, count = [], 0
    for s in subjects:
        rows = train_df[train_df[subject_col] == s]
        if count + len(rows) <= target_n or len(chosen) == 0:
            chosen.append(s); count += len(rows)
        if count >= target_n: break
    sub = train_df[train_df[subject_col].isin(chosen)].copy()
    log.info(f"[subset] requested={target_n} got={len(sub)} unique_subjects={sub[subject_col].nunique()}")
    return sub

# ---------------- Dataset (pre-tokenized) ----------------
class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int, label_col: str):
        self.tok = tok
        self.max_len = max_len
        self.tgt_reserve = max(8, int(tgt_reserve))
        self.label_col = label_col
        self.rows = rows

        prompts = rows["input_text"].astype(str).tolist()

        targets = []
        for codes in rows[label_col].tolist():
            formatted_codes = [format_icd9_properly(str(c)) for c in codes]
            formatted_codes = [c for c in formatted_codes if is_valid_icd9(c)]
            targets.append(" ".join(sorted(set(formatted_codes))))

        self.prompt_ids = [tok.encode(p + "\n", add_special_tokens=True) for p in prompts]
        eos = (tok.eos_token or "")
        self.ans_ids    = [tok.encode(t + eos, add_special_tokens=False) for t in targets]

    def __len__(self): return len(self.prompt_ids)

    def __getitem__(self, i):
        prompt_ids = self.prompt_ids[i]
        ans_ids    = self.ans_ids[i]
        max_prompt_len = max(1, self.max_len - self.tgt_reserve)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]
        remaining = max(1, self.max_len - len(prompt_ids))
        if len(ans_ids) > remaining:
            ans_ids = ans_ids[:remaining]
        input_ids = prompt_ids + ans_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + ans_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ---- Hand-rolled collator (no tok.pad -> no warning) ----
def pad_collate(features, tok):
    pad_id = tok.pad_token_id
    batch_size = len(features)
    lengths = [f["input_ids"].size(0) if torch.is_tensor(f["input_ids"]) else len(f["input_ids"]) for f in features]
    max_len = max(lengths)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        ids = f["input_ids"]; am = f["attention_mask"]; lab = f["labels"]
        if not torch.is_tensor(ids): ids = torch.tensor(ids, dtype=torch.long)
        if not torch.is_tensor(am):  am  = torch.tensor(am, dtype=torch.long)
        if not torch.is_tensor(lab): lab = torch.tensor(lab, dtype=torch.long)
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = am
        labels[i, :L] = lab
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------- Model ----------------
def unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def load_lm_and_tokenizer(model_name, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora_cfg)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    if is_main_process():
        model.print_trainable_parameters()
        
        # Calculate adapter size
        params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Trainable parameters: {params_count:,} ({params_count*2/1024/1024:.2f} MB)")
        
    return model, tok

# ---------------- Generation & metrics ----------------
@torch.no_grad()
def generate_codes(model, tok, prompts: List[str], labels_vocab: List[str],
                   max_new=96, batch_size=16, max_len=3072):
    """Optimized code generation with proper device handling and memory management"""
    # Ensure model is unwrapped properly
    model = unwrap_model(model)
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    allowed = set(labels_vocab)
    preds = []
    
    # Process in batches with timing and progress tracking
    total_samples = len(prompts)
    if is_main_process():
        rank0_print(f"Generating predictions for {total_samples} samples in batches of {batch_size}...")
    
    start_time = time.time()
    last_time = start_time
    
    for i in range(0, total_samples, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        curr_batch_size = len(batch_prompts)
        
        # Tokenize and move to device
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, 
                     truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with mixed precision if available
        with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and 
                                torch.cuda.get_device_capability(0)[0] >= 8)):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False, 
                num_beams=1,
                no_repeat_ngram_size=2,
                eos_token_id=tok.eos_token_id, 
                pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True, 
                output_scores=False,
            )
            
        # Extract generated tokens
        seq = out.sequences
        gen_only = seq[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        
        # Process and filter predicted codes
        for t in texts:
            tokens = re.split(r"[^A-Za-z0-9\.]+", t)
            cand = [normalize_code(z) for z in tokens if z]
            seen, keep = set(), []
            for c in cand:
                if c in allowed and is_valid_icd9(c) and c not in seen:
                    seen.add(c); keep.append(c)
            preds.append(keep)
        
        # Log progress periodically
        if is_main_process() and ((i + curr_batch_size) % (10 * batch_size) == 0 or 
                                 (i + curr_batch_size) >= total_samples):
            current_time = time.time()
            elapsed = current_time - start_time
            batch_time = current_time - last_time
            progress = (i + curr_batch_size) / total_samples
            remaining = elapsed / progress - elapsed if progress > 0 else 0
            
            rank0_print(f"Generated {i + curr_batch_size}/{total_samples} samples " 
                       f"({progress:.1%}) - Batch time: {batch_time:.2f}s - "
                       f"Est. remaining: {remaining:.2f}s")
            last_time = current_time
            
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and (i + curr_batch_size) % (20 * batch_size) == 0:
            torch.cuda.empty_cache()
            
    total_time = time.time() - start_time
    if is_main_process():
        rank0_print(f"Generation complete in {total_time:.2f}s "
                   f"({total_time/total_samples:.3f}s per sample)")
                
    return preds

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

def hierarchical_eval(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> Dict[str, float]:
    code_to_parent = {code: get_icd9_parent(code) for code in label_vocab}
    parent_to_idx = {}
    for idx, code in enumerate(label_vocab):
        parent = code_to_parent[code]
        parent_to_idx.setdefault(parent, []).append(idx)

    n_samples = y_true.shape[0]
    parent_hits = 0
    partial_matches = 0
    total_true_parents = 0

    for i in range(n_samples):
        pred_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_pred[i])[0]}
        true_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_true[i])[0]}
        parent_hits += len(pred_parents & true_parents)
        total_true_parents += len(true_parents)
        for parent in pred_parents:
            if parent in true_parents:
                child_indices = parent_to_idx.get(parent, [])
                exact_match = any(y_true[i, idx] == 1 and y_pred[i, idx] == 1 for idx in child_indices)
                if not exact_match:
                    partial_matches += 1

    parent_recall = (parent_hits / total_true_parents) if total_true_parents > 0 else 0
    return {
        "hierarchical_parent_recall": parent_recall,
        "hierarchical_partial_matches": partial_matches,
        "hierarchical_partial_per_sample": partial_matches / n_samples if n_samples > 0 else 0
    }

# ---------------- Custom training callbacks ----------------
class OptunaPruningCallback(TrainerCallback):
    """Simple Optuna pruning callback that doesn't rely on PyTorch Lightning."""
    def __init__(self, trial):
        self.trial = trial
        self.step = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        # Report eval metrics to Optuna for pruning decisions
        self.step += 1
        
        # Report eval loss for pruning
        if "eval_loss" in metrics:
            self.trial.report(metrics["eval_loss"], self.step)
            
            # Let Optuna decide whether to prune this trial
            if self.trial.should_prune():
                message = f"Trial {self.trial.number} pruned at step {self.step}"
                if is_main_process():
                    rank0_print(message)
                raise optuna.exceptions.TrialPruned(message)

# ---------------- HPO Components ----------------
class MetricTrackingCallback(TrainerCallback):
    """Callback to track metrics during training."""
    def __init__(self):
        self.train_metrics = []
        self.eval_metrics = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        # Determine if this is a training or evaluation log
        if 'loss' in logs and 'eval_loss' not in logs:
            # Training metrics
            self.train_metrics.append({
                'step': state.global_step,
                'epoch': logs.get('epoch', 0),
                'loss': logs.get('loss', 0),
                'learning_rate': logs.get('learning_rate', 0),
                'timestamp': datetime.datetime.now().isoformat(),
            })
        elif 'eval_loss' in logs:
            # Evaluation metrics
            self.eval_metrics.append({
                'step': state.global_step,
                'epoch': logs.get('epoch', 0),
                'eval_loss': logs.get('eval_loss', 0),
                'timestamp': datetime.datetime.now().isoformat(),
            })

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def make_run_dir(base, tag):
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, tag)
    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{path}__r{i}"): i += 1
        path = f"{path}__r{i}"
    if is_main_process():
        os.makedirs(path, exist_ok=False)
        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    barrier()
    return path

def save_json(path: str, obj: dict):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def show_test_predictions(df: pd.DataFrame, preds: List[List[str]],
                          label_col: str, label_vocab: List[str],
                          n_show: int = 5, seed: int = 0):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
    idx_map = {c:i for i,c in enumerate(label_vocab)}
    
    rank0_print(f"Showing {n_show} random test examples:")
    for i in idxs:
        row = df.iloc[i]
        gold = sorted({format_icd9_properly(str(c)) for c in row[label_col]
                       if is_valid_icd9(format_icd9_properly(str(c)))})
        pred = preds[i]
        missing = sorted([c for c in gold if c not in pred])
        extra   = sorted([c for c in pred if c not in gold])

        gold_parents = {get_icd9_parent(c) for c in gold}
        pred_parents = {get_icd9_parent(c) for c in pred}
        parent_matches = sorted([f"{c} (parent)" for c in pred
                                 if get_icd9_parent(c) in gold_parents and c not in gold])

        y_true = np.zeros(len(label_vocab))
        y_pred = np.zeros(len(label_vocab))
        for code in gold:
            j = idx_map.get(code);  y_true[j] = 1 if j is not None else 0
        for code in pred:
            j = idx_map.get(code);  y_pred[j] = 1 if j is not None else 0

        precision = precision_score([y_true], [y_pred], average='micro', zero_division=0)
        recall    = recall_score([y_true], [y_pred], average='micro', zero_division=0)
        f1        = f1_score([y_true], [y_pred], average='micro', zero_division=0)
        parent_recall = (len(pred_parents & gold_parents) / len(gold_parents)) if gold_parents else 0

        rank0_print("\n" + "="*80)
        rank0_print(f"Example {i}:")
        rank0_print(f"- METRICS: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, parent_recall={parent_recall:.4f}")
        rank0_print("- GOLD:", " ".join(gold) if gold else "(none)")
        rank0_print("- PRED:", " ".join(pred) if pred else "(none)")
        rank0_print(f"- FALSE NEGATIVES ({len(missing)}):", " ".join(missing) if missing else "(none)")
        rank0_print(f"- FALSE POSITIVES ({len(extra)}):", " ".join(extra) if extra else "(none)")
        rank0_print(f"- PARENT MATCHES ({len(parent_matches)}):", " ".join(parent_matches) if parent_matches else "(none)")

# ---------------- Hyperparameter Search ----------------
def define_search_space(trial, args):
    """Define hyperparameter search space for Optuna."""
    return {
        "epochs": trial.suggest_int("epochs", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "per_device_train_batch_size": 1, # Fixed due to memory constraints
        "per_device_eval_batch_size": 1,  # Fixed due to memory constraints
        "grad_accum": trial.suggest_categorical("grad_accum", [8, 16, 32]),
        "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64]),
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
        "early_stop": 1,
        "patience": trial.suggest_int("patience", 1, 3),
    }

# Global variables for tracking best performance
best_score = 0.0
best_trials = []

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    global best_score, best_trials
    
    # Only main process runs trials
    if not is_main_process():
        return 0.0
    
    # Get command line args
    args = get_args()
    
    # Set seed for reproducibility (different for each trial)
    trial_seed = args.seed + trial.number
    set_seed(trial_seed)
    
    # Sample hyperparameters
    params = define_search_space(trial, args)
    rank0_print(f"\n=== Starting Trial #{trial.number} ===")
    rank0_print(f"Parameters: {params}")
    
    # Report disk usage before trial
    output_dir_size = get_dir_size(args.output_dir)
    rank0_print(f"Current disk usage of output directory: {format_size(output_dir_size)}")
    
    try:
        # Load data
        if args.train_pickle and args.val_pickle and args.test_pickle:
            train_df = pickle.load(open(args.train_pickle, "rb"))
            val_df = pickle.load(open(args.val_pickle, "rb"))
            test_df = pickle.load(open(args.test_pickle, "rb"))
        elif args.data_pickle:
            full_df = pickle.load(open(args.data_pickle, "rb"))
            train_df, val_df, test_df = subject_splits(full_df, subject_col=args.subject_col)
        else:
            raise ValueError("No data files provided")
        
        # Subset training data if requested
        train_df = nested_subject_sample(train_df, args.train_size, subject_col=args.subject_col, seed=trial_seed)
        
        # Build prompts
        for df_, name in ((train_df, 'train'), (val_df, 'val'), (test_df, 'test')):
            df_["input_text"] = df_.apply(lambda r: build_input_text(
                r, args.use_structured==1, args.use_notes==1, args.subject_col), axis=1)
        
        # Create label space
        mlb = lock_label_space([train_df, val_df, test_df], args.label_col, 
                              args.icd9_pickle, bool(args.use_complete_icd9))
        labels_vocab = mlb.classes_.tolist()
        y_val = y_multi_hot(mlb, val_df[args.label_col].tolist())
        
        # Create model and tokenizer with trial-specific LoRA config
        model, tok = load_lm_and_tokenizer(
            args.llama_model,
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"]
        )
        
        # Create datasets
        train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col)
        val_ds = GenCodesDataset(val_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col)
        
        # Create trial directory
        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "checkpoints"), exist_ok=True)
        
        # Save params for reference
        save_json(os.path.join(trial_dir, "params.json"), params)
        
        # Setup callbacks
        metric_tracker = MetricTrackingCallback()
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=params["patience"]),
            metric_tracker
        ]
        
        # Add Optuna pruning callback if enabled
        if args.pruning:
            callbacks.append(OptunaPruningCallback(trial))
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(trial_dir, "checkpoints"),
            num_train_epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_eval_batch_size"],
            gradient_accumulation_steps=params["grad_accum"],
            warmup_ratio=params["warmup_ratio"],
            weight_decay=params["weight_decay"],
            logging_strategy="epoch",
            eval_strategy="epoch", 
            prediction_loss_only=True,
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=bool(params["early_stop"]),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            gradient_checkpointing=True,
            remove_unused_columns=False,
            fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
            bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
            optim="adamw_torch",
            dataloader_num_workers=2,
            run_name=f"trial_{trial.number}",
            disable_tqdm=True,
            ddp_find_unused_parameters=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=lambda feats: pad_collate(feats, tok),
            callbacks=callbacks
        )
        
        # Train model
        t0 = time.time()
        try:
            trainer.train()
        except optuna.exceptions.TrialPruned:
            # If the trial was pruned, re-raise to let Optuna handle it
            raise
            
        train_duration = time.time() - t0
        rank0_print(f"Training completed in {train_duration:.1f}s")
        
        # Select eval subset (use full val set if small enough)
        eval_subset = val_df
        if args.eval_sample_size > 0 and args.eval_sample_size < len(val_df):
            eval_subset = val_df.sample(args.eval_sample_size, random_state=trial_seed)
        
        # Generate predictions
        rank0_print(f"Generating predictions on {len(eval_subset)} validation samples...")
        eval_prompts = eval_subset["input_text"].astype(str).tolist()
        y_true_subset = y_multi_hot(mlb, eval_subset[args.label_col].tolist())
        
        t1 = time.time()
        pred_code_lists = generate_codes(
            trainer.model, tok, eval_prompts, labels_vocab,
            max_new=args.gen_max_new, batch_size=args.test_batch_size
        )
        generation_duration = time.time() - t1
        
        # Calculate metrics
        Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
        metrics = eval_sets(y_true_subset, Y_pred)
        hier_metrics = hierarchical_eval(y_true_subset, Y_pred, labels_vocab)
        metrics.update(hier_metrics)
        
        # Add timing metrics
        metrics.update({
            "train_seconds": train_duration,
            "generation_seconds": generation_duration,
            "samples_per_second": len(eval_subset) / generation_duration if generation_duration > 0 else 0
        })
        
        # Save metrics and training history
        metrics["training_history"] = metric_tracker.train_metrics
        metrics["evaluation_history"] = metric_tracker.eval_metrics
        save_json(os.path.join(trial_dir, "metrics.json"), metrics)
        
        # Save adapter weights (space-efficient)
        rank0_print("Saving adapter weights...")
        tok.save_pretrained(os.path.join(trial_dir, "tokenizer"))
        
        # Save according to user preference
        if args.save_full_model:
            trainer.model.save_pretrained(os.path.join(trial_dir, "model"))
        else:
            # Save only adapter weights (much smaller)
            trainer.model.save_pretrained(
                os.path.join(trial_dir, "adapter_only"),
                save_embedding_layers=False,
                save_base_model=False
            )
            
        # Report disk usage for this trial
        trial_size = get_dir_size(trial_dir)
        rank0_print(f"Trial {trial.number} disk usage: {format_size(trial_size)}")
        
        # Get the main objective metric
        objective_value = metrics.get(args.metric, 0.0)
        
        # Update best trials list
        if objective_value >= best_score * 0.95:  # Within 5% of best score
            best_trials.append((trial.number, objective_value))
            best_trials.sort(key=lambda x: x[1], reverse=True)
            if len(best_trials) > args.keep_trials:
                best_trials = best_trials[:args.keep_trials]
                
        # Update best score if needed
        if objective_value > best_score:
            best_score = objective_value
            rank0_print(f"New best score: {best_score:.4f}")
            
            # Create a symlink to the best trial for convenience
            best_link_path = os.path.join(args.output_dir, "best_trial")
            try:
                if os.path.exists(best_link_path):
                    if os.path.islink(best_link_path):
                        os.remove(best_link_path)
                os.symlink(trial_dir, best_link_path, target_is_directory=True)
                rank0_print(f"Updated best_trial symlink -> trial_{trial.number}")
            except Exception as e:
                rank0_print(f"Error creating symlink: {e}")
        
        # Cleanup checkpoints in any case to save space
        cleanup_trial_artifacts(trial_dir, keep_best=True)
            
        # If this trial performed poorly, clean up more aggressively
        if objective_value < best_score * args.auto_cleanup_threshold:
            rank0_print(f"Trial {trial.number} performed below threshold ({objective_value:.4f} < {best_score*args.auto_cleanup_threshold:.4f})")
            rank0_print("Cleaning up model weights to save space")
            cleanup_trial_artifacts(trial_dir, keep_best=False)
        
        # Save all metrics as trial attributes
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                trial.set_user_attr(k, v)
                
        # Compress older trials if enabled
        if args.compress_trials and trial.number > 5:
            best_trial_nums = [t[0] for t in best_trials]
            
            # Find trials that can be compressed
            for old_trial_num in range(0, trial.number - 3):
                old_trial_dir = os.path.join(args.output_dir, f"trial_{old_trial_num}")
                old_trial_archive = f"{old_trial_dir}.tar.gz"
                
                # Don't compress best trials or already compressed trials
                if (old_trial_num in best_trial_nums) or os.path.exists(old_trial_archive):
                    continue
                
                # Compress if the trial directory exists
                if os.path.exists(old_trial_dir):
                    compress_trial_dir(old_trial_dir)
        
        # Clear GPU memory
        del model, trainer
        torch.cuda.empty_cache()
                
        rank0_print(f"Trial {trial.number} {args.metric}: {objective_value:.4f}")
        return objective_value
        
    except optuna.exceptions.TrialPruned:
        rank0_print(f"Trial {trial.number} pruned by Optuna")
        # Try to clean up the trial directory to save space
        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
        if os.path.exists(trial_dir):
            rank0_print(f"Cleaning up pruned trial directory: {trial_dir}")
            shutil.rmtree(trial_dir)
        raise  # Re-raise to let Optuna handle it properly
        
    except Exception as e:
        rank0_print(f"Error in trial {trial.number}: {str(e)}")
        return float('-inf')  # Return worst possible score on error

# ---------------- Main ----------------
def main():
    args = get_args()
    
    # Seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Track initial disk usage
        initial_size = get_dir_size(args.output_dir)
        rank0_print(f"Initial output directory size: {format_size(initial_size)}")
        
        rank0_print(f"Output directory: {args.output_dir}")
        rank0_print(f"Optimizing for metric: {args.metric}")
        rank0_print(f"Number of trials: {args.n_trials}")
        rank0_print(f"Space saving: {'Enabled - adapters only' if not args.save_full_model else 'Disabled - saving full models'}")
        rank0_print(f"Auto cleanup threshold: {args.auto_cleanup_threshold}")
        rank0_print(f"Keeping top {args.keep_trials} trials uncompressed")
        
        # Create pruner based on args
        if args.pruning:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
            rank0_print("Using MedianPruner for early stopping of trials")
        else:
            pruner = optuna.pruners.NopPruner()
            rank0_print("No pruning enabled")
            
        # Create storage if specified
        if args.storage:
            rank0_print(f"Using storage: {args.storage}")
            study = optuna.create_study(
                storage=args.storage,
                study_name=args.study_name,
                direction="maximize",
                pruner=pruner,
                load_if_exists=True
            )
        else:
            rank0_print("Using in-memory storage")
            study = optuna.create_study(
                study_name=args.study_name,
                direction="maximize",
                pruner=pruner
            )
            
        # Run optimization
        rank0_print("Starting hyperparameter optimization...")
        study.optimize(objective, n_trials=args.n_trials)
        
        # Print results
        rank0_print("\n=== Optimization Completed ===")
        rank0_print(f"Best trial: #{study.best_trial.number}")
        rank0_print(f"Best {args.metric}: {study.best_value:.4f}")
        rank0_print(f"Best parameters: {study.best_params}")
        
        # Save results
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial.number
        
        results = {
            "best_trial": best_trial,
            "best_value": best_value,
            "best_metric": args.metric,
            "best_params": best_params,
            "timestamp": now_tag(),
            "total_trials": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            "pruned_trials": len([t for t in study.trials if t.state == TrialState.PRUNED]),
            "failed_trials": len([t for t in study.trials if t.state == TrialState.FAIL])
        }
        
        # Add all attributes from the best trial
        for k, v in study.best_trial.user_attrs.items():
            if isinstance(v, (int, float)):
                results[f"best_{k}"] = v
                
        # Save overall results
        save_json(os.path.join(args.output_dir, "best_results.json"), results)
        
        # Export all trials to CSV for analysis
        try:
            trials_df = study.trials_dataframe()
            trials_df.to_csv(os.path.join(args.output_dir, "trials.csv"), index=False)
        except Exception as e:
            rank0_print(f"Error exporting trials to CSV: {e}")
            
        # Final disk usage report
        final_size = get_dir_size(args.output_dir)
        rank0_print(f"Final output directory size: {format_size(final_size)}")
        rank0_print(f"Disk usage increase: {format_size(final_size - initial_size)}")
        
        rank0_print("HPO completed successfully!")

if __name__ == "__main__":
    main()