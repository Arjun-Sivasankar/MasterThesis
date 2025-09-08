# -*- coding: utf-8 -*-
"""
Generative ICD-9 code prediction with LoRA (DDP-safe, epoch timing, optimized generation)

- Rank detection works BEFORE DDP init (uses LOCAL_RANK/RANK env)
- Enhanced timing logs with timestamps at appropriate points
- Optimized test generation with configurable batch size
- Proper device placement and memory management
- Mixed precision inference for generation
"""

import os, re, json, random, logging, pickle, datetime, time, atexit, argparse, inspect
from typing import List, Any, Dict
import numpy as np
import pandas as pd

import torch
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

def load_lm_and_tokenizer(model_name):
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
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora_cfg)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    if is_main_process():
        model.print_trainable_parameters()
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
class DetailedEvalCallback(TrainerCallback):
    """
    Rank-0 only:
    - Pretty epoch logs with timestamps
    - Gen-eval each epoch on a fixed VAL subset
    - Per-epoch timing: train≈, gen-eval, total
    """
    def __init__(self, eval_dataset, tokenizer, label_vocab, eval_sample_size=100, seed=42):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.eval_sample_size = min(eval_sample_size, len(eval_dataset))
        self.best_micro_f1 = 0
        self.epoch_metrics = []
        self.epoch_times = {}  # epoch -> dict(times)
        self._epoch_t0 = None
        self.rng = np.random.RandomState(seed)
        self.subset_indices = self.rng.choice(len(self.eval_dataset), self.eval_sample_size, replace=False)
        self.epoch_start_time = None
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not is_main_process(): return
        self._epoch_t0 = time.time()
        self.epoch_start_time = datetime.datetime.now()
        self.current_epoch = getattr(state, 'epoch', 0)
        rank0_print(f"=== Starting epoch {self.current_epoch:.1f} ===")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not is_main_process(): return
        if 'loss' in logs and 'eval_loss' not in logs:
            rank0_print(f"[Train] epoch: {logs.get('epoch', 0):.1f}, loss: {logs.get('loss', 0):.4f}, "
                  f"lr: {logs.get('learning_rate', 0):.2e}, grad_norm: {logs.get('grad_norm', 0):.2f}")
        elif 'eval_loss' in logs:
            eval_loss = logs.get('eval_loss', 0)
            eval_runtime = logs.get('eval_runtime', 0) if 'eval_runtime' in logs else 0
            rank0_print(f"[Eval] epoch: {logs.get('epoch', 0):.1f}, loss: {eval_loss:.4f}, "
                  f"runtime: {eval_runtime:.1f}s")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not is_main_process() or (hasattr(state, "epoch") and state.epoch < 1.0): return
        model = kwargs.get("model")
        if not model: return
        
        ep = getattr(state, "epoch", 0)
        rank0_print(f"[GenEval] Starting generation evaluation for epoch {ep:.1f}...")
        
        subset_prompts, gold_codes_lists = [], []
        for idx in self.subset_indices:
            item = self.eval_dataset[idx]
            prompt_text = self.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            parts = prompt_text.split("[CODES]")
            if len(parts) > 1:
                subset_prompts.append(parts[0] + "[CODES]")
                target_text = parts[1].strip()
                gold_codes = [format_icd9_properly(c) for c in re.split(r"[^A-Za-z0-9\.]+", target_text) if c]
                gold_codes = [c for c in gold_codes if is_valid_icd9(c)]
                gold_codes_lists.append(gold_codes)
            else:
                subset_prompts.append(prompt_text)
                gold_codes_lists.append([])

        t_start = time.time()
        
        # Free up memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        preds = generate_codes(model, self.tokenizer, subset_prompts, self.label_vocab, 
                             max_new=96, batch_size=4)
        gen_time = time.time() - t_start

        y_true = codes_to_multihot(gold_codes_lists, self.label_vocab)
        y_pred = codes_to_multihot(preds, self.label_vocab)

        eval_metrics = eval_sets(y_true, y_pred)
        hier_metrics = hierarchical_eval(y_true, y_pred, self.label_vocab)
        eval_metrics.update(hier_metrics)

        if eval_metrics["micro_f1"] > self.best_micro_f1:
            self.best_micro_f1 = eval_metrics["micro_f1"]
            rank0_print(f"[GenEval] New best micro_f1: {self.best_micro_f1:.4f}")

        # store metrics + gen-eval time
        self.epoch_metrics.append({
            "epoch": ep,
            "eval_loss": metrics.get("eval_loss", 0) if metrics else 0,
            "micro_f1": eval_metrics["micro_f1"],
            "samples_f1": eval_metrics["samples_f1"],
            "parent_recall": hier_metrics["hierarchical_parent_recall"]
        })
        self.epoch_times[ep] = self.epoch_times.get(ep, {})
        self.epoch_times[ep]["gen_eval_seconds"] = gen_time

        rank0_print(f"[GenEval] epoch: {ep:.1f}, micro_f1: {eval_metrics['micro_f1']:.4f}, "
              f"samples_f1: {eval_metrics['samples_f1']:.4f}, "
              f"parent_recall: {hier_metrics['hierarchical_parent_recall']:.4f}, "
              f"gen_eval_time: {gen_time:.1f}s")

    def on_epoch_end(self, args, state, control, **kwargs):
        if not is_main_process(): return
        if self._epoch_t0 is None: return
        
        # Calculate timing at the end of epoch (where it makes sense)
        total = time.time() - self._epoch_t0
        ep = getattr(state, "epoch", 0)
        self.epoch_times[ep] = self.epoch_times.get(ep, {})
        gen_eval = self.epoch_times[ep].get("gen_eval_seconds", 0.0)
        train_approx = max(0.0, total - gen_eval)
        self.epoch_times[ep]["epoch_total_seconds"] = total
        self.epoch_times[ep]["train_seconds_approx"] = train_approx
        
        time_elapsed = datetime.datetime.now() - self.epoch_start_time
        
        # Log time BEFORE marking epoch as complete for proper sequence
        rank0_print(f"[Time] epoch: {ep:.1f}, train≈{train_approx:.1f}s, gen-eval={gen_eval:.1f}s, "
              f"total={total:.1f}s, wall_time={time_elapsed.total_seconds():.1f}s")
        rank0_print(f"=== Completed epoch {ep:.1f} ===")

    def on_train_end(self, args, state, control, **kwargs):
        if not is_main_process(): return
        rank0_print(f"\n===== Training Summary =====")
        rank0_print(f"Best validation micro F1: {self.best_micro_f1:.4f}")
        rank0_print(f"\nEpoch progression:")
        for m in self.epoch_metrics:
            ep = m['epoch']
            tinfo = self.epoch_times.get(ep, {})
            rank0_print(f"  Epoch {ep:.1f}: loss={m['eval_loss']:.4f}, micro_f1={m['micro_f1']:.4f}, "
                  f"samples_f1={m['samples_f1']:.4f}, parent_recall={m['parent_recall']:.4f}")
            rank0_print(f"          Time: train≈{tinfo.get('train_seconds_approx',0):.1f}s, "
                  f"gen-eval={tinfo.get('gen_eval_seconds',0):.1f}s, "
                  f"total={tinfo.get('epoch_total_seconds',0):.1f}s")

# ---------------- Run helpers ----------------
def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def make_run_dir(base, tag):
    os.makedirs(base, exist_ok=True)
    root = os.path.join(base, tag)
    path = root
    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{root}__r{i}"): i += 1
        path = f"{root}__r{i}"
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

# ---- TrainingArguments builder (prefers eval_strategy) ----
def make_training_args(args, RUN_DIR):
    TA = TrainingArguments
    sig = inspect.signature(TA.__init__).parameters
    def supports(name): return name in sig

    kwargs = dict(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
        prediction_loss_only=True,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=bool(args.early_stop),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim="adamw_torch",
        dataloader_num_workers=2,
        run_name=os.path.basename(RUN_DIR),
        disable_tqdm=True,
    )
    # eval/evaluation strategy
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        rank0_print("[WARN] transformers lacks eval/evaluation strategy; early stopping may not work.")

    # DDP knobs (best-effort)
    if "ddp_backend" in sig: kwargs["ddp_backend"] = "nccl"
    if "ddp_find_unused_parameters" in sig: kwargs["ddp_find_unused_parameters"] = bool(args.ddp_find_unused)
    if "ddp_timeout" in sig: kwargs["ddp_timeout"] = 28800  # 8h

    return TA(**kwargs)

# ---------------- Main ----------------
def main():
    args = get_args()
    set_seed(args.seed)

    rank0_print("CUDA:", torch.cuda.is_available(),
                "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Load data
    if args.train_pickle and args.val_pickle and args.test_pickle:
        train_df = pickle.load(open(args.train_pickle, "rb"))
        val_df   = pickle.load(open(args.val_pickle, "rb"))
        test_df  = pickle.load(open(args.test_pickle, "rb"))
    elif args.data_pickle:
        full_df = pickle.load(open(args.data_pickle, "rb"))
        train_df, val_df, test_df = subject_splits(full_df, subject_col=args.subject_col, test_size=0.10, val_size=0.10, seed=args.seed)
    else:
        raise ValueError("Provide either --data_pickle OR all of --train_pickle/--val_pickle/--test_pickle")

    for df_ in (train_df, val_df, test_df):
        assert args.label_col in df_.columns and args.subject_col in df_.columns

    # Subject-safe subsetting of TRAIN
    train_df = nested_subject_sample(train_df, args.train_size, subject_col=args.subject_col, seed=args.seed)

    # Build prompts
    for df_, name in ((train_df, 'train'), (val_df, 'val'), (test_df, 'test')):
        df_["input_text"] = df_.apply(lambda r: build_input_text(r, args.use_structured==1, args.use_notes==1, args.subject_col), axis=1)
        log.info(f"[{name}] rows with input_text: {df_['input_text'].notna().sum()}")

    # Label space
    mlb = lock_label_space([train_df, val_df, test_df], args.label_col,
                           icd9_pkl_path=args.icd9_pickle, use_complete=bool(args.use_complete_icd9))
    labels_vocab = mlb.classes_.tolist()
    y_val  = y_multi_hot(mlb, val_df[args.label_col].tolist())
    y_test = y_multi_hot(mlb, test_df[args.label_col].tolist())

    # Model & tokenizer
    model, tok = load_lm_and_tokenizer(args.llama_model)
    if args.compile:
        try: model = torch.compile(model)
        except Exception as e: log.warning(f"torch.compile failed: {e}")

    # Datasets
    train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col)
    val_ds   = GenCodesDataset(val_df,   tok, args.max_len, args.tgt_reserve_tok, args.label_col)

    # Run dir
    size_str = f"N{args.train_size}" if args.train_size > 0 else "full"
    tag = args.run_name or f"{now_tag()}_{size_str}_icd9{'_complete' if args.use_complete_icd9 else ''}"
    RUN_DIR = make_run_dir(args.run_root, tag)
    rank0_print(f"Run dir: {RUN_DIR}")

    if is_main_process():
        save_json(os.path.join(RUN_DIR, "config.json"), {
            "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate,
            "epochs": args.epochs, "gen_max_new": args.gen_max_new, "tgt_reserve_tok": args.tgt_reserve_tok,
            "seed": args.seed, "train_rows": len(train_df),
            "icd9_pickle": args.icd9_pickle, "use_complete_icd9": bool(args.use_complete_icd9),
            "total_label_space": len(labels_vocab),
            "test_batch_size": args.test_batch_size
        })
        save_json(os.path.join(RUN_DIR, "label_space.json"), {"labels": labels_vocab})
    barrier()

    # Training args
    train_args = make_training_args(args, RUN_DIR)

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    callbacks.append(DetailedEvalCallback(
        eval_dataset=val_ds, tokenizer=tok, label_vocab=labels_vocab,
        eval_sample_size=args.eval_sample_size, seed=args.seed
    ))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda feats: pad_collate(feats, tok),
        callbacks=callbacks
    )

    # Train
    t0 = time.perf_counter()
    train_start = datetime.datetime.now()
    rank0_print("Starting training…")
    trainer.train()
    train_secs = time.perf_counter() - t0
    train_duration = datetime.datetime.now() - train_start
    rank0_print(f"[TIME] Training completed in {train_secs:.2f}s ({train_duration})")
    barrier()

    # Save adapter/tokenizer (rank 0)
    if is_main_process():
        rank0_print("Saving model and tokenizer...")
        tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
        unwrap_model(trainer.model).save_pretrained(os.path.join(RUN_DIR, "adapter_best"))
        rank0_print("Model and tokenizer saved.")
    barrier()

    # Merge (optional, rank 0)
    if args.merge_after and is_main_process():
        try:
            rank0_print("Merging adapter with base model...")
            merged_dir = os.path.join(RUN_DIR, "model_merged")
            merged = unwrap_model(trainer.model).merge_and_unload()
            merged.save_pretrained(merged_dir)
            tok.save_pretrained(os.path.join(merged_dir, "tokenizer"))
            log.info(f"Merged model saved to: {merged_dir}")
        except Exception as e:
            log.warning(f"Could not merge adapters into base: {e}")
    barrier()

    # Final TEST generation (rank 0 only)
    if is_main_process():
        rank0_print(f"\n=== Starting TEST generation for {len(test_df)} samples ===")
        test_prompts = test_df["input_text"].astype(str).tolist()
        
        # Free up memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use specified batch size for test generation
        rank0_print(f"Using test batch size: {args.test_batch_size}")
        
        # Generate predictions with optimized batching
        test_start = datetime.datetime.now()
        t_gen = time.perf_counter()
        
        pred_code_lists = generate_codes(
            trainer.model, tok, test_prompts, labels_vocab,
            max_new=args.gen_max_new, 
            batch_size=args.test_batch_size, 
            max_len=args.max_len
        )
        test_gen_secs = time.perf_counter() - t_gen
        test_duration = datetime.datetime.now() - test_start
        
        rank0_print(f"=== Test Generation Summary ===")
        rank0_print(f"Total samples processed: {len(test_df)}")
        rank0_print(f"Batch size used: {args.test_batch_size}")
        rank0_print(f"Generation time: {test_gen_secs:.1f}s ({test_duration})")
        rank0_print(f"Average time per sample: {test_gen_secs/len(test_df):.3f}s")
        rank0_print(f"Samples per second: {len(test_df)/test_gen_secs:.2f}")

        # Calculate metrics
        rank0_print(f"Computing evaluation metrics...")
        Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
        metrics = eval_sets(y_test, Y_pred)
        metrics.update(hierarchical_eval(y_test, Y_pred, labels_vocab))
        metrics["train_seconds"] = train_secs
        metrics["train_duration_str"] = str(train_duration)
        metrics["test_generate_seconds"] = test_gen_secs
        metrics["test_duration_str"] = str(test_duration)
        metrics["test_samples"] = len(test_df)
        metrics["test_batch_size"] = args.test_batch_size
        metrics["samples_per_second"] = len(test_df)/test_gen_secs

        with open(os.path.join(RUN_DIR, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Calculate and print total experiment time
        total_experiment_time = datetime.datetime.now() - train_start
        rank0_print(f"\n=== Total Experiment Summary ===")
        rank0_print(f"Training time: {train_secs:.1f}s ({train_duration})")
        rank0_print(f"Testing time: {test_gen_secs:.1f}s ({test_duration})")
        rank0_print(f"Total experiment duration: {total_experiment_time}")
        
        rank0_print(f"\n=== Generative TEST metrics ===")
        rank0_print(f"Main metrics:")
        rank0_print(f"  - micro_f1: {metrics['micro_f1']:.4f}")
        rank0_print(f"  - macro_f1: {metrics['macro_f1']:.4f}")  
        rank0_print(f"  - samples_f1: {metrics['samples_f1']:.4f}")
        rank0_print(f"  - hierarchical_parent_recall: {metrics['hierarchical_parent_recall']:.4f}")
        
        rank0_print("\n=== Sample Predictions ===")
        show_test_predictions(test_df, pred_code_lists, args.label_col, labels_vocab,
                              n_show=args.test_examples, seed=args.seed)
    else:
        rank0_print("Skipping test generation on non-main ranks.")

if __name__ == "__main__":
    main()