# -*- coding: utf-8 -*-
"""
Generative ICD code prediction with LoRA (DDP-enabled)
- Adds multi-GPU training via DDP
- Rank-aware logging and barriers for synchronization
- Clean DDP shutdown and proper device handling
- Uses complete ICD9 codes from icd9.pkl but maintains original code handling
"""

import os, re, json, random, logging, pickle, datetime, time, atexit, argparse
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml

# ============== Env / logging / reproducibility ==============
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

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

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# Setup logging based on rank
logging.basicConfig(level=logging.INFO if is_main_process() else logging.WARNING, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def rank0_print(*a, **k):
    if is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}]", *a, **k)

# Enable TF32 on Ampere+ (H100 supported)
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

# Clean DDP/NCCL shutdown to silence destroy_process_group warning
def _cleanup_dist():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            torch.distributed.destroy_process_group()
            log.info("DDP/NCCL process group destroyed cleanly.")
    except Exception as e:
        log.debug(f"DDP cleanup skipped: {e}")
atexit.register(_cleanup_dist)

# ============== Args ===============
def get_args():
    parser = argparse.ArgumentParser()
    # Data and model config
    parser.add_argument("--data_pickle", default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_structured", type=int, default=1)
    parser.add_argument("--use_notes", type=int, default=1)
    parser.add_argument("--subject_col", default="subject_id_x")
    parser.add_argument("--label_col", default="icd_code")
    parser.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--icd9_pickle", default="MasterThesis/dataset/codes/icd9.pkl", 
                      help="Path to complete ICD-9 code list")
    parser.add_argument("--use_complete_icd9", type=int, default=1)
    
    # Training parameters
    parser.add_argument("--max_len", type=int, default=3072)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gen_max_new", type=int, default=96)
    parser.add_argument("--tgt_reserve_tok", type=int, default=128)
    parser.add_argument("--eval_gen_subset", type=int, default=500)
    parser.add_argument("--eval_gen_bs", type=int, default=4)
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--wandb_online", type=int, default=1)
    
    # Debug
    parser.add_argument("--smoke_test", action="store_true")
    
    return parser.parse_args()

# ============== Secrets (optional) ==============
def load_secrets(path="secrets.yaml"):
    if not os.path.isfile(path):
        return {"wandb_ok": False, "hf_ok": False, "wandb_cfg": {}}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    auth = (data.get("auth") or {})
    wb   = (data.get("wandb") or {})
    wandb_token = auth.get("wandb_token")
    hf_token    = auth.get("hf_token")
    if wandb_token: os.environ.setdefault("WANDB_API_KEY", wandb_token)
    if hf_token:    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    if "project" in wb: os.environ.setdefault("WANDB_PROJECT", wb["project"])
    return {"wandb_ok": bool(wandb_token), "hf_ok": bool(hf_token), "wandb_cfg": wb}

# ============== Helpers ==============
def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def make_run_dir(base="runs_gen", run_name=None):
    tag = run_name or f"{now_tag()}_llama1b_gen_ddp_len{args.max_len}_lr{args.learning_rate}"
    path = os.path.join(base, tag)
    
    # Only create directories on main process
    if is_main_process():
        os.makedirs(path, exist_ok=False)
        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    barrier()
    return path

def save_json(path: str, obj: dict):
    if is_main_process():
        with open(path, "w") as f: json.dump(obj, f, indent=2)

def log_seconds(tag, start):
    dur = time.perf_counter() - start
    log.info(f"[TIME] {tag}: {dur:.2f} seconds")
    return dur

# ============== Input serialization ==============
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
    TEXT_COLS_SAFE = [
        "Chief Complaint","History of Present Illness","Past Medical History",
        "Family History","Physical Exam","Pertinent Results",
        "Brief Hospital Course","Medications on Admission"
    ]
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t: chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)

def build_input_text(row: pd.Series, args) -> str:
    s = [f"[VISIT] subject_id={row.get(args.subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
    if args.use_structured: s.append(serialize_structured(row))
    if args.use_notes:
        t = serialize_notes(row)
        if t: s.append(t)
    s.append("[TASK] Predict ICD diagnosis codes (space-separated). Output ONLY the codes, separated by single spaces.")
    s.append("[CODES]")  # target delimiter
    return "\n".join([x for x in s if x])

# ============== Splits & labels ==============
def subject_splits(df: pd.DataFrame, subject_col,
                   test_size=0.10, val_size=0.10, seed=42):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    log.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

# Using lock_label_space from finetune_llama_gen_ddp.py
def lock_label_space(final_df: pd.DataFrame, label_col: str,
                     icd9_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
    """Set fixed label space (either data-driven or from icd9.pkl)"""
    # Collect all unique codes from the data
    all_codes = sorted({str(code) for codes in final_df[label_col] for code in codes})
    
    if use_complete and icd9_pkl_path and os.path.exists(icd9_pkl_path):
        try:
            # Load the comprehensive ICD9 code list
            if is_main_process():
                rank0_print(f"Loading complete ICD9 codes from {icd9_pkl_path}")
            with open(icd9_pkl_path, 'rb') as f:
                icd9_data = pickle.load(f)
            
            # Extract all codes from the comprehensive list
            complete_codes = set()
            for _, row in icd9_data.iterrows():
                code = str(row['icd_code']).strip()
                if code:
                    complete_codes.add(code)
            
            if is_main_process():
                rank0_print(f"Complete ICD9 codes: {len(complete_codes)}")
                rank0_print(f"Codes in data: {len(all_codes)}")
                missing = all_codes - complete_codes
                if missing:
                    rank0_print(f"Warning: {len(missing)} codes in data not in ICD9 list")
            
            # Use the union of complete codes and data codes
            all_codes = sorted(complete_codes.union(all_codes))
        except Exception as e:
            if is_main_process():
                rank0_print(f"Error loading ICD9 codes: {e}, using data-derived codes only")
            all_codes = sorted(all_codes)
    else:
        all_codes = sorted(all_codes)
    
    # Create and fit the MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=all_codes)
    mlb.fit([all_codes])
    if is_main_process():
        rank0_print(f"Total unique ICD codes in label space: {len(all_codes)}")
    return mlb

def y_multi_hot(mlb: MultiLabelBinarizer, lists):
    return mlb.transform([[str(c) for c in row] for row in lists])

# ============== Dataset (reserves target tokens) ==============
class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int = 128, label_col="icd_code"):
        self.prompts = rows["input_text"].astype(str).tolist()
        self.targets = [" ".join(sorted({str(c) for c in codes})) for codes in rows[label_col].tolist()]
        self.tok = tok; self.max_len = max_len; self.tgt_reserve = max(8, int(tgt_reserve))

    def __len__(self): return len(self.prompts)

    def __getitem__(self, i):
        prompt = self.prompts[i]
        answer = self.targets[i]

        # Tokenize in one call using the __call__ method
        prompt_with_newline = f"{prompt}\n"
        prompt_encoding = self.tok(
            prompt_with_newline,
            add_special_tokens=True,
            max_length=self.max_len - self.tgt_reserve,
            truncation=True,
            return_tensors=None
        )
        
        # Tokenize the answer
        answer_with_eos = f"{answer}{self.tok.eos_token or ''}"
        answer_encoding = self.tok(
            answer_with_eos,
            add_special_tokens=False,
            max_length=self.tgt_reserve,
            truncation=True,
            return_tensors=None
        )
        
        # Combine the encodings
        input_ids = prompt_encoding["input_ids"] + answer_encoding["input_ids"]
        attention_mask = prompt_encoding["attention_mask"] + [1] * len(answer_encoding["input_ids"])
        
        # Create the labels: -100 for prompt, actual ids for answer
        labels = [-100] * len(prompt_encoding["input_ids"]) + answer_encoding["input_ids"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pad_collate(features, tok):
    input_ids = [f["input_ids"] for f in features]
    attn = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]
    
    # Use the tokenizer's padding method in one call
    pad_out = tok.pad(
        {"input_ids": input_ids, "attention_mask": attn}, 
        return_tensors="pt",
        padding=True
    )
    
    # Handle labels separately since they contain -100 values
    max_len = pad_out["input_ids"].size(1)
    lab_pad = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, lab in enumerate(labels):
        lab_pad[i, :len(lab)] = lab
        
    return {
        "input_ids": pad_out["input_ids"], 
        "attention_mask": pad_out["attention_mask"], 
        "labels": lab_pad
    }

# ============== Unwrap for DDP model access ==============
def unwrap_model(m):
    # Handle various model wrappers: DDP, FSDP, DataParallel
    if hasattr(m, "module"): return unwrap_model(m.module)
    # Handle PEFT models
    if hasattr(m, "base_model"): return m
    return m

# ============== Model loader ==============
def load_lm_and_tokenizer(model_name):
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    # IMPORTANT: Fix for DDP error - don't use device_map="auto" with DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    base = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype,
        # Remove device_map="auto" for DDP compatibility
    )
    # Explicitly move model to correct device
    base = base.to(device)
    
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora_cfg)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    
    if is_main_process():
        model.print_trainable_parameters()
    return model, tok

# ============== Generation + vocabulary-filtered parsing ==============
def normalize_code(c: str) -> str:
    c = c.strip().upper()
    c = re.sub(r"\s+", "", c)
    return c[:-1] if c.endswith(".") else c

@torch.no_grad()
def generate_codes(model, tok, prompts: List[str], labels_vocab: List[str],
                   max_new=96, batch_size=4, max_len=3072):
    """Greedy decoding + strict vocab filter against label space."""
    # Use unwrapped model
    model = unwrap_model(model)
    model.eval()
    allowed = set(labels_vocab)
    preds = []
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Process in batches with proper device placement
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,        # greedy
                num_beams=1,
                no_repeat_ngram_size=2,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            
        texts = tok.batch_decode(out, skip_special_tokens=True)
        for t in texts:
            tail = t.split("[CODES]")[-1]
            tokens = re.split(r"[^A-Za-z0-9\.]+", tail)
            cand = [normalize_code(z) for z in tokens if z]
            # Keep only codes seen in training label space, de-dup in order
            seen, keep = set(), []
            for c in cand:
                if c in allowed and c not in seen:
                    seen.add(c); keep.append(c)
            preds.append(keep)
            
        # Clear cache periodically
        if torch.cuda.is_available() and i % (5 * batch_size) == 0:
            torch.cuda.empty_cache()
            
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
        "micro_recall":      recall_score(y_true, y_pred, average="micro",   zero_division=0),
        "macro_recall":      recall_score(y_true, y_pred, average="macro",   zero_division=0),
        "samples_recall":    recall_score(y_true, y_pred, average="samples", zero_division=0),
    }

# ============== Per-epoch gen metrics callback with timing ==============
# FIXED: Store gen_max_new and max_len as instance variables
class EvalGenCallback(TrainerCallback):
    def __init__(self, model_ref, tok, val_prompts: List[str], y_val: np.ndarray, label_vocab: List[str],
                 batch_size=4, max_items=500, seed=42, gen_max_new=96, max_len=3072):
        super().__init__()
        self.model_ref = model_ref
        self.tok = tok
        self.all_prompts = list(val_prompts)
        self.y_val_full = y_val
        self.label_vocab = label_vocab
        self.bs = batch_size
        self.max_items = max_items
        self.gen_max_new = gen_max_new  # Store as instance variable
        self.max_len = max_len          # Store as instance variable
        self.rng = np.random.default_rng(seed)
        self.sub_idx = None  # fixed subset across epochs

    def on_evaluate(self, args, state, control, **kwargs):
        # Skip on non-main processes
        if not is_main_process():
            return control
            
        if self.max_items is not None and self.sub_idx is None:
            n = min(self.max_items, len(self.all_prompts))
            self.sub_idx = self.rng.choice(len(self.all_prompts), size=n, replace=False)

        if self.sub_idx is None:
            prompts = self.all_prompts
            y_true  = self.y_val_full
        else:
            prompts = [self.all_prompts[i] for i in self.sub_idx]
            y_true  = self.y_val_full[self.sub_idx]

        t0 = time.perf_counter()
        preds = generate_codes(
            self.model_ref, self.tok, prompts, self.label_vocab,
            max_new=self.gen_max_new,  # Use instance variable instead of args
            batch_size=self.bs, 
            max_len=self.max_len       # Use instance variable instead of args
        )
        gen_secs = time.perf_counter() - t0
        Yh = codes_to_multihot(preds, self.label_vocab)
        metrics = eval_sets(y_true, Yh)
        prefixed = {f"eval_{k}": v for k, v in metrics.items()}
        prefixed["epoch"] = float(state.epoch or 0.0)
        prefixed["eval_gen_seconds"] = gen_secs
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.log(prefixed)
        else:
            log.info(f"[EvalGen] {prefixed}")
        return control

# ============== Pretty-print a few test predictions ==============
def show_test_predictions(df: pd.DataFrame,
                          preds: List[List[str]],
                          n_show: int = 8,
                          seed: int = 0,
                          prompt_tail_lines: int = 8,
                          tail_col: str = "input_text",
                          label_col: str = "icd_code"):
    if not is_main_process():
        return
        
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
    for i in idxs:
        row = df.iloc[i]
        gold = sorted({normalize_code(c) for c in row[label_col]})
        pred = preds[i]
        missing = sorted([c for c in gold if c not in pred])  # FN
        extra   = sorted([c for c in pred if c not in gold])  # FP
        print("\n" + "="*80)
        print(f"Example idx={i} | subject_id={row.get(args.subject_col)} | hadm_id={row.get('hadm_id')}")
        print("- GOLD:", " ".join(gold) if gold else "(none)")
        print("- PRED:", " ".join(pred) if pred else "(none)")
        print(f"- FN  ({len(missing)}):", " ".join(missing) if missing else "(none)")
        print(f"- FP  ({len(extra)}):", " ".join(extra) if extra else "(none)")
        tail_lines = str(row[tail_col]).splitlines()[-prompt_tail_lines:]
        print("\n--- Prompt tail ---")
        for line in tail_lines:
            line = line.strip()
            if line.startswith("[CODES]"):
                print("[CODES]  <-- target starts after this marker during training")
            else:
                print(line[:180])

# ============== Main function ==============
def main():
    global args
    args = get_args()
    set_seed(args.seed)
    
    # Logging information about environment
    rank0_print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        rank0_print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        rank0_print(f"GPU count: {torch.cuda.device_count()}")
    rank0_print(f"Process rank: {get_rank()}")

    # Debug settings for smoke test
    DEBUG_MAX_LEN = 768
    DEBUG_TRAIN_SUBJ = 300
    DEBUG_VAL_SUBJ = 60
    DEBUG_TEST_SUBJ = 60

    if args.smoke_test:
        args.max_len = DEBUG_MAX_LEN
        WANDB_ONLINE = False
    else:
        WANDB_ONLINE = bool(args.wandb_online)

    # Load secrets & data
    secrets_info = load_secrets("secrets.yaml")
    if not WANDB_ONLINE:
        os.environ["WANDB_MODE"] = "offline"

    # Load and prepare the data
    if is_main_process():
        rank0_print(f"Loading data from {args.data_pickle}")
        
    final_df = pickle.load(open(args.data_pickle, "rb"))
    assert args.label_col in final_df.columns and args.subject_col in final_df.columns

    df = final_df.copy()
    df["input_text"] = df.apply(lambda r: build_input_text(r, args), axis=1)
    df = df[df["input_text"].str.len() > 0]

    train_df, val_df, test_df = subject_splits(df, subject_col=args.subject_col, test_size=0.10, val_size=0.10, seed=42)

    if args.smoke_test:
        def limit_subjects(dx, col, n):
            subs = dx[col].dropna().unique().tolist()
            random.shuffle(subs); keep = set(subs[:min(n,len(subs))])
            return dx[dx[col].isin(keep)].copy()
        train_df = limit_subjects(train_df, args.subject_col, DEBUG_TRAIN_SUBJ)
        val_df   = limit_subjects(val_df,   args.subject_col, DEBUG_VAL_SUBJ)
        test_df  = limit_subjects(test_df,  args.subject_col, DEBUG_TEST_SUBJ)
        log.info(f"[SMOKE] visits -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Use the lock_label_space from finetune_llama_gen_ddp.py
    mlb = lock_label_space(
        final_df=df, 
        label_col=args.label_col,
        icd9_pkl_path=args.icd9_pickle,
        use_complete=bool(args.use_complete_icd9)
    )
    
    y_train = y_multi_hot(mlb, train_df[args.label_col].tolist())
    y_val = y_multi_hot(mlb, val_df[args.label_col].tolist())
    y_test = y_multi_hot(mlb, test_df[args.label_col].tolist())
    labels_vocab = mlb.classes_.tolist()

    # Create run directory
    RUN_DIR = make_run_dir(run_name=args.run_name)
    rank0_print("Run dir:", RUN_DIR)
    
    if is_main_process():
        save_json(os.path.join(RUN_DIR, "config.json"), {
            "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate, 
            "epochs": args.epochs, "gen_max_new": args.gen_max_new, "tgt_reserve_tok": args.tgt_reserve_tok,
            "smoke_test": args.smoke_test, "early_stop": args.early_stop, "patience": args.patience,
            "eval_gen_subset": args.eval_gen_subset, "ddp_enabled": True,
            "use_complete_icd9": args.use_complete_icd9
        })
        save_json(os.path.join(RUN_DIR, "label_space.json"), {"labels": labels_vocab})
    barrier()

    # Load model and tokenizer with fixed DDP-compatible loading
    model, tok = load_lm_and_tokenizer(args.llama_model)
    train_ds = GenCodesDataset(train_df, tok, args.max_len, tgt_reserve=args.tgt_reserve_tok, label_col=args.label_col)
    val_ds   = GenCodesDataset(val_df,   tok, args.max_len, tgt_reserve=args.tgt_reserve_tok, label_col=args.label_col)
    test_prompts = test_df["input_text"].astype(str).tolist()

    # W&B
    report_to = ["wandb"] if secrets_info["wandb_ok"] and WANDB_ONLINE else "none"

    # Training args for DDP
    train_args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03, 
        weight_decay=0.0,
        logging_strategy="epoch",
        eval_strategy="epoch",
        prediction_loss_only=True,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=bool(args.early_stop),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim="adamw_torch", 
        dataloader_num_workers=2,
        run_name=os.path.basename(RUN_DIR),
        disable_tqdm=True,
        # DDP-specific:
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
    )

    # Callbacks
    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    # Add per-epoch generation metrics callback (with timing) - FIXED by passing args to callback
    callbacks.append(
        EvalGenCallback(
            model_ref=model,
            tok=tok,
            val_prompts=val_df["input_text"].astype(str).tolist(),
            y_val=y_val,
            label_vocab=labels_vocab,
            batch_size=args.eval_gen_bs,
            max_items=args.eval_gen_subset,
            seed=42,
            gen_max_new=args.gen_max_new,  # Pass command-line args
            max_len=args.max_len           # to callback constructor
        )
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=lambda feats: pad_collate(feats, tok),
        compute_metrics=None,
        callbacks=callbacks
    )

    # Train (timed)
    t0 = time.perf_counter()
    rank0_print("Starting generative training…")
    trainer.train()
    train_time = log_seconds("train", t0)
    rank0_print("Training complete.")
    barrier()

    # Save artifacts (main process only)
    if is_main_process():
        tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
        trainer.model.save_pretrained(os.path.join(RUN_DIR, "adapter_best"))
        rank0_print("Saved tokenizer and model adapter.")

        # Merge model (optional)
        t_merge = time.perf_counter()
        try:
            merged_dir = os.path.join(RUN_DIR, "model_merged")
            merged = unwrap_model(trainer.model).merge_and_unload()
            merged.save_pretrained(merged_dir)
            tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
            log.info(f"Merged model saved to: {merged_dir}")
        except Exception as e:
            log.warning(f"Could not merge adapters into base: {e}")
        merge_time = log_seconds("merge_model", t_merge)
    barrier()

    # Final TEST evaluation (main process only)
    if is_main_process():
        # Free up memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        t_gen = time.perf_counter()
        rank0_print(f"Generating predictions for {len(test_prompts)} test samples...")
        pred_code_lists = generate_codes(
            model, tok, test_prompts, labels_vocab,
            max_new=args.gen_max_new, batch_size=args.eval_gen_bs, max_len=args.max_len
        )
        gen_time = log_seconds("test_generate", t_gen)

        t_metric = time.perf_counter()
        Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
        metrics = eval_sets(y_test, Y_pred)
        metrics["train_time"] = train_time
        metrics["merge_time"] = merge_time
        metrics["gen_time"] = gen_time
        metrics_time = log_seconds("test_metrics", t_metric)

        save_json(os.path.join(RUN_DIR, "test_metrics.json"), metrics)

        print("\n=== Generative Test metrics ===")
        print(json.dumps(metrics, indent=2))

        print("\n--- A few test predictions ---")
        show_test_predictions(test_df, pred_code_lists, n_show=5, seed=42, 
                             prompt_tail_lines=8, label_col=args.label_col)
    barrier()

if __name__ == "__main__":
    main()