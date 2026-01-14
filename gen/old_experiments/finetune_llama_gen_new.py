# -*- coding: utf-8 -*-
"""
Generative ICD code prediction with LoRA
"""

import os, re, json, random, logging, pickle, datetime, time, atexit
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

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
print("CUDA:", torch.cuda.is_available(),
      "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

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
            logging.info("DDP/NCCL process group destroyed cleanly.")
    except Exception as e:
        logging.debug(f"DDP cleanup skipped: {e}")
atexit.register(_cleanup_dist)

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

# ============== Config ==============
SMOKE_TEST   = False      # set True to sub-sample subjects and shrink MAX_LEN
EARLY_STOP   = True
PATIENCE     = 2
WANDB_ONLINE = True

RUN_NAME     = None
#DATA_PICKLE  = "mergeddf.pkl"
DATA_PICKLE  = "df.pkl"
ICD9_CODES_PICKLE = "MasterThesis/dataset/codes/icd9.pkl"  # Path to complete ICD-9 codes
print(f"Using data: {DATA_PICKLE}")
print(f"Using ICD-9 codes: {ICD9_CODES_PICKLE}")

USE_STRUCTURED, USE_NOTES = True, True
SUBJECT_COL, LABEL_COL = "subject_id_x", "icd_code"

TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

LLAMA_MODEL  = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LEN      = 3072
LR           = 2e-4
EPOCHS       = 10
GEN_MAX_NEW  = 96

# Always reserve space for labels in the sequence
TGT_RESERVE_TOK = 128

# eval-gen subset per epoch (to keep it fast/mem-safe)
EVAL_GEN_SUBSET = 500
EVAL_GEN_BS     = 4

# smoke-test knobs
DEBUG_MAX_LEN = 768; DEBUG_TRAIN_SUBJ = 300; DEBUG_VAL_SUBJ = 60; DEBUG_TEST_SUBJ = 60
MAX_STEPS = 200; EVAL_STEPS = 50; LOG_STEPS = 10

print(f"\nModel Configs: MAX_LEN:{MAX_LEN}, LR:{LR}, EPOCHS:{EPOCHS}, GEN_MAX_NEW:{GEN_MAX_NEW}, TGT_RESERVE_TOK:{TGT_RESERVE_TOK}")

# ============== Helpers ==============
def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def make_run_dir(base="runs_gen", run_name=None):
    tag = run_name or f"{now_tag()}_llama1b_gen_len{MAX_LEN}_lr{LR}"
    path = os.path.join(base, tag)
    os.makedirs(path, exist_ok=False)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path

def save_json(path: str, obj: dict):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def log_seconds(tag, start):
    dur = time.perf_counter() - start
    logging.info(f"[TIME] {tag}: {dur:.2f} seconds")
    return dur

# ============== ICD-9 Code Handling ==============
def format_icd9_properly(code: str) -> str:
    """Format ICD-9 code with proper decimal placement."""
    # Basic cleaning
    code = code.strip().upper()
    code = re.sub(r"\s+", "", code)
    
    # Remove trailing period if any
    if code.endswith("."):
        code = code[:-1]
        
    # Handle decimal point formatting for diagnosis codes
    if code and code[0].isdigit():
        # Regular ICD-9 diagnosis codes
        if '.' not in code and len(code) > 3:
            return code[:3] + '.' + code[3:]
    
    # Handle V and E codes and procedure codes
    elif code and len(code) > 1:
        if code[0] in ('V', 'E') and '.' not in code and len(code) > 3:
            return code[:3] + '.' + code[3:]
    
    return code

def is_valid_icd9(code: str) -> bool:
    """Validate if a string follows ICD-9-CM format patterns."""
    # If empty, not valid
    if not code:
        return False
    
    # Regular diagnosis codes (3 digits + optional 1-2 decimals)
    if code[0].isdigit():
        return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    
    # V codes (V + 2 digits + optional 1-2 decimals)
    elif code.startswith('V'):
        return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    
    # E codes (E + 3 digits + optional 1 decimal)
    elif code.startswith('E'):
        return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    
    return False

def normalize_code(c: str) -> str:
    """Normalize ICD code with format validation."""
    return format_icd9_properly(c)

def get_icd9_parent(code: str) -> str:
    """Get parent code (category) from an ICD-9 code."""
    if not code or len(code) < 3:
        return code
    
    # Regular codes - first 3 digits
    if code[0].isdigit():
        return code.split('.')[0][:3]
    
    # V codes - V + first 2 digits
    elif code.startswith('V'):
        base = code.split('.')[0]
        return base[:3]
    
    # E codes - E + first 3 digits
    elif code.startswith('E'):
        base = code.split('.')[0]
        return base[:4] if len(base) >= 4 else base
    
    return code

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
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t: chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)

def format_icd9_codes_list(codes_list):
    """Format a list of ICD-9 codes with proper formatting."""
    return [format_icd9_properly(str(c)) for c in codes_list]

def build_input_text(row: pd.Series) -> str:
    s = [f"[VISIT] subject_id={row.get(SUBJECT_COL,'?')} hadm_id={row.get('hadm_id','?')}"]
    if USE_STRUCTURED: s.append(serialize_structured(row))
    if USE_NOTES:
        t = serialize_notes(row)
        if t: s.append(t)
    
    # Improved task prompt with more explicit instructions
    s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
    s.append("1. List only the ICD-9 codes separated by spaces")
    s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
    s.append("3. Include only codes directly supported by the clinical information")
    s.append("4. Do not include any explanations or text besides the codes themselves")
    s.append("[CODES]")  # target delimiter
    return "\n".join([x for x in s if x])

# ============== Splits & labels (with complete ICD-9 code space) ==============
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

def lock_label_space(full_df: pd.DataFrame, icd9_pkl_path: str) -> MultiLabelBinarizer:
    """Create label space using complete ICD-9 code list from external file."""
    try:
        # Load the complete ICD-9 code list
        icd9_df = pd.read_pickle(icd9_pkl_path)
        complete_codes = sorted(icd9_df['icd_code'].astype(str).tolist())
        
        # Format the complete codes list with proper decimal points
        complete_codes = [format_icd9_properly(code) for code in complete_codes]
        
        # Filter to only include valid codes
        complete_codes = [code for code in complete_codes if is_valid_icd9(code)]
        
        logging.info(f"Loaded {len(complete_codes)} complete ICD-9 codes from {icd9_pkl_path}")
        
        # Create and fit the MultiLabelBinarizer with the complete code list
        mlb = MultiLabelBinarizer(classes=complete_codes)
        mlb.fit([complete_codes])
        
        # Optionally, check how many codes from training data are in complete set
        training_codes = {format_icd9_properly(str(code)) for codes in full_df[LABEL_COL] for code in codes}
        training_codes = {c for c in training_codes if is_valid_icd9(c)}
        
        codes_in_complete = sum(1 for c in training_codes if c in set(complete_codes))
        codes_not_in_complete = len(training_codes) - codes_in_complete
        
        logging.info(f"Training data has {len(training_codes)} unique codes")
        logging.info(f"- {codes_in_complete} codes present in complete set")
        logging.info(f"- {codes_not_in_complete} codes not in complete set")
        
        if codes_not_in_complete > 0:
            logging.warning(f"Some codes in training data not found in complete ICD-9 set!")
        
        return mlb
        
    except Exception as e:
        logging.error(f"Error loading complete ICD-9 codes: {e}")
        logging.warning("Falling back to training-data-only label space")
        
        # Fallback to original approach if loading fails
        all_codes = sorted({format_icd9_properly(str(code)) 
                           for codes in full_df[LABEL_COL] 
                           for code in codes})
        # Remove any invalid codes
        all_codes = [code for code in all_codes if is_valid_icd9(code)]
        
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        logging.info(f"Created fallback label space with {len(all_codes)} codes from training data")
        return mlb

def y_multi_hot(mlb: MultiLabelBinarizer, lists):
    # Format the codes consistently before conversion to multi-hot
    formatted_lists = []
    for row in lists:
        formatted_row = [format_icd9_properly(str(c)) for c in row]
        # Only keep valid codes
        formatted_row = [c for c in formatted_row if is_valid_icd9(c)]
        formatted_lists.append(formatted_row)
    
    return mlb.transform(formatted_lists)

# ============== Dataset (reserves target tokens) ==============
class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int = 128):
        self.prompts = rows["input_text"].astype(str).tolist()
        
        # Format ICD codes consistently in training targets
        self.targets = []
        for codes in rows[LABEL_COL].tolist():
            # Format each code properly with decimal points
            formatted_codes = [format_icd9_properly(str(c)) for c in codes]
            # Keep only valid ICD-9 codes
            formatted_codes = [c for c in formatted_codes if is_valid_icd9(c)]
            # Join with spaces to create the target string
            self.targets.append(" ".join(sorted(set(formatted_codes))))
            
        self.tok = tok; self.max_len = max_len; self.tgt_reserve = max(8, int(tgt_reserve))

    def __len__(self): return len(self.prompts)

    def __getitem__(self, i):
        prompt = self.prompts[i]
        answer = self.targets[i]

        # encode separately
        prompt_ids = self.tok.encode(prompt + "\n", add_special_tokens=True)
        ans_ids    = self.tok.encode(answer + (self.tok.eos_token or ""), add_special_tokens=False)

        # 1) truncate prompt first to leave room for target
        max_prompt_len = max(1, self.max_len - self.tgt_reserve)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        # 2) then fit as much answer as possible in the remaining space
        remaining = max(1, self.max_len - len(prompt_ids))
        if len(ans_ids) > remaining:
            ans_ids = ans_ids[:remaining]

        input_ids = prompt_ids + ans_ids
        attention_mask = [1] * len(input_ids)
        # labels: ignore prompt, supervise answer
        labels = ([-100] * len(prompt_ids)) + ans_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def pad_collate(features, tok):
    input_ids = [f["input_ids"] for f in features]
    attn     = [f["attention_mask"] for f in features]
    labels   = [f["labels"] for f in features]
    pad_out  = tok.pad({"input_ids": input_ids, "attention_mask": attn}, return_tensors="pt")
    max_len = pad_out["input_ids"].size(1)
    lab_pad = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, lab in enumerate(labels):
        lab_pad[i, :lab.size(0)] = lab
    return {"input_ids": pad_out["input_ids"], "attention_mask": pad_out["attention_mask"], "labels": lab_pad}

# ============== Model loader ==============
def load_lm_and_tokenizer():
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL, torch_dtype=dtype, device_map="auto"
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora_cfg)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tok

# ============== Generation + vocabulary-filtered parsing ==============
@torch.no_grad()
def generate_codes(model, tok, prompts: List[str], labels_vocab: List[str],
                   max_new=96, batch_size=4, max_len=3072):
    """Greedy decoding + strict vocab filter against label space with ICD-9 validation."""
    model.eval()
    allowed = set(labels_vocab)
    preds = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tok(batch_prompts, return_tensors="pt",
                     padding=True, truncation=True, max_length=max_len).to(model.device)
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
            
            # Keep only codes that:
            # 1. Are seen in training label space
            # 2. Pass ICD-9 format validation 
            # 3. Haven't been seen before (de-dupe)
            seen, keep = set(), []
            for c in cand:
                if c in allowed and is_valid_icd9(c) and c not in seen:
                    seen.add(c)
                    keep.append(c)
            preds.append(keep)
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

def hierarchical_eval(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> Dict[str, float]:
    """Evaluate predictions with hierarchical consideration of ICD-9 codes."""
    # Get standard metrics first
    std_metrics = {}
    
    # Build a mapping of codes to their parent categories
    code_to_parent = {code: get_icd9_parent(code) for code in label_vocab}
    parent_to_idx = {}
    for idx, code in enumerate(label_vocab):
        parent = code_to_parent[code]
        if parent not in parent_to_idx:
            parent_to_idx[parent] = []
        parent_to_idx[parent].append(idx)
    
    # For each sample, check hierarchical matches
    n_samples = y_true.shape[0]
    parent_hits = 0
    partial_matches = 0
    total_true_parents = 0
    
    for i in range(n_samples):
        # Find predicted categories that match actual categories at parent level
        pred_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_pred[i])[0]}
        true_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_true[i])[0]}
        
        # Count parent-level hits
        parent_hits += len(pred_parents & true_parents)
        total_true_parents += len(true_parents)
        
        # Count predictions with correct parent but wrong specific code
        for parent in pred_parents:
            if parent in true_parents:
                # Get all child indices for this parent
                child_indices = parent_to_idx.get(parent, [])
                
                # Check if any child codes match exactly
                exact_match = False
                for idx in child_indices:
                    if y_true[i, idx] == 1 and y_pred[i, idx] == 1:
                        exact_match = True
                        break
                
                if not exact_match:
                    partial_matches += 1
    
    # Calculate hierarchical metrics
    parent_recall = total_true_parents > 0 and parent_hits / total_true_parents or 0
    
    # Add hierarchical metrics
    std_metrics["hierarchical_parent_recall"] = parent_recall
    std_metrics["hierarchical_partial_matches"] = partial_matches
    std_metrics["hierarchical_partial_per_sample"] = partial_matches / n_samples if n_samples > 0 else 0
    
    return std_metrics

# ============== Per-epoch gen metrics callback with timing ==============
class EvalGenCallback(TrainerCallback):
    def __init__(self, model_ref, tok, val_prompts: List[str], y_val: np.ndarray, label_vocab: List[str],
                 batch_size=EVAL_GEN_BS, max_items=EVAL_GEN_SUBSET, seed=42):
        super().__init__()
        self.model_ref = model_ref
        self.tok = tok
        self.all_prompts = list(val_prompts)
        self.y_val_full = y_val
        self.label_vocab = label_vocab
        self.bs = batch_size
        self.max_items = max_items
        self.rng = np.random.default_rng(seed)
        self.sub_idx = None  # fixed subset across epochs

    def on_evaluate(self, args, state, control, **kwargs):
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
            max_new=GEN_MAX_NEW, batch_size=self.bs, max_len=MAX_LEN
        )
        gen_secs = time.perf_counter() - t0
        Yh = codes_to_multihot(preds, self.label_vocab)
        
        # Get standard metrics
        metrics = eval_sets(y_true, Yh)
        
        # Add hierarchical metrics
        hier_metrics = hierarchical_eval(y_true, Yh, self.label_vocab)
        metrics.update({k: v for k, v in hier_metrics.items()})
        
        prefixed = {f"eval_{k}": v for k, v in metrics.items()}
        prefixed["epoch"] = float(state.epoch or 0.0)
        prefixed["eval_gen_seconds"] = gen_secs
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.log(prefixed)
        else:
            logging.info(f"[EvalGen] {prefixed}")
        return control

# ============== Pretty-print a few test predictions ==============
def show_test_predictions(df: pd.DataFrame,
                          preds: List[List[str]],
                          n_show: int = 8,
                          seed: int = 0,
                          prompt_tail_lines: int = 8,
                          tail_col: str = "input_text"):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
    for i in idxs:
        row = df.iloc[i]
        # Format gold codes consistently
        gold = sorted({format_icd9_properly(str(c)) for c in row[LABEL_COL] 
                      if is_valid_icd9(format_icd9_properly(str(c)))})
        pred = preds[i]
        missing = sorted([c for c in gold if c not in pred])  # FN
        extra   = sorted([c for c in pred if c not in gold])  # FP
        
        # Check for parent-level matches (partial matches)
        gold_parents = {get_icd9_parent(c) for c in gold}
        pred_parents = {get_icd9_parent(c) for c in pred}
        parent_matches = sorted([f"{c} (parent)" for c in pred if get_icd9_parent(c) in gold_parents and c not in gold])
        
        print("\n" + "="*80)
        print(f"Example idx={i} | subject_id={row.get(SUBJECT_COL)} | hadm_id={row.get('hadm_id')}")
        print("- GOLD:", " ".join(gold) if gold else "(none)")
        print("- PRED:", " ".join(pred) if pred else "(none)")
        print(f"- FN  ({len(missing)}):", " ".join(missing) if missing else "(none)")
        print(f"- FP  ({len(extra)}):", " ".join(extra) if extra else "(none)")
        print(f"- PARENT MATCHES ({len(parent_matches)}):", " ".join(parent_matches) if parent_matches else "(none)")
        tail_lines = str(row[tail_col]).splitlines()[-prompt_tail_lines:]
        print("\n--- Prompt tail ---")
        for line in tail_lines:
            line = line.strip()
            if line.startswith("[CODES]"):
                print("[CODES]  <-- target starts after this marker during training")
            else:
                print(line[:180])

# ============== Load secrets & data ==============
secrets_info = load_secrets("secrets.yaml")
if not WANDB_ONLINE:
    os.environ["WANDB_MODE"] = "offline"

final_df = pickle.load(open(DATA_PICKLE, "rb"))
assert LABEL_COL in final_df.columns and SUBJECT_COL in final_df.columns

df = final_df.copy()
df["input_text"] = df.apply(lambda r: build_input_text(r), axis=1)
df = df[df["input_text"].str.len() > 0]

train_df, val_df, test_df = subject_splits(df, subject_col=SUBJECT_COL, test_size=0.10, val_size=0.10, seed=42)

if SMOKE_TEST:
    MAX_LEN = DEBUG_MAX_LEN
    def limit_subjects(dx, col, n):
        subs = dx[col].dropna().unique().tolist()
        random.shuffle(subs); keep = set(subs[:min(n,len(subs))])
        return dx[dx[col].isin(keep)].copy()
    train_df = limit_subjects(train_df, SUBJECT_COL, DEBUG_TRAIN_SUBJ)
    val_df   = limit_subjects(val_df,   SUBJECT_COL, DEBUG_VAL_SUBJ)
    test_df  = limit_subjects(test_df,  SUBJECT_COL, DEBUG_TEST_SUBJ)
    logging.info(f"[SMOKE] visits -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Use the complete ICD-9 code set for label space
mlb = lock_label_space(df, ICD9_CODES_PICKLE)
y_train = y_multi_hot(mlb, train_df[LABEL_COL].tolist())
y_val   = y_multi_hot(mlb, val_df[LABEL_COL].tolist())
y_test  = y_multi_hot(mlb, test_df[LABEL_COL].tolist())
labels_vocab = mlb.classes_.tolist()

# ============== Run dir & save config ==============
RUN_DIR = make_run_dir(run_name=RUN_NAME)
print("Run dir:", RUN_DIR)
save_json(os.path.join(RUN_DIR, "config.json"), {
    "model": LLAMA_MODEL, "max_len": MAX_LEN, "lr": LR, "epochs": EPOCHS,
    "gen_max_new": GEN_MAX_NEW, "tgt_reserve_tok": TGT_RESERVE_TOK,
    "smoke_test": SMOKE_TEST, "early_stop": EARLY_STOP, "patience": PATIENCE,
    "eval_gen_subset": EVAL_GEN_SUBSET,
    "icd9_codes_file": ICD9_CODES_PICKLE,
    "total_label_space": len(labels_vocab)
})
save_json(os.path.join(RUN_DIR, "label_space.json"), {"labels": labels_vocab})

# ============== Model, tokenizer, datasets ==============
model, tok = load_lm_and_tokenizer()
train_ds = GenCodesDataset(train_df, tok, MAX_LEN, tgt_reserve=TGT_RESERVE_TOK)
val_ds   = GenCodesDataset(val_df,   tok, MAX_LEN, tgt_reserve=TGT_RESERVE_TOK)
test_prompts = test_df["input_text"].astype(str).tolist()

# W&B
report_to = ["wandb"] if secrets_info["wandb_ok"] and WANDB_ONLINE else "none"

# ============== Training args ==============
if SMOKE_TEST:
    args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=100, max_steps=MAX_STEPS,
        learning_rate=LR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.0, weight_decay=0.0,
        logging_strategy="steps", logging_steps=LOG_STEPS,
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        prediction_loss_only=True,
        save_strategy="no", load_best_model_at_end=False,
        report_to=report_to,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim="adamw_torch", dataloader_num_workers=2,
        run_name=os.path.basename(RUN_DIR),
        disable_tqdm=True,
    )
    callbacks = []
else:
    args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.03, weight_decay=0.0,
        logging_strategy="epoch",
        eval_strategy="epoch",
        prediction_loss_only=True,         # memory-safe
        save_strategy="epoch",
        load_best_model_at_end=EARLY_STOP,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim="adamw_torch", dataloader_num_workers=2,
        run_name=os.path.basename(RUN_DIR),
        disable_tqdm=True,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)] if EARLY_STOP else []

# Add per-epoch generation metrics callback (with timing)
callbacks.append(
    EvalGenCallback(
        model_ref=model,
        tok=tok,
        val_prompts=val_df["input_text"].astype(str).tolist(),
        y_val=y_val,
        label_vocab=labels_vocab,
        batch_size=EVAL_GEN_BS,
        max_items=EVAL_GEN_SUBSET,
        seed=42
    )
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,                          # ok (deprecation warning is harmless)
    data_collator=lambda feats: pad_collate(feats, tok),
    compute_metrics=None,                   # gen-F1 logged via callback
    callbacks=callbacks
)

# ============== Train (timed) ==============
t0 = time.perf_counter()
logging.info("Starting generative training…")
trainer.train()
log_seconds("train", t0)
logging.info("Training complete.")

# ============== Save artifacts ==============
tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
trainer.model.save_pretrained(os.path.join(RUN_DIR, "adapter_best"))

t_merge = time.perf_counter()
try:
    merged_dir = os.path.join(RUN_DIR, "model_merged")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(os.path.join(merged_dir, "tokenizer"))
    logging.info(f"Merged model saved to: {merged_dir}")
except Exception as e:
    logging.warning(f"Could not merge adapters into base: {e}")
log_seconds("merge_model", t_merge)

# ============== Final TEST evaluation (timed) ==============
t_gen = time.perf_counter()
pred_code_lists = generate_codes(
    model, tok, test_prompts, labels_vocab,
    max_new=GEN_MAX_NEW, batch_size=EVAL_GEN_BS, max_len=MAX_LEN
)
log_seconds("test_generate", t_gen)

t_metric = time.perf_counter()
Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
metrics = eval_sets(y_test, Y_pred)

# Add hierarchical metrics to final evaluation
hier_metrics = hierarchical_eval(y_test, Y_pred, labels_vocab)
metrics.update({k: v for k, v in hier_metrics.items()})

log_seconds("test_metrics", t_metric)

save_json(os.path.join(RUN_DIR, "test_metrics.json"), metrics)

print("\n=== Generative Test metrics ===")
print(json.dumps(metrics, indent=2))

print("\n--- A few test predictions ---")
show_test_predictions(test_df, pred_code_lists, n_show=5, seed=42, prompt_tail_lines=8)