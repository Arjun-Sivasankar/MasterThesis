"""
Complete training script for ICD-9 parent code prediction
All utility functions included - no external dependencies
"""

import os, sys, time, json, argparse, datetime, inspect, logging, pickle, re, atexit, ast
import torch, torch.distributed as dist
import pandas as pd
import numpy as np
from typing import List, Any, Dict, Optional
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

# ---------------- Environment setup ----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----- DDP helpers -----
def dist_is_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _env_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

def get_rank():
    if dist_is_initialized():
        return torch.distributed.get_rank()
    return _env_rank()

def is_main_process():
    return get_rank() == 0

def barrier():
    if dist_is_initialized():
        torch.distributed.barrier()

def rank0_print(*a, **k):
    if is_main_process():
        print(*a, **k)
        sys.stdout.flush()

# Quiet non-main logs
if not is_main_process():
    logging.getLogger().setLevel(logging.WARNING)

def _cleanup_dist():
    if dist_is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

atexit.register(_cleanup_dist)

# ---------------- ICD Handling ----------------
_SPLIT = re.compile(r"[,\s;]+")

def _split_codes(s: str) -> List[str]:
    return [x.strip() for x in _SPLIT.split(s.strip()) if x.strip()]

def _clean(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _is_container(x) -> bool:
    return hasattr(x, '__iter__') and not isinstance(x, (str, bytes))

def _is_na_scalar(x) -> bool:
    """True only for scalar NA; never treats arrays/containers as NA."""
    if _is_container(x):
        return False
    try:
        r = pd.isna(x)
        if not (np.isscalar(r) or isinstance(r, (bool, np.bool_))):
            return False
        if bool(r):
            return True
    except Exception:
        pass
    
    s = str(x).strip().lower()
    return s in ("", "nan", "none", "null")

def to_list(x) -> List[str]:
    """
    Robust parser for label/feature fields.
    """
    if _is_na_scalar(x) or x is None:
        return []
    if isinstance(x, np.ndarray):
        return [str(v).strip() for v in x.ravel().tolist() if str(v).strip()]
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, pd.Series):
        return [str(v).strip() for v in x.tolist() if str(v).strip()]
    if isinstance(x, dict):
        return [str(v).strip() for v in x.values() if str(v).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s: return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = ast.literal_eval(s)
                return to_list(obj)
            except Exception:
                pass
        return [t for t in _split_codes(s) if t]
    sx = str(x).strip()
    return [sx] if sx else []

# --- ICD regexes and formatting ---
def format_icd9_properly(code: str) -> str:
    """Format ICD-9 code properly with decimal placement"""
    code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
    if not code: return ""
    if code[0].isdigit():
        if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        return code
    if code[0] == "V":
        if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        return code
    if code[0] == "E":
        if len(code)>4 and "." not in code: return code[:4]+"."+code[4:]
        return code
    return code

def is_valid_icd9(code: str) -> bool:
    """Check if code is a valid ICD-9 format"""
    if not code: 
        return False
    
    if code[0].isdigit(): 
        return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    if code.startswith('V'): 
        return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    if code.startswith('E'): 
        return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    return False

def get_icd9_parent(code: str) -> str:
    """Get parent code (first 3 characters for most, 4 for E codes)"""
    if not code or len(code) < 3: 
        return code
    
    if code[0].isdigit(): 
        return code.split('.')[0][:3]
    if code.startswith('V'):
        base = code.split('.')[0]
        return base[:3]
    if code.startswith('E'):
        base = code.split('.')[0]
        return base[:4] if len(base) >= 4 else base
    return code

def format_icd_leaf(code: str, scheme: str) -> str:
    """Format a leaf-level ICD code"""
    if scheme == "icd9cm":
        formatted = format_icd9_properly(code)
        return formatted if is_valid_icd9(formatted) else ""
    elif scheme == "icd10cm":
        code = code.strip().upper()
        if _RX_ICD10.match(code):
            return code
    return ""

def is_valid_icd_leaf(code: str, scheme: str) -> bool:
    """Check if code is valid at leaf level"""
    if scheme == "icd9cm":
        formatted = format_icd9_properly(code)
        return is_valid_icd9(formatted)
    elif scheme == "icd10cm":
        return bool(_RX_ICD10.match(code.strip().upper()))
    return False

def parent_first3(code: str) -> str:
    """Get parent code - updated to use proper ICD-9 parent logic"""
    if not code:
        return code
    return get_icd9_parent(code)

def to_level(codes: List[str], level: str, scheme: str) -> List[str]:
    """Convert codes to specified level"""
    if level == "leaf":
        valid_codes = []
        for c in codes:
            formatted = format_icd_leaf(c, scheme)
            if formatted:
                valid_codes.append(formatted)
        return valid_codes
    elif level == "parent":
        valid_leaves = []
        for c in codes:
            formatted = format_icd_leaf(c, scheme)
            if formatted:
                valid_leaves.append(formatted)
        # Get unique parent codes
        parents = list(set(parent_first3(c) for c in valid_leaves))
        return [p for p in parents if p]  # Filter out empty strings
    return []

# ---------------- Text processing ----------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x: Any) -> str:
    if _is_na_scalar(x):
        return ""
    text = str(x).strip()
    text = re.sub(r'\s+', ' ', text)
    return text[:2000]

def serialize_structured(row: pd.Series) -> str:
    parts = []
    for col in TEXT_COLS_SAFE:
        if col in row:
            val = clean_text(row[col])
            if val:
                parts.append(f"{col}: {val}")
    return "\n".join(parts)

def serialize_notes(row: pd.Series) -> str:
    note_cols = [c for c in row.index if "note" in c.lower() or "text" in c.lower()]
    parts = []
    for col in note_cols[:3]:
        val = clean_text(row[col])
        if val:
            parts.append(val)
    return "\n".join(parts)

def build_input_text(row: pd.Series, use_structured: bool, use_notes: bool, 
                    subject_col: str, icd_level: str, icd_scheme: str) -> str:
    parts = [f"Patient ID: {row.get(subject_col, 'Unknown')}"]
    
    if use_structured:
        struct = serialize_structured(row)
        if struct:
            parts.append("Clinical Data:")
            parts.append(struct)
    
    if use_notes:
        notes = serialize_notes(row)
        if notes:
            parts.append("Clinical Notes:")
            parts.append(notes)
    
    level_desc = "parent categories" if icd_level == "parent" else "specific codes"
    scheme_desc = "ICD-9-CM" if icd_scheme == "icd9cm" else "ICD-10-CM"
    
    parts.append(f"\nPredict {scheme_desc} {level_desc} for this patient:")
    parts.append("[CODES]")
    
    return "\n".join(parts)

# ---------------- Data handling ----------------
def subject_splits(df: pd.DataFrame, subject_col: str = "subject_id_x", 
                  test_size: float = 0.10, val_size: float = 0.10, seed: int = 42) -> tuple:
    subjects = df[subject_col].unique()
    
    train_subj, temp_subj = train_test_split(subjects, test_size=(test_size + val_size), 
                                           random_state=seed, shuffle=True)
    
    if val_size > 0:
        val_subj, test_subj = train_test_split(temp_subj, test_size=(test_size / (test_size + val_size)), 
                                             random_state=seed, shuffle=True)
    else:
        val_subj = []
        test_subj = temp_subj
    
    train_df = df[df[subject_col].isin(train_subj)].copy()
    val_df = df[df[subject_col].isin(val_subj)].copy()
    test_df = df[df[subject_col].isin(test_subj)].copy()
    
    return train_df, val_df, test_df

def nested_subject_sample(df: pd.DataFrame, n_samples: int, subject_col: str = "subject_id_x", 
                         seed: int = 42) -> pd.DataFrame:
    if n_samples <= 0 or n_samples >= len(df):
        return df
    
    subjects = df[subject_col].unique()
    np.random.seed(seed)
    selected_subjects = np.random.choice(subjects, size=min(n_samples//2, len(subjects)), replace=False)
    subset = df[df[subject_col].isin(selected_subjects)]
    
    if len(subset) < n_samples:
        remaining = df[~df[subject_col].isin(selected_subjects)]
        additional = remaining.sample(n=min(n_samples - len(subset), len(remaining)), random_state=seed)
        subset = pd.concat([subset, additional])
    
    return subset.sample(n=min(n_samples, len(subset)), random_state=seed)

def collapse_codes_in_df(df: pd.DataFrame, label_col: str, icd_level: str, icd_scheme: str) -> List[List[str]]:
    result = []
    for _, row in df.iterrows():
        codes = to_list(row[label_col])
        level_codes = to_level(codes, icd_level, icd_scheme)
        result.append(level_codes)
    return result

def lock_label_space(dfs: List[pd.DataFrame], label_col: str, icd_level: str, icd_scheme: str,
                    codes_pkl_path: Optional[str] = None, use_complete: bool = False) -> MultiLabelBinarizer:
    all_codes = set()
    
    if use_complete and codes_pkl_path and os.path.exists(codes_pkl_path):
        try:
            with open(codes_pkl_path, 'rb') as f:
                complete_codes = pickle.load(f)
            
            # Handle different types of loaded data
            if isinstance(complete_codes, pd.DataFrame):
                rank0_print(f"[LabelSpace] Complete codes is DataFrame, skipping")
            elif complete_codes:  # Check if not empty
                level_codes = to_level(complete_codes, icd_level, icd_scheme)
                all_codes.update(level_codes)
                rank0_print(f"[LabelSpace] Using COMPLETE code set ({len(all_codes)} labels) from {codes_pkl_path} at level={icd_level}")
            else:
                rank0_print(f"[LabelSpace] Complete codes file is empty, falling back to observed codes")
        except Exception as e:
            rank0_print(f"[LabelSpace] Could not load complete codes: {e}")
    
    # Always collect from observed data
    for df in dfs:
        for _, row in df.iterrows():
            codes = to_list(row[label_col])
            level_codes = to_level(codes, icd_level, icd_scheme)
            all_codes.update(level_codes)
    
    if not all_codes:
        raise ValueError(f"No valid {icd_level} codes found in data or complete codes file")
    
    # Simpler condition check
    rank0_print(f"[LabelSpace] Final code set ({len(all_codes)} labels) at level={icd_level}")
    
    mlb = MultiLabelBinarizer()
    mlb.fit([sorted(all_codes)])
    return mlb

def y_multi_hot(y_lists: List[List[str]], mlb: MultiLabelBinarizer) -> np.ndarray:
    return mlb.transform(y_lists).astype(np.float32)

# ---------------- Model loading ----------------
def load_lm_and_tokenizer(model_name: str):
    rank0_print(f"Loading model and tokenizer...")
    rank0_print(f"Using model: {model_name}")
    
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    # Load model with proper device map for PEFT
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map=None,  # Important: Don't use device_map with PEFT
        low_cpu_mem_usage=True
    )
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply PEFT
    model = get_peft_model(model, lora_config)
    
    # CRITICAL: Enable input gradients for gradient checkpointing with PEFT
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        rank0_print("Enabled input gradients for PEFT model")
    
    # Set pad token
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False  # Required for training
    
    # Ensure PEFT parameters require gradients
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    rank0_print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")
    
    # Verify we have trainable parameters
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! LoRA setup failed.")
    
    return model, tok

# ---------------- Dataset ----------------
class GenCodesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, tgt_reserve_tok: int,
                 label_col: str, icd_level: str, icd_scheme: str,
                 use_structured: bool, use_notes: bool, subject_col: str):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tgt_reserve_tok = tgt_reserve_tok
        self.label_col = label_col
        self.icd_level = icd_level
        self.icd_scheme = icd_scheme
        self.use_structured = use_structured
        self.use_notes = use_notes
        self.subject_col = subject_col
        
        # Pre-build inputs and targets
        self._build_samples()
    
    def _build_samples(self):
        self.samples = []
        for idx, row in self.df.iterrows():
            # Build input text
            input_text = build_input_text(row, self.use_structured, self.use_notes, 
                                        self.subject_col, self.icd_level, self.icd_scheme)
            
            # Build target codes
            codes = to_list(row[self.label_col])
            level_codes = to_level(codes, self.icd_level, self.icd_scheme)
            target_text = " ".join(sorted(level_codes))
            
            # Combine for training
            full_text = input_text + " " + target_text + self.tokenizer.eos_token
            
            self.samples.append(full_text)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]

        # print(f"Sample {idx} text length: {len(text)} characters")
        # print(f"Sample {idx} text preview: {text}...")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create labels (copy of input_ids but with -100 for input portion)
        labels = input_ids.clone()
        
        # Find where [CODES] appears to mask input portion
        codes_token = "[CODES]"
        try:
            codes_tokens = self.tokenizer.encode(codes_token, add_special_tokens=False)
            if codes_tokens:
                # Find the position and mask everything before it
                for i in range(len(input_ids) - len(codes_tokens) + 1):
                    if input_ids[i:i+len(codes_tokens)].tolist() == codes_tokens:
                        labels[:i+len(codes_tokens)] = -100
                        break
        except:
            # Fallback: mask first 80% of tokens
            mask_len = int(0.8 * len(input_ids))
            labels[:mask_len] = -100
        
        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "labels": labels.long()
        }

def pad_collate(features, tokenizer):
    # Extract tensors
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ---------------- Generation and evaluation ----------------
@torch.no_grad()
def generate_codes(model, tokenizer, prompts: List[str], labels_vocab: List[str],
                  icd_level: str, icd_scheme: str, max_new: int = 96, 
                  batch_size: int = 4, max_len: int = 3072) -> List[List[str]]:
    model.eval()
    device = next(model.parameters()).device
    allowed = set(labels_vocab)
    preds = []
    
    total = len(prompts)
    for i in range(0, total, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize batch
        encoding = tokenizer(
            batch_prompts,
            max_length=max_len - max_new,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model.generate(
                **encoding,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode predictions
        for j, output in enumerate(outputs):
            # Extract only the new tokens
            input_len = len(encoding["input_ids"][j])
            new_tokens = output[input_len:]
            
            # Decode and parse
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred_codes = pred_text.strip().split()
            
            # Filter to valid codes
            valid_codes = []
            for code in pred_codes:
                clean_code = code.strip()
                if clean_code in allowed:
                    valid_codes.append(clean_code)
            
            preds.append(valid_codes)
        
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= total:
            progress = min(i + batch_size, total)
            rank0_print(f"Generated {progress}/{total} ({100*progress/total:.1f}%)")
    
    return preds

def codes_to_multihot(code_lists: List[List[str]], vocab: List[str]) -> np.ndarray:
    mlb = MultiLabelBinarizer(classes=vocab)
    return mlb.fit_transform(code_lists).astype(np.float32)

def eval_sets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
    }

# ---------------- Callbacks ----------------
class DetailedEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, label_vocab, icd_level, icd_scheme,
                 eval_sample_size=100, seed=42, gen_max_new=96):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.icd_level = icd_level
        self.icd_scheme = icd_scheme
        self.eval_sample_size = min(eval_sample_size, len(eval_dataset))
        self.best_micro_f1 = 0
        self.epoch_metrics = []
        self.rng = np.random.RandomState(seed)
        self.subset_indices = self.rng.choice(len(self.eval_dataset), self.eval_sample_size, replace=False)
        self._gen_max_new = gen_max_new

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not is_main_process(): return
        ep = getattr(state, 'epoch', 0)
        rank0_print(f"=== Starting epoch {ep + 1:.1f} ===")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not is_main_process(): return
        if 'loss' in logs and 'eval_loss' not in logs:
            rank0_print(f"[Train] epoch: {logs.get('epoch', 0):.1f}, loss: {logs.get('loss', 0):.4f}, "
                  f"lr: {logs.get('learning_rate', 0):.2e}, grad_norm: {logs.get('grad_norm', 0):.2f}")
        elif 'eval_loss' in logs:
            rank0_print(f"[Eval] epoch: {logs.get('epoch', 0):.1f}, loss: {logs.get('eval_loss', 0):.4f}, "
                  f"runtime: {logs.get('eval_runtime', 0):.1f}s")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        barrier()
        if not is_main_process():
            barrier()
            return

        model = kwargs.get("model")
        if not model:
            barrier()
            return

        ep = getattr(state, "epoch", 0)
        rank0_print(f"[GenEval] Starting generation evaluation for epoch {ep:.1f}...")

        # Build subset prompts and gold labels
        subset_prompts, gold_codes_lists = [], []
        for idx in self.subset_indices:
            item = self.eval_dataset[idx]
            prompt_text = self.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            parts = prompt_text.split("[CODES]")
            prompt = parts[0] + "[CODES]" if len(parts) > 1 else prompt_text
            subset_prompts.append(prompt)
            
            target_ids = item["labels"].tolist()
            tgt_only = [tid for tid in target_ids if tid != -100]
            gold_text = self.tokenizer.decode(tgt_only, skip_special_tokens=True)
            gold_codes_lists.append(gold_text.split())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        preds = generate_codes(model, self.tokenizer, subset_prompts, self.label_vocab,
                               self.icd_level, self.icd_scheme,
                               max_new=self._gen_max_new, batch_size=4)
        
        y_true = codes_to_multihot(gold_codes_lists, self.label_vocab)
        y_pred = codes_to_multihot(preds, self.label_vocab)
        eval_metrics = eval_sets(y_true, y_pred)

        if eval_metrics["micro_f1"] > self.best_micro_f1:
            self.best_micro_f1 = eval_metrics["micro_f1"]
            rank0_print(f"[GenEval] New best micro_f1: {self.best_micro_f1:.4f}")

        self.epoch_metrics.append({
            "epoch": ep,
            "eval_loss": metrics.get("eval_loss", 0) if metrics else 0,
            "micro_f1": eval_metrics["micro_f1"],
            "macro_f1": eval_metrics["macro_f1"],
            "samples_f1": eval_metrics["samples_f1"],
        })

        rank0_print(f"[GenEval] epoch: {ep:.1f} | micro_f1={eval_metrics['micro_f1']:.4f} "
                    f"| macro_f1={eval_metrics['macro_f1']:.4f} | samples_f1={eval_metrics['samples_f1']:.4f}")

        barrier()

    def on_epoch_end(self, args, state, control, **kwargs):
        if not is_main_process(): return
        ep = getattr(state, "epoch", 0)
        rank0_print(f"=== Completed epoch {ep:.1f} ===")

# ---------------- Utilities ----------------
def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def unwrap_model(model):
    """Extract the base model from PEFT wrapper"""
    if hasattr(model, 'module'):
        return model.module
    return model

# ---------------- Main functions ----------------
def get_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", default=None, help="Full dataset pickle for subject splitting")
    ap.add_argument("--train_pickle", default=None)
    ap.add_argument("--val_pickle", default=None)
    ap.add_argument("--codes_pickle", default=None, help="Complete code list pickle")
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--use_complete_codes", type=int, default=0)

    # model/prompt
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--tgt_reserve_tok", type=int, default=128)
    ap.add_argument("--gen_max_new", type=int, default=96)

    # ICD options - fixed for parent level
    ap.add_argument("--icd_scheme", choices=["icd9cm", "icd10cm"], default="icd9cm")
    ap.add_argument("--icd_level", default="parent", help="Fixed to parent level")

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

    # size & seed
    ap.add_argument("--train_size", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)

    # run dirs
    ap.add_argument("--run_root", default="runs_gen/icd9_parent")
    ap.add_argument("--run_name", default=None)

    # eval
    ap.add_argument("--eval_sample_size", type=int, default=100)
    
    # misc
    ap.add_argument("--compile", type=int, default=0)
    return ap.parse_args()

def make_training_args(args, run_dir):
    TA = TrainingArguments
    sig = inspect.signature(TA.__init__).parameters
    
    kwargs = dict(
        output_dir=os.path.join(run_dir, "checkpoints"),
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
        bf16=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
        run_name=os.path.basename(run_dir),
        disable_tqdm=True,
        local_rank=-1,
    )
    
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"
        
    return TA(**kwargs)

def main():
    args = get_args()
    set_seed(args.seed)
    
    rank0_print(f"Starting ICD-{args.icd_scheme.upper()} PARENT training...")
    rank0_print(f"ICD setup: scheme={args.icd_scheme} level={args.icd_level}")

    # Load data
    if args.train_pickle and args.val_pickle:
        train_df = pickle.load(open(args.train_pickle, "rb"))
        val_df = pickle.load(open(args.val_pickle, "rb"))
    elif args.data_pickle:
        full_df = pickle.load(open(args.data_pickle, "rb"))
        train_df, val_df, _ = subject_splits(full_df, subject_col=args.subject_col, 
                                           test_size=0.10, val_size=0.10, seed=args.seed)
    else:
        raise ValueError("Provide either --data_pickle OR both --train_pickle/--val_pickle")

    # Subject-safe subsetting
    train_df = nested_subject_sample(train_df, args.train_size, 
                                   subject_col=args.subject_col, seed=args.seed)

    for name, df_ in (("train", train_df), ("val", val_df)):
        rank0_print(f"[{name}] rows: {len(df_)}")

    # Lock label space for parent codes
    mlb = lock_label_space([train_df, val_df], args.label_col,
                          args.icd_level, args.icd_scheme,
                          codes_pkl_path=args.codes_pickle, 
                          use_complete=bool(args.use_complete_codes))
    labels_vocab = mlb.classes_.tolist()

    # Load model and tokenizer
    model, tok = load_lm_and_tokenizer(args.llama_model)
    if args.compile:
        try: 
            model = torch.compile(model)
        except Exception as e: 
            log.warning(f"torch.compile failed: {e}")

    # Create datasets
    train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok,
                               args.label_col, args.icd_level, args.icd_scheme,
                               bool(args.use_structured), bool(args.use_notes), 
                               args.subject_col)
    val_ds = GenCodesDataset(val_df, tok, args.max_len, args.tgt_reserve_tok,
                             args.label_col, args.icd_level, args.icd_scheme,
                             bool(args.use_structured), bool(args.use_notes), 
                             args.subject_col)

    # Setup run directory
    size_str = f"N{args.train_size}" if args.train_size > 0 else "full"
    tag = args.run_name or f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{size_str}_{args.icd_scheme}_{args.icd_level}"
    run_dir = os.path.join(args.run_root, tag)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    rank0_print(f"Run dir: {run_dir}")

    # Save configuration
    if is_main_process():
        save_json(os.path.join(run_dir, "config.json"), {
            "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate,
            "epochs": args.epochs, "gen_max_new": args.gen_max_new, 
            "tgt_reserve_tok": args.tgt_reserve_tok,
            "seed": args.seed, "train_rows": len(train_df),
            "codes_pickle": args.codes_pickle, 
            "use_complete_codes": bool(args.use_complete_codes),
            "total_label_space": len(labels_vocab),
            "icd_scheme": args.icd_scheme, "icd_level": args.icd_level
        })
        save_json(os.path.join(run_dir, "label_space.json"), {"labels": labels_vocab})

    # Setup training
    train_args = make_training_args(args, run_dir)
    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    callbacks.append(DetailedEvalCallback(
        eval_dataset=val_ds, tokenizer=tok, label_vocab=labels_vocab,
        icd_level=args.icd_level, icd_scheme=args.icd_scheme,
        eval_sample_size=args.eval_sample_size, seed=args.seed, 
        gen_max_new=args.gen_max_new
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
    rank0_print("Starting training...")
    t0 = time.perf_counter()
    train_start = datetime.datetime.now()
    trainer.train()
    train_secs = time.perf_counter() - t0
    train_duration = datetime.datetime.now() - train_start
    rank0_print(f"[TIME] Training completed in {train_secs:.2f}s ({train_duration})")

    # Save model
    if is_main_process():
        rank0_print("Saving model and tokenizer...")
        tok.save_pretrained(os.path.join(run_dir, "tokenizer"))
        unwrap_model(trainer.model).save_pretrained(os.path.join(run_dir, "adapter_best"))
        save_json(os.path.join(run_dir, "train_summary.json"), {
            "train_seconds": train_secs,
            "train_duration_str": str(train_duration)
        })
        rank0_print("Model and tokenizer saved.")
        rank0_print(f"To test this model, use:")
        rank0_print(f"python gen/test_parents.py --adapter_dir {run_dir}/adapter_best --labels_json {run_dir}/label_space.json")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())