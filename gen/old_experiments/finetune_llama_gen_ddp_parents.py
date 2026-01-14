# -*- coding: utf-8 -*-
"""
Generative ICD code prediction (ICD-9/ICD-10) with LoRA
(DDP-safe, parent/leaf toggle, epoch timing, optimized generation)
"""

import os, re, json, random, logging, pickle, datetime, time, atexit, argparse, inspect, ast
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

# ---------------- CUDA safety helpers ----------------
def set_device_safely():
    """
    Select a valid CUDA device for this process based on LOCAL_RANK.
    Returns (has_cuda: bool, current_device_idx: int or -1).
    """
    if not torch.cuda.is_available():
        return False, -1
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        local_rank = 0
    n = torch.cuda.device_count()
    if n == 0:
        return False, -1
    dev = local_rank % n  # clamp/wrap so we never go out of range
    torch.cuda.set_device(dev)
    return True, dev

def gpu_major_capability():
    try:
        return torch.cuda.get_device_capability(torch.cuda.current_device())[0]
    except Exception:
        return 0

def is_ampere_plus():
    return torch.cuda.is_available() and gpu_major_capability() >= 8

def enable_tf32_if_available():
    """Enable TF32 on Ampere+ for the *current* device, if safe."""
    if not torch.cuda.is_available():
        return
    try:
        if gpu_major_capability() >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass

# ---------------- Args ----------------
def get_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", default=None, help="If provided, will subject-split into train/val/test.")
    ap.add_argument("--train_pickle", default=None)
    ap.add_argument("--val_pickle", default=None)
    ap.add_argument("--test_pickle", default=None)
    ap.add_argument("--codes_pickle", default=None, help="Optional complete code list (ICD-9 or ICD-10)")
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

    # ICD options
    ap.add_argument("--icd_scheme", choices=["icd9cm", "icd10cm"], default="icd10cm")
    ap.add_argument("--icd_level", choices=["parent", "leaf"], default="parent",
                    help="parent: first 3 of pre-decimal head; leaf: full specific codes")

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

# ---------------- ICD Handling (ICD-9/10 + parent/leaf) ----------------
_SPLIT = re.compile(r"[,\s;]+")

def _split_codes(s: str) -> List[str]:
    return [t for t in _SPLIT.split(s) if t]

def _clean(x: Any) -> str:
    if x is None: return ""
    s = str(x).upper().strip()
    s = re.sub(r"[^\w\.\-]", "", s)
    if s.endswith("."): s = s[:-1]
    return s

def _is_container(x) -> bool:
    return isinstance(x, (list, tuple, set, np.ndarray, pd.Series, dict))

def _is_na_scalar(x) -> bool:
    """True only for scalar NA; never treats arrays/containers as NA."""
    if _is_container(x):
        return False
    try:
        r = pd.isna(x)
        return bool(r) if np.isscalar(r) or isinstance(r, (bool, np.bool_)) else False
    except Exception:
        return False

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

# --- ICD regexes ---
_RX_ICD9_NUM  = re.compile(r"^(?:\d{3})(?:\.\d{1,2})?$")
_RX_ICD9_V    = re.compile(r"^V\d{2}(?:\.\d{1,2})?$")
_RX_ICD9_E    = re.compile(r"^E\d{3}(?:\.\d{1})?$")
_RX_ICD10     = re.compile(r"^[A-Z][0-9A-Z][0-9A-Z](?:\.[0-9A-Z]{1,4})?$")

def format_icd_leaf(code: str, scheme: str) -> str:
    c = _clean(code)
    if not c: return c
    if scheme == "icd9cm":
        if c[0].isdigit() and "." not in c and len(c) > 3:
            return c[:3] + "." + c[3:]
        if c[0] in ("V","E") and "." not in c and len(c) > 3:
            return c[:3] + "." + c[3:]
        return c
    else:
        return c

def is_valid_icd_leaf(code: str, scheme: str) -> bool:
    c = _clean(code)
    if not c: return False
    if scheme == "icd9cm":
        return bool(_RX_ICD9_NUM.match(c) or _RX_ICD9_V.match(c) or _RX_ICD9_E.match(c))
    else:
        return bool(_RX_ICD10.match(c))

def parent_first3(code: str) -> str:
    c = _clean(code)
    if not c: return ""
    head = c.split(".", 1)[0]
    head = re.sub(r"[^A-Z0-9]", "", head)
    return head[:3] if len(head) >= 3 else head

def to_level(codes: List[str], level: str, scheme: str) -> List[str]:
    seen, out = set(), []
    for code in codes or []:
        if not code: continue
        if level == "leaf":
            c = format_icd_leaf(code, scheme)
            if is_valid_icd_leaf(c, scheme):
                if c not in seen:
                    seen.add(c); out.append(c)
        else:
            p = parent_first3(code)
            if p and p not in seen:
                seen.add(p); out.append(p)
    return out

# ---------------- Prompting helpers ----------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x: Any) -> str:
    if _is_na_scalar(x): return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()

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
                     subject_col="subject_id_x", icd_level="parent", icd_scheme="icd10cm") -> str:
    s = [f"[VISIT] subject_id={row.get(subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
    if use_structured: s.append(serialize_structured(row))
    if use_notes:
        t = serialize_notes(row)
        if t: s.append(t)
    if icd_level == "parent":
        s.append(f"[TASK] You are a medical coding expert. Based on the information above, "
                 f"generate the appropriate *parent* ICD codes for {icd_scheme.upper()}. Guidelines:")
        s.append("1. Output ONLY parent codes (3-character categories, pre-decimal), separated by spaces, e.g., E11 I10 J45")
        s.append("2. Include only codes directly supported by the clinical information")
        s.append("3. No explanations or extra text besides the codes")
    else:
        ex = "E11.9 I10" if icd_scheme == "icd10cm" else "250.00 401.9"
        s.append(f"[TASK] You are a medical coding expert. Generate specific (leaf) ICD codes for {icd_scheme.upper()}. Guidelines:")
        s.append(f"1. Output ONLY codes separated by spaces, e.g., {ex}")
        s.append("2. Use proper code formatting (decimals if applicable)")
        s.append("3. No explanations or extra text besides the codes")
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

def _extract_codes_from_pickle(pkl_path: str) -> List[str]:
    obj = pd.read_pickle(pkl_path)
    if isinstance(obj, pd.Series):
        return [str(v).strip() for v in obj.tolist() if str(v).strip()]
    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            vals = [str(v).strip() for v in obj[col].tolist() if str(v).strip()]
            if vals: return vals
        return []
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        arr = obj.tolist() if isinstance(obj, np.ndarray) else obj
        return [str(v).strip() for v in arr if str(v).strip()]
    s = str(obj).strip()
    return [s] if s else []

def collapse_codes_in_df(df: pd.DataFrame, label_col: str, icd_level: str, icd_scheme: str) -> List[List[str]]:
    collapsed = []
    for codes in df[label_col].values:
        raw_list = to_list(codes)
        collapsed.append(to_level(raw_list, icd_level, icd_scheme))
    return collapsed

def lock_label_space(frames: List[pd.DataFrame], label_col: str,
                     icd_level: str, icd_scheme: str,
                     codes_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
    if use_complete and codes_pkl_path:
        try:
            all_codes = _extract_codes_from_pickle(codes_pkl_path)
            collapsed_complete = to_level(all_codes, icd_level, icd_scheme)
            classes = sorted(set(collapsed_complete))
            mlb = MultiLabelBinarizer(classes=classes)
            mlb.fit([classes])
            log.info(f"[LabelSpace] Using COMPLETE code set ({len(classes)} labels) from {codes_pkl_path} at level={icd_level}")
            return mlb
        except Exception as e:
            log.warning(f"[LabelSpace] Could not load complete code list: {e}. Falling back to training data.")

    label_set = set()
    for fr in frames:
        for codes in collapse_codes_in_df(fr, label_col, icd_level, icd_scheme):
            label_set.update(codes)
    classes = sorted(label_set)
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    log.info(f"[LabelSpace] Using {len(classes)} labels from data at level={icd_level}")
    return mlb

def y_multi_hot(mlb: MultiLabelBinarizer, lists: List[List[str]]) -> np.ndarray:
    return mlb.transform(lists)

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

# ---------------- Dataset ----------------
class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int,
                 raw_label_col: str, icd_level: str, icd_scheme: str, use_structured: bool, use_notes: bool, subject_col: str):
        self.tok = tok
        self.max_len = max_len
        self.tgt_reserve = max(8, int(tgt_reserve))
        self.raw_label_col = raw_label_col
        self.icd_level = icd_level
        self.icd_scheme = icd_scheme
        self.use_structured = use_structured
        self.use_notes = use_notes
        self.subject_col = subject_col
        self.rows = rows

        prompts = rows.apply(lambda r: build_input_text(
            r, use_structured, use_notes, subject_col, icd_level, icd_scheme
        ), axis=1).astype(str).tolist()

        targets = []
        for codes in rows[raw_label_col].tolist():
            collapsed = to_level(to_list(codes), icd_level, icd_scheme)
            targets.append(" ".join(sorted(set(collapsed))))

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

# ---- Hand-rolled collator ----
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
        try:
            use_bf16 = (gpu_major_capability() >= 8)
        except Exception:
            use_bf16 = False
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    print('-'*80)
    print("Loading model and tokenizer...")
    print(f"Using model: {model_name} with dtype={dtype}")

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
def _parts_from_text(t: str) -> List[str]:
    return [p for p in re.split(r"[^\w\.]+", t.upper()) if p]

@torch.no_grad()
def generate_codes(model, tok, prompts: List[str], label_vocab: List[str],
                   icd_level="parent", icd_scheme="icd10cm",
                   max_new=96, batch_size=16, max_len=3072):
    model = unwrap_model(model)
    model.eval()
    device = next(model.parameters()).device
    allowed = set(label_vocab)
    preds = []

    if is_main_process():
        rank0_print(f"Generating predictions for {len(prompts)} samples (batch={batch_size})...")

    t0 = time.time(); last = t0
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.amp.autocast('cuda', enabled=is_ampere_plus()):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False, num_beams=1,
                no_repeat_ngram_size=2,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True, output_scores=False,
            )

        seq = out.sequences
        gen_only = seq[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)

        for t in texts:
            toks = _parts_from_text(t)
            if icd_level == "parent":
                mapped = [parent_first3(x) for x in toks]
            else:
                mapped = [format_icd_leaf(x, icd_scheme) for x in toks if is_valid_icd_leaf(x, icd_scheme)]
            seen, keep = set(), []
            for c in mapped:
                if c and (c in allowed) and c not in seen:
                    seen.add(c); keep.append(c)
            preds.append(keep)

        if is_main_process() and ((i + batch_size) % (10 * batch_size) == 0 or (i + batch_size) >= len(prompts)):
            now = time.time()
            done = min(i + batch_size, len(prompts))
            pct = done / len(prompts)
            eta = (now - t0) / pct - (now - t0) if pct > 0 else 0
            rank0_print(f"Generated {done}/{len(prompts)} ({pct:.1%}) | last batch {now-last:.2f}s | ETA {eta:.1f}s")
            last = now

        if torch.cuda.is_available() and (i // batch_size) % 20 == 0:
            torch.cuda.empty_cache()

    if is_main_process():
        elapsed = time.time() - t0
        rank0_print(f"Generation complete in {elapsed:.1f}s ({elapsed/max(1,len(prompts)):.3f}s/sample)")
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

# ---------------- Custom training callbacks ----------------
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
        self.epoch_times = {}
        self._epoch_t0 = None
        self.rng = np.random.RandomState(seed)
        self.subset_indices = self.rng.choice(len(self.eval_dataset), self.eval_sample_size, replace=False)
        self.epoch_start_time = None
        self.current_epoch = 0
        self._gen_max_new = gen_max_new

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not is_main_process(): return
        self._epoch_t0 = time.time()
        self.epoch_start_time = datetime.datetime.now()
        self.current_epoch = getattr(state, 'epoch', 0)
        rank0_print(f"=== Starting epoch {self.current_epoch + 1:.1f} ===")

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
        # ---- All ranks hit a barrier BEFORE gen-eval ----
        barrier()

        # Only rank-0 performs generation; others wait at the post barrier and return
        if not is_main_process():
            barrier()
            return

        model = kwargs.get("model")
        if not model:
            barrier()
            return

        ep = getattr(state, "epoch", 0)
        rank0_print(f"[GenEval] Starting generation evaluation for epoch {ep:.1f}...")

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
            gparts = _parts_from_text(gold_text)
            if self.icd_level == "parent":
                gold_codes_lists.append(sorted(set(parent_first3(x) for x in gparts if parent_first3(x))))
            else:
                gl = []
                for x in gparts:
                    x2 = format_icd_leaf(x, self.icd_scheme)
                    if is_valid_icd_leaf(x2, self.icd_scheme): gl.append(x2)
                gold_codes_lists.append(sorted(set(gl)))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        preds = generate_codes(model, self.tokenizer, subset_prompts, self.label_vocab,
                               icd_level=self.icd_level, icd_scheme=self.icd_scheme,
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

        now = time.time()
        if self._epoch_t0 is not None:
            self.epoch_times[ep] = self.epoch_times.get(ep, {})
            self.epoch_times[ep]["gen_eval_seconds"] = now - self._epoch_t0

        rank0_print(f"[GenEval] epoch: {ep:.1f} | micro_f1={eval_metrics['micro_f1']:.4f} "
                    f"| macro_f1={eval_metrics['macro_f1']:.4f} | samples_f1={eval_metrics['samples_f1']:.4f}")

        # ---- All ranks hit a barrier AFTER gen-eval ----
        barrier()

    def on_epoch_end(self, args, state, control, **kwargs):
        if not is_main_process(): return
        if self._epoch_t0 is None: return
        total = time.time() - self._epoch_t0
        ep = getattr(state, "epoch", 0)
        gen_eval = self.epoch_times.get(ep, {}).get("gen_eval_seconds", total * 0.25)
        train_approx = max(0.0, total - gen_eval)
        self.epoch_times[ep] = {
            "epoch_total_seconds": total,
            "train_seconds_approx": train_approx,
            "gen_eval_seconds": gen_eval
        }
        rank0_print(f"[Time] epoch: {ep:.1f}, train≈{train_approx:.1f}s, gen-eval≈{gen_eval:.1f}s, total={total:.1f}s")
        rank0_print(f"=== Completed epoch {ep:.1f} ===")

    def on_train_end(self, args, state, control, **kwargs):
        if not is_main_process(): return
        rank0_print(f"\n===== Training Summary =====")
        rank0_print(f"Best validation micro F1: {self.best_micro_f1:.4f}")
        for m in self.epoch_metrics:
            ep = m['epoch']
            tinfo = self.epoch_times.get(ep, {})
            rank0_print(f"  Epoch {ep:.1f}: eval_loss={m['eval_loss']:.4f}, micro_f1={m['micro_f1']:.4f}, "
                        f"macro_f1={m['macro_f1']:.4f}, samples_f1={m['samples_f1']:.4f}")
            rank0_print(f"          Time: train≈{tinfo.get('train_seconds_approx',0):.1f}s, "
                        f"gen-eval≈{tinfo.get('gen_eval_seconds',0):.1f}s, "
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
                          icd_level: str, icd_scheme: str, n_show: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
    idx_map = {c:i for i,c in enumerate(label_vocab)}

    rank0_print(f"Showing {min(n_show,len(df))} random test examples (level={icd_level}):")
    for i in idxs:
        gold = collapse_codes_in_df(df.iloc[[i]], label_col, icd_level, icd_scheme)[0]
        pred = preds[i]
        missing = sorted([c for c in gold if c not in pred])
        extra   = sorted([c for c in pred if c not in gold])

        y_true = np.zeros(len(label_vocab))
        y_pred = np.zeros(len(label_vocab))
        for code in gold:
            j = idx_map.get(code);  y_true[j] = 1 if j is not None else 0
        for code in pred:
            j = idx_map.get(code);  y_pred[j] = 1 if j is not None else 0

        precision = precision_score([y_true], [y_pred], average='micro', zero_division=0)
        recall    = recall_score([y_true], [y_pred], average='micro', zero_division=0)
        f1        = f1_score([y_true], [y_pred], average='micro', zero_division=0)

        rank0_print("\n" + "="*80)
        rank0_print(f"Example {i}:")
        rank0_print(f"- METRICS: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        rank0_print("- GOLD:", " ".join(sorted(gold)) if gold else "(none)")
        rank0_print("- PRED:", " ".join(sorted(pred)) if pred else "(none)")
        rank0_print(f"- FALSE NEGATIVES ({len(missing)}):", " ".join(missing) if missing else "(none)")
        rank0_print(f"- FALSE POSITIVES ({len(extra)}):", " ".join(extra) if extra else "(none)")

# ---- TrainingArguments builder ----
def make_training_args(args, RUN_DIR):
    TA = TrainingArguments
    sig = inspect.signature(TA.__init__).parameters

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
        fp16=(torch.cuda.is_available() and not is_ampere_plus()),
        bf16=is_ampere_plus(),
        optim="adamw_torch",
        dataloader_num_workers=2,
        run_name=os.path.basename(RUN_DIR),
        disable_tqdm=True,
    )
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"

    if "ddp_backend" in sig: kwargs["ddp_backend"] = "nccl"
    if "ddp_find_unused_parameters" in sig: kwargs["ddp_find_unused_parameters"] = bool(args.ddp_find_unused)
    if "ddp_timeout" in sig: kwargs["ddp_timeout"] = 28800  # 8h

    return TA(**kwargs)

# ---------------- Main ----------------
def main():
    args = get_args()
    set_seed(args.seed)

    # Set device BEFORE any cuda queries
    has_cuda, dev = set_device_safely()
    enable_tf32_if_available()

    rank0_print(
        "CUDA:", torch.cuda.is_available(),
        "| Device:", (torch.cuda.get_device_name(dev) if has_cuda else "CPU"),
        "| Visible GPUs:", (torch.cuda.device_count() if has_cuda else 0),
        "| LOCAL_RANK:", os.environ.get("LOCAL_RANK", "0")
    )
    rank0_print(f"ICD setup: scheme={args.icd_scheme} level={args.icd_level}")

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

    for name, df_ in (("train", train_df), ("val", val_df), ("test", test_df)):
        rank0_print(f"[{name}] rows: {len(df_)}")

    # Label space at requested level
    mlb = lock_label_space([train_df, val_df, test_df], args.label_col,
                           args.icd_level, args.icd_scheme,
                           codes_pkl_path=args.codes_pickle, use_complete=bool(args.use_complete_codes))
    labels_vocab = mlb.classes_.tolist()
    y_val  = y_multi_hot(mlb, collapse_codes_in_df(val_df,  args.label_col, args.icd_level, args.icd_scheme))
    y_test = y_multi_hot(mlb, collapse_codes_in_df(test_df, args.label_col, args.icd_level, args.icd_scheme))

    # Model & tokenizer
    model, tok = load_lm_and_tokenizer(args.llama_model)
    if args.compile:
        try: model = torch.compile(model)
        except Exception as e: log.warning(f"torch.compile failed: {e}")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Datasets
    train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok,
                               args.label_col, args.icd_level, args.icd_scheme,
                               bool(args.use_structured), bool(args.use_notes), args.subject_col)
    val_ds   = GenCodesDataset(val_df,   tok, args.max_len, args.tgt_reserve_tok,
                               args.label_col, args.icd_level, args.icd_scheme,
                               bool(args.use_structured), bool(args.use_notes), args.subject_col)

    # Run dir
    size_str = f"N{args.train_size}" if args.train_size > 0 else "full"
    tag = args.run_name or f"{now_tag()}_{size_str}_{args.icd_scheme}_{args.icd_level}"
    RUN_DIR = make_run_dir(args.run_root, tag)
    rank0_print(f"Run dir: {RUN_DIR}")

    if is_main_process():
        save_json(os.path.join(RUN_DIR, "config.json"), {
            "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate,
            "epochs": args.epochs, "gen_max_new": args.gen_max_new, "tgt_reserve_tok": args.tgt_reserve_tok,
            "seed": args.seed, "train_rows": len(train_df),
            "codes_pickle": args.codes_pickle, "use_complete_codes": bool(args.use_complete_codes),
            "total_label_space": len(labels_vocab),
            "test_batch_size": args.test_batch_size,
            "icd_scheme": args.icd_scheme, "icd_level": args.icd_level
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
        icd_level=args.icd_level, icd_scheme=args.icd_scheme,
        eval_sample_size=args.eval_sample_size, seed=args.seed, gen_max_new=args.gen_max_new
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

    # ---- DEBUG: Print rank info before test ----
    print(f"DEBUG: get_rank()={get_rank()}, is_main_process()={is_main_process()}, RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")

    # Final TEST generation (rank 0 only)
    # Use explicit rank check as fallback
    run_test = is_main_process() or os.environ.get("RANK", "0") == "0"
    if run_test:
        rank0_print(f"\n=== Starting TEST generation for {len(test_df)} samples ===")
        test_prompts = test_df.apply(lambda r: build_input_text(
            r, bool(args.use_structured), bool(args.use_notes), args.subject_col, args.icd_level, args.icd_scheme
        ), axis=1).astype(str).tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rank0_print(f"Using test batch size: {args.test_batch_size}")

        test_start = datetime.datetime.now()
        t_gen = time.perf_counter()

        pred_code_lists = generate_codes(
            trainer.model, tok, test_prompts, labels_vocab,
            icd_level=args.icd_level, icd_scheme=args.icd_scheme,
            max_new=args.gen_max_new, batch_size=args.test_batch_size, max_len=args.max_len
        )
        test_gen_secs = time.perf_counter() - t_gen
        test_duration = datetime.datetime.now() - test_start

        rank0_print(f"=== Test Generation Summary ===")
        rank0_print(f"Total samples processed: {len(test_df)}")
        rank0_print(f"Batch size used: {args.test_batch_size}")
        rank0_print(f"Generation time: {test_gen_secs:.1f}s ({test_duration})")
        rank0_print(f"Average time per sample: {test_gen_secs/max(1,len(test_df)):.3f}s")
        rank0_print(f"Samples per second: {len(test_df)/max(test_gen_secs,1e-9):.2f}")

        gold_lists = collapse_codes_in_df(test_df, args.label_col, args.icd_level, args.icd_scheme)
        Y_true = y_multi_hot(mlb, gold_lists)
        Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
        metrics = eval_sets(Y_true, Y_pred)
        metrics.update({
            "train_seconds": train_secs,
            "train_duration_str": str(train_duration),
            "test_generate_seconds": test_gen_secs,
            "test_duration_str": str(test_duration),
            "test_samples": len(test_df),
            "test_batch_size": args.test_batch_size,
            "samples_per_second": len(test_df)/max(test_gen_secs,1e-9),
            "icd_scheme": args.icd_scheme,
            "icd_level": args.icd_level,
            "label_space_size": len(labels_vocab)
        })

        with open(os.path.join(RUN_DIR, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        rank0_print(f"\n=== Generative TEST metrics (level={args.icd_level}) ===")
        rank0_print(f"  - micro_f1: {metrics['micro_f1']:.4f}")
        rank0_print(f"  - macro_f1: {metrics['macro_f1']:.4f}")
        rank0_print(f"  - samples_f1: {metrics['samples_f1']:.4f}")

        rank0_print("\n=== Sample Predictions ===")
        show_test_predictions(test_df, pred_code_lists, args.label_col, labels_vocab,
                              icd_level=args.icd_level, icd_scheme=args.icd_scheme,
                              n_show=args.test_examples, seed=args.seed)
    else:
        rank0_print("Skipping test generation on non-main ranks.")

if __name__ == "__main__":
    # Optional NCCL settings; uncomment if your cluster benefits:
    # os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    # os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    # os.environ.setdefault("NCCL_DEBUG", "WARN")
    main()
