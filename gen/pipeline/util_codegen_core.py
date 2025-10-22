# pipelines/util_codegen_core.py
import os, re, json, random, logging, datetime, time
from typing import List, Any, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers.utils import logging as hf_logging

# ---------------- logging ----------------
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def is_main_process() -> bool:
    return True  # single-process in this utility

def rank0_print(*a, **k):
    if is_main_process():
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}]", *a, **k)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------- ICD-9 helpers ----------------
# def format_icd9_properly(code: str) -> str:
#     code = str(code).strip().upper()
#     code = re.sub(r"\s+", "", code)
#     if code.endswith("."):
#         code = code[:-1]
#     if code and code[0].isdigit():
#         if '.' not in code and len(code) > 3:
#             return code[:3] + '.' + code[3:]
#     elif code and len(code) > 1:
#         if code[0] in ('V', 'E') and '.' not in code and len(code) > 3:
#             return code[:3] + '.' + code[3:]
#     return code

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
    if not code: return False
    if code[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    if code.startswith('V'): return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    if code.startswith('E'): return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    return False

def normalize_code(c: str) -> str:
    return format_icd9_properly(c)

def get_icd9_parent(code: str) -> str:
    code = str(code).upper().strip()
    if not code or len(code) < 3:
        return code
    if code[0].isdigit():
        return code.split('.')[0][:3]
    if code.startswith('V'):
        base = code.split('.')[0]; return base[:3]
    if code.startswith('E'):
        base = code.split('.')[0]; return base[:4] if len(base) >= 4 else base
    return code

# ---------------- text helpers ----------------
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
    s.append("[TASK] You are a medical coding expert. Based on the patient information above, "
             "generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
    s.append("1. List only the ICD-9 codes separated by spaces")
    s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
    s.append("3. Include only codes directly supported by the clinical information")
    s.append("4. Do not include any explanations or text besides the codes themselves")
    s.append("[CODES]")
    return "\n".join([x for x in s if x])

# ---------------- splits ----------------
from sklearn.model_selection import train_test_split
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

# ---------------- dataset & collator ----------------
class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int, label_col: str):
        self.tok = tok
        self.max_len = max_len
        self.tgt_reserve = max(8, int(tgt_reserve))
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
        if len(prompt_ids) > max_prompt_len: prompt_ids = prompt_ids[:max_prompt_len]
        remaining = max(1, self.max_len - len(prompt_ids))
        if len(ans_ids) > remaining: ans_ids = ans_ids[:remaining]
        input_ids = prompt_ids + ans_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + ans_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def pad_collate(features, tok):
    pad_id = tok.pad_token_id
    batch_size = len(features)
    lengths = [f["input_ids"].size(0) for f in features]
    max_len = max(lengths)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        ids = f["input_ids"]; am = f["attention_mask"]; lab = f["labels"]
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = am
        labels[i, :L] = lab
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------- model loader ----------------
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
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
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

# ---------------- metrics ----------------
from sklearn.metrics import f1_score, precision_score, recall_score
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

# ---------------- IO helpers ----------------
def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def show_test_predictions(df: pd.DataFrame, preds: List[List[str]],
                          label_col: str, label_vocab: List[str],
                          n_show: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
    idx_map = {c:i for i,c in enumerate(label_vocab)}
    from sklearn.metrics import precision_score, recall_score, f1_score
    rank0_print(f"Showing {n_show} random test examples:")
    for i in idxs:
        row = df.iloc[i]
        gold = sorted({format_icd9_properly(str(c)) for c in row[label_col] if is_valid_icd9(format_icd9_properly(str(c)))})
        pred = preds[i]
        missing = sorted([c for c in gold if c not in pred])
        extra   = sorted([c for c in pred if c not in gold])

        gold_parents = {get_icd9_parent(c) for c in gold}
        pred_parents = {get_icd9_parent(c) for c in pred}
        parent_matches = sorted([f"{c} (parent)" for c in pred if get_icd9_parent(c) in gold_parents and c not in gold])

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
