import os, re, json, time, argparse, datetime, logging, pickle, random, atexit
from typing import List, Any, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

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

# ----------------- DDP helpers -----------------
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

def _cleanup_dist():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try: torch.distributed.barrier()
            except Exception: pass
            torch.distributed.destroy_process_group()
    except Exception:
        pass
atexit.register(_cleanup_dist)

# Only log at INFO level for the main process
logging.basicConfig(level=logging.INFO if is_main_process() else logging.ERROR, 
                   format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

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
    if is_main_process():
        os.makedirs(path, exist_ok=False)
    barrier()
    return path

def save_json(path: str, obj: dict):
    if is_main_process():
        with open(path, "w") as f: json.dump(obj, f, indent=2)
    barrier()

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
    s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
    s.append("1. List only the ICD-9 codes separated by spaces")
    s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
    s.append("3. Include only codes directly supported by the clinical information")
    s.append("4. Do not include any explanations or text besides the codes themselves")
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
    if is_main_process():
        log.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

# ----------------- ICD-9 Code Handling -----------------
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



def normalize_code_icd9(c: str) -> str:
    return format_icd9_properly(c)

def get_icd9_parent(code: str) -> str:
    if not code or len(code) < 3: return code
    if code[0].isdigit(): return code.split('.')[0][:3]
    if code.startswith('V'):
        base = code.split('.')[0]; return base[:3]
    if code.startswith('E'):
        base = code.split('.')[0]; return base[:4] if len(base) >= 4 else base
    return code

def format_icd10_properly(code: str) -> str:
    """
    Format ICD-10 codes to standard format with dot in the right position.
    Handles category codes, full codes, and alphanumeric extensions.
    """
    code = str(code).strip().upper()
    code = re.sub(r"\s+", "", code)
    
    # Handle empty category codes (like A00.)
    if code.endswith('.'):
        # Category code with dot, preserve as is
        return code
        
    # Standard ICD-10 format has a dot after the 3rd character if not already present
    if code and len(code) >= 3 and '.' not in code:
        # If it's exactly 3 chars (category code), add the dot
        if len(code) == 3:
            return code + '.'
        # Otherwise, it's a full code with implicit dot
        return code[:3] + '.' + code[3:]
        
    return code

def is_valid_icd10(code: str) -> bool:
    """
    Check if code follows valid ICD-10-CM format, handling all cases including:
    - Standard codes (A01.1)
    - Category codes (A00.)
    - Alphanumeric extensions (C49.A0)
    - Empty subcategory markers (A00.)
    - Seventh character extensions (T82.855A)
    """
    if not code: return False
    
    # Clean up the code first
    code = code.strip()
    
    # Handle category codes (A00.)
    if re.match(r"^[A-Z]\d{1,2}\.$", code):
        return True
        
    # Handle standard codes and extensions
    return bool(re.match(r"^[A-Z]\d{1,2}(\.([A-Z0-9]{0,4}))?$", code))

def normalize_code_icd10(c: str) -> str:
    """Normalize ICD-10 code format"""
    return format_icd10_properly(c)

def get_icd10_parent(code: str) -> str:
    """Extract parent code (category level) from ICD-10 code"""
    if not code or len(code) < 3: return code
    # Category level is first 3 characters (letter + 2 digits)
    return code[:3]

# ---------------- Improved lock_label_space from DDP script ----------------
def lock_label_space(frames: List[pd.DataFrame], label_col: str,
                     icd9_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
    train_codes = set()
    for fr in frames:
        for codes in fr[label_col]:
            train_codes.update(format_icd9_properly(str(c)) for c in codes)
    train_codes = {c for c in train_codes if is_valid_icd9(c)}
    if is_main_process():
        log.info(f"Found {len(train_codes)} unique valid ICD codes in training data")

    if not use_complete or not icd9_pkl_path:
        all_codes = sorted(train_codes)
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        if is_main_process():
            log.info(f"Using {len(all_codes)} codes from training data only")
        return mlb

    try:
        icd9_df = pd.read_pickle(icd9_pkl_path)
        complete_codes = sorted(icd9_df['icd_code'].astype(str).tolist())
        complete_codes = [format_icd9_properly(code) for code in complete_codes]
        complete_codes = [code for code in complete_codes if is_valid_icd9(code)]
        if is_main_process():
            log.info(f"Loaded {len(complete_codes)} complete ICD-9 codes from {icd9_pkl_path}")
        mlb = MultiLabelBinarizer(classes=complete_codes)
        mlb.fit([complete_codes])

        if is_main_process():
            codes_in_complete = sum(1 for c in train_codes if c in set(complete_codes))
            codes_not_in_complete = len(train_codes) - codes_in_complete
            log.info(f"Training data coverage: in={codes_in_complete}, missing={codes_not_in_complete}")
            if codes_not_in_complete > 0:
                log.warning("Some training codes not found in complete ICD-9 set.")
        return mlb

    except Exception as e:
        if is_main_process():
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

icd9_code = "276.69" # Example ICD-9 code
icd10_code = "I21.19" # Example ICD-10 code
print("Parent ICD-9 Code: ", get_icd9_parent(icd9_code))
print("Parent ICD-10 Code: ", get_icd10_parent(icd10_code))
