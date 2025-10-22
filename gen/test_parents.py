"""
Complete test script for ICD-9 parent code prediction
All utility functions included - no external dependencies
"""

import os, sys, json, time, argparse, pickle, re
import numpy as np
import pandas as pd
import torch
import ast
from typing import List, Any, Dict, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings

# -------- Hygiene & perf toggles --------
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_DISABLE_PROGRESS_BAR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=".*top_p.*")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rank0_print(*a, **k):
    print(*a, **k)
    sys.stdout.flush()

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

# --- ICD regexes ---
_RX_ICD9_NUM  = re.compile(r"^(?:\d{3})(?:\.\d{1,2})?$")
_RX_ICD9_V    = re.compile(r"^V\d{2}(?:\.\d{1,2})?$")
_RX_ICD9_E    = re.compile(r"^E\d{3}(?:\.\d{1})?$")
_RX_ICD10     = re.compile(r"^[A-Z][0-9A-Z][0-9A-Z](?:\.[0-9A-Z]{1,4})?$")

def format_icd_leaf(code: str, scheme: str) -> str:
    code = code.strip().upper()
    if scheme == "icd9cm":
        if _RX_ICD9_NUM.match(code) or _RX_ICD9_V.match(code) or _RX_ICD9_E.match(code):
            return code
    elif scheme == "icd10cm":
        if _RX_ICD10.match(code):
            return code
    return ""

def is_valid_icd_leaf(code: str, scheme: str) -> bool:
    return bool(format_icd_leaf(code, scheme))

def parent_first3(code: str) -> str:
    clean_code = code.strip().upper()
    if len(clean_code) >= 3:
        return clean_code[:3]
    return clean_code

def to_level(codes: List[str], level: str, scheme: str) -> List[str]:
    if level == "leaf":
        return [format_icd_leaf(c, scheme) for c in codes if is_valid_icd_leaf(c, scheme)]
    elif level == "parent":
        valid_leaves = [format_icd_leaf(c, scheme) for c in codes if is_valid_icd_leaf(c, scheme)]
        return list(set(parent_first3(c) for c in valid_leaves))
    return []

# ---------------- Text processing ----------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x: Any) -> str:
    if pd.isna(x):
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

# ---------------- Data processing ----------------
def collapse_codes_in_df(df: pd.DataFrame, label_col: str, icd_level: str, icd_scheme: str) -> List[List[str]]:
    result = []
    for _, row in df.iterrows():
        codes = to_list(row[label_col])
        level_codes = to_level(codes, icd_level, icd_scheme)
        result.append(level_codes)
    return result

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
    rank0_print(f"Generating predictions for {total} samples (batch={batch_size})...")
    
    start_time = time.perf_counter()
    for i in range(0, total, batch_size):
        batch_start = time.perf_counter()
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
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
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
        
        # Progress reporting
        batch_time = time.perf_counter() - batch_start
        progress = min(i + batch_size, total)
        if progress % (batch_size * 10) == 0 or progress == total:
            elapsed = time.perf_counter() - start_time
            remaining = total - progress
            if progress > 0:
                eta = elapsed * remaining / progress
                rank0_print(f"Generated {progress}/{total} ({100*progress/total:.1f}%) | last batch {batch_time:.2f}s | ETA {eta:.1f}s")
    
    total_time = time.perf_counter() - start_time
    rank0_print(f"Generation complete in {total_time:.1f}s ({total_time/total:.3f}s/sample)")
    
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

def show_test_predictions(test_df: pd.DataFrame, preds: List[List[str]], 
                         label_col: str, labels_vocab: List[str],
                         icd_level: str, icd_scheme: str, 
                         n_show: int = 5, seed: int = 42) -> None:
    """Show example predictions vs ground truth"""
    np.random.seed(seed)
    indices = np.random.choice(len(test_df), min(n_show, len(test_df)), replace=False)
    
    gold_lists = collapse_codes_in_df(test_df, label_col, icd_level, icd_scheme)
    
    for i, idx in enumerate(indices):
        row = test_df.iloc[idx]
        pred_codes = preds[idx]
        gold_codes = gold_lists[idx]
        
        rank0_print(f"\n--- Example {i+1} (Sample {idx}) ---")
        rank0_print(f"Subject ID: {row.get('subject_id_x', 'Unknown')}")
        
        # Show a snippet of the input
        if 'Chief Complaint' in row and not pd.isna(row['Chief Complaint']):
            chief_complaint = str(row['Chief Complaint'])[:200]
            rank0_print(f"Chief Complaint: {chief_complaint}...")
        
        rank0_print(f"Gold codes ({len(gold_codes)}): {sorted(gold_codes)}")
        rank0_print(f"Pred codes ({len(pred_codes)}): {sorted(pred_codes)}")
        
        # Calculate overlap
        gold_set = set(gold_codes)
        pred_set = set(pred_codes)
        intersection = gold_set & pred_set
        union = gold_set | pred_set
        
        if union:
            jaccard = len(intersection) / len(union)
            rank0_print(f"Jaccard similarity: {jaccard:.3f}")
        
        if gold_set:
            precision = len(intersection) / len(pred_set) if pred_set else 0
            recall = len(intersection) / len(gold_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            rank0_print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# ---------------- Main functions ----------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_pickle", required=True, help="Test dataset pickle")
    ap.add_argument("--adapter_dir", required=True, help="Directory with trained adapter")
    ap.add_argument("--labels_json", required=True, help="Label space JSON from training")
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--test_batch_size", type=int, default=16)
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--icd_scheme", choices=["icd9cm", "icd10cm"], default="icd9cm")
    ap.add_argument("--icd_level", default="parent")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs_gen/test_results")
    ap.add_argument("--test_examples", type=int, default=5)
    return ap.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    rank0_print(f"=== Testing ICD-{args.icd_scheme.upper()} PARENT Model ===")
    rank0_print(f"Test pickle: {args.test_pickle}")
    rank0_print(f"Adapter dir: {args.adapter_dir}")
    rank0_print(f"Labels JSON: {args.labels_json}")

    # Load test data
    test_df = pickle.load(open(args.test_pickle, "rb"))
    rank0_print(f"Test samples: {len(test_df)}")

    # Load label vocabulary
    labels_data = json.load(open(args.labels_json))
    labels_vocab = labels_data["labels"]
    rank0_print(f"Label vocabulary size: {len(labels_vocab)}")

    # Setup tokenizer
    adapter_parent = os.path.dirname(args.adapter_dir)
    tok_src = os.path.join(adapter_parent, "tokenizer") \
        if os.path.exists(os.path.join(adapter_parent, "tokenizer")) else args.llama_model
    
    rank0_print(f"Loading tokenizer from: {tok_src}")
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # For efficient generation

    # Setup device and model
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        use_bf16 = cc[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        device = torch.device("cuda")
        rank0_print(f"CUDA device: {torch.cuda.get_device_name(0)}, bf16: {use_bf16}")
    else:
        dtype = torch.float32
        device = torch.device("cpu")
        rank0_print("Using CPU (slow)")

    # Load model
    rank0_print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.llama_model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = True

    rank0_print(f"Loading adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.use_cache = True
    model.to(device).eval()

    # Build test prompts
    rank0_print("Building test prompts...")
    test_prompts = test_df.apply(lambda r: build_input_text(
        r, bool(args.use_structured), bool(args.use_notes), args.subject_col,
        args.icd_level, args.icd_scheme
    ), axis=1).astype(str).tolist()

    # Prepare gold labels
    gold_lists = collapse_codes_in_df(test_df, args.label_col, 
                                     args.icd_level, args.icd_scheme)

    # Sort by length for efficient batching
    rank0_print("Sorting prompts by length for efficient batching...")
    lens = [len(tok.encode(p, add_special_tokens=True)) for p in test_prompts]
    order = np.argsort(lens)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    prompts_sorted = [test_prompts[i] for i in order]

    # Generate predictions
    rank0_print(f"Starting generation with batch size {args.test_batch_size}...")
    test_start = time.perf_counter()
    
    preds_sorted = generate_codes(
        model, tok, prompts_sorted, labels_vocab,
        icd_level=args.icd_level, icd_scheme=args.icd_scheme,
        max_new=args.gen_max_new, batch_size=args.test_batch_size, 
        max_len=args.max_len
    )
    
    # Restore original order
    preds = [preds_sorted[i] for i in inv_order]
    test_gen_secs = time.perf_counter() - test_start

    rank0_print(f"Generation completed in {test_gen_secs/60:.2f} minutes")

    # Calculate metrics
    rank0_print("Calculating metrics...")
    Y_true = codes_to_multihot(gold_lists, labels_vocab)
    Y_pred = codes_to_multihot(preds, labels_vocab)
    metrics = eval_sets(Y_true, Y_pred)

    # Add test metadata
    metrics.update({
        "test_samples": len(test_df),
        "test_batch_size": args.test_batch_size,
        "test_generate_seconds": test_gen_secs,
        "samples_per_second": len(test_df) / max(test_gen_secs, 1e-9),
        "icd_scheme": args.icd_scheme,
        "icd_level": args.icd_level,
        "label_space_size": len(labels_vocab)
    })

    # Save results
    results_file = os.path.join(args.out_dir, "test_metrics.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    rank0_print(f"Results saved to: {results_file}")

    # Print main metrics
    rank0_print(f"\n=== TEST RESULTS ({args.icd_level} level) ===")
    rank0_print(f"Test samples: {len(test_df)}")
    rank0_print(f"Generation time: {test_gen_secs:.1f}s")
    rank0_print(f"Samples/second: {len(test_df)/test_gen_secs:.2f}")
    rank0_print(f"Label space: {len(labels_vocab)} codes")
    rank0_print(f"\nPerformance Metrics:")
    rank0_print(f"  Micro F1:    {metrics['micro_f1']:.4f}")
    rank0_print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    rank0_print(f"  Samples F1:  {metrics['samples_f1']:.4f}")
    rank0_print(f"  Micro Prec:  {metrics['micro_precision']:.4f}")
    rank0_print(f"  Micro Rec:   {metrics['micro_recall']:.4f}")

    # Show example predictions
    rank0_print(f"\n=== Sample Predictions ===")
    show_test_predictions(test_df, preds, args.label_col, labels_vocab,
                         args.icd_level, args.icd_scheme,
                         n_show=args.test_examples, seed=args.seed)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())