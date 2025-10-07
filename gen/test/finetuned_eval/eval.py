# -*- coding: utf-8 -*-
"""
Evaluation script for generative ICD-9 code prediction.
Loads fine-tuned LoRA adapter and evaluates on test set.
"""

import os, re, json, random, logging, pickle, datetime, time, argparse
from typing import List, Any, Dict
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------- Env & logging ----------------
from transformers import logging as hf_logging
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

def rank0_print(*a, **k):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *a, **k)

# ---------------- Args ----------------
def get_args():
    ap = argparse.ArgumentParser()
    # required
    ap.add_argument("--run_dir", required=True, help="Directory with saved adapter_best and tokenizer")
    ap.add_argument("--test_pickle", required=True, help="Path to test data pickle")

    # model/prompt (can override config.json)
    ap.add_argument("--llama_model", default=None, help="Base model path (overrides config)")
    ap.add_argument("--use_structured", type=int, default=None)
    ap.add_argument("--use_notes", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--gen_max_new", type=int, default=None)

    # data cols
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")

    # eval
    ap.add_argument("--test_batch_size", type=int, default=16)
    ap.add_argument("--test_examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # misc
    ap.add_argument("--output_json", default=None, help="Path to save metrics (default: run_dir/test_metrics_eval.json)")
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

# ---------------- Labels ----------------
def y_multi_hot(mlb: MultiLabelBinarizer, lists):
    formatted_lists = []
    for row in lists:
        formatted_row = [format_icd9_properly(str(c)) for c in row]
        formatted_row = [c for c in formatted_row if is_valid_icd9(c)]
        formatted_lists.append(formatted_row)
    return mlb.transform(formatted_lists)

# ---------------- Generation & metrics ----------------
@torch.no_grad()
def generate_codes(model, tok, prompts: List[str], labels_vocab: List[str],
                   max_new=96, batch_size=16, max_len=3072):
    """Optimized code generation with proper device handling and memory management"""
    model.eval()
    
    device = next(model.parameters()).device
    allowed = set(labels_vocab)
    preds = []
    
    total_samples = len(prompts)
    rank0_print(f"Generating predictions for {total_samples} samples in batches of {batch_size}...")
    
    start_time = time.time()
    last_time = start_time
    
    for i in range(0, total_samples, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        curr_batch_size = len(batch_prompts)
        
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, 
                     truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
            
        seq = out.sequences
        gen_only = seq[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        
        for t in texts:
            tokens = re.split(r"[^A-Za-z0-9\.]+", t)
            cand = [normalize_code(z) for z in tokens if z]
            seen, keep = set(), []
            for c in cand:
                if c in allowed and is_valid_icd9(c) and c not in seen:
                    seen.add(c); keep.append(c)
            preds.append(keep)
        
        if ((i + curr_batch_size) % (10 * batch_size) == 0 or 
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
            
        if torch.cuda.is_available() and (i + curr_batch_size) % (20 * batch_size) == 0:
            torch.cuda.empty_cache()
            
    total_time = time.time() - start_time
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

def show_test_predictions(df: pd.DataFrame, preds: List[List[str]],
                          label_col: str, label_vocab: List[str],
                          n_show: int = 5, seed: int = 0):
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

# ---------------- Main ----------------
def main():
    args = get_args()
    set_seed(args.seed)

    rank0_print("CUDA:", torch.cuda.is_available(),
                "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Load config from run_dir
    config_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"config.json not found in {args.run_dir}")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Override with args if provided
    llama_model = args.llama_model or config.get("model")
    max_len = args.max_len or config.get("max_len", 3072)
    gen_max_new = args.gen_max_new or config.get("gen_max_new", 96)
    # For prompt building, assume from args or default to 1
    use_structured = args.use_structured if args.use_structured is not None else 1
    use_notes = args.use_notes if args.use_notes is not None else 1

    # Load label space
    label_path = os.path.join(args.run_dir, "label_space.json")
    if not os.path.exists(label_path):
        raise ValueError(f"label_space.json not found in {args.run_dir}")
    with open(label_path, "r") as f:
        label_data = json.load(f)
    labels_vocab = label_data["labels"]
    rank0_print(f"Loaded {len(labels_vocab)} labels from label_space.json")

    # Create MLB
    mlb = MultiLabelBinarizer(classes=labels_vocab)
    mlb.fit([labels_vocab])

    # Load test data
    test_df = pickle.load(open(args.test_pickle, "rb"))
    assert args.label_col in test_df.columns and args.subject_col in test_df.columns
    rank0_print(f"Loaded test data: {len(test_df)} rows")

    # Build prompts
    test_df["input_text"] = test_df.apply(
        lambda r: build_input_text(r, use_structured==1, use_notes==1, args.subject_col), axis=1
    )
    test_prompts = test_df["input_text"].astype(str).tolist()

    # Load tokenizer
    tok_path = os.path.join(args.run_dir, "tokenizer")
    if not os.path.exists(tok_path):
        raise ValueError(f"Tokenizer not found in {args.run_dir}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Load model
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32
    
    rank0_print(f"Loading base model: {llama_model} with dtype={dtype}")
    base = AutoModelForCausalLM.from_pretrained(
        llama_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    adapter_path = os.path.join(args.run_dir, "adapter_best")
    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter not found at {adapter_path}")
    rank0_print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    rank0_print(f"Using test batch size: {args.test_batch_size}")
    test_start = datetime.datetime.now()
    t_gen = time.perf_counter()
    
    pred_code_lists = generate_codes(
        model, tok, test_prompts, labels_vocab,
        max_new=gen_max_new, 
        batch_size=args.test_batch_size, 
        max_len=max_len
    )
    test_gen_secs = time.perf_counter() - t_gen
    test_duration = datetime.datetime.now() - test_start
    
    rank0_print(f"=== Test Generation Summary ===")
    rank0_print(f"Total samples processed: {len(test_df)}")
    rank0_print(f"Batch size used: {args.test_batch_size}")
    rank0_print(f"Generation time: {test_gen_secs:.1f}s ({test_duration})")
    rank0_print(f"Average time per sample: {test_gen_secs/len(test_df):.3f}s")
    rank0_print(f"Samples per second: {len(test_df)/test_gen_secs:.2f}")

    # Ground truth
    y_test = y_multi_hot(mlb, test_df[args.label_col].tolist())

    # Predictions
    Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)

    # Metrics
    rank0_print(f"Computing evaluation metrics...")
    metrics = eval_sets(y_test, Y_pred)
    metrics.update(hierarchical_eval(y_test, Y_pred, labels_vocab))
    metrics["test_generate_seconds"] = test_gen_secs
    metrics["test_duration_str"] = str(test_duration)
    metrics["test_samples"] = len(test_df)
    metrics["test_batch_size"] = args.test_batch_size
    metrics["samples_per_second"] = len(test_df)/test_gen_secs

    output_json = args.output_json or os.path.join(args.run_dir, "test_metrics_eval.json")
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    rank0_print(f"Metrics saved to {output_json}")

    rank0_print(f"\n=== Generative TEST metrics ===")
    rank0_print(f"Main metrics:")
    rank0_print(f"  - micro_f1: {metrics['micro_f1']:.4f}")
    rank0_print(f"  - macro_f1: {metrics['macro_f1']:.4f}")  
    rank0_print(f"  - samples_f1: {metrics['samples_f1']:.4f}")
    rank0_print(f"  - hierarchical_parent_recall: {metrics['hierarchical_parent_recall']:.4f}")
    
    rank0_print("\n=== Sample Predictions ===")
    show_test_predictions(test_df, pred_code_lists, args.label_col, labels_vocab,
                          n_show=args.test_examples, seed=args.seed)

if __name__ == "__main__":
    main()
