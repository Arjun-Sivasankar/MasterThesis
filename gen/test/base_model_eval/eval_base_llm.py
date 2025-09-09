# # -*- coding: utf-8 -*-
# """
# Fast BASE-model evaluation for generative ICD code prediction (no finetuning, no LoRA).

# Features:
# - DDP support for multi-GPU evaluation
# - Single-pass generation -> parse twice (filtered & unfiltered)
# - Left padding for generation (more efficient with uneven prompts)
# - Optional early stop on double newline to avoid rambling
# - Higher default generation batch size
# - Proper ICD-9 code formatting and validation
# - Support for complete ICD-9 code space

# Defaults
# - BOTH modalities ON (structured + notes)
# - Filter mode BOTH (but still single-pass generation)

# Examples
# --------
# # Full eval, both metrics, both modalities:
# python eval_base_llm.py --data_pickle mergeddf.pkl --model meta-llama/Llama-3.2-1B-Instruct

# # Multi-GPU parallel evaluation:
# torchrun --nproc_per_node=4 eval_base_llm.py --data_pickle mergeddf.pkl --model meta-llama/Llama-3.2-1B-Instruct

# # Faster: raise batch size (if VRAM allows), still single-pass:
# python eval_base_llm.py --gen_batch_size 8

# # Filtered metrics only (same single-pass gen, slightly less postproc):
# python eval_base_llm.py --filter_mode filtered
# """

# import os, re, json, time, argparse, datetime, logging, pickle, random, atexit
# from typing import List, Any, Dict, Optional
# import numpy as np
# import pandas as pd
# import torch
# import torch.distributed as dist

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score, precision_score, recall_score
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.utils import logging as hf_logging
# from transformers import StoppingCriteria, StoppingCriteriaList

# # ----------------- Quiet & deterministic -----------------
# os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# hf_logging.set_verbosity_error()

# def set_seed(seed=42):
#     random.seed(seed); np.random.seed(seed)
#     torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# # ----------------- DDP helpers -----------------
# def dist_is_initialized():
#     return torch.distributed.is_available() and torch.distributed.is_initialized()

# def _env_rank():
#     for k in ("LOCAL_RANK", "RANK"):
#         v = os.environ.get(k)
#         if v is not None:
#             try: return int(v)
#             except: pass
#     return 0

# def get_rank():
#     return torch.distributed.get_rank() if dist_is_initialized() else _env_rank()

# def is_main_process():
#     return get_rank() == 0

# def barrier():
#     if dist_is_initialized():
#         try: torch.distributed.barrier()
#         except Exception: pass

# def rank0_print(*a, **k):
#     if is_main_process():
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(f"[{timestamp}]", *a, **k)

# def _cleanup_dist():
#     try:
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             try: torch.distributed.barrier()
#             except Exception: pass
#             torch.distributed.destroy_process_group()
#     except Exception:
#         pass
# atexit.register(_cleanup_dist)

# # Only log at INFO level for the main process
# logging.basicConfig(level=logging.INFO if is_main_process() else logging.ERROR, 
#                    format="%(asctime)s - %(levelname)s - %(message)s")
# log = logging.getLogger(__name__)

# # ----------------- Columns / text sections -----------------
# SUBJECT_COL = "subject_id_x"
# LABEL_COL   = "icd_code"
# TEXT_COLS_SAFE = [
#     "Chief Complaint","History of Present Illness","Past Medical History",
#     "Family History","Physical Exam","Pertinent Results",
#     "Brief Hospital Course","Medications on Admission"
# ]

# # ----------------- Helpers -----------------
# def now_tag():
#     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# def make_run_dir(base="runs_base_eval_fast", run_name=None):
#     tag = run_name or f"{now_tag()}_base_eval_fast"
#     path = os.path.join(base, tag)
#     if is_main_process():
#         os.makedirs(path, exist_ok=False)
#     barrier()
#     return path

# def save_json(path: str, obj: dict):
#     if is_main_process():
#         with open(path, "w") as f: json.dump(obj, f, indent=2)
#     barrier()

# def clean_text(x: Any) -> str:
#     if isinstance(x, (list, tuple, set, dict, np.ndarray, pd.Series)): return ""
#     try:
#         if pd.isna(x): return ""
#     except Exception:
#         pass
#     s = str(x).replace("\x00"," ").replace("\r"," ")
#     s = re.sub(r"_+"," ", s)
#     return re.sub(r"\s+"," ", s).strip()

# def to_list(x) -> List[str]:
#     """Robust label coercion -> list[str], safe for arrays/lists/strings/NaN."""
#     def _norm(z):
#         s = str(z); s = re.sub(r"\s+","", s.upper())
#         return s[:-1] if s.endswith(".") else s
#     if isinstance(x, (list, tuple, set)):
#         return [_norm(v) for v in x if str(v).strip()]
#     if isinstance(x, np.ndarray):
#         return [_norm(v) for v in x.reshape(-1).tolist() if str(v).strip()]
#     if isinstance(x, pd.Series):
#         return [_norm(v) for v in x.tolist() if str(v).strip()]
#     try:
#         if pd.isna(x): return []
#     except Exception:
#         pass
#     if isinstance(x, str):
#         s = x.strip()
#         if not s: return []
#         if s.startswith("[") and s.endswith("]"):
#             try:
#                 import ast
#                 v = ast.literal_eval(s)
#                 if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
#                     if isinstance(v, np.ndarray): v = v.tolist()
#                     if isinstance(v, pd.Series):  v = v.tolist()
#                     return [_norm(z) for z in v if str(z).strip()]
#             except Exception:
#                 pass
#         return [_norm(t) for t in re.split(r"[,\s]+", s) if t]
#     return [_norm(x)]

# def serialize_structured(row: pd.Series) -> str:
#     parts = []
#     parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
#                  f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
#     ndc  = to_list(row.get("ndc", []))
#     proc = to_list(row.get("pro_code", []))
#     labs = to_list(row.get("lab_test", []))
#     if ndc:  parts.append("[NDC] "  + " ".join(ndc[:32]))
#     if proc: parts.append("[PROC] " + " ".join(proc[:32]))
#     if labs: parts.append("[LAB] "  + " ".join(labs[:64]))
#     return "\n".join(parts)

# def serialize_notes(row: pd.Series, text_cols: List[str]) -> str:
#     chunks=[]
#     for col in text_cols:
#         if col in row:
#             t = clean_text(row[col])
#             if t: chunks.append(f"[{col.upper()}] {t}")
#     return "\n".join(chunks)

# def build_input_text(row: pd.Series, use_structured=True, use_notes=True, text_cols=TEXT_COLS_SAFE) -> str:
#     s = [f"[VISIT] subject_id={row.get(SUBJECT_COL,'?')} hadm_id={row.get('hadm_id','?')}"]
#     if use_structured: s.append(serialize_structured(row))
#     if use_notes:
#         t = serialize_notes(row, text_cols)
#         if t: s.append(t)
#     s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
#     s.append("1. List only the ICD-9 codes separated by spaces")
#     s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
#     s.append("3. Include only codes directly supported by the clinical information")
#     s.append("4. Do not include any explanations or text besides the codes themselves")
#     s.append("[CODES]")
#     return "\n".join([x for x in s if x])

# def subject_splits(df: pd.DataFrame, subject_col=SUBJECT_COL,
#                    test_size=0.10, val_size=0.10, seed=42):
#     subs = df[subject_col].dropna().unique()
#     train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
#     train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
#     tr = df[df[subject_col].isin(train_subs)].copy()
#     va = df[df[subject_col].isin(val_subs)].copy()
#     te = df[df[subject_col].isin(test_subs)].copy()
#     if is_main_process():
#         log.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
#     return tr, va, te

# # ----------------- ICD-9 Code Handling -----------------
# def format_icd9_properly(code: str) -> str:
#     code = code.strip().upper()
#     code = re.sub(r"\s+", "", code)
#     if code.endswith("."): code = code[:-1]
#     if code and code[0].isdigit():
#         if '.' not in code and len(code) > 3:
#             return code[:3] + '.' + code[3:]
#     elif code and len(code) > 1:
#         if code[0] in ('V', 'E') and '.' not in code and len(code) > 3:
#             return code[:3] + '.' + code[3:]
#     return code

# def is_valid_icd9(code: str) -> bool:
#     if not code: return False
#     if code[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
#     if code.startswith('V'): return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
#     if code.startswith('E'): return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
#     return False

# def normalize_code(c: str) -> str:
#     return format_icd9_properly(c)

# def get_icd9_parent(code: str) -> str:
#     if not code or len(code) < 3: return code
#     if code[0].isdigit(): return code.split('.')[0][:3]
#     if code.startswith('V'):
#         base = code.split('.')[0]; return base[:3]
#     if code.startswith('E'):
#         base = code.split('.')[0]; return base[:4] if len(base) >= 4 else base
#     return code

# # ---------------- Improved lock_label_space from DDP script ----------------
# def lock_label_space(frames: List[pd.DataFrame], label_col: str,
#                      icd9_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
#     train_codes = set()
#     for fr in frames:
#         for codes in fr[label_col]:
#             train_codes.update(format_icd9_properly(str(c)) for c in codes)
#     train_codes = {c for c in train_codes if is_valid_icd9(c)}
#     if is_main_process():
#         log.info(f"Found {len(train_codes)} unique valid ICD codes in training data")

#     if not use_complete or not icd9_pkl_path:
#         all_codes = sorted(train_codes)
#         mlb = MultiLabelBinarizer(classes=all_codes)
#         mlb.fit([all_codes])
#         if is_main_process():
#             log.info(f"Using {len(all_codes)} codes from training data only")
#         return mlb

#     try:
#         icd9_df = pd.read_pickle(icd9_pkl_path)
#         complete_codes = sorted(icd9_df['icd_code'].astype(str).tolist())
#         complete_codes = [format_icd9_properly(code) for code in complete_codes]
#         complete_codes = [code for code in complete_codes if is_valid_icd9(code)]
#         if is_main_process():
#             log.info(f"Loaded {len(complete_codes)} complete ICD-9 codes from {icd9_pkl_path}")
#         mlb = MultiLabelBinarizer(classes=complete_codes)
#         mlb.fit([complete_codes])

#         if is_main_process():
#             codes_in_complete = sum(1 for c in train_codes if c in set(complete_codes))
#             codes_not_in_complete = len(train_codes) - codes_in_complete
#             log.info(f"Training data coverage: in={codes_in_complete}, missing={codes_not_in_complete}")
#             if codes_not_in_complete > 0:
#                 log.warning("Some training codes not found in complete ICD-9 set.")
#         return mlb

#     except Exception as e:
#         if is_main_process():
#             log.error(f"Error loading complete ICD-9 codes: {e}")
#             log.warning("Falling back to training-data-only label space")
#         all_codes = sorted(train_codes)
#         mlb = MultiLabelBinarizer(classes=all_codes)
#         mlb.fit([all_codes])
#         return mlb

# def y_multi_hot(mlb: MultiLabelBinarizer, lists):
#     formatted_lists = []
#     for row in lists:
#         formatted_row = [format_icd9_properly(str(c)) for c in row]
#         formatted_row = [c for c in formatted_row if is_valid_icd9(c)]
#         formatted_lists.append(formatted_row)
#     return mlb.transform(formatted_lists)

# # ----------------- Optimized generation -----------------
# class DoubleNewlineStop(StoppingCriteria):
#     def __init__(self, tok, max_new_tokens: int, lookback: int = 96):
#         self.tok = tok; self.max_new = max_new_tokens; self.lookback = lookback
#     def __call__(self, input_ids, scores, **kwargs):
#         last = input_ids[0].tolist()[-min(self.lookback, len(input_ids[0])):]
#         tail = self.tok.decode(last, skip_special_tokens=True)
#         return "\n\n" in tail or len(last) >= self.max_new

# @torch.inference_mode()
# def generate_texts(model, tok, prompts: List[str],
#                    max_new=96, batch_size=8, max_len=3072,
#                    stop_on_double_newline=True) -> List[str]:
#     """Generate texts in a memory-efficient manner"""
#     texts = []
#     old_side = tok.padding_side
#     tok.padding_side = "left"  # faster for varied prompt lengths
#     stops = StoppingCriteriaList([DoubleNewlineStop(tok, max_new)]) if stop_on_double_newline else None
    
#     # Get device from model
#     device = next(model.parameters()).device
    
#     # Process in batches with timing
#     total_samples = len(prompts)
#     if is_main_process():
#         rank0_print(f"Generating predictions for {total_samples} samples in batches of {batch_size}...")
    
#     start_time = time.time()
#     last_time = start_time
    
#     for i in range(0, total_samples, batch_size):
#         batch_prompts = prompts[i:i+batch_size]
#         curr_batch_size = len(batch_prompts)
        
#         # Tokenize and move to device
#         inputs = tok(batch_prompts, return_tensors="pt", padding=True, 
#                      truncation=True, max_length=max_len)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         # Generate with mixed precision if available
#         with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and 
#                                 torch.cuda.get_device_capability(0)[0] >= 8)):
#             out = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new,
#                 do_sample=False, 
#                 num_beams=1,
#                 no_repeat_ngram_size=2,
#                 eos_token_id=tok.eos_token_id, 
#                 pad_token_id=tok.pad_token_id,
#                 use_cache=True, 
#                 stopping_criteria=stops
#             )
        
#         texts.extend(tok.batch_decode(out, skip_special_tokens=True))
        
#         # Log progress periodically
#         if is_main_process() and ((i + curr_batch_size) % (10 * batch_size) == 0 or 
#                                  (i + curr_batch_size) >= total_samples):
#             current_time = time.time()
#             elapsed = current_time - start_time
#             batch_time = current_time - last_time
#             progress = (i + curr_batch_size) / total_samples
#             remaining = elapsed / progress - elapsed if progress > 0 else 0
            
#             rank0_print(f"Generated {i + curr_batch_size}/{total_samples} samples " 
#                        f"({progress:.1%}) - Batch time: {batch_time:.2f}s - "
#                        f"Est. remaining: {remaining:.2f}s")
#             last_time = current_time
        
#         # Clear CUDA cache periodically
#         if torch.cuda.is_available() and (i + curr_batch_size) % (20 * batch_size) == 0:
#             torch.cuda.empty_cache()
            
#     tok.padding_side = old_side
#     return texts

# def parse_codes(text: str, labels_vocab: Optional[List[str]] = None) -> List[str]:
#     tail = text.split("[CODES]")[-1]
#     tokens = re.split(r"[^A-Za-z0-9\.]+", tail)
#     cand = [format_icd9_properly(z) for z in tokens if z]
#     cand = [c for c in cand if is_valid_icd9(c)]
#     allowed = set(labels_vocab) if labels_vocab is not None else None
#     seen, keep = set(), []
#     for c in cand:
#         if (allowed is None or c in allowed) and c not in seen:
#             seen.add(c); keep.append(c)
#     return keep

# def codes_to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
#     idx = {c:i for i,c in enumerate(label_vocab)}
#     Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
#     for i, lst in enumerate(code_lists):
#         for c in lst:
#             j = idx.get(c)
#             if j is not None: Y[i, j] = 1
#     return Y

# def eval_sets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     return {
#         "micro_f1":   f1_score(y_true, y_pred, average="micro",   zero_division=0),
#         "macro_f1":   f1_score(y_true, y_pred, average="macro",   zero_division=0),
#         "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
#         "micro_precision":   precision_score(y_true, y_pred, average="micro",   zero_division=0),
#         "macro_precision":   precision_score(y_true, y_pred, average="macro",   zero_division=0),
#         "samples_precision": precision_score(y_true, y_pred, average="samples", zero_division=0),
#         "micro_recall":      recall_score(y_true, y_pred, average="micro",      zero_division=0),
#         "macro_recall":      recall_score(y_true, y_pred, average="macro",      zero_division=0),
#         "samples_recall":    recall_score(y_true, y_pred, average="samples",    zero_division=0),
#     }

# def hierarchical_eval(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> Dict[str, float]:
#     code_to_parent = {code: get_icd9_parent(code) for code in label_vocab}
#     parent_to_idx = {}
#     for idx, code in enumerate(label_vocab):
#         parent = code_to_parent[code]
#         parent_to_idx.setdefault(parent, []).append(idx)

#     n_samples = y_true.shape[0]
#     parent_hits = 0
#     partial_matches = 0
#     total_true_parents = 0

#     for i in range(n_samples):
#         pred_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_pred[i])[0]}
#         true_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_true[i])[0]}
#         parent_hits += len(pred_parents & true_parents)
#         total_true_parents += len(true_parents)
#         for parent in pred_parents:
#             if parent in true_parents:
#                 child_indices = parent_to_idx.get(parent, [])
#                 exact_match = any(y_true[i, idx] == 1 and y_pred[i, idx] == 1 for idx in child_indices)
#                 if not exact_match:
#                     partial_matches += 1

#     parent_recall = (parent_hits / total_true_parents) if total_true_parents > 0 else 0
#     return {
#         "hierarchical_parent_recall": parent_recall,
#         "hierarchical_partial_matches": partial_matches,
#         "hierarchical_partial_per_sample": partial_matches / n_samples if n_samples > 0 else 0
#     }

# def show_examples(df: pd.DataFrame, preds: List[List[str]], n=5, seed=42):
#     if not is_main_process():
#         return
        
#     rng = np.random.default_rng(seed)
#     idxs = rng.choice(len(df), size=min(n, len(df)), replace=False)
#     for i in idxs:
#         row = df.iloc[i]
#         gold = [format_icd9_properly(str(c)) for c in row[LABEL_COL]]
#         gold = sorted([c for c in gold if is_valid_icd9(c)])
#         pred = preds[i]
        
#         missing = sorted([c for c in gold if c not in pred])
#         extra   = sorted([c for c in pred if c not in gold])

#         gold_parents = {get_icd9_parent(c) for c in gold}
#         pred_parents = {get_icd9_parent(c) for c in pred}
#         parent_matches = sorted([f"{c} (parent)" for c in pred
#                                 if get_icd9_parent(c) in gold_parents and c not in gold])
        
#         print("\n" + "="*80)
#         print(f"idx={i} subject_id={row.get(SUBJECT_COL)} hadm_id={row.get('hadm_id')}")
#         print("- GOLD:", " ".join(gold) if gold else "(none)")
#         print("- PRED:", " ".join(pred) if pred else "(none)")
#         print(f"- FALSE NEGATIVES ({len(missing)}):", " ".join(missing) if missing else "(none)")
#         print(f"- FALSE POSITIVES ({len(extra)}):", " ".join(extra) if extra else "(none)")
#         print(f"- PARENT MATCHES ({len(parent_matches)}):", " ".join(parent_matches) if parent_matches else "(none)")

# # ----------------- Distributed Data Handling -----------------
# def shard_dataset(df: pd.DataFrame):
#     """Split dataset for distributed evaluation"""
#     if not dist_is_initialized():
#         return df
        
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
    
#     # Calculate shard indices
#     shard_size = len(df) // world_size
#     start_idx = rank * shard_size
#     end_idx = start_idx + shard_size if rank < world_size - 1 else len(df)
    
#     # Get our shard
#     shard = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
#     log.info(f"Rank {rank}/{world_size}: Processing {len(shard)} samples (indices {start_idx} to {end_idx-1})")
#     return shard

# def gather_predictions(local_preds):
#     """Gather predictions from all processes"""
#     if not dist_is_initialized():
#         return local_preds
        
#     world_size = dist.get_world_size()
    
#     # Create a list to hold predictions from all processes
#     all_preds = [None] * world_size
    
#     # Gather all predictions with error handling
#     try:
#         dist.all_gather_object(all_preds, local_preds)
#     except Exception as e:
#         log.error(f"Error during all_gather_object: {e}")
#         # If we can't gather, return only local predictions and warn
#         if is_main_process():
#             log.warning("Failed to gather predictions from all processes. Using partial results.")
#         return local_preds
    
#     # Flatten the list of lists
#     combined_preds = []
#     for p in all_preds:
#         if p is not None:  # Sanity check
#             combined_preds.extend(p)
            
#     # Sort by original order (if we have a way to track it)
#     return combined_preds

# # ----------------- Load model/tokenizer -----------------
# def load_base(model_name: str):
#     if torch.cuda.is_available():
#         use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
#         dtype = torch.bfloat16 if use_bf16 else torch.float16
#         torch.backends.cuda.matmul.allow_tf32 = True
#         try: torch.set_float32_matmul_precision("high")
#         except Exception: pass
#     else:
#         dtype = torch.float32
#     tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     if tok.pad_token is None: tok.pad_token = tok.eos_token
#     tok.padding_side = "right"  # we switch to left only during generation
    
#     # Select appropriate device map based on DDP initialization
#     if dist_is_initialized():
#         # In DDP mode, each process loads model to its specific GPU
#         local_rank = int(os.environ.get("LOCAL_RANK", 0))
#         device_map = {"": local_rank}
#     else:
#         # Auto-distribute in single process mode
#         device_map = "auto"
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, 
#         torch_dtype=dtype, 
#         device_map=device_map
#     )
#     model.config.pad_token_id = tok.pad_token_id
#     model.config.use_cache = True
#     return model, tok

# # ----------------- Init DDP -----------------
# def init_distributed():
#     """Initialize distributed training if running with torchrun"""
#     if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
#         log.info("Not running in distributed mode")
#         return False
        
#     rank = int(os.environ["RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
#     if world_size <= 1:
#         log.info("Only one process found, not initializing distributed mode")
#         return False
    
#     if torch.cuda.device_count() < world_size:
#         log.warning(f"Requested world_size={world_size} but only have {torch.cuda.device_count()} GPUs")
    
#     # Set device before initializing process group
#     torch.cuda.set_device(local_rank)
    
#     # Initialize process group with specific device
#     log.info(f"Initializing process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
#     init_method = "env://"
#     backend = "nccl" if torch.cuda.is_available() else "gloo"
    
#     # Initialize with timeout and explicit device IDs
#     timeout = datetime.timedelta(minutes=60)  # Increase timeout to avoid early termination
    
#     try:
#         dist.init_process_group(
#             backend=backend,
#             init_method=init_method,
#             timeout=timeout,
#             rank=rank,
#             world_size=world_size
#         )
#         log.info(f"Distributed initialization complete: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
#     except Exception as e:
#         log.error(f"Failed to initialize process group: {e}")
#         return False
    
#     return True

# # ----------------- Main -----------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_pickle", type=str, default="mergeddf.pkl")
#     ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
#     ap.add_argument("--run_name", type=str, default=None)

#     ap.add_argument("--max_len", type=int, default=3072)
#     ap.add_argument("--gen_max_new", type=int, default=96)
#     ap.add_argument("--gen_batch_size", type=int, default=8)
#     ap.add_argument("--stop_on_double_newline", action="store_true")
#     ap.add_argument("--no_stop_on_double_newline", dest="stop_on_double_newline", action="store_false")
#     ap.set_defaults(stop_on_double_newline=True)

#     # Modalities: ON by default
#     group_struct = ap.add_mutually_exclusive_group()
#     group_struct.add_argument("--structured", dest="use_structured", action="store_true")
#     group_struct.add_argument("--no-structured", dest="use_structured", action="store_false")
#     group_notes = ap.add_mutually_exclusive_group()
#     group_notes.add_argument("--notes", dest="use_notes", action="store_true")
#     group_notes.add_argument("--no-notes", dest="use_notes", action="store_false")
#     ap.set_defaults(use_structured=True, use_notes=True)

#     ap.add_argument("--filter_mode", choices=["filtered","unfiltered","both"], default="both")
#     ap.add_argument("--subset", type=int, default=None, help="limit #test samples for quick runs")
#     ap.add_argument("--save_samples", type=int, default=5)
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--compile", action="store_true", help="try torch.compile for small extra speed")
    
#     # New options for ICD-9 code handling
#     ap.add_argument("--icd9_pickle", default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl", 
#                     help="Path to complete ICD-9 code list")
#     ap.add_argument("--use_complete_icd9", type=int, default=1, help="Use complete ICD-9 code space")

#     args = ap.parse_args()

#     # Initialize distributed processing
#     is_distributed = init_distributed()
#     barrier()

#     set_seed(args.seed)
#     rank0_print(f"CUDA: {torch.cuda.is_available()} | Model: {args.model}")
#     if is_distributed:
#         rank0_print(f"Running in distributed mode with {dist.get_world_size()} processes")
    
#     rank0_print(f"Config -> MAX_LEN:{args.max_len} GEN_MAX_NEW:{args.gen_max_new} GEN_BS:{args.gen_batch_size} "
#           f"STRUCTURED:{args.use_structured} NOTES:{args.use_notes} FILTER_MODE:{args.filter_mode}")

#     # Data
#     df = pickle.load(open(args.data_pickle, "rb"))
#     assert LABEL_COL in df.columns and SUBJECT_COL in df.columns
#     df[LABEL_COL] = df[LABEL_COL].apply(to_list)
#     df["input_text"] = df.apply(lambda r: build_input_text(
#         r, use_structured=args.use_structured, use_notes=args.use_notes), axis=1)
#     df = df[df["input_text"].str.len() > 0].reset_index(drop=True)

#     train_df, val_df, test_df = subject_splits(df, subject_col=SUBJECT_COL, seed=args.seed)
#     mlb = lock_label_space(
#         [train_df, val_df, test_df], LABEL_COL,
#         icd9_pkl_path=args.icd9_pickle, use_complete=bool(args.use_complete_icd9)
#     )
#     labels_vocab = mlb.classes_.tolist()
    
#     # Create a copy of the full test set before sharding
#     full_test_df = test_df.copy()
#     y_test_full = y_multi_hot(mlb, full_test_df[LABEL_COL].tolist())
    
#     if args.subset is not None:
#         test_df = test_df.iloc[:min(args.subset, len(test_df))].copy()
#         full_test_df = test_df.copy()  # Update full test too if using subset
#         y_test_full = y_test_full[:len(test_df)]

#     # Run dir
#     run_dir = make_run_dir(run_name=args.run_name)
#     rank0_print("Run dir:", run_dir)
#     save_json(os.path.join(run_dir, "config.json"), {
#         **vars(args), "labels": len(labels_vocab), "test_size": len(test_df)
#     })

#     # Model
#     model, tok = load_base(args.model)
#     if args.compile:
#         try:
#             model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
#             rank0_print("torch.compile: enabled")
#         except Exception as e:
#             rank0_print(f"torch.compile failed ({e}); continuing without it.")

#     # For tracking subject IDs along with predictions
#     subject_ids = []
    
#     # --------- Distributed data sharding if needed ---------
#     if is_distributed:
#         # Shard the test set for distributed processing
#         orig_test_len = len(test_df)
        
#         # Store subject IDs before sharding to help with reassembly
#         test_df['orig_idx'] = range(len(test_df))
        
#         # Shard the dataset
#         test_df = shard_dataset(test_df)
        
#         # Keep track of subject IDs and indices for proper reassembly later
#         subject_ids = list(zip(test_df[SUBJECT_COL].tolist(), test_df['orig_idx'].tolist()))
        
#         rank0_print(f"Sharded test data: {orig_test_len} total -> {len(test_df)} per process")
    
#     # --------- Single-pass generation ---------
#     test_prompts = test_df["input_text"].astype(str).tolist()
#     t0 = time.perf_counter()
#     texts = generate_texts(
#         model, tok, test_prompts,
#         max_new=args.gen_max_new, batch_size=args.gen_batch_size,
#         max_len=args.max_len, stop_on_double_newline=args.stop_on_double_newline
#     )
#     gen_secs = time.perf_counter() - t0
#     rank0_print(f"[TIME] generation (single pass): {gen_secs/60:.2f} min for {len(test_prompts)} samples")

#     # Process predictions locally
#     if args.filter_mode in ("filtered","both"):
#         preds_f = [parse_codes(t, labels_vocab) for t in texts]
#     if args.filter_mode in ("unfiltered","both"):
#         preds_u = [parse_codes(t, None) for t in texts]
    
#     # Create prediction packages with indices for reassembly
#     pred_packages = []
#     if is_distributed:
#         if args.filter_mode == "filtered":
#             pred_packages = list(zip(subject_ids, preds_f))
#         elif args.filter_mode == "unfiltered": 
#             pred_packages = list(zip(subject_ids, preds_u))
#         else:  # both
#             pred_packages = list(zip(subject_ids, preds_f, preds_u))
    
#     # --------- Gather predictions from all processes if distributed ---------
#     if is_distributed:
#         # First gather the pred packages from all processes
#         combined_packages = gather_predictions(pred_packages)
        
#         barrier()  # Ensure all processes finish gather before proceeding
        
#         if not is_main_process():
#             # Non-main processes don't need to continue
#             return
            
#         # Main process reassembles predictions in the correct order
#         if args.filter_mode == "filtered":
#             # Extract and sort by original index
#             sorted_packages = sorted(combined_packages, key=lambda x: x[0][1])
#             # Extract just the predictions
#             preds_f = [p[1] for p in sorted_packages]
            
#         elif args.filter_mode == "unfiltered":
#             # Extract and sort by original index
#             sorted_packages = sorted(combined_packages, key=lambda x: x[0][1])
#             # Extract just the predictions
#             preds_u = [p[1] for p in sorted_packages]
            
#         else:  # both
#             # Extract and sort by original index
#             sorted_packages = sorted(combined_packages, key=lambda x: x[0][1])
#             # Extract both filtered and unfiltered predictions
#             preds_f = [p[1] for p in sorted_packages]
#             preds_u = [p[2] for p in sorted_packages]
            
#         # Use the full test set and labels for evaluation
#         test_df = full_test_df
#         y_test = y_test_full
        
#     else:
#         # In single process mode, just use the normal test set
#         y_test = y_test_full
        
#     results = {}

#     # Parse/score: filtered
#     if args.filter_mode in ("filtered","both"):
#         Yf = codes_to_multihot(preds_f, labels_vocab)
#         m_f = eval_sets(y_test, Yf)
#         m_f.update(hierarchical_eval(y_test, Yf, labels_vocab))
#         m_f["gen_seconds"] = gen_secs
#         results["filtered"] = m_f
#         save_json(os.path.join(run_dir, "metrics_filtered.json"), m_f)
#         rank0_print("\n=== BASE metrics (STRICT VOCAB FILTER) ===")
#         rank0_print(json.dumps(m_f, indent=2))
#         rank0_print("\n--- Sample predictions (filtered) ---")
#         show_examples(test_df, preds_f, n=args.save_samples, seed=args.seed)

#     # Parse/score: unfiltered
#     if args.filter_mode in ("unfiltered","both"):
#         Yu = codes_to_multihot(preds_u, labels_vocab)  # off-vocab codes drop out naturally
#         m_u = eval_sets(y_test, Yu)
#         m_u.update(hierarchical_eval(y_test, Yu, labels_vocab))
#         m_u["gen_seconds"] = gen_secs
#         results["unfiltered"] = m_u
#         save_json(os.path.join(run_dir, "metrics_unfiltered.json"), m_u)
#         rank0_print("\n=== BASE metrics (NO FILTER) ===")
#         rank0_print(json.dumps(m_u, indent=2))
#         rank0_print("\n--- Sample predictions (unfiltered) ---")
#         show_examples(test_df, preds_u, n=args.save_samples, seed=args.seed)

#     save_json(os.path.join(run_dir, "metrics_summary.json"), results)
#     rank0_print(f"\nDone. Results saved under: {run_dir}")
    
#     # Clean up distributed resources
#     if is_distributed:
#         dist.destroy_process_group()

# if __name__ == "__main__":
#     main()

import os, re, json, time, argparse, datetime, logging, pickle, random, atexit, signal
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

# ----------------- Timeout handling -----------------
class DdpTimeoutHandler:
    def __init__(self, timeout_seconds=3600):
        self.timeout_seconds = timeout_seconds
        self.original_handler = None
        
    def register(self):
        # Only register in Unix systems where SIGALRM is available
        try:
            self.original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout_seconds)
            print(f"⏱️ Timeout handler registered for {self.timeout_seconds}s")
        except AttributeError:
            print("❌ SIGALRM not available, timeout handler not registered")
            
    def cancel(self):
        try:
            signal.alarm(0)
            if self.original_handler:
                signal.signal(signal.SIGALRM, self.original_handler)
            print("⏱️ Timeout handler canceled")
        except AttributeError:
            pass
            
    def handler(self, signum, frame):
        print("⚠️ OPERATION TIMED OUT - Emergency cleanup...")
        # Try to dump debug info
        try:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                print(f"⚠️ Process rank {rank} timed out")
                # Try to terminate dist group gracefully
                try: torch.distributed.destroy_process_group()
                except: pass
        except:
            pass
        # Exit with error code
        os._exit(124)  # Exit with timeout status

# ----------------- Robust DDP helpers -----------------
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
    """Safer barrier operation without timeout (older PyTorch compatibility)"""
    if dist_is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            logging.warning(f"Barrier error (continuing anyway): {e}")

def rank0_print(*a, **k):
    if is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}]", *a, **k, flush=True)  # Force flush for better logging

def _cleanup_dist():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try: barrier()
            except Exception: pass
            torch.distributed.destroy_process_group()
    except Exception as e:
        print(f"Error during cleanup: {e}")
        pass
atexit.register(_cleanup_dist)

# Log config
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

def make_run_dir(base="runs_base_eval_improved", run_name=None):
    tag = run_name or f"{now_tag()}_base_eval_improved"
    path = os.path.join(base, tag)
    if is_main_process():
        os.makedirs(path, exist_ok=False)
    barrier()  # Wait for directory to be created
    return path

def save_json(path: str, obj: dict):
    if is_main_process():
        with open(path, "w") as f: json.dump(obj, f, indent=2)

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

# ----------------- ICD-9 Code Handling (Improved) -----------------
def format_icd9_properly(code: str) -> str:
    """Format ICD-9 codes with proper decimal point placement"""
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
    """Check if code follows valid ICD-9-CM format"""
    if not code: return False
    if code[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    if code.startswith('V'): return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    if code.startswith('E'): return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    return False

def get_icd9_parent(code: str) -> str:
    """Extract parent code (category level) from ICD-9 code"""
    if not code or len(code) < 3: return code
    if code[0].isdigit(): return code.split('.')[0][:3]
    if code.startswith('V'):
        base = code.split('.')[0]; return base[:3]
    if code.startswith('E'):
        base = code.split('.')[0]; return base[:4] if len(base) >= 4 else base
    return code

# ---------------- Improved lock_label_space function ----------------
def lock_label_space(frames: List[pd.DataFrame], label_col: str,
                     icd9_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
    """Create label space from training data, with option to use complete ICD-9 code set"""
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
    """Convert lists of ICD codes to multi-hot encoding, with proper formatting"""
    formatted_lists = []
    for row in lists:
        formatted_row = [format_icd9_properly(str(c)) for c in row]
        formatted_row = [c for c in formatted_row if is_valid_icd9(c)]
        formatted_lists.append(formatted_row)
    return mlb.transform(formatted_lists)

# ----------------- Hierarchical evaluation -----------------
def hierarchical_eval(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> Dict[str, float]:
    """Evaluate predictions with hierarchical metrics for ICD-9 codes"""
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

# ----------------- Optimized generation -----------------
class DoubleNewlineStop(StoppingCriteria):
    def __init__(self, tok, max_new_tokens: int, lookback: int = 96):
        self.tok = tok; self.max_new = max_new_tokens; self.lookback = lookback
    def __call__(self, input_ids, scores, **kwargs):
        last = input_ids[0].tolist()[-min(self.lookback, len(input_ids[0])):]
        tail = self.tok.decode(last, skip_special_tokens=True)
        return "\n\n" in tail or len(last) >= self.max_new

@torch.inference_mode()
def generate_texts(model, tok, prompts: List[str],
                   max_new=96, batch_size=8, max_len=3072,
                   stop_on_double_newline=True) -> List[str]:
    texts = []
    old_side = tok.padding_side
    tok.padding_side = "left"  # faster for varied prompt lengths
    stops = StoppingCriteriaList([DoubleNewlineStop(tok, max_new)]) if stop_on_double_newline else None
    
    # Get device from model
    device = next(model.parameters()).device
    rank = get_rank()
    
    # Process in batches with timing
    total_samples = len(prompts)
    log.info(f"Rank {rank}: Generating predictions for {total_samples} samples in batches of {batch_size}...")
    
    start_time = time.time()
    
    for i in range(0, total_samples, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        curr_batch_size = len(batch_prompts)
        
        # Log progress for tracking on all ranks
        if (i + curr_batch_size) % (5 * batch_size) == 0 or (i + curr_batch_size) >= total_samples:
            log.info(f"Rank {rank}: Generated {i + curr_batch_size}/{total_samples} samples ({(i + curr_batch_size)/total_samples:.1%})")
        
        # Tokenize and move to device
        try:
            inputs = tok(batch_prompts, return_tensors="pt", padding=True, 
                        truncation=True, max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with mixed precision if available
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False, 
                    num_beams=1,
                    no_repeat_ngram_size=2,
                    eos_token_id=tok.eos_token_id, 
                    pad_token_id=tok.pad_token_id,
                    use_cache=True, 
                    stopping_criteria=stops
                )
            
            batch_texts = tok.batch_decode(out, skip_special_tokens=True)
            texts.extend(batch_texts)
            
        except Exception as e:
            log.error(f"Error in batch {i}: {e}")
            # Add empty texts as fallback for this batch
            texts.extend(["[CODES] ERROR"] * curr_batch_size)
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and (i + curr_batch_size) % (10 * batch_size) == 0:
            torch.cuda.empty_cache()
            
    tok.padding_side = old_side
    
    # Log completion
    log.info(f"Rank {rank}: Generation completed, produced {len(texts)} texts")
    return texts

def parse_codes(text: str, labels_vocab: Optional[List[str]] = None) -> List[str]:
    """Extract and parse ICD codes from generated text with improved validation"""
    tail = text.split("[CODES]")[-1]
    tokens = re.split(r"[^A-Za-z0-9\.]+", tail)
    cand = [format_icd9_properly(z) for z in tokens if z]
    cand = [c for c in cand if is_valid_icd9(c)]  # Filter invalid codes
    allowed = set(labels_vocab) if labels_vocab is not None else None
    seen, keep = set(), []
    for c in cand:
        if (allowed is None or c in allowed) and c not in seen:
            seen.add(c); keep.append(c)
    return keep

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

def show_examples(df: pd.DataFrame, preds: List[List[str]], n=5, seed=42, show_prompts=False):
    """Show sample predictions with improved error analysis"""
    if not is_main_process():
        return
        
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n, len(df)), replace=False)
    for i in idxs:
        row = df.iloc[i]
        gold = [format_icd9_properly(str(c)) for c in row[LABEL_COL]]
        gold = sorted([c for c in gold if is_valid_icd9(c)])
        pred = preds[i]
        
        missing = sorted([c for c in gold if c not in pred])
        extra   = sorted([c for c in pred if c not in gold])

        gold_parents = {get_icd9_parent(c) for c in gold}
        pred_parents = {get_icd9_parent(c) for c in pred}
        parent_matches = sorted([f"{c} (parent)" for c in pred
                                if get_icd9_parent(c) in gold_parents and c not in gold])
        
        print("\n" + "="*80)
        print(f"idx={i} subject_id={row.get(SUBJECT_COL)} hadm_id={row.get('hadm_id')}")
        
        if show_prompts and "input_text" in row:
            print("\n----- PROMPT -----")
            print(row["input_text"])
            print("----- END PROMPT -----\n")
            
        print("- GOLD:", " ".join(gold) if gold else "(none)")
        print("- PRED:", " ".join(pred) if pred else "(none)")
        print(f"- FALSE NEGATIVES ({len(missing)}):", " ".join(missing) if missing else "(none)")
        print(f"- FALSE POSITIVES ({len(extra)}):", " ".join(extra) if extra else "(none)")
        print(f"- PARENT MATCHES ({len(parent_matches)}):", " ".join(parent_matches) if parent_matches else "(none)")

# ----------------- Distributed Data Handling -----------------
def shard_dataset(df: pd.DataFrame):
    """Split dataset for distributed evaluation"""
    if not dist_is_initialized():
        return df
        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Calculate shard indices
    total_size = len(df)
    shard_size = total_size // world_size
    remainder = total_size % world_size
    
    # Distribute remainder across first few ranks
    start_idx = rank * shard_size + min(rank, remainder)
    end_idx = start_idx + shard_size + (1 if rank < remainder else 0)
    
    # Get our shard
    shard = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    log.info(f"Rank {rank}/{world_size}: Processing {len(shard)}/{total_size} samples (indices {start_idx} to {end_idx-1})")
    return shard

def gather_predictions(local_preds):
    """Gather predictions from all processes safely without timeouts"""
    if not dist_is_initialized():
        return local_preds
        
    world_size = dist.get_world_size()
    rank = get_rank()
    
    log.info(f"Rank {rank}: Starting prediction gathering...")
    
    # Create a list to hold predictions from all processes
    all_preds = [None] * world_size
    
    # Gather all predictions with error handling
    try:
        # Sync first to ensure all processes are ready
        log.info(f"Rank {rank}: Entering barrier before gather...")
        barrier()
        
        log.info(f"Rank {rank}: Proceeding with all_gather_object...")
        dist.all_gather_object(all_preds, local_preds)
        log.info(f"Rank {rank}: Successfully completed all_gather_object")
    except Exception as e:
        log.error(f"Error during gather: {e}")
        
        # If we can't gather, return only local predictions and warn
        if is_main_process():
            log.warning("Failed to gather predictions from all processes. Using partial results.")
        return local_preds
    
    # Flatten the list of lists
    combined_preds = []
    for p in all_preds:
        if p is not None:  # Sanity check
            combined_preds.extend(p)
    
    log.info(f"Rank {rank}: Gathered {len(combined_preds)} total predictions from {world_size} processes")
    return combined_preds

# ----------------- Load model/tokenizer -----------------
def load_base(model_name: str):
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    else:
        dtype = torch.float32
    
    # Fixed kwargs - removed flash attention
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True
    }
    
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # we switch to left only during generation
    
    # Select appropriate device map based on DDP initialization
    if dist_is_initialized():
        # In DDP mode, each process loads model to its specific GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}
        load_kwargs["device_map"] = device_map
    else:
        # Auto-distribute in single process mode
        load_kwargs["device_map"] = "auto"
    
    log.info(f"Loading model with kwargs: {load_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = True
    
    # Log device placement
    if torch.cuda.is_available():
        device_id = next(model.parameters()).device.index
        log.info(f"Model loaded on device: {next(model.parameters()).device} (ID: {device_id})")
    
    return model, tok

# ----------------- Init DDP -----------------
def init_distributed():
    """Initialize distributed training with explicit device assignment"""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        log.info("Not running in distributed mode")
        return False
        
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size <= 1:
        log.info("Only one process found, not initializing distributed mode")
        return False
    
    if torch.cuda.device_count() < world_size:
        log.warning(f"Requested world_size={world_size} but only have {torch.cuda.device_count()} GPUs")
    
    # First set CUDA device - this must come before process group init
    log.info(f"Setting CUDA device {local_rank} for rank {rank}")
    torch.cuda.set_device(local_rank)
    
    # Initialize process group with specific device
    log.info(f"Initializing process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    init_method = "env://"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Set environment variables for better NCCL
    os.environ["NCCL_DEBUG"] = "WARN"
    
    # Initialize without timeout (older PyTorch compatibility)
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        log.info(f"Distributed initialization complete: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    except Exception as e:
        log.error(f"Failed to initialize process group: {e}")
        return False
    
    # Perform an initial barrier to check if everyone can communicate
    try:
        log.info("Testing communication with initial barrier...")
        dist.barrier()
        log.info("Initial barrier completed successfully - communication working")
    except Exception as e:
        log.error(f"Failed initial barrier test: {e}")
        try:
            dist.destroy_process_group()
        except:
            pass
        return False
    
    return True

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pickle", type=str, default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--run_name", type=str, default=None)

    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--gen_batch_size", type=int, default=16)  # Smaller default batch size for stability
    ap.add_argument("--stop_on_double_newline", action="store_true")
    ap.add_argument("--no_stop_on_double_newline", dest="stop_on_double_newline", action="store_false")
    ap.set_defaults(stop_on_double_newline=True)

    # Modalities: ON by default
    group_struct = ap.add_mutually_exclusive_group()
    group_struct.add_argument("--structured", dest="use_structured", action="store_true")
    group_struct.add_argument("--no-structured", dest="use_structured", action="store_false")
    group_notes = ap.add_mutually_exclusive_group()
    group_notes.add_argument("--notes", dest="use_notes", action="store_true")
    group_notes.add_argument("--no-notes", dest="use_notes", action="store_false")
    ap.set_defaults(use_structured=True, use_notes=True)

    ap.add_argument("--filter_mode", choices=["filtered","unfiltered","both"], default="both")
    ap.add_argument("--subset", type=int, default=None, help="limit #test samples for quick runs")
    ap.add_argument("--save_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true", help="try torch.compile for small extra speed")
    ap.add_argument("--show_prompts", action="store_true", help="show prompts in example outputs")
    
    # New options for ICD-9 code handling
    ap.add_argument("--icd9_pickle", default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl", 
                    help="Path to complete ICD-9 code list")
    ap.add_argument("--use_complete_icd9", type=int, default=1, help="Use complete ICD-9 code space")
    
    # Timeout handling options
    ap.add_argument("--operation_timeout", type=int, default=600,
                    help="Timeout in seconds for critical operations (default 10 minutes)")

    args = ap.parse_args()

    # Register timeout handler
    timeout_handler = DdpTimeoutHandler(timeout_seconds=args.operation_timeout)
    timeout_handler.register()
    
    try:
        # Initialize distributed processing
        is_distributed = init_distributed()
        barrier()  # Ensure all processes finish initialization

        set_seed(args.seed)
        rank0_print(f"CUDA: {torch.cuda.is_available()} | Model: {args.model}")
        if is_distributed:
            rank0_print(f"Running in distributed mode with {dist.get_world_size()} processes")
        
        rank0_print(f"Config -> MAX_LEN:{args.max_len} GEN_MAX_NEW:{args.gen_max_new} GEN_BS:{args.gen_batch_size} "
              f"STRUCTURED:{args.use_structured} NOTES:{args.use_notes} FILTER_MODE:{args.filter_mode}")

        # Data loading
        log.info(f"Loading data from {args.data_pickle}")
        df = pickle.load(open(args.data_pickle, "rb"))
        assert LABEL_COL in df.columns and SUBJECT_COL in df.columns
        df[LABEL_COL] = df[LABEL_COL].apply(to_list)
        
        # Create input texts
        log.info("Generating input texts")
        df["input_text"] = df.apply(lambda r: build_input_text(
            r, use_structured=args.use_structured, use_notes=args.use_notes), axis=1)
        df = df[df["input_text"].str.len() > 0].reset_index(drop=True)

        # Split datasets
        log.info("Splitting dataset")
        train_df, val_df, test_df = subject_splits(df, subject_col=SUBJECT_COL, seed=args.seed)
        
        # Set up label space - using the improved implementation
        log.info("Setting up label space")
        mlb = lock_label_space(
            [train_df, val_df, test_df], LABEL_COL,
            icd9_pkl_path=args.icd9_pickle, use_complete=bool(args.use_complete_icd9)
        )
        labels_vocab = mlb.classes_.tolist()
        
        # Create a copy of the full test set before sharding
        log.info("Preparing full test set")
        full_test_df = test_df.copy()
        y_test_full = y_multi_hot(mlb, full_test_df[LABEL_COL].tolist())
        
        if args.subset is not None:
            test_df = test_df.iloc[:min(args.subset, len(test_df))].copy()
            full_test_df = test_df.copy()  # Update full test too if using subset
            y_test_full = y_test_full[:len(test_df)]
            log.info(f"Using subset of {len(test_df)} test samples")

        # Run dir
        run_dir = make_run_dir(run_name=args.run_name)
        rank0_print("Run dir:", run_dir)
        save_json(os.path.join(run_dir, "config.json"), {
            **vars(args), "labels": len(labels_vocab), "test_size": len(test_df)
        })

        # Model loading - with explicit error handling
        log.info(f"Loading model: {args.model}")
        try:
            model, tok = load_base(args.model)
            if args.compile:
                try:
                    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                    rank0_print("torch.compile: enabled")
                except Exception as e:
                    rank0_print(f"torch.compile failed ({e}); continuing without it.")
            log.info("Model loaded successfully")
        except Exception as e:
            log.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

        # For tracking subject IDs along with predictions
        subject_ids = []
        
        # --------- Distributed data sharding if needed ---------
        if is_distributed:
            # Shard the test set for distributed processing
            orig_test_len = len(test_df)
            
            # Store subject IDs before sharding to help with reassembly
            test_df['orig_idx'] = range(len(test_df))
            
            # Shard the dataset
            test_df = shard_dataset(test_df)
            
            # Keep track of subject IDs and indices for proper reassembly later
            subject_ids = list(zip(test_df[SUBJECT_COL].tolist(), test_df['orig_idx'].tolist()))
            
            rank0_print(f"Sharded test data: {orig_test_len} total -> {len(test_df)} per process")
        
        # --------- Single-pass generation ---------
        log.info("Starting text generation")
        test_prompts = test_df["input_text"].astype(str).tolist()
        t0 = time.perf_counter()
        texts = generate_texts(
            model, tok, test_prompts,
            max_new=args.gen_max_new, batch_size=args.gen_batch_size,
            max_len=args.max_len, stop_on_double_newline=args.stop_on_double_newline
        )
        gen_secs = time.perf_counter() - t0
        rank0_print(f"[TIME] generation (single pass): {gen_secs/60:.2f} min for {len(test_prompts)} samples")

        # Process predictions locally
        log.info("Parsing codes")
        if args.filter_mode in ("filtered","both"):
            preds_f = [parse_codes(t, labels_vocab) for t in texts]
        if args.filter_mode in ("unfiltered","both"):
            preds_u = [parse_codes(t, None) for t in texts]
        
        # Create prediction packages with indices for reassembly
        pred_packages = []
        if is_distributed:
            if args.filter_mode == "filtered":
                pred_packages = list(zip(subject_ids, preds_f))
            elif args.filter_mode == "unfiltered": 
                pred_packages = list(zip(subject_ids, preds_u))
            else:  # both
                pred_packages = list(zip(subject_ids, preds_f, preds_u))
        
        # --------- Gather predictions from all processes if distributed ---------
        if is_distributed:
            # First gather the pred packages from all processes
            log.info("Starting gather operation")
            combined_packages = gather_predictions(pred_packages)
            
            barrier()  # Ensure all processes finish gather
            
            if not is_main_process():
                # Non-main processes don't need to continue
                log.info("Non-main process finished its work")
                return
                
            # Main process reassembles predictions in the correct order
            log.info(f"Main process reassembling predictions from {len(combined_packages)} entries")
            
            if args.filter_mode == "filtered":
                # Extract and sort by original index
                sorted_packages = sorted(combined_packages, key=lambda x: x[0][1])
                # Extract just the predictions
                preds_f = [p[1] for p in sorted_packages]
                
            elif args.filter_mode == "unfiltered":
                # Extract and sort by original index
                sorted_packages = sorted(combined_packages, key=lambda x: x[0][1])
                # Extract just the predictions
                preds_u = [p[1] for p in sorted_packages]
                
            else:  # both
                # Extract and sort by original index
                sorted_packages = sorted(combined_packages, key=lambda x: x[0][1])
                # Extract both filtered and unfiltered predictions
                preds_f = [p[1] for p in sorted_packages]
                preds_u = [p[2] for p in sorted_packages]
                
            # Use the full test set and labels for evaluation
            test_df = full_test_df
            y_test = y_test_full
            
        else:
            # In single process mode, just use the normal test set
            y_test = y_test_full
            
        results = {}

        # Parse/score: filtered
        if args.filter_mode in ("filtered","both"):
            log.info("Evaluating filtered predictions")
            Yf = codes_to_multihot(preds_f, labels_vocab)
            m_f = eval_sets(y_test, Yf)
            # Add hierarchical evaluation with the improved implementation
            m_f.update(hierarchical_eval(y_test, Yf, labels_vocab))
            m_f["gen_seconds"] = gen_secs
            results["filtered"] = m_f
            save_json(os.path.join(run_dir, "metrics_filtered.json"), m_f)
            rank0_print("\n=== BASE metrics (STRICT VOCAB FILTER) ===")
            rank0_print(json.dumps(m_f, indent=2))
            rank0_print("\n--- Sample predictions (filtered) ---")
            show_examples(test_df, preds_f, n=args.save_samples, seed=args.seed, show_prompts=args.show_prompts)

        # Parse/score: unfiltered
        if args.filter_mode in ("unfiltered","both"):
            log.info("Evaluating unfiltered predictions")
            Yu = codes_to_multihot(preds_u, labels_vocab)  # off-vocab codes drop out naturally
            m_u = eval_sets(y_test, Yu)
            # Add hierarchical evaluation with the improved implementation
            m_u.update(hierarchical_eval(y_test, Yu, labels_vocab))
            m_u["gen_seconds"] = gen_secs
            results["unfiltered"] = m_u
            save_json(os.path.join(run_dir, "metrics_unfiltered.json"), m_u)
            rank0_print("\n=== BASE metrics (NO FILTER) ===")
            rank0_print(json.dumps(m_u, indent=2))
            rank0_print("\n--- Sample predictions (unfiltered) ---")
            show_examples(test_df, preds_u, n=args.save_samples, seed=args.seed, show_prompts=args.show_prompts)

        save_json(os.path.join(run_dir, "metrics_summary.json"), results)
        rank0_print(f"\nDone. Results saved under: {run_dir}")
        
    except Exception as e:
        log.error(f"Error in main: {e}", exc_info=True)
        raise
    finally:
        # Cancel timeout handler
        timeout_handler.cancel()
        
        # Always clean up distributed resources
        if dist_is_initialized():
            try:
                log.info("Cleaning up distributed resources")
                dist.destroy_process_group()
            except:
                pass

if __name__ == "__main__":
    main()