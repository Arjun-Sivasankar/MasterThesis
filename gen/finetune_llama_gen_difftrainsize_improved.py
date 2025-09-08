# # -*- coding: utf-8 -*-
# """
# Simple finetuning script with a single flag to control TRAIN SIZE.

# - Subject-safe subsetting of the train split with --train_size and --seed
# - Keeps your original prompt style and LoRA setup
# - Enhanced ICD-9 formatting with proper decimal points
# - Option to use complete ICD-9 code label space from reference file
# - Minimal eval: val loss for early stop; final TEST generation with vocab filter

# Example run (change --train_size per job on HPC):

# python finetune_llama_gen_difftrainsize.py \
#   --train_pickle /path/train_62k.pkl \
#   --val_pickle /path/val_7k.pkl \
#   --test_pickle /path/test_7k.pkl \
#   --icd9_pickle MasterThesis/dataset/codes/icd9.pkl \
#   --train_size 10000 \
#   --epochs 3 --learning_rate 2e-4 \
#   --run_root runs_simple

# Set --train_size to any integer; if it exceeds the available rows, full train is used.
# If you only have a single merged pickle, pass --data_pickle instead of the three files.
# """

# import os, re, json, random, logging, pickle, datetime, time, atexit, math, argparse
# from typing import List, Any, Dict
# import numpy as np
# import pandas as pd

# import torch
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score, precision_score, recall_score

# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM,
#     TrainingArguments, EarlyStoppingCallback, Trainer
# )
# from transformers.utils import logging as hf_logging
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# # ---------------- Env & logging ----------------
# os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# hf_logging.set_verbosity_error()

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# def set_seed(seed=42):
#     random.seed(seed); np.random.seed(seed)
#     torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# print("CUDA:", torch.cuda.is_available(),
#       "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# # Enable TF32 on Ampere+ (A100/H100, etc.)
# if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     try: torch.set_float32_matmul_precision("high")
#     except Exception: pass


# def _cleanup_dist():
#     try:
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             try: torch.distributed.barrier()
#             except Exception: pass
#             torch.distributed.destroy_process_group()
#     except Exception:
#         pass
# atexit.register(_cleanup_dist)

# # ---------------- Args ----------------

# def get_args():
#     ap = argparse.ArgumentParser()
#     # data
#     ap.add_argument("--data_pickle", default=None, help="If provided, will subject-split into train/val/test.")
#     ap.add_argument("--train_pickle", default=None)
#     ap.add_argument("--val_pickle", default=None)
#     ap.add_argument("--test_pickle", default=None)
#     ap.add_argument("--icd9_pickle", default="MasterThesis/dataset/codes/icd9.pkl", 
#                     help="Path to complete ICD-9 code list")
#     ap.add_argument("--subject_col", default="subject_id_x")
#     ap.add_argument("--label_col", default="icd_code")
#     ap.add_argument("--use_complete_icd9", type=int, default=1,
#                     help="Use the complete ICD-9 reference (1) or only codes from training (0)")

#     # model/prompt
#     ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
#     ap.add_argument("--use_structured", type=int, default=1)
#     ap.add_argument("--use_notes", type=int, default=1)
#     ap.add_argument("--max_len", type=int, default=3072)
#     ap.add_argument("--tgt_reserve_tok", type=int, default=128)
#     ap.add_argument("--gen_max_new", type=int, default=96)

#     # training
#     ap.add_argument("--epochs", type=int, default=6)
#     ap.add_argument("--per_device_train_batch_size", type=int, default=1)
#     ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
#     ap.add_argument("--grad_accum", type=int, default=16)
#     ap.add_argument("--learning_rate", type=float, default=2e-4)
#     ap.add_argument("--weight_decay", type=float, default=0.0)
#     ap.add_argument("--warmup_ratio", type=float, default=0.03)
#     ap.add_argument("--early_stop", type=int, default=1)
#     ap.add_argument("--patience", type=int, default=2)

#     # size & seed
#     ap.add_argument("--train_size", type=int, default=-1, help="Number of training rows to use (subject-safe subset). -1=all")
#     ap.add_argument("--seed", type=int, default=42)

#     # run dirs
#     ap.add_argument("--run_root", default="runs_gen/diffsize")
#     ap.add_argument("--run_name", default=None)

#     # misc
#     ap.add_argument("--compile", type=int, default=0)
#     ap.add_argument("--merge_after", type=int, default=0)

#     return ap.parse_args()

# # ---------------- ICD-9 Code Handling ----------------

# def format_icd9_properly(code: str) -> str:
#     """Format ICD-9 code with proper decimal placement."""
#     # Basic cleaning
#     code = code.strip().upper()
#     code = re.sub(r"\s+", "", code)
    
#     # Remove trailing period if any
#     if code.endswith("."):
#         code = code[:-1]
        
#     # Handle decimal point formatting for diagnosis codes
#     if code and code[0].isdigit():
#         # Regular ICD-9 diagnosis codes
#         if '.' not in code and len(code) > 3:
#             return code[:3] + '.' + code[3:]
    
#     # Handle V and E codes and procedure codes
#     elif code and len(code) > 1:
#         if code[0] in ('V', 'E') and '.' not in code and len(code) > 3:
#             return code[:3] + '.' + code[3:]
    
#     return code

# def is_valid_icd9(code: str) -> str:
#     """Validate if a string follows ICD-9-CM format patterns."""
#     # If empty, not valid
#     if not code:
#         return False
    
#     # Regular diagnosis codes (3 digits + optional 1-2 decimals)
#     if code[0].isdigit():
#         return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    
#     # V codes (V + 2 digits + optional 1-2 decimals)
#     elif code.startswith('V'):
#         return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    
#     # E codes (E + 3 digits + optional 1 decimal)
#     elif code.startswith('E'):
#         return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    
#     return False

# def normalize_code(c: str) -> str:
#     """Normalize ICD code with format validation."""
#     return format_icd9_properly(c)

# def get_icd9_parent(code: str) -> str:
#     """Get parent code (category) from an ICD-9 code."""
#     if not code or len(code) < 3:
#         return code
    
#     # Regular codes - first 3 digits
#     if code[0].isdigit():
#         return code.split('.')[0][:3]
    
#     # V codes - V + first 2 digits
#     elif code.startswith('V'):
#         base = code.split('.')[0]
#         return base[:3]
    
#     # E codes - E + first 3 digits
#     elif code.startswith('E'):
#         base = code.split('.')[0]
#         return base[:4] if len(base) >= 4 else base
    
#     return code

# # ---------------- Prompting helpers ----------------

# TEXT_COLS_SAFE = [
#     "Chief Complaint","History of Present Illness","Past Medical History",
#     "Family History","Physical Exam","Pertinent Results",
#     "Brief Hospital Course","Medications on Admission"
# ]


# def clean_text(x: Any) -> str:
#     if pd.isna(x): return ""
#     s = str(x).replace("\x00"," ").replace("\r"," ")
#     s = re.sub(r"_+"," ", s)
#     return re.sub(r"\s+"," ", s).strip()


# def to_list(x) -> List[str]:
#     if isinstance(x, list): return [str(v) for v in x]
#     if pd.isna(x): return []
#     s = str(x).strip()
#     if s.startswith("[") and s.endswith("]"):
#         try:
#             import ast; v = ast.literal_eval(s)
#             if isinstance(v, list): return [str(z) for z in v]
#         except Exception: pass
#     return [t for t in re.split(r"[,\s]+", s) if t]


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


# def serialize_notes(row: pd.Series) -> str:
#     chunks=[]
#     for col in TEXT_COLS_SAFE:
#         if col in row:
#             t = clean_text(row[col])
#             if t: chunks.append(f"[{col.upper()}] {t}")
#     return "\n".join(chunks)


# def build_input_text(row: pd.Series, use_structured=True, use_notes=True,
#                      subject_col="subject_id_x") -> str:
#     s = [f"[VISIT] subject_id={row.get(subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
#     if use_structured: s.append(serialize_structured(row))
#     if use_notes:
#         t = serialize_notes(row)
#         if t: s.append(t)
    
#     # Improved task prompt with more explicit instructions
#     s.append("[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:")
#     s.append("1. List only the ICD-9 codes separated by spaces")
#     s.append("2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)")
#     s.append("3. Include only codes directly supported by the clinical information")
#     s.append("4. Do not include any explanations or text besides the codes themselves")
#     s.append("[CODES]")
    
#     return "\n".join([x for x in s if x])

# # ---------------- Splits & labels ----------------

# def subject_splits(df: pd.DataFrame, subject_col: str,
#                    test_size=0.10, val_size=0.10, seed=42):
#     subs = df[subject_col].dropna().unique()
#     train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
#     train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
#     tr = df[df[subject_col].isin(train_subs)].copy()
#     va = df[df[subject_col].isin(val_subs)].copy()
#     te = df[df[subject_col].isin(test_subs)].copy()
#     logging.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
#     return tr, va, te


# def lock_label_space(frames: List[pd.DataFrame], label_col: str, 
#                      icd9_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
#     """Create label space from training data or complete ICD-9 code reference."""
    
#     # Get codes from training data with proper formatting
#     train_codes = set()
#     for fr in frames:
#         for codes in fr[label_col]:
#             train_codes.update(format_icd9_properly(str(c)) for c in codes)
    
#     # Filter out invalid codes
#     train_codes = {c for c in train_codes if is_valid_icd9(c)}
#     logging.info(f"Found {len(train_codes)} unique valid ICD codes in training data")
    
#     # If not using complete reference, just use the training codes
#     if not use_complete or not icd9_pkl_path:
#         all_codes = sorted(train_codes)
#         mlb = MultiLabelBinarizer(classes=all_codes)
#         mlb.fit([all_codes])
#         logging.info(f"Using {len(all_codes)} codes from training data only")
#         return mlb
    
#     # Try to load the complete ICD-9 code set
#     try:
#         icd9_df = pd.read_pickle(icd9_pkl_path)
#         complete_codes = sorted(icd9_df['icd_code'].astype(str).tolist())
        
#         # Format the complete codes list with proper decimal points
#         complete_codes = [format_icd9_properly(code) for code in complete_codes]
        
#         # Filter to only include valid codes
#         complete_codes = [code for code in complete_codes if is_valid_icd9(code)]
        
#         logging.info(f"Loaded {len(complete_codes)} complete ICD-9 codes from {icd9_pkl_path}")
        
#         # Create and fit the MultiLabelBinarizer with the complete code list
#         mlb = MultiLabelBinarizer(classes=complete_codes)
#         mlb.fit([complete_codes])
        
#         # Check how many codes from training data are in complete set
#         codes_in_complete = sum(1 for c in train_codes if c in set(complete_codes))
#         codes_not_in_complete = len(train_codes) - codes_in_complete
        
#         logging.info(f"Training data coverage:")
#         logging.info(f"- {codes_in_complete} codes present in complete ICD-9 set")
#         logging.info(f"- {codes_not_in_complete} codes not in complete ICD-9 set")
        
#         if codes_not_in_complete > 0:
#             logging.warning(f"Some codes in training data not found in complete ICD-9 set!")
#             missing_codes = sorted(c for c in train_codes if c not in set(complete_codes))
#             logging.debug(f"First 10 missing codes: {missing_codes[:10]}")
            
#         return mlb
        
#     except Exception as e:
#         logging.error(f"Error loading complete ICD-9 codes: {e}")
#         logging.warning("Falling back to training-data-only label space")
        
#         # Fallback to original approach
#         all_codes = sorted(train_codes)
#         mlb = MultiLabelBinarizer(classes=all_codes)
#         mlb.fit([all_codes])
#         logging.info(f"Using {len(all_codes)} codes from training data")
#         return mlb


# def y_multi_hot(mlb: MultiLabelBinarizer, lists):
#     """Convert lists of ICD codes to multi-hot vectors with proper formatting."""
#     # Format the codes consistently before conversion to multi-hot
#     formatted_lists = []
#     for row in lists:
#         formatted_row = [format_icd9_properly(str(c)) for c in row]
#         # Only keep valid codes
#         formatted_row = [c for c in formatted_row if is_valid_icd9(c)]
#         formatted_lists.append(formatted_row)
    
#     return mlb.transform(formatted_lists)

# # ---------------- Subject-safe subsetting ----------------

# def nested_subject_sample(train_df, target_n, subject_col="subject_id_x", seed=13):
#     if target_n is None or target_n < 0 or target_n >= len(train_df):
#         return train_df.copy()
#     rng = np.random.default_rng(seed)
#     # subjects = train_df[subject_col].drop_duplicates().tolist()
#     subjects = train_df[subject_col].tolist()
#     rng.shuffle(subjects)
#     chosen, count = [], 0
#     for s in subjects:
#         rows = train_df[train_df[subject_col] == s]
#         if count + len(rows) <= target_n or len(chosen) == 0:
#             chosen.append(s)
#             count += len(rows)
#         if count >= target_n: break
#     sub = train_df[train_df[subject_col].isin(chosen)].copy()
#     logging.info(f"[subset] requested={target_n} got={len(sub)} unique_subjects={sub[subject_col].nunique()}")
#     return sub

# # ---------------- Dataset (pre-tokenized) ----------------

# class GenCodesDataset(Dataset):
#     def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int, label_col: str):
#         self.tok = tok
#         self.max_len = max_len
#         self.tgt_reserve = max(8, int(tgt_reserve))
#         self.label_col = label_col
        
#         prompts = rows["input_text"].astype(str).tolist()
        
#         # Format ICD codes consistently in training targets
#         targets = []
#         for codes in rows[label_col].tolist():
#             # Format each code properly with decimal points
#             formatted_codes = [format_icd9_properly(str(c)) for c in codes]
#             # Keep only valid ICD-9 codes
#             formatted_codes = [c for c in formatted_codes if is_valid_icd9(c)]
#             # Join with spaces to create the target string
#             targets.append(" ".join(sorted(set(formatted_codes))))
        
#         self.prompt_ids = [tok.encode(p + "\n", add_special_tokens=True) for p in prompts]
#         eos = (tok.eos_token or "")
#         self.ans_ids    = [tok.encode(t + eos, add_special_tokens=False) for t in targets]

#     def __len__(self): return len(self.prompt_ids)

#     def __getitem__(self, i):
#         prompt_ids = self.prompt_ids[i]
#         ans_ids    = self.ans_ids[i]
#         max_prompt_len = max(1, self.max_len - self.tgt_reserve)
#         if len(prompt_ids) > max_prompt_len:
#             prompt_ids = prompt_ids[:max_prompt_len]
#         remaining = max(1, self.max_len - len(prompt_ids))
#         if len(ans_ids) > remaining:
#             ans_ids = ans_ids[:remaining]
#         input_ids = prompt_ids + ans_ids
#         attention_mask = [1] * len(input_ids)
#         labels = ([-100] * len(prompt_ids)) + ans_ids
#         return {
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#             "labels": torch.tensor(labels, dtype=torch.long),
#         }


# def pad_collate(features, tok):
#     input_ids = [f["input_ids"] for f in features]
#     attn     = [f["attention_mask"] for f in features]
#     labels   = [f["labels"] for f in features]
#     pad_out  = tok.pad({"input_ids": input_ids, "attention_mask": attn}, return_tensors="pt")
#     max_len = pad_out["input_ids"].size(1)
#     lab_pad = torch.full((len(labels), max_len), -100, dtype=torch.long)
#     for i, lab in enumerate(labels):
#         lab_pad[i, :lab.size(0)] = lab
#     return {"input_ids": pad_out["input_ids"], "attention_mask": pad_out["attention_mask"], "labels": lab_pad}

# # ---------------- Model ----------------

# def load_lm_and_tokenizer(model_name):
#     if torch.cuda.is_available():
#         use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
#         dtype = torch.bfloat16 if use_bf16 else torch.float16
#     else:
#         dtype = torch.float32
#     tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     if tok.pad_token is None: tok.pad_token = tok.eos_token
#     tok.padding_side = "right"
#     base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
#     base.config.pad_token_id = tok.pad_token_id
#     base.config.use_cache = False
#     base = prepare_model_for_kbit_training(base)
#     lora_cfg = LoraConfig(
#         r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
#         target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
#     )
#     model = get_peft_model(base, lora_cfg)
#     if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
#     model.print_trainable_parameters()
#     return model, tok

# # ---------------- Generation & metrics ----------------

# @torch.no_grad()
# def generate_codes(model, tok, prompts: List[str], labels_vocab: List[str],
#                    max_new=96, batch_size=4, max_len=3072):
#     """Greedy decoding + strict vocab filter against label space with ICD-9 validation."""
#     model.eval()
#     allowed = set(labels_vocab)
#     preds = []
#     for i in range(0, len(prompts), batch_size):
#         batch_prompts = prompts[i:i+batch_size]
#         inputs = tok(batch_prompts, return_tensors="pt",
#                      padding=True, truncation=True, max_length=max_len).to(model.device)
#         out = model.generate(
#             **inputs,
#             max_new_tokens=max_new,
#             do_sample=False,
#             num_beams=1,
#             no_repeat_ngram_size=2,
#             eos_token_id=tok.eos_token_id,
#             pad_token_id=tok.pad_token_id,
#             return_dict_in_generate=True,
#             output_scores=False,
#         )
#         seq = out.sequences
#         gen_only = seq[:, inputs["input_ids"].shape[1]:]  # only new tokens
#         texts = tok.batch_decode(gen_only, skip_special_tokens=True)
#         for t in texts:
#             tokens = re.split(r"[^A-Za-z0-9\.]+", t)
#             cand = [normalize_code(z) for z in tokens if z]
            
#             # Keep only codes that:
#             # 1. Are seen in training label space
#             # 2. Pass ICD-9 format validation 
#             # 3. Haven't been seen before (de-dupe)
#             seen, keep = set(), []
#             for c in cand:
#                 if c in allowed and is_valid_icd9(c) and c not in seen:
#                     seen.add(c)
#                     keep.append(c)
#             preds.append(keep)
#     return preds


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
#         "micro_recall":      recall_score(y_true, y_pred, average="micro",   zero_division=0),
#         "macro_recall":      recall_score(y_true, y_pred, average="macro",   zero_division=0),
#         "samples_recall":    recall_score(y_true, y_pred, average="samples", zero_division=0),
#     }


# def hierarchical_eval(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> Dict[str, float]:
#     """Evaluate predictions with hierarchical consideration of ICD-9 codes."""
#     # Get standard metrics first
#     std_metrics = {}
    
#     # Build a mapping of codes to their parent categories
#     code_to_parent = {code: get_icd9_parent(code) for code in label_vocab}
#     parent_to_idx = {}
#     for idx, code in enumerate(label_vocab):
#         parent = code_to_parent[code]
#         if parent not in parent_to_idx:
#             parent_to_idx[parent] = []
#         parent_to_idx[parent].append(idx)
    
#     # For each sample, check hierarchical matches
#     n_samples = y_true.shape[0]
#     parent_hits = 0
#     partial_matches = 0
#     total_true_parents = 0
    
#     for i in range(n_samples):
#         # Find predicted categories that match actual categories at parent level
#         pred_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_pred[i])[0]}
#         true_parents = {code_to_parent[label_vocab[j]] for j in np.where(y_true[i])[0]}
        
#         # Count parent-level hits
#         parent_hits += len(pred_parents & true_parents)
#         total_true_parents += len(true_parents)
        
#         # Count predictions with correct parent but wrong specific code
#         for parent in pred_parents:
#             if parent in true_parents:
#                 # Get all child indices for this parent
#                 child_indices = parent_to_idx.get(parent, [])
                
#                 # Check if any child codes match exactly
#                 exact_match = False
#                 for idx in child_indices:
#                     if y_true[i, idx] == 1 and y_pred[i, idx] == 1:
#                         exact_match = True
#                         break
                
#                 if not exact_match:
#                     partial_matches += 1
    
#     # Calculate hierarchical metrics
#     parent_recall = total_true_parents > 0 and parent_hits / total_true_parents or 0
    
#     # Add hierarchical metrics
#     std_metrics["hierarchical_parent_recall"] = parent_recall
#     std_metrics["hierarchical_partial_matches"] = partial_matches
#     std_metrics["hierarchical_partial_per_sample"] = partial_matches / n_samples if n_samples > 0 else 0
    
#     return std_metrics

# # ---------------- Run helpers ----------------

# def now_tag():
#     return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# def make_run_dir(base, tag):
#     path = os.path.join(base, tag)
#     os.makedirs(path, exist_ok=False)
#     os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
#     return path


# def save_json(path: str, obj: dict):
#     with open(path, "w") as f: json.dump(obj, f, indent=2)


# def show_test_predictions(df: pd.DataFrame,
#                           preds: List[List[str]],
#                           label_col: str,
#                           n_show: int = 5,
#                           seed: int = 0):
#     """Display example predictions with properly formatted ICD-9 codes."""
#     rng = np.random.default_rng(seed)
#     idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
#     for i in idxs:
#         row = df.iloc[i]
#         # Format gold codes consistently
#         gold = sorted({format_icd9_properly(str(c)) for c in row[label_col] 
#                       if is_valid_icd9(format_icd9_properly(str(c)))})
#         pred = preds[i]
#         missing = sorted([c for c in gold if c not in pred])  # FN
#         extra   = sorted([c for c in pred if c not in gold])  # FP
        
#         # Check for parent-level matches (partial matches)
#         gold_parents = {get_icd9_parent(c) for c in gold}
#         pred_parents = {get_icd9_parent(c) for c in pred}
#         parent_matches = sorted([f"{c} (parent)" for c in pred 
#                               if get_icd9_parent(c) in gold_parents and c not in gold])
        
#         print("\n" + "="*80)
#         print(f"Example {i+1}:")
#         print("- GOLD:", " ".join(gold) if gold else "(none)")
#         print("- PRED:", " ".join(pred) if pred else "(none)")
#         print(f"- FALSE NEGATIVES ({len(missing)}):", " ".join(missing) if missing else "(none)")
#         print(f"- FALSE POSITIVES ({len(extra)}):", " ".join(extra) if extra else "(none)")
#         print(f"- PARENT MATCHES ({len(parent_matches)}):", " ".join(parent_matches) if parent_matches else "(none)")

# # ---------------- Main ----------------

# def main():
#     args = get_args()
#     set_seed(args.seed)

#     # Load data
#     if args.train_pickle and args.val_pickle and args.test_pickle:
#         train_df = pickle.load(open(args.train_pickle, "rb"))
#         val_df   = pickle.load(open(args.val_pickle, "rb"))
#         test_df  = pickle.load(open(args.test_pickle, "rb"))
#     elif args.data_pickle:
#         full_df = pickle.load(open(args.data_pickle, "rb"))
#         train_df, val_df, test_df = subject_splits(full_df, subject_col=args.subject_col, test_size=0.10, val_size=0.10, seed=args.seed)
#     else:
#         raise ValueError("Provide either --data_pickle OR all of --train_pickle/--val_pickle/--test_pickle")

#     for df_ in (train_df, val_df, test_df):
#         assert args.label_col in df_.columns and args.subject_col in df_.columns

#     # Subject-safe subsetting of TRAIN
#     train_df = nested_subject_sample(train_df, args.train_size, subject_col=args.subject_col, seed=args.seed)

#     # Build prompts with improved instructions
#     for df_, name in ((train_df, 'train'), (val_df, 'val'), (test_df, 'test')):
#         df_["input_text"] = df_.apply(lambda r: build_input_text(r, args.use_structured==1, args.use_notes==1, args.subject_col), axis=1)
#         logging.info(f"[{name}] rows with input_text: {df_['input_text'].notna().sum()}")

#     # Label space - either from training data only or complete ICD-9 reference
#     mlb = lock_label_space(
#         [train_df, val_df, test_df], 
#         args.label_col, 
#         icd9_pkl_path=args.icd9_pickle, 
#         use_complete=bool(args.use_complete_icd9)
#     )
    
#     labels_vocab = mlb.classes_.tolist()
    
#     # Convert labels to multi-hot vectors with proper formatting
#     y_val  = y_multi_hot(mlb, val_df[args.label_col].tolist())
#     y_test = y_multi_hot(mlb, test_df[args.label_col].tolist())

#     # Model & tokenizer
#     model, tok = load_lm_and_tokenizer(args.llama_model)
#     if args.compile:
#         try:
#             model = torch.compile(model)
#         except Exception as e:
#             logging.warning(f"torch.compile failed: {e}")

#     # Datasets with properly formatted ICD-9 codes
#     train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col)
#     val_ds   = GenCodesDataset(val_df,   tok, args.max_len, args.tgt_reserve_tok, args.label_col)

#     # Run dir
#     size_str = f"N{args.train_size}" if args.train_size > 0 else "full"
#     tag = args.run_name or f"{now_tag()}_{size_str}_icd9"
#     RUN_DIR = make_run_dir(args.run_root, tag)
#     logging.info(f"Run dir: {RUN_DIR}")

#     save_json(os.path.join(RUN_DIR, "config.json"), {
#         "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate,
#         "epochs": args.epochs, "gen_max_new": args.gen_max_new, "tgt_reserve_tok": args.tgt_reserve_tok,
#         "seed": args.seed, "train_rows": len(train_df),
#         "icd9_pickle": args.icd9_pickle, 
#         "use_complete_icd9": bool(args.use_complete_icd9),
#         "total_label_space": len(labels_vocab)
#     })
#     save_json(os.path.join(RUN_DIR, "label_space.json"), {"labels": labels_vocab})

#     # Training args (simple)
#     train_args = TrainingArguments(
#         output_dir=os.path.join(RUN_DIR, "checkpoints"),
#         num_train_epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         per_device_train_batch_size=args.per_device_train_batch_size,
#         per_device_eval_batch_size=args.per_device_eval_batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
#         logging_strategy="epoch",
#         eval_strategy="epoch",
#         prediction_loss_only=True,
#         save_strategy="epoch",
#         save_total_limit=1,
#         load_best_model_at_end=bool(args.early_stop),
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#         report_to="none",
#         gradient_checkpointing=True,
#         remove_unused_columns=False,
#         fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
#         bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
#         optim="adamw_torch", dataloader_num_workers=2,
#         run_name=os.path.basename(RUN_DIR),
#         disable_tqdm=True,
#     )

#     callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.early_stop else []

#     trainer = Trainer(
#         model=model,
#         args=train_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         tokenizer=tok,
#         data_collator=lambda feats: pad_collate(feats, tok),
#         callbacks=callbacks
#     )

#     # Train
#     t0 = time.perf_counter()
#     logging.info("Starting training…")
#     trainer.train()
#     train_secs = time.perf_counter() - t0
#     logging.info(f"[TIME] train: {train_secs:.2f}s")

#     # Save adapter
#     tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
#     trainer.model.save_pretrained(os.path.join(RUN_DIR, "adapter_best"))

#     if args.merge_after:
#         try:
#             merged_dir = os.path.join(RUN_DIR, "model_merged")
#             merged = trainer.model.merge_and_unload()
#             merged.save_pretrained(merged_dir)
#             tok.save_pretrained(os.path.join(merged_dir, "tokenizer"))
#             logging.info(f"Merged model saved to: {merged_dir}")
#         except Exception as e:
#             logging.warning(f"Could not merge adapters into base: {e}")

#     # Final TEST generation
#     test_prompts = test_df["input_text"].astype(str).tolist()
#     t_gen = time.perf_counter()
#     pred_code_lists = generate_codes(
#         model, tok, test_prompts, labels_vocab,
#         max_new=args.gen_max_new, batch_size=args.per_device_eval_batch_size, max_len=args.max_len
#     )
#     test_gen_secs = time.perf_counter() - t_gen

#     Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
#     metrics = eval_sets(y_test, Y_pred)
    
#     # Add hierarchical evaluation metrics
#     hier_metrics = hierarchical_eval(y_test, Y_pred, labels_vocab)
#     metrics.update(hier_metrics)
    
#     metrics["train_seconds"] = train_secs
#     metrics["test_generate_seconds"] = test_gen_secs

#     with open(os.path.join(RUN_DIR, "test_metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=2)

#     print("\n=== Generative TEST metrics ===")
#     print(json.dumps(metrics, indent=2))
    
#     # Show sample predictions with proper ICD-9 formatting
#     print("\n=== Sample Predictions ===")
#     show_test_predictions(test_df, pred_code_lists, args.label_col, n_show=5, seed=args.seed)


# if __name__ == "__main__":
#     main()

import os, re, json, random, logging, pickle, datetime, time, atexit, math, argparse
from typing import List, Any, Dict
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, EarlyStoppingCallback, Trainer, TrainerCallback
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- Env & logging ----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


print("CUDA:", torch.cuda.is_available(),
      "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Enable TF32 on Ampere+ (A100/H100, etc.)
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass


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
    ap.add_argument("--icd9_pickle", default="MasterThesis/dataset/codes/icd9.pkl", 
                    help="Path to complete ICD-9 code list")
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--use_complete_icd9", type=int, default=1,
                    help="Use the complete ICD-9 reference (1) or only codes from training (0)")

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

    # size & seed
    ap.add_argument("--train_size", type=int, default=-1, help="Number of training rows to use (subject-safe subset). -1=all")
    ap.add_argument("--seed", type=int, default=42)

    # run dirs
    ap.add_argument("--run_root", default="runs_gen/diffsize")
    ap.add_argument("--run_name", default=None)

    # eval reporting
    ap.add_argument("--eval_sample_size", type=int, default=100, help="Number of samples to evaluate during training for metrics")
    ap.add_argument("--test_examples", type=int, default=5, help="Number of test examples to show")

    # misc
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--merge_after", type=int, default=0)

    return ap.parse_args()

# ---------------- ICD-9 Code Handling ----------------

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
    
    # Improved task prompt with more explicit instructions
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
    logging.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te


def lock_label_space(frames: List[pd.DataFrame], label_col: str, 
                     icd9_pkl_path: str = None, use_complete: bool = False) -> MultiLabelBinarizer:
    """Create label space from training data or complete ICD-9 code reference."""
    
    # Get codes from training data with proper formatting
    train_codes = set()
    for fr in frames:
        for codes in fr[label_col]:
            train_codes.update(format_icd9_properly(str(c)) for c in codes)
    
    # Filter out invalid codes
    train_codes = {c for c in train_codes if is_valid_icd9(c)}
    logging.info(f"Found {len(train_codes)} unique valid ICD codes in training data")
    
    # If not using complete reference, just use the training codes
    if not use_complete or not icd9_pkl_path:
        all_codes = sorted(train_codes)
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        logging.info(f"Using {len(all_codes)} codes from training data only")
        return mlb
    
    # Try to load the complete ICD-9 code set
    try:
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
        
        # Check how many codes from training data are in complete set
        codes_in_complete = sum(1 for c in train_codes if c in set(complete_codes))
        codes_not_in_complete = len(train_codes) - codes_in_complete
        
        logging.info(f"Training data coverage:")
        logging.info(f"- {codes_in_complete} codes present in complete ICD-9 set")
        logging.info(f"- {codes_not_in_complete} codes not in complete ICD-9 set")
        
        if codes_not_in_complete > 0:
            logging.warning(f"Some codes in training data not found in complete ICD-9 set!")
            missing_codes = sorted(c for c in train_codes if c not in set(complete_codes))
            logging.debug(f"First 10 missing codes: {missing_codes[:10]}")
            
        return mlb
        
    except Exception as e:
        logging.error(f"Error loading complete ICD-9 codes: {e}")
        logging.warning("Falling back to training-data-only label space")
        
        # Fallback to original approach
        all_codes = sorted(train_codes)
        mlb = MultiLabelBinarizer(classes=all_codes)
        mlb.fit([all_codes])
        logging.info(f"Using {len(all_codes)} codes from training data")
        return mlb


def y_multi_hot(mlb: MultiLabelBinarizer, lists):
    """Convert lists of ICD codes to multi-hot vectors with proper formatting."""
    # Format the codes consistently before conversion to multi-hot
    formatted_lists = []
    for row in lists:
        formatted_row = [format_icd9_properly(str(c)) for c in row]
        # Only keep valid codes
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
            chosen.append(s)
            count += len(rows)
        if count >= target_n: break
    sub = train_df[train_df[subject_col].isin(chosen)].copy()
    logging.info(f"[subset] requested={target_n} got={len(sub)} unique_subjects={sub[subject_col].nunique()}")
    return sub

# ---------------- Dataset (pre-tokenized) ----------------

class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int, label_col: str):
        self.tok = tok
        self.max_len = max_len
        self.tgt_reserve = max(8, int(tgt_reserve))
        self.label_col = label_col
        self.rows = rows  # Store the original rows for evaluation
        
        prompts = rows["input_text"].astype(str).tolist()
        
        # Format ICD codes consistently in training targets
        targets = []
        for codes in rows[label_col].tolist():
            # Format each code properly with decimal points
            formatted_codes = [format_icd9_properly(str(c)) for c in codes]
            # Keep only valid ICD-9 codes
            formatted_codes = [c for c in formatted_codes if is_valid_icd9(c)]
            # Join with spaces to create the target string
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

# ---------------- Model ----------------

def load_lm_and_tokenizer(model_name):
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
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

# ---------------- Generation & metrics ----------------

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
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=2,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        seq = out.sequences
        gen_only = seq[:, inputs["input_ids"].shape[1]:]  # only new tokens
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        for t in texts:
            tokens = re.split(r"[^A-Za-z0-9\.]+", t)
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
    """Calculate standard evaluation metrics for multi-label classification."""
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
    
    return {
        "hierarchical_parent_recall": parent_recall,
        "hierarchical_partial_matches": partial_matches,
        "hierarchical_partial_per_sample": partial_matches / n_samples if n_samples > 0 else 0
    }

# ---------------- Custom training callbacks for detailed logging ----------------

class DetailedEvalCallback(TrainerCallback):
    """Callback for enhanced training logs with per-epoch metrics and generation evaluation."""
    def __init__(self, eval_dataset, tokenizer, label_vocab, eval_sample_size=100, seed=42):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.eval_sample_size = min(eval_sample_size, len(eval_dataset))
        self.best_micro_f1 = 0
        self.epoch_metrics = []
        self.rng = np.random.RandomState(seed)
        
        # Sample a fixed subset for consistency across epochs
        self.subset_indices = self.rng.choice(
            len(self.eval_dataset), 
            self.eval_sample_size, 
            replace=False
        )
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Format logs to clearly distinguish between train and eval phases."""
        if logs is None:
            return
        
        # Add prefixes to make it clear which phase we're in
        if 'loss' in logs and 'eval_loss' not in logs:
            print(f"[Train] epoch: {logs.get('epoch', 0):.1f}, loss: {logs.get('loss', 0):.4f}, "
                  f"lr: {logs.get('learning_rate', 0):.2e}, grad_norm: {logs.get('grad_norm', 0):.2f}")
        elif 'eval_loss' in logs:
            eval_loss = logs.get('eval_loss', 0)
            print(f"[Eval] epoch: {logs.get('epoch', 0):.1f}, loss: {eval_loss:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Run generation-based evaluation on a subset of validation data."""
        # Skip during early epochs to save time
        if state.epoch < 1.0:
            return
            
        model = kwargs.get("model")
        if not model:
            return
            
        # Extract prompts for subset
        subset_prompts = []
        gold_codes_lists = []
        
        for idx in self.subset_indices:
            # Get prompt from dataset
            item = self.eval_dataset[idx]
            prompt_text = self.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            # Extract the prompt part (before the target)
            prompt_parts = prompt_text.split("[CODES]")
            if len(prompt_parts) > 1:
                subset_prompts.append(prompt_parts[0] + "[CODES]")
                
                # Extract gold codes from target
                target_text = prompt_parts[1].strip()
                gold_codes = [format_icd9_properly(c) for c in re.split(r"[^A-Za-z0-9\.]+", target_text) if c]
                gold_codes = [c for c in gold_codes if is_valid_icd9(c)]
                gold_codes_lists.append(gold_codes)
            else:
                # Fallback
                subset_prompts.append(prompt_text)
                gold_codes_lists.append([])
        
        # Generate predictions
        t_start = time.time()
        preds = generate_codes(
            model, self.tokenizer, subset_prompts, self.label_vocab,
            max_new=96, batch_size=4
        )
        gen_time = time.time() - t_start
        
        # Convert to multi-hot
        y_true = codes_to_multihot(gold_codes_lists, self.label_vocab)
        y_pred = codes_to_multihot(preds, self.label_vocab)
        
        # Calculate metrics
        eval_metrics = eval_sets(y_true, y_pred)
        hier_metrics = hierarchical_eval(y_true, y_pred, self.label_vocab)
        eval_metrics.update(hier_metrics)
        
        # Store best F1
        if eval_metrics["micro_f1"] > self.best_micro_f1:
            self.best_micro_f1 = eval_metrics["micro_f1"]
        
        # Store epoch metrics
        self.epoch_metrics.append({
            "epoch": state.epoch,
            "eval_loss": metrics.get("eval_loss", 0),
            "micro_f1": eval_metrics["micro_f1"],
            "samples_f1": eval_metrics["samples_f1"],
            "parent_recall": hier_metrics["hierarchical_parent_recall"]
        })
        
        # Print gen-eval metrics
        print(f"[GenEval] epoch: {state.epoch:.1f}, micro_f1: {eval_metrics['micro_f1']:.4f}, "
              f"samples_f1: {eval_metrics['samples_f1']:.4f}, "
              f"parent_recall: {hier_metrics['hierarchical_parent_recall']:.4f}, "
              f"time: {gen_time:.1f}s")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Print a summary of training progress at the end."""
        print("\n===== Training Summary =====")
        print(f"Best validation micro F1: {self.best_micro_f1:.4f}")
        print("Epoch progression:")
        for m in self.epoch_metrics:
            print(f"  Epoch {m['epoch']:.1f}: loss={m['eval_loss']:.4f}, micro_f1={m['micro_f1']:.4f}, "
                  f"samples_f1={m['samples_f1']:.4f}, parent_recall={m['parent_recall']:.4f}")

# ---------------- Run helpers ----------------

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(base, tag):
    path = os.path.join(base, tag)
    os.makedirs(path, exist_ok=False)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


def save_json(path: str, obj: dict):
    with open(path, "w") as f: json.dump(obj, f, indent=2)


def show_test_predictions(df: pd.DataFrame, preds: List[List[str]], 
                          label_col: str, label_vocab: List[str],
                          n_show: int = 5, seed: int = 0):
    """Display example predictions with properly formatted ICD-9 codes and per-example metrics."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(n_show, len(df)), replace=False)
    
    for i in idxs:
        row = df.iloc[i]
        # Format gold codes consistently
        gold = sorted({format_icd9_properly(str(c)) for c in row[label_col] 
                      if is_valid_icd9(format_icd9_properly(str(c)))})
        pred = preds[i]
        missing = sorted([c for c in gold if c not in pred])  # FN
        extra   = sorted([c for c in pred if c not in gold])  # FP
        
        # Check for parent-level matches (partial matches)
        gold_parents = {get_icd9_parent(c) for c in gold}
        pred_parents = {get_icd9_parent(c) for c in pred}
        parent_matches = sorted([f"{c} (parent)" for c in pred 
                              if get_icd9_parent(c) in gold_parents and c not in gold])
        
        # Calculate per-example metrics
        # Convert to single-example multi-hot vectors
        y_true = np.zeros(len(label_vocab))
        y_pred = np.zeros(len(label_vocab))
        
        for code in gold:
            if code in label_vocab:
                idx = label_vocab.index(code)
                y_true[idx] = 1
                
        for code in pred:
            if code in label_vocab:
                idx = label_vocab.index(code)
                y_pred[idx] = 1
                
        # Calculate metrics for this example
        precision = precision_score([y_true], [y_pred], average='micro', zero_division=0)
        recall = recall_score([y_true], [y_pred], average='micro', zero_division=0)
        f1 = f1_score([y_true], [y_pred], average='micro', zero_division=0)
        
        # Calculate hierarchical metrics
        parent_recall = 0
        if gold_parents:
            parent_hits = len(pred_parents & gold_parents)
            parent_recall = parent_hits / len(gold_parents)
            
        print("\n" + "="*80)
        print(f"Example {i}:")
        print("- METRICS: precision={:.4f}, recall={:.4f}, f1={:.4f}, parent_recall={:.4f}".format(
            precision, recall, f1, parent_recall))
        print("- GOLD:", " ".join(gold) if gold else "(none)")
        print("- PRED:", " ".join(pred) if pred else "(none)")
        print(f"- FALSE NEGATIVES ({len(missing)}):", " ".join(missing) if missing else "(none)")
        print(f"- FALSE POSITIVES ({len(extra)}):", " ".join(extra) if extra else "(none)")
        print(f"- PARENT MATCHES ({len(parent_matches)}):", " ".join(parent_matches) if parent_matches else "(none)")

# ---------------- Main ----------------

def main():
    args = get_args()
    set_seed(args.seed)

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

    # Build prompts with improved instructions
    for df_, name in ((train_df, 'train'), (val_df, 'val'), (test_df, 'test')):
        df_["input_text"] = df_.apply(lambda r: build_input_text(r, args.use_structured==1, args.use_notes==1, args.subject_col), axis=1)
        logging.info(f"[{name}] rows with input_text: {df_['input_text'].notna().sum()}")

    # Label space - either from training data only or complete ICD-9 reference
    mlb = lock_label_space(
        [train_df, val_df, test_df], 
        args.label_col, 
        icd9_pkl_path=args.icd9_pickle, 
        use_complete=bool(args.use_complete_icd9)
    )
    
    labels_vocab = mlb.classes_.tolist()
    
    # Convert labels to multi-hot vectors with proper formatting
    y_val  = y_multi_hot(mlb, val_df[args.label_col].tolist())
    y_test = y_multi_hot(mlb, test_df[args.label_col].tolist())

    # Model & tokenizer
    model, tok = load_lm_and_tokenizer(args.llama_model)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")

    # Datasets with properly formatted ICD-9 codes
    train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col)
    val_ds   = GenCodesDataset(val_df,   tok, args.max_len, args.tgt_reserve_tok, args.label_col)

    # Run dir
    size_str = f"N{args.train_size}" if args.train_size > 0 else "full"
    tag = args.run_name or f"{now_tag()}_{size_str}_icd9{'_complete' if args.use_complete_icd9 else ''}"
    RUN_DIR = make_run_dir(args.run_root, tag)
    logging.info(f"Run dir: {RUN_DIR}")

    save_json(os.path.join(RUN_DIR, "config.json"), {
        "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate,
        "epochs": args.epochs, "gen_max_new": args.gen_max_new, "tgt_reserve_tok": args.tgt_reserve_tok,
        "seed": args.seed, "train_rows": len(train_df),
        "icd9_pickle": args.icd9_pickle, 
        "use_complete_icd9": bool(args.use_complete_icd9),
        "total_label_space": len(labels_vocab)
    })
    save_json(os.path.join(RUN_DIR, "label_space.json"), {"labels": labels_vocab})

    # Training args (simple)
    train_args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        logging_strategy="epoch",
        eval_strategy="epoch",
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
        optim="adamw_torch", dataloader_num_workers=2,
        run_name=os.path.basename(RUN_DIR),
        disable_tqdm=True,
    )

    # Add both the early stopping and detailed evaluation callbacks
    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    
    # Add our custom detailed evaluation callback
    detailed_callback = DetailedEvalCallback(
        eval_dataset=val_ds,
        tokenizer=tok,
        label_vocab=labels_vocab,
        eval_sample_size=args.eval_sample_size,
        seed=args.seed
    )
    callbacks.append(detailed_callback)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=lambda feats: pad_collate(feats, tok),
        callbacks=callbacks
    )

    # Train
    t0 = time.perf_counter()
    logging.info("Starting training…")
    trainer.train()
    train_secs = time.perf_counter() - t0
    logging.info(f"[TIME] train: {train_secs:.2f}s")

    # Save adapter
    tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
    trainer.model.save_pretrained(os.path.join(RUN_DIR, "adapter_best"))

    if args.merge_after:
        try:
            merged_dir = os.path.join(RUN_DIR, "model_merged")
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir)
            tok.save_pretrained(os.path.join(merged_dir, "tokenizer"))
            logging.info(f"Merged model saved to: {merged_dir}")
        except Exception as e:
            logging.warning(f"Could not merge adapters into base: {e}")

    # Final TEST generation
    test_prompts = test_df["input_text"].astype(str).tolist()
    t_gen = time.perf_counter()
    pred_code_lists = generate_codes(
        model, tok, test_prompts, labels_vocab,
        max_new=args.gen_max_new, batch_size=args.per_device_eval_batch_size, max_len=args.max_len
    )
    test_gen_secs = time.perf_counter() - t_gen

    Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
    metrics = eval_sets(y_test, Y_pred)
    
    # Add hierarchical evaluation metrics
    hier_metrics = hierarchical_eval(y_test, Y_pred, labels_vocab)
    metrics.update(hier_metrics)
    
    metrics["train_seconds"] = train_secs
    metrics["test_generate_seconds"] = test_gen_secs

    with open(os.path.join(RUN_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Generative TEST metrics ===")
    print(json.dumps(metrics, indent=2))
    
    # Show sample predictions with proper ICD-9 formatting and per-example metrics
    print("\n=== Sample Predictions ===")
    show_test_predictions(test_df, pred_code_lists, args.label_col, labels_vocab, 
                         n_show=args.test_examples, seed=args.seed)


if __name__ == "__main__":
    main()