# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# test_textgen_ragKG.py
# Testing script for RAG-enhanced diagnosis generation models.
# ✓ Maps generated text to ICD-9 codes using SapBERT encoder (like baseline)
# ✓ Extracts gold codes from 'target_codes' field in JSONL
# ✓ Supports baseline, RAG unweighted, and RAG weighted models
# ✓ Subset evaluation support
# """

# import os, json, time, argparse, logging, sys, glob, pickle
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from collections import Counter

# import torch
# import torch.distributed as dist
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# # Import from common_textgen (same as baseline)
# from common_textgen import (
#     log, is_main_process, world_size, local_rank,
#     ICDMapper, format_icd9, is_valid_icd9,
#     restrict_to, multihot, eval_pack, add_parent_macro_f1,
#     get_icd9_parent
# )

# # ============================================================================
# # DISTRIBUTED UTILS
# # ============================================================================

# def maybe_init_dist():
#     if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
#         dist.init_process_group(backend="nccl")
#     return dist.is_initialized()

# def shard_indices(N: int, rank: int, W: int):
#     return list(range(rank, N, W))

# def barrier():
#     if dist.is_available() and dist.is_initialized():
#         try:
#             dist.barrier()
#         except Exception:
#             pass

# def cleanup_dist():
#     if dist.is_available() and dist.is_initialized():
#         try:
#             dist.destroy_process_group()
#         except Exception:
#             pass

# # ============================================================================
# # CODE EXTRACTION & FORMATTING
# # ============================================================================

# def extract_terms_from_generation(text: str, max_terms: int = 12) -> list:
#     """
#     Extract diagnosis terms from generated text.
#     Each line is treated as a potential diagnosis term.
#     """
#     terms = []
#     for line in text.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
#         # Remove common prefixes (bullet points, numbers, etc.)
#         line = line.lstrip('- ')
#         line = line.lstrip('* ')
#         # Remove leading numbers like "1. ", "2. ", etc.
#         import re
#         line = re.sub(r'^\d+\.\s*', '', line)
#         line = re.sub(r'^\(\d+\)\s*', '', line)
        
#         if line:
#             terms.append(line)
        
#         if len(terms) >= max_terms:
#             break
    
#     return terms

# # ✓ FIXED: Extract gold codes from "target_codes" field
# def extract_gold_codes_from_example(example: dict) -> list:
#     """
#     Extract gold ICD-9 codes from JSONL example.
#     Priority order:
#     1. 'target_codes' (RAG dataset format)
#     2. 'gold_codes'
#     3. 'codes'
#     4. 'labels'
#     5. 'icd9_codes'
#     """
#     # Priority 1: target_codes (your RAG dataset format)
#     if 'target_codes' in example:
#         codes = example['target_codes']
#         if isinstance(codes, list):
#             return [format_icd9(c) for c in codes if c]
#         elif isinstance(codes, str):
#             # Handle string formats: "410.71,414.01,401.9" or "410.71;414.01"
#             for sep in [',', ';', ' ', '|']:
#                 if sep in codes:
#                     return [format_icd9(c.strip()) for c in codes.split(sep) if c.strip()]
#             # Single code as string
#             return [format_icd9(codes)] if codes.strip() else []
    
#     # Priority 2-5: Try other common field names
#     for field in ['gold_codes', 'codes', 'labels', 'icd9_codes', 'icd_codes']:
#         if field in example:
#             codes = example[field]
#             if isinstance(codes, list):
#                 return [format_icd9(c) for c in codes if c]
#             elif isinstance(codes, str):
#                 # Handle string formats
#                 for sep in [',', ';', ' ', '|']:
#                     if sep in codes:
#                         return [format_icd9(c.strip()) for c in codes.split(sep) if c.strip()]
#                 return [format_icd9(codes)] if codes.strip() else []
    
#     return []  # No codes found

# # ============================================================================
# # METRICS
# # ============================================================================

# def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
#     """Add parent-level metrics (micro, macro, samples)."""
#     from sklearn.metrics import precision_score, recall_score, f1_score
    
#     g = [[get_icd9_parent(c) for c in lst] for lst in gold_lists]
#     p = [[get_icd9_parent(c) for c in lst] for lst in pred_lists]
#     labels = sorted({x for lst in g for x in lst})
    
#     if not labels:
#         # No parent labels, return zeros
#         metrics_dict.update({
#             "precision_macro_parent": 0.0,
#             "recall_macro_parent": 0.0,
#             "f1_macro_parent": 0.0,
#             "precision_micro_parent": 0.0,
#             "recall_micro_parent": 0.0,
#             "f1_micro_parent": 0.0,
#             "precision_samples_parent": 0.0,
#             "recall_samples_parent": 0.0,
#             "f1_samples_parent": 0.0,
#         })
#         return labels, None, None
    
#     Yg = multihot(g, labels)
#     Yp = multihot(p, labels)
    
#     metrics_dict.update({
#         "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
#         "recall_macro_parent": float(recall_score(Yg, Yp, average="macro", zero_division=0)),
#         "f1_macro_parent": float(f1_score(Yg, Yp, average="macro", zero_division=0)),
#         "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
#         "recall_micro_parent": float(recall_score(Yg, Yp, average="micro", zero_division=0)),
#         "f1_micro_parent": float(f1_score(Yg, Yp, average="micro", zero_division=0)),
#         "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
#         "recall_samples_parent": float(recall_score(Yg, Yp, average="samples", zero_division=0)),
#         "f1_samples_parent": float(f1_score(Yg, Yp, average="samples", zero_division=0)),
#     })
#     return labels, Yg, Yp

# def sample_level_metrics(gold_lists, pred_lists):
#     """Compute per-sample P/R/F1 and return average."""
#     vals = []
#     for g, p in zip(gold_lists, pred_lists):
#         G, P = set(g), set(p)
#         tp = len(G & P)
#         fp = len(P - G)
#         fn = len(G - P)
#         prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
#         vals.append((prec, rec, f1))
    
#     arr = np.array(vals) if vals else np.zeros((0, 3))
#     return {
#         "precision_sample_avg": float(arr[:, 0].mean() if arr.size else 0.0),
#         "recall_sample_avg": float(arr[:, 1].mean() if arr.size else 0.0),
#         "f1_sample_avg": float(arr[:, 2].mean() if arr.size else 0.0)
#     }

# # ============================================================================
# # GENERATION
# # ============================================================================

# def generate_batch(model, tokenizer, prompts, max_len=5120, gen_max_new=128, 
#                    decoding="greedy", num_beams=1, temperature=1.0, 
#                    top_p=0.95, top_k=50):
#     """Generate text for a batch of prompts."""
#     # Prepare inputs
#     conversations = [[{"role": "user", "content": p}] for p in prompts]
    
#     input_texts = [
#         tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
#         for conv in conversations
#     ]
    
#     encodings = tokenizer(
#         input_texts,
#         padding=True,
#         truncation=True,
#         max_length=max_len,
#         return_tensors="pt"
#     )
    
#     input_ids = encodings['input_ids'].to(model.device)
#     attention_mask = encodings['attention_mask'].to(model.device)
    
#     # Generation kwargs
#     gen_kwargs = {
#         "max_new_tokens": gen_max_new,
#         "pad_token_id": tokenizer.pad_token_id,
#         "eos_token_id": tokenizer.eos_token_id,
#         "do_sample": False,
#     }
    
#     if decoding == "beam":
#         gen_kwargs["num_beams"] = num_beams
#         gen_kwargs["early_stopping"] = True
#     elif decoding == "sample":
#         gen_kwargs["do_sample"] = True
#         gen_kwargs["temperature"] = temperature
#         gen_kwargs["top_p"] = top_p
#         gen_kwargs["top_k"] = top_k
    
#     # Generate
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             **gen_kwargs
#         )
    
#     # Decode
#     generated_texts = []
#     for i, output in enumerate(outputs):
#         # Remove input prompt
#         input_len = input_ids[i].shape[0]
#         generated = output[input_len:]
#         text = tokenizer.decode(generated, skip_special_tokens=True)
#         generated_texts.append(text)
    
#     return generated_texts

# # ============================================================================
# # PRETTY PRINT
# # ============================================================================

# def _pretty_print_block(title: str, d: dict):
#     log.info(f"\n--- {title} ---")
#     for k in sorted(d.keys()):
#         v = d[k]
#         if isinstance(v, float):
#             log.info(f"  {k:>35s}: {v:.6f}")
#         else:
#             log.info(f"  {k:>35s}: {v}")

# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     parser = argparse.ArgumentParser()
    
#     # Data
#     parser.add_argument("--test_jsonl", required=True, help="Test JSONL file")
#     parser.add_argument("--subset_size", type=int, default=None, 
#                         help="Test on subset (e.g., 100 samples). None = full test set")
#     parser.add_argument("--subset_seed", type=int, default=42,
#                         help="Random seed for subset sampling")
    
#     # Model
#     parser.add_argument("--base_model", required=True, help="Base model path")
#     parser.add_argument("--adapter_dir", required=True, help="Adapter directory")
    
#     # Generation
#     parser.add_argument("--max_len", type=int, default=5120)
#     parser.add_argument("--gen_max_new", type=int, default=128)
#     parser.add_argument("--gen_batch_size", type=int, default=4)
#     parser.add_argument("--decoding", choices=["greedy", "beam", "sample"], default="greedy")
#     parser.add_argument("--num_beams", type=int, default=1)
#     parser.add_argument("--temperature", type=float, default=1.0)
#     parser.add_argument("--top_p", type=float, default=0.95)
#     parser.add_argument("--top_k", type=int, default=50)
#     parser.add_argument("--N_max_terms", type=int, default=12, 
#                         help="Max diagnosis terms to extract from generation")
    
#     # SapBERT Mapper arguments (same as baseline)
#     parser.add_argument("--icd_index_dir", required=True, 
#                         help="Directory with FAISS index for ICD-9 codes")
#     parser.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#     parser.add_argument("--faiss_rows", type=int, default=50)
#     parser.add_argument("--tau_cos", type=float, default=0.40)
#     parser.add_argument("--tau_final", type=float, default=0.60)
#     parser.add_argument("--w_cos", type=float, default=0.6)
#     parser.add_argument("--w_fuz", type=float, default=0.4)
    
#     # Multi-GPU
#     parser.add_argument("--distributed", action="store_true")
#     parser.add_argument("--tmp_dir", default="runs_textgen_rag/test_shards")
    
#     # Output
#     parser.add_argument("--out_metrics", required=True, help="Output metrics JSON path")
#     parser.add_argument("--print_samples", type=int, default=5)
    
#     # Device
#     parser.add_argument("--use_bf16", action="store_true")
    
#     args = parser.parse_args()
    
#     # ========================================================================
#     # SETUP
#     # ========================================================================
    
#     if is_main_process():
#         log.info("=" * 80)
#         log.info("RAG TEXT GENERATION - TESTING (with SapBERT Mapping)")
#         log.info("=" * 80)
#         log.info(f"Test data: {args.test_jsonl}")
#         log.info(f"Base model: {args.base_model}")
#         log.info(f"Adapter: {args.adapter_dir}")
#         log.info(f"Decoding: {args.decoding}")
#         log.info(f"ICD Index: {args.icd_index_dir}")
#         log.info(f"Encoder: {args.encoder_model}")
#         if args.subset_size:
#             log.info(f"⚠️  SUBSET MODE: Testing on {args.subset_size} samples only")
#         log.info("=" * 80)
    
#     # Load test data
#     examples = []
#     with open(args.test_jsonl, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 examples.append(json.loads(line))
    
#     total_examples = len(examples)
    
#     if is_main_process():
#         log.info(f"\n📂 Loaded {total_examples} examples from {args.test_jsonl}")
#         # Show available fields in first example
#         if examples:
#             example_keys = list(examples[0].keys())
#             log.info(f"   Available fields: {example_keys}")
#             # Show sample target_codes if present
#             if 'target_codes' in examples[0]:
#                 sample_codes = examples[0]['target_codes']
#                 if isinstance(sample_codes, list):
#                     log.info(f"   Sample target_codes (first example): {sample_codes[:5]}...")
#                 else:
#                     log.info(f"   Sample target_codes (first example): {sample_codes}")
    
#     # Subset sampling
#     if args.subset_size and args.subset_size < len(examples):
#         np.random.seed(args.subset_seed)
#         subset_indices = np.random.choice(len(examples), args.subset_size, replace=False)
#         examples = [examples[i] for i in sorted(subset_indices)]
        
#         if is_main_process():
#             log.info(f"\n🎯 Subset sampling:")
#             log.info(f"   Total in file: {total_examples}")
#             log.info(f"   Testing on: {len(examples)} samples (seed={args.subset_seed})")
#     else:
#         if is_main_process():
#             log.info(f"\n📊 Testing on full set: {len(examples)} samples")
    
#     # ✓ FIXED: Extract prompts, targets, and gold codes properly
#     prompts = [ex['prompt'] for ex in examples]
#     targets = [ex.get('target', '') for ex in examples]
    
#     # Extract gold codes from JSONL examples
#     if is_main_process():
#         log.info(f"\n🔍 Extracting gold codes from examples...")
    
#     gold_codes = []
#     extraction_stats = {'target_codes': 0, 'other_fields': 0, 'empty': 0}
    
#     for ex in examples:
#         codes = extract_gold_codes_from_example(ex)
#         gold_codes.append(codes)
        
#         # Track extraction source for diagnostics
#         if 'target_codes' in ex and ex['target_codes']:
#             extraction_stats['target_codes'] += 1
#         elif any(field in ex for field in ['gold_codes', 'codes', 'labels', 'icd9_codes']):
#             extraction_stats['other_fields'] += 1
#         else:
#             extraction_stats['empty'] += 1
    
#     # Build label space
#     all_codes = sorted(set([c for lst in gold_codes for c in lst]))
    
#     if is_main_process():
#         log.info(f"\n📋 Gold code extraction results:")
#         log.info(f"   Total unique codes: {len(all_codes)}")
#         log.info(f"   Extraction sources:")
#         log.info(f"     - target_codes field: {extraction_stats['target_codes']}")
#         log.info(f"     - other fields: {extraction_stats['other_fields']}")
#         log.info(f"     - empty: {extraction_stats['empty']}")
#         log.info(f"   Avg codes per sample: {np.mean([len(g) for g in gold_codes]):.2f}")
        
#         if len(all_codes) == 0:
#             log.error("❌ ERROR: No gold codes found in test set!")
#             log.error("   Expected JSONL format:")
#             log.error('   {"prompt": "...", "target": "...", "target_codes": ["410.71", "414.01", ...]}')
#             log.error(f"   Found fields: {list(examples[0].keys())}")
#             return 1  # Exit with error
        
#         # Show sample of extracted codes
#         sample_gold = [g for g in gold_codes[:3] if g]
#         if sample_gold:
#             log.info(f"\n   Sample gold codes (first few examples):")
#             for i, codes in enumerate(sample_gold[:3]):
#                 log.info(f"     Example {i+1}: {codes[:5]}{'...' if len(codes) > 5 else ''}")
    
#     # ========================================================================
#     # MODEL LOADING
#     # ========================================================================
    
#     dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and 
#                                torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    
#     if is_main_process():
#         log.info(f"\n🤖 Loading tokenizer from {args.adapter_dir}...")
    
#     tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.padding_side = "left"
    
#     if is_main_process():
#         log.info(f"🤖 Loading base model: {args.base_model}")
#         log.info(f"   dtype: {dtype}")
    
#     base_model = AutoModelForCausalLM.from_pretrained(
#         args.base_model,
#         torch_dtype=dtype,
#         low_cpu_mem_usage=True
#     )
    
#     if is_main_process():
#         log.info(f"📦 Loading adapter from {args.adapter_dir}")
    
#     model = PeftModel.from_pretrained(base_model, args.adapter_dir)
#     model.config.use_cache = True
    
#     dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
#     model.to(dev).eval()
    
#     if is_main_process():
#         log.info(f"✅ Model loaded on device: {dev}")
    
#     # ========================================================================
#     # INITIALIZE SAPBERT MAPPER
#     # ========================================================================
    
#     if is_main_process():
#         log.info(f"\n🔍 Initializing SapBERT mapper...")
#         log.info(f"   Index dir: {args.icd_index_dir}")
#         log.info(f"   Encoder: {args.encoder_model}")
#         log.info(f"   Weights: w_cos={args.w_cos}, w_fuz={args.w_fuz}")
    
#     mapper = ICDMapper(
#         index_dir=args.icd_index_dir,
#         encoder_model_cli=args.encoder_model,
#         tau_cos=args.tau_cos,
#         tau_final=args.tau_final,
#         w_cos=args.w_cos,
#         w_fuz=args.w_fuz,
#         faiss_rows=args.faiss_rows
#     )
    
#     if is_main_process():
#         log.info("✅ Mapper initialized")
    
#     # ========================================================================
#     # DISTRIBUTED SHARDING
#     # ========================================================================
    
#     if args.distributed:
#         maybe_init_dist()
#         rank = int(os.environ.get("RANK", "0"))
#         W = world_size()
#         idxs = shard_indices(len(prompts), rank, W)
#     else:
#         rank, W = 0, 1
#         idxs = list(range(len(prompts)))
    
#     shard_prompts = [prompts[i] for i in idxs]
#     shard_gold = [gold_codes[i] for i in idxs]
    
#     # ========================================================================
#     # GENERATION
#     # ========================================================================
    
#     if is_main_process():
#         log.info(f"\n🚀 Generating predictions for {len(shard_prompts)} examples...")
    
#     generated_texts = []
#     bs = args.gen_batch_size
#     t0 = time.time()
    
#     for i in range(0, len(shard_prompts), bs):
#         batch_prompts = shard_prompts[i:i+bs]
#         batch_outputs = generate_batch(
#             model, tokenizer, batch_prompts,
#             max_len=args.max_len,
#             gen_max_new=args.gen_max_new,
#             decoding=args.decoding,
#             num_beams=args.num_beams,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             top_k=args.top_k
#         )
#         generated_texts.extend(batch_outputs)
        
#         if is_main_process() and (i + bs) % (bs * 10) == 0:
#             log.info(f"  Progress: {min(i+bs, len(shard_prompts))}/{len(shard_prompts)}")
    
#     elapsed = time.time() - t0
#     if is_main_process():
#         log.info(f"✅ Generation completed in {elapsed:.2f}s ({elapsed/max(1,len(shard_prompts)):.2f}s/sample)")
    
#     # ========================================================================
#     # EXTRACT TERMS AND MAP TO ICD-9
#     # ========================================================================
    
#     if is_main_process():
#         log.info(f"\n📝 Extracting diagnosis terms from generated text...")
    
#     # Extract terms from generated text
#     terms_lists = []
#     for text in generated_texts:
#         terms = extract_terms_from_generation(text, max_terms=args.N_max_terms)
#         terms_lists.append(terms)
    
#     if is_main_process():
#         log.info(f"✅ Extracted terms from {len(terms_lists)} generations")
#         log.info(f"\n🔍 Mapping terms to ICD-9 codes using SapBERT...")
    
#     # Map terms to ICD-9 codes using SapBERT
#     pred_codes = mapper.map_terms(terms_lists)
    
#     if is_main_process():
#         log.info(f"✅ Mapping completed")
        
#         # Mapper diagnostics
#         if mapper.last_stats:
#             n_terms = np.array([n for (n, m) in mapper.last_stats], dtype=np.float32)
#             n_map = np.array([m for (n, m) in mapper.last_stats], dtype=np.float32)
#             log.info(f"   Avg terms per sample: {n_terms.mean():.2f}")
#             log.info(f"   Avg mapped per sample: {n_map.mean():.2f}")
#             unmappable_rate = np.mean(np.where(n_terms > 0, 1.0 - (n_map / np.maximum(n_terms, 1)), 0.0))
#             log.info(f"   Unmappable term rate: {unmappable_rate:.4f}")
    
#     # ========================================================================
#     # SAVE SHARD
#     # ========================================================================
    
#     os.makedirs(args.tmp_dir, exist_ok=True)
#     shard_path = os.path.join(args.tmp_dir, f"shard_{rank:03d}_of_{W:03d}.pkl")
    
#     with open(shard_path, "wb") as f:
#         pickle.dump({
#             "idxs": idxs,
#             "generated": generated_texts,
#             "terms": terms_lists,
#             "predicted": pred_codes,
#             "gold": shard_gold,
#         }, f)
    
#     if is_main_process():
#         log.info(f"💾 Saved shard to {shard_path}")
    
#     barrier()
    
#     # ========================================================================
#     # RANK-0: MERGE AND COMPUTE METRICS
#     # ========================================================================
    
#     if rank == 0:
#         log.info("\n" + "=" * 80)
#         log.info("MERGING SHARDS AND COMPUTING METRICS")
#         log.info("=" * 80)
        
#         # Merge all shards
#         shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
#         all_idx, all_gen, all_terms, all_pred, all_gold = [], [], [], [], []
        
#         for sp in shards:
#             with open(sp, "rb") as f:
#                 D = pickle.load(f)
#             all_idx.extend(D["idxs"])
#             all_gen.extend(D["generated"])
#             all_terms.extend(D["terms"])
#             all_pred.extend(D["predicted"])
#             all_gold.extend(D["gold"])
        
#         # Restore original order
#         order = np.argsort(np.array(all_idx))
#         generated_all = [all_gen[i] for i in order]
#         terms_all = [all_terms[i] for i in order]
#         pred_all = [all_pred[i] for i in order]
#         gold_all = [all_gold[i] for i in order]
        
#         # Restrict to evaluation label space
#         gold_eval = restrict_to(gold_all, all_codes)
#         pred_eval = restrict_to(pred_all, all_codes)
        
#         # Convert to multi-hot
#         y_true = multihot(gold_eval, all_codes)
#         y_pred = multihot(pred_eval, all_codes)
        
#         # Compute metrics
#         metrics = eval_pack(y_true, y_pred)
        
#         # Parent metrics
#         parent_metrics = {}
#         add_parent_metrics_full(parent_metrics, gold_eval, pred_eval)
#         metrics.update(parent_metrics)
        
#         # Legacy parent macro F1
#         add_parent_macro_f1(metrics, gold_eval, pred_eval)
        
#         # Sample-level metrics
#         sample_metrics = sample_level_metrics(gold_eval, pred_eval)
#         metrics.update(sample_metrics)
        
#         # Additional stats
#         metrics["num_samples"] = len(gold_all)
#         metrics["num_unique_codes"] = len(all_codes)
#         metrics["avg_gold_codes_per_sample"] = float(np.mean([len(g) for g in gold_all]))
#         metrics["avg_pred_codes_per_sample"] = float(np.mean([len(p) for p in pred_all]))
        
#         # Mapper stats
#         if mapper.last_stats:
#             n_terms = np.array([n for (n, m) in mapper.last_stats], dtype=np.float32)
#             n_map = np.array([m for (n, m) in mapper.last_stats], dtype=np.float32)
#             metrics["mean_terms_per_visit"] = float(n_terms.mean())
#             metrics["mean_mapped_terms_per_visit"] = float(n_map.mean())
#             metrics["unmappable_term_rate"] = float(np.mean(np.where(n_terms > 0, 1.0 - (n_map / np.maximum(n_terms, 1)), 0.0)))
        
#         # Subset info
#         if args.subset_size:
#             metrics["total_test_examples"] = total_examples
#             metrics["subset_size"] = len(gold_all)
#             metrics["subset_seed"] = args.subset_seed
        
#         # ====================================================================
#         # PRINT SAMPLE PREDICTIONS
#         # ====================================================================
        
#         log.info("\n" + "=" * 80)
#         log.info("SAMPLE PREDICTIONS")
#         log.info("=" * 80)
        
#         n_show = min(args.print_samples, len(generated_all))
#         for i in range(n_show):
#             log.info(f"\n{'─' * 76}")
#             log.info(f"Sample {i+1}/{n_show}")
#             log.info(f"{'─' * 76}")
            
#             # Show first 300 chars of prompt
#             prompt_preview = prompts[order[i]][:300].replace('\n', ' ')
#             log.info(f"📝 PROMPT (first 300 chars):")
#             log.info(f"   {prompt_preview}...")
            
#             log.info(f"\n🤖 GENERATED TEXT:")
#             for line in generated_all[i].split('\n')[:10]:
#                 if line.strip():
#                     log.info(f"   {line}")
            
#             log.info(f"\n📋 EXTRACTED TERMS:")
#             if terms_all[i]:
#                 for term in terms_all[i]:
#                     log.info(f"   - {term}")
#             else:
#                 log.info(f"   (none)")
            
#             log.info(f"\n🎯 MAPPED ICD-9 CODES:")
#             pred_str = ', '.join(sorted(pred_all[i])) if pred_all[i] else '(none)'
#             gold_str = ', '.join(sorted(gold_all[i])) if gold_all[i] else '(none)'
#             log.info(f"   Predicted: {pred_str}")
#             log.info(f"   Gold:      {gold_str}")
            
#             # Per-sample metrics
#             if gold_all[i] or pred_all[i]:
#                 G = set(gold_all[i])
#                 P = set(pred_all[i])
#                 tp = len(G & P)
#                 fp = len(P - G)
#                 fn = len(G - P)
#                 prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#                 rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#                 f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                
#                 log.info(f"\n📊 SAMPLE METRICS:")
#                 log.info(f"   TP={tp}, FP={fp}, FN={fn}")
#                 log.info(f"   Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
        
#         # ====================================================================
#         # PRINT OVERALL METRICS
#         # ====================================================================
        
#         log.info("\n" + "=" * 80)
#         log.info("OVERALL METRICS")
#         log.info("=" * 80)
        
#         # Group metrics
#         micro_metrics = {k: v for k, v in metrics.items() if 'micro' in k}
#         macro_metrics = {k: v for k, v in metrics.items() if 'macro' in k}
#         sample_metrics_dict = {k: v for k, v in metrics.items() if 'sample' in k}
#         other_metrics = {k: v for k, v in metrics.items() 
#                         if 'micro' not in k and 'macro' not in k and 'sample' not in k}
        
#         _pretty_print_block("MICRO METRICS (global)", micro_metrics)
#         _pretty_print_block("MACRO METRICS (per-label avg)", macro_metrics)
#         _pretty_print_block("SAMPLE-LEVEL METRICS (per-sample avg)", sample_metrics_dict)
#         _pretty_print_block("STATISTICS", other_metrics)
        
#         log.info("=" * 80)
        
#         # ====================================================================
#         # SAVE METRICS
#         # ====================================================================
        
#         output_data = {
#             "config": {
#                 "test_jsonl": args.test_jsonl,
#                 "base_model": args.base_model,
#                 "adapter_dir": args.adapter_dir,
#                 "decoding": args.decoding,
#                 "num_beams": args.num_beams,
#                 "max_len": args.max_len,
#                 "gen_max_new": args.gen_max_new,
#                 "N_max_terms": args.N_max_terms,
#                 "icd_index_dir": args.icd_index_dir,
#                 "encoder_model": args.encoder_model,
#                 "w_cos": args.w_cos,
#                 "w_fuz": args.w_fuz,
#                 "subset_size": args.subset_size,
#                 "subset_seed": args.subset_seed if args.subset_size else None,
#             },
#             "metrics": metrics,
#         }
        
#         os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
#         with open(args.out_metrics, 'w') as f:
#             json.dump(output_data, f, indent=2)
        
#         log.info(f"\n✅ Metrics saved to: {args.out_metrics}")
#         log.info("=" * 80)
    
#     cleanup_dist()
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_textgen_ragKG.py
Testing script for RAG-enhanced diagnosis generation models.
✓ Maps generated text to ICD-9 codes using SapBERT encoder (like baseline)
✓ Extracts gold codes from 'target_codes' field in JSONL
✓ FIXED: Properly handles semicolon-separated multi-diagnosis format
✓ Supports baseline, RAG unweighted, and RAG weighted models
✓ Subset evaluation support
"""

import os, json, time, argparse, logging, sys, glob, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import from common_textgen (same as baseline)
from common_textgen import (
    log, is_main_process, world_size, local_rank,
    ICDMapper, format_icd9, is_valid_icd9,
    restrict_to, multihot, eval_pack, add_parent_macro_f1,
    get_icd9_parent
)

# ============================================================================
# DISTRIBUTED UTILS
# ============================================================================

def maybe_init_dist():
    if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl")
    return dist.is_initialized()

def shard_indices(N: int, rank: int, W: int):
    return list(range(rank, N, W))

def barrier():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass

def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

# ============================================================================
# CODE EXTRACTION & FORMATTING
# ============================================================================

def _clean_diagnosis_term(term: str) -> str:
    """
    Clean up a single diagnosis term.
    Removes prefixes, numbering, and extra whitespace.
    """
    import re
    
    term = term.strip()
    if not term:
        return ""
    
    # Remove common prefixes
    term = term.lstrip('- ')
    term = term.lstrip('* ')
    term = term.lstrip('• ')
    
    # Remove leading numbers like "1. ", "2. ", "(1)", etc.
    term = re.sub(r'^\d+\.\s*', '', term)
    term = re.sub(r'^\(\d+\)\s*', '', term)
    term = re.sub(r'^\d+\)\s*', '', term)
    
    # Remove any remaining leading/trailing whitespace
    term = term.strip()
    
    # Remove trailing punctuation (except period in medical terms)
    if term and term[-1] in [',', ';', ':']:
        term = term[:-1].strip()
    
    return term


def extract_terms_from_generation(text: str, max_terms: int = 12) -> list:
    """
    Extract diagnosis terms from generated text.
    Handles both line-by-line format AND semicolon-separated format.
    
    Example formats:
    1. Line-by-line:
       "Acute myocardial infarction
        Hypertension
        Diabetes mellitus"
    
    2. Semicolon-separated (RAG model output):
       "Acute myocardial infarction; Hypertension; Diabetes mellitus"
    """
    terms = []
    
    # First, split by newlines
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line contains semicolons (multi-diagnosis format)
        if ';' in line:
            # Split by semicolon and process each term
            parts = line.split(';')
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # Clean up the term
                part = _clean_diagnosis_term(part)
                
                if part and part not in terms:  # Avoid duplicates
                    terms.append(part)
                
                if len(terms) >= max_terms:
                    return terms
        else:
            # Single diagnosis per line
            line = _clean_diagnosis_term(line)
            
            if line and line not in terms:  # Avoid duplicates
                terms.append(line)
            
            if len(terms) >= max_terms:
                return terms
    
    return terms


def extract_gold_codes_from_example(example: dict) -> list:
    """
    Extract gold ICD-9 codes from JSONL example.
    Priority order:
    1. 'target_codes' (RAG dataset format)
    2. 'gold_codes'
    3. 'codes'
    4. 'labels'
    5. 'icd9_codes'
    """
    # Priority 1: target_codes (your RAG dataset format)
    if 'target_codes' in example:
        codes = example['target_codes']
        if isinstance(codes, list):
            return [format_icd9(c) for c in codes if c]
        elif isinstance(codes, str):
            # Handle string formats: "410.71,414.01,401.9" or "410.71;414.01"
            for sep in [',', ';', ' ', '|']:
                if sep in codes:
                    return [format_icd9(c.strip()) for c in codes.split(sep) if c.strip()]
            # Single code as string
            return [format_icd9(codes)] if codes.strip() else []
    
    # Priority 2-5: Try other common field names
    for field in ['gold_codes', 'codes', 'labels', 'icd9_codes', 'icd_codes']:
        if field in example:
            codes = example[field]
            if isinstance(codes, list):
                return [format_icd9(c) for c in codes if c]
            elif isinstance(codes, str):
                # Handle string formats
                for sep in [',', ';', ' ', '|']:
                    if sep in codes:
                        return [format_icd9(c.strip()) for c in codes.split(sep) if c.strip()]
                return [format_icd9(codes)] if codes.strip() else []
    
    return []  # No codes found

# ============================================================================
# METRICS
# ============================================================================

def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
    """Add parent-level metrics (micro, macro, samples)."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    g = [[get_icd9_parent(c) for c in lst] for lst in gold_lists]
    p = [[get_icd9_parent(c) for c in lst] for lst in pred_lists]
    labels = sorted({x for lst in g for x in lst})
    
    if not labels:
        # No parent labels, return zeros
        metrics_dict.update({
            "precision_macro_parent": 0.0,
            "recall_macro_parent": 0.0,
            "f1_macro_parent": 0.0,
            "precision_micro_parent": 0.0,
            "recall_micro_parent": 0.0,
            "f1_micro_parent": 0.0,
            "precision_samples_parent": 0.0,
            "recall_samples_parent": 0.0,
            "f1_samples_parent": 0.0,
        })
        return labels, None, None
    
    Yg = multihot(g, labels)
    Yp = multihot(p, labels)
    
    metrics_dict.update({
        "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
        "recall_macro_parent": float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent": float(f1_score(Yg, Yp, average="macro", zero_division=0)),
        "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
        "recall_micro_parent": float(recall_score(Yg, Yp, average="micro", zero_division=0)),
        "f1_micro_parent": float(f1_score(Yg, Yp, average="micro", zero_division=0)),
        "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
        "recall_samples_parent": float(recall_score(Yg, Yp, average="samples", zero_division=0)),
        "f1_samples_parent": float(f1_score(Yg, Yp, average="samples", zero_division=0)),
    })
    return labels, Yg, Yp

def sample_level_metrics(gold_lists, pred_lists):
    """Compute per-sample P/R/F1 and return average."""
    vals = []
    for g, p in zip(gold_lists, pred_lists):
        G, P = set(g), set(p)
        tp = len(G & P)
        fp = len(P - G)
        fn = len(G - P)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        vals.append((prec, rec, f1))
    
    arr = np.array(vals) if vals else np.zeros((0, 3))
    return {
        "precision_sample_avg": float(arr[:, 0].mean() if arr.size else 0.0),
        "recall_sample_avg": float(arr[:, 1].mean() if arr.size else 0.0),
        "f1_sample_avg": float(arr[:, 2].mean() if arr.size else 0.0)
    }

# ============================================================================
# GENERATION
# ============================================================================

def generate_batch(model, tokenizer, prompts, max_len=5120, gen_max_new=128, 
                   decoding="greedy", num_beams=1, temperature=1.0, 
                   top_p=0.95, top_k=50):
    """Generate text for a batch of prompts."""
    # Prepare inputs
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    
    input_texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    
    encodings = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    input_ids = encodings['input_ids'].to(model.device)
    attention_mask = encodings['attention_mask'].to(model.device)
    
    # Generation kwargs
    gen_kwargs = {
        "max_new_tokens": gen_max_new,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,
    }
    
    if decoding == "beam":
        gen_kwargs["num_beams"] = num_beams
        gen_kwargs["early_stopping"] = True
    elif decoding == "sample":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    
    # Decode
    generated_texts = []
    for i, output in enumerate(outputs):
        # Remove input prompt
        input_len = input_ids[i].shape[0]
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

# ============================================================================
# PRETTY PRINT
# ============================================================================

def _pretty_print_block(title: str, d: dict):
    log.info(f"\n--- {title} ---")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            log.info(f"  {k:>35s}: {v:.6f}")
        else:
            log.info(f"  {k:>35s}: {v}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--test_jsonl", required=True, help="Test JSONL file")
    parser.add_argument("--subset_size", type=int, default=None, 
                        help="Test on subset (e.g., 100 samples). None = full test set")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Random seed for subset sampling")
    
    # Model
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--adapter_dir", required=True, help="Adapter directory")
    
    # Generation
    parser.add_argument("--max_len", type=int, default=5120)
    parser.add_argument("--gen_max_new", type=int, default=128)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--decoding", choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--N_max_terms", type=int, default=12, 
                        help="Max diagnosis terms to extract from generation")
    
    # SapBERT Mapper arguments (same as baseline)
    parser.add_argument("--icd_index_dir", required=True, 
                        help="Directory with FAISS index for ICD-9 codes")
    parser.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--faiss_rows", type=int, default=50)
    parser.add_argument("--tau_cos", type=float, default=0.40)
    parser.add_argument("--tau_final", type=float, default=0.60)
    parser.add_argument("--w_cos", type=float, default=0.6)
    parser.add_argument("--w_fuz", type=float, default=0.4)
    
    # Multi-GPU
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--tmp_dir", default="runs_textgen_rag/test_shards")
    
    # Output
    parser.add_argument("--out_metrics", required=True, help="Output metrics JSON path")
    parser.add_argument("--print_samples", type=int, default=5)
    
    # Device
    parser.add_argument("--use_bf16", action="store_true")
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    if is_main_process():
        log.info("=" * 80)
        log.info("RAG TEXT GENERATION - TESTING (with SapBERT Mapping)")
        log.info("=" * 80)
        log.info(f"Test data: {args.test_jsonl}")
        log.info(f"Base model: {args.base_model}")
        log.info(f"Adapter: {args.adapter_dir}")
        log.info(f"Decoding: {args.decoding}")
        log.info(f"ICD Index: {args.icd_index_dir}")
        log.info(f"Encoder: {args.encoder_model}")
        if args.subset_size:
            log.info(f"⚠️  SUBSET MODE: Testing on {args.subset_size} samples only")
        log.info("=" * 80)
    
    # Load test data
    examples = []
    with open(args.test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    total_examples = len(examples)
    
    if is_main_process():
        log.info(f"\n📂 Loaded {total_examples} examples from {args.test_jsonl}")
        # Show available fields in first example
        if examples:
            example_keys = list(examples[0].keys())
            log.info(f"   Available fields: {example_keys}")
            # Show sample target_codes if present
            if 'target_codes' in examples[0]:
                sample_codes = examples[0]['target_codes']
                if isinstance(sample_codes, list):
                    log.info(f"   Sample target_codes (first example): {sample_codes[:5]}...")
                else:
                    log.info(f"   Sample target_codes (first example): {sample_codes}")
    
    # Subset sampling
    if args.subset_size and args.subset_size < len(examples):
        np.random.seed(args.subset_seed)
        subset_indices = np.random.choice(len(examples), args.subset_size, replace=False)
        examples = [examples[i] for i in sorted(subset_indices)]
        
        if is_main_process():
            log.info(f"\n🎯 Subset sampling:")
            log.info(f"   Total in file: {total_examples}")
            log.info(f"   Testing on: {len(examples)} samples (seed={args.subset_seed})")
    else:
        if is_main_process():
            log.info(f"\n📊 Testing on full set: {len(examples)} samples")
    
    # ✓ Extract prompts, targets, and gold codes properly
    prompts = [ex['prompt'] for ex in examples]
    targets = [ex.get('target', '') for ex in examples]
    
    # Extract gold codes from JSONL examples
    if is_main_process():
        log.info(f"\n🔍 Extracting gold codes from examples...")
    
    gold_codes = []
    extraction_stats = {'target_codes': 0, 'other_fields': 0, 'empty': 0}
    
    for ex in examples:
        codes = extract_gold_codes_from_example(ex)
        gold_codes.append(codes)
        
        # Track extraction source for diagnostics
        if 'target_codes' in ex and ex['target_codes']:
            extraction_stats['target_codes'] += 1
        elif any(field in ex for field in ['gold_codes', 'codes', 'labels', 'icd9_codes']):
            extraction_stats['other_fields'] += 1
        else:
            extraction_stats['empty'] += 1
    
    # Build label space
    all_codes = sorted(set([c for lst in gold_codes for c in lst]))
    
    if is_main_process():
        log.info(f"\n📋 Gold code extraction results:")
        log.info(f"   Total unique codes: {len(all_codes)}")
        log.info(f"   Extraction sources:")
        log.info(f"     - target_codes field: {extraction_stats['target_codes']}")
        log.info(f"     - other fields: {extraction_stats['other_fields']}")
        log.info(f"     - empty: {extraction_stats['empty']}")
        log.info(f"   Avg codes per sample: {np.mean([len(g) for g in gold_codes]):.2f}")
        
        if len(all_codes) == 0:
            log.error("❌ ERROR: No gold codes found in test set!")
            log.error("   Expected JSONL format:")
            log.error('   {"prompt": "...", "target": "...", "target_codes": ["410.71", "414.01", ...]}')
            log.error(f"   Found fields: {list(examples[0].keys())}")
            return 1  # Exit with error
        
        # Show sample of extracted codes
        sample_gold = [g for g in gold_codes[:3] if g]
        if sample_gold:
            log.info(f"\n   Sample gold codes (first few examples):")
            for i, codes in enumerate(sample_gold[:3]):
                log.info(f"     Example {i+1}: {codes[:5]}{'...' if len(codes) > 5 else ''}")
    
    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and 
                               torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    
    if is_main_process():
        log.info(f"\n🤖 Loading tokenizer from {args.adapter_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if is_main_process():
        log.info(f"🤖 Loading base model: {args.base_model}")
        log.info(f"   dtype: {dtype}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    if is_main_process():
        log.info(f"📦 Loading adapter from {args.adapter_dir}")
    
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.config.use_cache = True
    
    dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    
    if is_main_process():
        log.info(f"✅ Model loaded on device: {dev}")
    
    # ========================================================================
    # INITIALIZE SAPBERT MAPPER
    # ========================================================================
    
    if is_main_process():
        log.info(f"\n🔍 Initializing SapBERT mapper...")
        log.info(f"   Index dir: {args.icd_index_dir}")
        log.info(f"   Encoder: {args.encoder_model}")
        log.info(f"   Weights: w_cos={args.w_cos}, w_fuz={args.w_fuz}")
    
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos,
        tau_final=args.tau_final,
        w_cos=args.w_cos,
        w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    
    if is_main_process():
        log.info("✅ Mapper initialized")
    
    # ========================================================================
    # DISTRIBUTED SHARDING
    # ========================================================================
    
    if args.distributed:
        maybe_init_dist()
        rank = int(os.environ.get("RANK", "0"))
        W = world_size()
        idxs = shard_indices(len(prompts), rank, W)
    else:
        rank, W = 0, 1
        idxs = list(range(len(prompts)))
    
    shard_prompts = [prompts[i] for i in idxs]
    shard_gold = [gold_codes[i] for i in idxs]
    
    # ========================================================================
    # GENERATION
    # ========================================================================
    
    if is_main_process():
        log.info(f"\n🚀 Generating predictions for {len(shard_prompts)} examples...")
    
    generated_texts = []
    bs = args.gen_batch_size
    t0 = time.time()
    
    for i in range(0, len(shard_prompts), bs):
        batch_prompts = shard_prompts[i:i+bs]
        batch_outputs = generate_batch(
            model, tokenizer, batch_prompts,
            max_len=args.max_len,
            gen_max_new=args.gen_max_new,
            decoding=args.decoding,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        generated_texts.extend(batch_outputs)
        
        if is_main_process() and (i + bs) % (bs * 10) == 0:
            log.info(f"  Progress: {min(i+bs, len(shard_prompts))}/{len(shard_prompts)}")
    
    elapsed = time.time() - t0
    if is_main_process():
        log.info(f"✅ Generation completed in {elapsed:.2f}s ({elapsed/max(1,len(shard_prompts)):.2f}s/sample)")
    
    # ========================================================================
    # EXTRACT TERMS AND MAP TO ICD-9
    # ========================================================================
    
    if is_main_process():
        log.info(f"\n📝 Extracting diagnosis terms from generated text...")
    
    # Extract terms from generated text
    terms_lists = []
    for text in generated_texts:
        terms = extract_terms_from_generation(text, max_terms=args.N_max_terms)
        terms_lists.append(terms)
    
    if is_main_process():
        log.info(f"✅ Extracted terms from {len(terms_lists)} generations")
        log.info(f"\n🔍 Mapping terms to ICD-9 codes using SapBERT...")
    
    # Map terms to ICD-9 codes using SapBERT
    pred_codes = mapper.map_terms(terms_lists)
    
    if is_main_process():
        log.info(f"✅ Mapping completed")
        
        # Mapper diagnostics
        if mapper.last_stats:
            n_terms = np.array([n for (n, m) in mapper.last_stats], dtype=np.float32)
            n_map = np.array([m for (n, m) in mapper.last_stats], dtype=np.float32)
            log.info(f"   Avg terms per sample: {n_terms.mean():.2f}")
            log.info(f"   Avg mapped per sample: {n_map.mean():.2f}")
            unmappable_rate = np.mean(np.where(n_terms > 0, 1.0 - (n_map / np.maximum(n_terms, 1)), 0.0))
            log.info(f"   Unmappable term rate: {unmappable_rate:.4f}")
    
    # ========================================================================
    # SAVE SHARD
    # ========================================================================
    
    os.makedirs(args.tmp_dir, exist_ok=True)
    shard_path = os.path.join(args.tmp_dir, f"shard_{rank:03d}_of_{W:03d}.pkl")
    
    with open(shard_path, "wb") as f:
        pickle.dump({
            "idxs": idxs,
            "generated": generated_texts,
            "terms": terms_lists,
            "predicted": pred_codes,
            "gold": shard_gold,
        }, f)
    
    if is_main_process():
        log.info(f"💾 Saved shard to {shard_path}")
    
    barrier()
    
    # ========================================================================
    # RANK-0: MERGE AND COMPUTE METRICS
    # ========================================================================
    
    if rank == 0:
        log.info("\n" + "=" * 80)
        log.info("MERGING SHARDS AND COMPUTING METRICS")
        log.info("=" * 80)
        
        # Merge all shards
        shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
        all_idx, all_gen, all_terms, all_pred, all_gold = [], [], [], [], []
        
        for sp in shards:
            with open(sp, "rb") as f:
                D = pickle.load(f)
            all_idx.extend(D["idxs"])
            all_gen.extend(D["generated"])
            all_terms.extend(D["terms"])
            all_pred.extend(D["predicted"])
            all_gold.extend(D["gold"])
        
        # Restore original order
        order = np.argsort(np.array(all_idx))
        generated_all = [all_gen[i] for i in order]
        terms_all = [all_terms[i] for i in order]
        pred_all = [all_pred[i] for i in order]
        gold_all = [all_gold[i] for i in order]
        
        # Restrict to evaluation label space
        gold_eval = restrict_to(gold_all, all_codes)
        pred_eval = restrict_to(pred_all, all_codes)
        
        # Convert to multi-hot
        y_true = multihot(gold_eval, all_codes)
        y_pred = multihot(pred_eval, all_codes)
        
        # Compute metrics
        metrics = eval_pack(y_true, y_pred)
        
        # Parent metrics
        parent_metrics = {}
        add_parent_metrics_full(parent_metrics, gold_eval, pred_eval)
        metrics.update(parent_metrics)
        
        # Legacy parent macro F1
        add_parent_macro_f1(metrics, gold_eval, pred_eval)
        
        # Sample-level metrics
        sample_metrics = sample_level_metrics(gold_eval, pred_eval)
        metrics.update(sample_metrics)
        
        # Additional stats
        metrics["num_samples"] = len(gold_all)
        metrics["num_unique_codes"] = len(all_codes)
        metrics["avg_gold_codes_per_sample"] = float(np.mean([len(g) for g in gold_all]))
        metrics["avg_pred_codes_per_sample"] = float(np.mean([len(p) for p in pred_all]))
        
        # Mapper stats
        if mapper.last_stats:
            n_terms = np.array([n for (n, m) in mapper.last_stats], dtype=np.float32)
            n_map = np.array([m for (n, m) in mapper.last_stats], dtype=np.float32)
            metrics["mean_terms_per_visit"] = float(n_terms.mean())
            metrics["mean_mapped_terms_per_visit"] = float(n_map.mean())
            metrics["unmappable_term_rate"] = float(np.mean(np.where(n_terms > 0, 1.0 - (n_map / np.maximum(n_terms, 1)), 0.0)))
        
        # Subset info
        if args.subset_size:
            metrics["total_test_examples"] = total_examples
            metrics["subset_size"] = len(gold_all)
            metrics["subset_seed"] = args.subset_seed
        
        # ====================================================================
        # PRINT SAMPLE PREDICTIONS
        # ====================================================================
        
        log.info("\n" + "=" * 80)
        log.info("SAMPLE PREDICTIONS")
        log.info("=" * 80)
        
        n_show = min(args.print_samples, len(generated_all))
        for i in range(n_show):
            log.info(f"\n{'─' * 76}")
            log.info(f"Sample {i+1}/{n_show}")
            log.info(f"{'─' * 76}")
            
            # Show first 300 chars of prompt
            prompt_preview = prompts[order[i]][:300].replace('\n', ' ')
            log.info(f"📝 PROMPT (first 300 chars):")
            log.info(f"   {prompt_preview}...")
            
            log.info(f"\n🤖 GENERATED TEXT:")
            gen_preview = generated_all[i][:500]  # First 500 chars
            for line in gen_preview.split('\n')[:3]:
                if line.strip():
                    log.info(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
            
            log.info(f"\n📋 EXTRACTED TERMS ({len(terms_all[i])} terms):")
            if terms_all[i]:
                for j, term in enumerate(terms_all[i][:8]):  # Show first 8
                    log.info(f"   {j+1}. {term[:80]}{'...' if len(term) > 80 else ''}")
                if len(terms_all[i]) > 8:
                    log.info(f"   ... and {len(terms_all[i])-8} more")
            else:
                log.info(f"   (none)")
            
            log.info(f"\n🎯 MAPPED ICD-9 CODES ({len(pred_all[i])} codes):")
            pred_str = ', '.join(sorted(pred_all[i])) if pred_all[i] else '(none)'
            gold_str = ', '.join(sorted(gold_all[i])) if gold_all[i] else '(none)'
            log.info(f"   Predicted: {pred_str}")
            log.info(f"   Gold:      {gold_str}")
            
            # Per-sample metrics
            if gold_all[i] or pred_all[i]:
                G = set(gold_all[i])
                P = set(pred_all[i])
                tp = len(G & P)
                fp = len(P - G)
                fn = len(G - P)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                
                log.info(f"\n📊 SAMPLE METRICS:")
                log.info(f"   TP={tp}, FP={fp}, FN={fn}")
                log.info(f"   Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
        
        # ====================================================================
        # PRINT OVERALL METRICS
        # ====================================================================
        
        log.info("\n" + "=" * 80)
        log.info("OVERALL METRICS")
        log.info("=" * 80)
        
        # Group metrics
        micro_metrics = {k: v for k, v in metrics.items() if 'micro' in k}
        macro_metrics = {k: v for k, v in metrics.items() if 'macro' in k}
        sample_metrics_dict = {k: v for k, v in metrics.items() if 'sample' in k}
        other_metrics = {k: v for k, v in metrics.items() 
                        if 'micro' not in k and 'macro' not in k and 'sample' not in k}
        
        _pretty_print_block("MICRO METRICS (global)", micro_metrics)
        _pretty_print_block("MACRO METRICS (per-label avg)", macro_metrics)
        _pretty_print_block("SAMPLE-LEVEL METRICS (per-sample avg)", sample_metrics_dict)
        _pretty_print_block("STATISTICS", other_metrics)
        
        log.info("=" * 80)
        
        # ====================================================================
        # SAVE METRICS
        # ====================================================================
        
        output_data = {
            "config": {
                "test_jsonl": args.test_jsonl,
                "base_model": args.base_model,
                "adapter_dir": args.adapter_dir,
                "decoding": args.decoding,
                "num_beams": args.num_beams,
                "max_len": args.max_len,
                "gen_max_new": args.gen_max_new,
                "N_max_terms": args.N_max_terms,
                "icd_index_dir": args.icd_index_dir,
                "encoder_model": args.encoder_model,
                "w_cos": args.w_cos,
                "w_fuz": args.w_fuz,
                "subset_size": args.subset_size,
                "subset_seed": args.subset_seed if args.subset_size else None,
            },
            "metrics": metrics,
        }
        
        os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
        with open(args.out_metrics, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        log.info(f"\n✅ Metrics saved to: {args.out_metrics}")
        log.info("=" * 80)
    
    cleanup_dist()
    return 0

if __name__ == "__main__":
    sys.exit(main())