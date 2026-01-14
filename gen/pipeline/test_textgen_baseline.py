#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_textgen_baseline.py
Testing script for baseline diagnosis generation models (no KG retrieval).
Maps generated text to ICD-9 codes using SapBERT encoder.
Extracts gold codes from 'target_codes' field in JSONL.
Supports subset evaluation and comprehensive metrics.
"""

import os, json, time, argparse, logging, sys, glob, pickle, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

sys.path.insert(0, '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/pipeline/')

from common_textgen_util import (
    log, is_main_process, world_size, local_rank,
    ICDMapper, format_icd9, is_valid_icd9,
    restrict_to, multihot, eval_pack,
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

def update_prompt_k(prompt: str, new_k: int) -> str:
    """Update the 'Maximum: X lines' instruction in prompt to new k value."""
    import re
    pattern = r'(- Maximum:\s*)\d+(\s*lines?)'
    replacement = rf'\g<1>{new_k}\g<2>'
    updated = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
    
    if not re.search(pattern, prompt):
        log.warning(f"Could not find 'Maximum: X lines' pattern in prompt")
    elif updated == prompt:
        log.warning(f"Prompt k update failed - prompt unchanged")
    
    return updated

def _clean_diagnosis_term(term: str) -> str:
    term = term.strip()
    if not term:
        return ""
    term = term.lstrip('- ')
    term = term.lstrip('* ')
    term = term.lstrip('• ')
    term = re.sub(r'^\d+\.\s*', '', term)
    term = re.sub(r'^\(\d+\)\s*', '', term)
    term = re.sub(r'^\d+\)\s*', '', term)
    term = term.strip()
    if term and term[-1] in [',', ';', ':']:
        term = term[:-1].strip()
    return term

def extract_terms_from_generation(text: str, max_terms: int = 12) -> list:
    terms = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ';' in line:
            parts = line.split(';')
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                part = _clean_diagnosis_term(part)
                if part and part not in terms:
                    terms.append(part)
                if len(terms) >= max_terms:
                    return terms
        else:
            line = _clean_diagnosis_term(line)
            if line and line not in terms:
                terms.append(line)
            if len(terms) >= max_terms:
                return terms
    return terms

def extract_gold_codes_from_example(example: dict) -> list:
    if 'target_codes' in example:
        codes = example['target_codes']
        if isinstance(codes, list):
            return [format_icd9(c) for c in codes if c]
        elif isinstance(codes, str):
            for sep in [',', ';', ' ', '|']:
                if sep in codes:
                    return [format_icd9(c.strip()) for c in codes.split(sep) if c.strip()]
            return [format_icd9(codes)] if codes.strip() else []
    for field in ['gold_codes', 'codes', 'labels', 'icd9_codes', 'icd_codes']:
        if field in example:
            codes = example[field]
            if isinstance(codes, list):
                return [format_icd9(c) for c in codes if c]
            elif isinstance(codes, str):
                for sep in [',', ';', ' ', '|']:
                    if sep in codes:
                        return [format_icd9(c.strip()) for c in codes.split(sep) if c.strip()]
                return [format_icd9(codes)] if codes.strip() else []
    return []

# ============================================================================
# METRICS HELPERS
# ============================================================================

def _read_first_col_codes(path):
    """Read codes from first column of CSV."""
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return []
        col = df.columns[0]
        vals = [format_icd9(str(x)) for x in df[col].tolist()]
        return sorted({v for v in vals if is_valid_icd9(v)})
    except Exception as e:
        log.warning(f"Could not read codes from {path}: {e}")
        return []

def _read_first_col_parents(path):
    """Read parent codes from first column of CSV."""
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return []
        col = df.columns[0]
        raw = [format_icd9(str(x)) for x in df[col].tolist()]
        parents = sorted({get_icd9_parent(v) for v in raw if v})
        return [p for p in parents if p]
    except Exception as e:
        log.warning(f"Could not read parent codes from {path}: {e}")
        return []

def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
    """Add comprehensive parent-level metrics."""
    g = [[get_icd9_parent(c) for c in lst] for lst in gold_lists]
    p = [[get_icd9_parent(c) for c in lst] for lst in pred_lists]
    labels = sorted({x for lst in g for x in lst})
    if not labels:
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
    """Calculate sample-level set metrics."""
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
        "precision_samples_set": float(arr[:, 0].mean() if arr.size else 0.0),
        "recall_samples_set": float(arr[:, 1].mean() if arr.size else 0.0),
        "f1_samples_set": float(arr[:, 2].mean() if arr.size else 0.0)
    }

def per_label_table(y_true, y_pred, labels, out_csv_path=None):
    """Generate per-label metrics table."""
    if len(labels) == 0:
        return pd.DataFrame()
    
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    if len(p) != len(labels):
        log.warning(f"Mismatch: {len(labels)} labels but {len(p)} metrics. Truncating.")
        min_len = min(len(labels), len(p))
        labels = labels[:min_len]
        p = p[:min_len]
        r = r[:min_len]
        f1 = f1[:min_len]
        support = support[:min_len]
    
    df = pd.DataFrame({
        "code": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support
    })
    
    if out_csv_path:
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        df.to_csv(out_csv_path, index=False)
    return df

# ============================================================================
# GENERATION
# ============================================================================

def generate_batch(model, tokenizer, prompts, max_len=5120, gen_max_new=128, 
                   decoding="greedy", num_beams=1, temperature=1.0, 
                   top_p=0.95, top_k=50):
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
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    generated_texts = []
    for i, output in enumerate(outputs):
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
            log.info(f"  {k:>28s}: {v:.6f}")
        else:
            log.info(f"  {k:>28s}: {v}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_jsonl", required=True, help="Test JSONL file")
    parser.add_argument("--subset_size", type=int, default=None, help="Test on subset. None = full test set")
    parser.add_argument("--subset_seed", type=int, default=42, help="Random seed for subset sampling")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--adapter_dir", default="", help="Adapter directory (optional for base_model_only mode)")
    parser.add_argument("--base_model_only", action="store_true", help="Evaluate base model without adapter")
    parser.add_argument("--max_len", type=int, default=5120)
    parser.add_argument("--gen_max_new", type=int, default=128)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--decoding", choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--N_max_terms", type=int, default=12, help="Max diagnosis terms to extract")
    parser.add_argument("--icd_index_dir", required=True, help="FAISS index for ICD-9 codes")
    parser.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--faiss_rows", type=int, default=50)
    parser.add_argument("--tau_cos", type=float, default=0.40)
    parser.add_argument("--tau_final", type=float, default=0.60)
    parser.add_argument("--w_cos", type=float, default=0.6)
    parser.add_argument("--w_fuz", type=float, default=0.4)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--tmp_dir", default="runs_textgen/baseline_test_shards")
    parser.add_argument("--out_metrics", required=True, help="Output metrics JSON path")
    parser.add_argument("--print_samples", type=int, default=5)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--update_prompt_k", action="store_true",
                       help="Update prompt k value to match N_max_terms (for k-ablation)")
    
    # Bucket evaluation arguments
    parser.add_argument("--top_codes_csv", default="", help="Path to top 50 codes CSV")
    parser.add_argument("--bottom_codes_csv", default="", help="Path to bottom 50 codes CSV")
    parser.add_argument("--top_parent_csv", default="", help="Path to top 50 parent codes CSV")
    
    args = parser.parse_args()

    # Validate arguments
    if not args.base_model_only and not args.adapter_dir:
        log.error("ERROR: --adapter_dir is required unless --base_model_only is set")
        return 1

    # ========================================================================
    # SETUP
    # ========================================================================
    if is_main_process():
        log.info("=" * 80)
        if args.base_model_only:
            log.info("BASELINE TESTING - BASE MODEL ONLY (ABLATION)")
        else:
            log.info("BASELINE TESTING (with SapBERT Mapping)")
        log.info("=" * 80)
        log.info(f"Test data: {args.test_jsonl}")
        log.info(f"Base model: {args.base_model}")
        if args.base_model_only:
            log.info(f"ABLATION MODE: Using base model only (no adapter)")
        else:
            log.info(f"Adapter: {args.adapter_dir}")
        log.info(f"Decoding: {args.decoding}")
        log.info(f"ICD Index: {args.icd_index_dir}")
        log.info(f"Encoder: {args.encoder_model}")
        if args.subset_size:
            log.info(f"SUBSET MODE: Testing on {args.subset_size} samples only")
        log.info("=" * 80)

    # Load test data
    examples = []
    with open(args.test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    total_examples = len(examples)

    if is_main_process():
        log.info(f"\nLoaded {total_examples} examples from {args.test_jsonl}")
        if examples:
            example_keys = list(examples[0].keys())
            log.info(f"  Available fields: {example_keys}")
            if 'target_codes' in examples[0]:
                sample_codes = examples[0]['target_codes']
                if isinstance(sample_codes, list):
                    log.info(f"  Sample target_codes: {sample_codes[:5]}...")
                else:
                    log.info(f"  Sample target_codes: {sample_codes}")

            # Check original k value
            if 'prompt' in examples[0]:
                sample_prompt = examples[0]['prompt']
                match = re.search(r'- Maximum:\s*(\d+)\s*lines?', sample_prompt, re.IGNORECASE)
                if match:
                    original_k = int(match.group(1))
                    log.info(f"  Original k in prompts: {original_k}")
                    if args.update_prompt_k and args.N_max_terms != original_k:
                        log.info(f"  Will update prompts from k={original_k} to k={args.N_max_terms}")

    # Subset sampling
    if args.subset_size and args.subset_size < len(examples):
        np.random.seed(args.subset_seed)
        subset_indices = np.random.choice(len(examples), args.subset_size, replace=False)
        examples = [examples[i] for i in sorted(subset_indices)]
        if is_main_process():
            log.info(f"\nSubset sampling:")
            log.info(f"  Total in file: {total_examples}")
            log.info(f"  Testing on: {len(examples)} samples (seed={args.subset_seed})")
    else:
        if is_main_process():
            log.info(f"\nTesting on full set: {len(examples)} samples")

    # Extract prompts and gold codes
    prompts = [ex['prompt'] for ex in examples]
    targets = [ex.get('target', '') for ex in examples]

    # Update k value if requested
    if args.update_prompt_k:
        prompts = [update_prompt_k(p, args.N_max_terms) for p in prompts]

    # Extract gold codes
    if is_main_process():
        log.info(f"\nExtracting gold codes from examples...")
    gold_codes = []
    extraction_stats = {'target_codes': 0, 'other_fields': 0, 'empty': 0}
    for ex in examples:
        codes = extract_gold_codes_from_example(ex)
        gold_codes.append(codes)
        if 'target_codes' in ex and ex['target_codes']:
            extraction_stats['target_codes'] += 1
        elif any(field in ex for field in ['gold_codes', 'codes', 'labels', 'icd9_codes']):
            extraction_stats['other_fields'] += 1
        else:
            extraction_stats['empty'] += 1
    all_codes = sorted(set([c for lst in gold_codes for c in lst]))
    if is_main_process():
        log.info(f"[DATA] Test size: {len(examples)}")
        log.info(f"[DATA] Eval label space: {len(all_codes)} codes (FULL)")
        if len(all_codes) == 0:
            log.error("ERROR: No gold codes found in test set!")
            log.error("  Expected JSONL format:")
            log.error('  {"prompt": "...", "target": "...", "target_codes": ["410.71", "414.01", ...]}')
            log.error(f"  Found fields: {list(examples[0].keys())}")
            return 1

    if is_main_process() and args.update_prompt_k:
        log.info("\nSample prompt transformation:")
        log.info("=" * 80)
        original = examples[0]['prompt']
        updated = prompts[0]
        
        orig_match = re.search(r'\[FORMAT\].*?\[OUTPUT\]', original, re.DOTALL)
        upd_match = re.search(r'\[FORMAT\].*?\[OUTPUT\]', updated, re.DOTALL)
        
        if orig_match and upd_match:
            log.info("ORIGINAL:")
            log.info(orig_match.group(0))
            log.info("\nUPDATED:")
            log.info(upd_match.group(0))
        log.info("=" * 80)

    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    
    if is_main_process():
        log.info(f"\nLoading tokenizer from base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if is_main_process():
        log.info(f"Loading base model: {args.base_model}")
        log.info(f"  dtype: {dtype}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    if args.base_model_only:
        model = base_model
        if is_main_process():
            log.info(f"Using BASE MODEL ONLY (no adapter) for ablation")
    else:
        if is_main_process():
            log.info(f"Loading adapter from {args.adapter_dir}")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    
    model.config.use_cache = True
    dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    if is_main_process():
        log.info(f"Model loaded on device: {dev}")

    # ========================================================================
    # INITIALIZE SAPBERT MAPPER
    # ========================================================================
    if is_main_process():
        log.info(f"\nInitializing SapBERT mapper...")
        log.info(f"  Index dir: {args.icd_index_dir}")
        log.info(f"  Encoder: {args.encoder_model}")
        log.info(f"  Weights: w_cos={args.w_cos}, w_fuz={args.w_fuz}")
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
        log.info("Mapper initialized")

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
        log.info(f"\nGenerating predictions for {len(shard_prompts)} examples...")
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
        log.info(f"[GEN] Generation done ({elapsed/max(1,len(shard_prompts)):.2f}s/sample).")

    # ========================================================================
    # EXTRACT TERMS AND MAP TO ICD-9
    # ========================================================================
    if is_main_process():
        log.info(f"\nExtracting diagnosis terms from generated text...")
    terms_lists = []
    for text in generated_texts:
        terms = extract_terms_from_generation(text, max_terms=args.N_max_terms)
        terms_lists.append(terms)
    if is_main_process():
        log.info(f"Extracted terms from {len(terms_lists)} generations")
        log.info(f"\nMapping terms to ICD-9 codes using SapBERT...")
    pred_codes = mapper.map_terms(terms_lists)
    if is_main_process():
        log.info(f"Mapping completed")

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
        log.info(f"Saved shard to {shard_path}")
    barrier()

    # ========================================================================
    # RANK-0: MERGE AND COMPUTE METRICS
    # ========================================================================
    if rank == 0:
        log.info("\n" + "=" * 80)
        log.info("MERGING SHARDS AND COMPUTING METRICS")
        log.info("=" * 80)
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
        order = np.argsort(np.array(all_idx))
        generated_all = [all_gen[i] for i in order]
        terms_all = [all_terms[i] for i in order]
        pred_all = [all_pred[i] for i in order]
        gold_all = [all_gold[i] for i in order]
        
        gold_eval = restrict_to(gold_all, all_codes)
        pred_eval = restrict_to(pred_all, all_codes)
        y_true = multihot(gold_eval, all_codes)
        y_pred = multihot(pred_eval, all_codes)
        
        # Main metrics
        metrics = eval_pack(y_true, y_pred)
        
        # Parent metrics
        parent_metrics = {}
        parent_labels, Yg_par, Yp_par = add_parent_metrics_full(parent_metrics, gold_eval, pred_eval)
        metrics.update(parent_metrics)
        
        # Sample-level metrics
        sample_metrics = sample_level_metrics(gold_eval, pred_eval)
        metrics.update(sample_metrics)
        
        # Statistics
        if mapper.last_stats:
            n_terms = np.array([n for (n, m) in mapper.last_stats], dtype=np.float32)
            n_map = np.array([m for (n, m) in mapper.last_stats], dtype=np.float32)
            metrics["mean_terms_per_visit"] = float(n_terms.mean())
            metrics["mean_mapped_terms_per_visit"] = float(n_map.mean())
            metrics["unmappable_term_rate"] = float(np.mean(np.where(n_terms > 0, 1.0 - (n_map / np.maximum(n_terms, 1)), 0.0)))
        
        # Save per-label table
        out_dir = os.path.dirname(args.out_metrics)
        per_label_table(y_true, y_pred, all_codes, os.path.join(out_dir, "per_label_FULL.csv"))
        log.info(f"[INFO] Per-label table saved to {os.path.join(out_dir, 'per_label_FULL.csv')}")
        
        # ========================================================================
        # BUCKET EVALUATIONS
        # ========================================================================
        top_codes = _read_first_col_codes(args.top_codes_csv)
        bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
        top_parents = _read_first_col_parents(args.top_parent_csv)
        results_ext = {}

        def restrict_and_eval(bucket_codes, bucket_name):
            """Only evaluate codes that exist in test set."""
            valid_bucket_codes = [c for c in bucket_codes if c in all_codes]
            
            if not valid_bucket_codes:
                log.warning(f"[BUCKET] {bucket_name}: No codes from bucket found in test set. Skipping.")
                return None
            
            idx = {c: i for i, c in enumerate(all_codes)}
            keep = [idx[c] for c in valid_bucket_codes]
            
            yt = y_true[:, keep]
            yp = y_pred[:, keep]
            result = eval_pack(yt, yp)
            
            per_label_table(yt, yp, valid_bucket_codes, os.path.join(out_dir, f"per_label_{bucket_name}.csv"))
            
            log.info(f"[BUCKET] {bucket_name}: Evaluated {len(valid_bucket_codes)}/{len(bucket_codes)} codes present in test set")
            return result

        if top_codes:
            r = restrict_and_eval(top_codes, "TOP_50_CODES")
            if r:
                results_ext["TOP_50_CODES"] = r

        if bottom_codes:
            r = restrict_and_eval(bottom_codes, "BOTTOM_50_CODES")
            if r:
                results_ext["BOTTOM_50_CODES"] = r

        if top_parents:
            def to_parents(lists):
                return [[get_icd9_parent(c) for c in row] for row in lists]

            gold_parents_all = set()
            for lst in gold_eval:
                for c in lst:
                    p = get_icd9_parent(c)
                    if p:
                        gold_parents_all.add(p)
            
            valid_top_parents = [p for p in top_parents if p in gold_parents_all]
            
            if valid_top_parents:
                def mh(L, labels):
                    idx = {c: i for i, c in enumerate(labels)}
                    Y = np.zeros((len(L), len(labels)), dtype=np.int32)
                    for i, lst in enumerate(L):
                        for c in lst:
                            j = idx.get(c)
                            if j is not None:
                                Y[i, j] = 1
                    return Y

                YgP = mh(to_parents(gold_eval), valid_top_parents)
                YpP = mh(to_parents(pred_eval), valid_top_parents)
                results_ext["TOP_50_PARENTS"] = eval_pack(YgP, YpP)
                per_label_table(YgP, YpP, valid_top_parents, os.path.join(out_dir, "per_label_TOP_50_PARENTS.csv"))
                log.info(f"[BUCKET] TOP_50_PARENTS: Evaluated {len(valid_top_parents)}/{len(top_parents)} parents present in test set")
            else:
                log.warning("[BUCKET] TOP_50_PARENTS: No parent codes from bucket found in test set. Skipping.")

        # Save bucket metrics
        if results_ext:
            bucket_path = os.path.join(out_dir, "test_metrics_buckets.json")
            with open(bucket_path, 'w') as f:
                json.dump(results_ext, f, indent=2)
            log.info(f"[INFO] Bucket metrics saved to {bucket_path}")
        
        # ========================================================================
        # SAMPLE PREDICTIONS
        # ========================================================================
        log.info("\n" + "=" * 80)
        log.info("=== Sample predictions (with per-sample metrics) ===")
        log.info("=" * 80)
        n_show = min(args.print_samples, len(generated_all))
        for i in range(n_show):
            log.info(f"\n[Sample {i+1}]")
            
            gold_str = ', '.join(sorted(gold_all[i])) if gold_all[i] else '(none)'
            log.info(f"  GOLD codes: {gold_str}")
            
            log.info(f"  FREE-TEXT terms:")
            gen_lines = generated_all[i].split('\n')[:3]
            for line in gen_lines:
                if line.strip():
                    log.info(f"    - {line.strip()[:100]}{'...' if len(line.strip()) > 100 else ''}")
            
            pred_str = ', '.join(sorted(pred_all[i])) if pred_all[i] else '(none)'
            log.info(f"  MAPPED ICD-9: {pred_str}")
            
            if gold_all[i] or pred_all[i]:
                G = set(gold_all[i])
                P = set(pred_all[i])
                tp = len(G & P)
                fp = len(P - G)
                fn = len(G - P)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                log.info(f"  Sample metrics -> P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        
        # ========================================================================
        # PRINT METRICS
        # ========================================================================
        log.info(f"\n[INFO] Metrics saved to {args.out_metrics}")
        _pretty_print_block("OVERALL (code-level)", metrics)
        
        if "TOP_50_CODES" in results_ext:
            _pretty_print_block("TOP_50_CODES (code-level)", results_ext["TOP_50_CODES"])
        
        if "BOTTOM_50_CODES" in results_ext:
            _pretty_print_block("BOTTOM_50_CODES (code-level)", results_ext["BOTTOM_50_CODES"])
        
        if "TOP_50_PARENTS" in results_ext:
            _pretty_print_block("TOP_50_PARENTS (parent-level)", results_ext["TOP_50_PARENTS"])
        
        # Save final metrics
        output_data = {
            "config": {
                "test_jsonl": args.test_jsonl,
                "base_model": args.base_model,
                "adapter_dir": args.adapter_dir if not args.base_model_only else "N/A (base model only)",
                "base_model_only": args.base_model_only,
                "mode": "baseline",
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
        
        log.info("=" * 80)
    
    cleanup_dist()
    return 0

if __name__ == "__main__":
    sys.exit(main())