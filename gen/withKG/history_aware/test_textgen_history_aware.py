#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_textgen_history_aware.py
Comprehensive test script for history-aware diagnosis generation models.
"""

import os, json, time, argparse, logging, sys, re
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

from common_textgen import (
    ICDMapper, restrict_to, multihot, eval_pack,
    format_icd9, is_valid_icd9, get_icd9_parent
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - INFO - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

def is_main_process():
    return int(os.environ.get('RANK', 0)) == 0

# ============================================================================
# HELPER FUNCTIONS FOR COMPREHENSIVE METRICS
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
    g_parents = [[get_icd9_parent(c) for c in lst] for lst in gold_lists]
    p_parents = [[get_icd9_parent(c) for c in lst] for lst in pred_lists]
    
    all_parents = sorted({p for lst in g_parents for p in lst if p})
    
    if not all_parents:
        metrics_dict.update({
            "precision_macro_parent": 0.0,
            "recall_macro_parent": 0.0,
            "f1_macro_parent": 0.0,
            "precision_micro_parent": 0.0,
            "recall_micro_parent": 0.0,
            "f1_micro_parent": 0.0,
            # "precision_samples_parent": 0.0,
            # "recall_samples_parent": 0.0,
            # "f1_samples_parent": 0.0,
        })
        return all_parents, None, None
    
    def to_multihot(lists, labels):
        idx_map = {c: i for i, c in enumerate(labels)}
        Y = np.zeros((len(lists), len(labels)), dtype=np.int32)
        for i, lst in enumerate(lists):
            for code in lst:
                if code in idx_map:
                    Y[i, idx_map[code]] = 1
        return Y
    
    Yg = to_multihot(g_parents, all_parents)
    Yp = to_multihot(p_parents, all_parents)
    
    metrics_dict.update({
        "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
        "recall_macro_parent": float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent": float(f1_score(Yg, Yp, average="macro", zero_division=0)),
        "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
        "recall_micro_parent": float(recall_score(Yg, Yp, average="micro", zero_division=0)),
        "f1_micro_parent": float(f1_score(Yg, Yp, average="micro", zero_division=0)),
        # "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
        # "recall_samples_parent": float(recall_score(Yg, Yp, average="samples", zero_division=0)),
        # "f1_samples_parent": float(f1_score(Yg, Yp, average="samples", zero_division=0)),
    })
    return all_parents, Yg, Yp

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
        "precision_sample_avg": float(arr[:, 0].mean() if arr.size else 0.0),
        "recall_sample_avg": float(arr[:, 1].mean() if arr.size else 0.0),
        "f1_sample_avg": float(arr[:, 2].mean() if arr.size else 0.0),
        "precision_samples": float(arr[:, 0].mean() if arr.size else 0.0),
        "recall_samples": float(arr[:, 1].mean() if arr.size else 0.0),
        "f1_samples": float(arr[:, 2].mean() if arr.size else 0.0),
        "precision_samples_set": float(arr[:, 0].mean() if arr.size else 0.0),
        "recall_samples_set": float(arr[:, 1].mean() if arr.size else 0.0),
        "f1_samples_set": float(arr[:, 2].mean() if arr.size else 0.0),
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

def extract_terms_from_generation(text: str, max_terms: int = 12) -> list:
    """Extract diagnosis terms from generated text."""
    terms = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = line.lstrip('-* ')
        line = re.sub(r'^\d+\.\s*', '', line)
        line = re.sub(r'^\(\d+\)\s*', '', line)
        for part in line.split(' | '):
            part = part.strip()
            if part and part not in terms:
                terms.append(part)
            if len(terms) >= max_terms:
                break
        if len(terms) >= max_terms:
            break
    return terms

def _pretty_print_block(title: str, d: dict):
    """Pretty print metrics block."""
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
    parser.add_argument("--test_tsv", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--max_len", type=int, default=5120)
    parser.add_argument("--gen_max_new", type=int, default=128)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--icd_index_dir", required=True)
    parser.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--faiss_rows", type=int, default=50)
    parser.add_argument("--tau_cos", type=float, default=0.40)
    parser.add_argument("--tau_final", type=float, default=0.60)
    parser.add_argument("--w_cos", type=float, default=0.6)
    parser.add_argument("--w_fuz", type=float, default=0.4)
    parser.add_argument("--N_max_terms", type=int, default=12)
    parser.add_argument("--print_samples", type=int, default=5)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--top_codes_csv", default="")
    parser.add_argument("--bottom_codes_csv", default="")
    parser.add_argument("--top_parent_csv", default="")
    args = parser.parse_args()

    if is_main_process():
        log.info("=" * 80)
        log.info("HISTORY-AWARE TEXT GENERATION - TESTING")
        log.info("=" * 80)
        log.info(f"Test data: {args.test_tsv}")
        log.info(f"Gold codes: {args.test_jsonl}")
        log.info(f"Base model: {args.base_model}")
        log.info(f"Adapter: {args.adapter_dir}")
        if args.subset_size:
            log.info(f"  SUBSET MODE: {args.subset_size} samples")
        log.info("=" * 80)

    # Load data
    df = pd.read_csv(args.test_tsv, sep='\t')
    prompts = df['prompt'].tolist()
    targets = df['target'].tolist() if 'target' in df.columns else [''] * len(df)

    gold_codes = []
    with open(args.test_jsonl, "r") as f:
        for line in f:
            ex = json.loads(line)
            codes = ex.get('target_icd_codes', [])
            if isinstance(codes, str):
                for sep in [',', ';', ' ', '|']:
                    if sep in codes:
                        codes = [format_icd9(c.strip()) for c in codes.split(sep) if c.strip()]
                        break
                else:
                    codes = [format_icd9(codes)] if codes.strip() else []
            elif isinstance(codes, list):
                codes = [format_icd9(c) for c in codes if c]
            gold_codes.append(codes)

    # Subset sampling
    total_examples = len(prompts)
    if args.subset_size and args.subset_size < total_examples:
        np.random.seed(args.subset_seed)
        subset_indices = np.random.choice(total_examples, args.subset_size, replace=False)
        subset_indices = sorted(subset_indices)
        prompts = [prompts[i] for i in subset_indices]
        targets = [targets[i] for i in subset_indices]
        gold_codes = [gold_codes[i] for i in subset_indices]
        if is_main_process():
            log.info(f"\n Subset: {len(prompts)} samples (seed={args.subset_seed})")
    else:
        if is_main_process():
            log.info(f"\n Full test set: {len(prompts)} samples")

    all_codes = sorted(set([c for lst in gold_codes for c in lst]))
    if is_main_process():
        log.info(f"   Unique codes: {len(all_codes)}")
        log.info(f"   Avg codes/sample: {np.mean([len(g) for g in gold_codes]):.2f}")

    # ============================================================================
    # MODEL LOADING - EXACTLY AS YOUR ORIGINAL
    # ============================================================================
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    
    if is_main_process():
        log.info(f"\n Loading tokenizer from adapter: {args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    if is_main_process():
        log.info(f" Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    if is_main_process():
        log.info(f" Loading adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    
    model.config.use_cache = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    if is_main_process():
        log.info(f" Model ready on {dev}")

    # ============================================================================
    # SAPBERT MAPPER
    # ============================================================================
    if is_main_process():
        log.info(f"\n Initializing SapBERT mapper...")
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
        log.info(" Mapper ready")

    # ============================================================================
    # GENERATION
    # ============================================================================
    if is_main_process():
        log.info(f"\n Generating predictions...")
    t0 = time.time()
    generations = []
    bs = args.gen_batch_size
    for i in range(0, len(prompts), bs):
        batch_prompts = prompts[i:i+bs]
        conversations = [[{"role": "user", "content": p}] for p in batch_prompts]
        input_texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        encodings = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].to(model.device)
        attention_mask = encodings['attention_mask'].to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.gen_max_new,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        for j, output in enumerate(outputs):
            input_len = input_ids[j].shape[0]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            generations.append(text)
        if is_main_process() and (i + bs) % (bs * 10) == 0:
            log.info(f"  Progress: {min(i+bs, len(prompts))}/{len(prompts)}")
    elapsed = time.time() - t0
    if is_main_process():
        log.info(f" Generation done ({elapsed:.2f}s, {elapsed/max(1,len(prompts)):.2f}s/sample)")

    # ============================================================================
    # TERM EXTRACTION AND MAPPING
    # ============================================================================
    if is_main_process():
        log.info(f"\n Extracting terms...")
    terms_lists = [extract_terms_from_generation(gen, max_terms=args.N_max_terms) for gen in generations]
    if is_main_process():
        log.info(f" Mapping to ICD-9 codes...")
    pred_codes = mapper.map_terms(terms_lists)
    if is_main_process():
        log.info(f" Mapping done")

    # ============================================================================
    # COMPUTE METRICS - COMPREHENSIVE
    # ============================================================================
    gold_eval = restrict_to(gold_codes, all_codes)
    pred_eval = restrict_to(pred_codes, all_codes)
    y_true = multihot(gold_eval, all_codes)
    y_pred = multihot(pred_eval, all_codes)
    
    metrics = eval_pack(y_true, y_pred)
    parent_labels, Yg_par, Yp_par = add_parent_metrics_full(metrics, gold_eval, pred_eval)
    metrics.update(sample_level_metrics(gold_eval, pred_eval))
    
    metrics["num_samples"] = len(gold_codes)
    metrics["num_unique_codes"] = len(all_codes)
    metrics["avg_gold_codes_per_sample"] = float(np.mean([len(g) for g in gold_codes]))
    metrics["avg_pred_codes_per_sample"] = float(np.mean([len(p) for p in pred_codes]))
    
    if hasattr(mapper, "last_stats") and mapper.last_stats:
        n_terms = np.array([n for (n, m) in mapper.last_stats], dtype=np.float32)
        n_map = np.array([m for (n, m) in mapper.last_stats], dtype=np.float32)
        metrics["mean_terms_per_visit"] = float(n_terms.mean())
        metrics["mean_mapped_terms_per_visit"] = float(n_map.mean())
        metrics["unmappable_term_rate"] = float(np.mean(np.where(n_terms > 0, 1.0 - (n_map / np.maximum(n_terms, 1)), 0.0)))

    # Per-label table: FULL
    out_dir = os.path.dirname(args.out_metrics)
    per_label_table(y_true, y_pred, all_codes, os.path.join(out_dir, "per_label_FULL.csv"))

    # ============================================================================
    # BUCKET EVALUATIONS
    # ============================================================================
    top_codes = _read_first_col_codes(args.top_codes_csv)
    bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
    top_parents = _read_first_col_parents(args.top_parent_csv)
    results_ext = {}

    def restrict_and_eval(bucket_codes, bucket_name):
        valid_bucket_codes = [c for c in bucket_codes if c in all_codes]
        if not valid_bucket_codes:
            return None
        idx = {c: i for i, c in enumerate(all_codes)}
        keep = [idx[c] for c in valid_bucket_codes]
        yt = y_true[:, keep]
        yp = y_pred[:, keep]
        result = eval_pack(yt, yp)
        per_label_table(yt, yp, valid_bucket_codes, os.path.join(out_dir, f"per_label_{bucket_name}.csv"))
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

    if results_ext:
        bucket_path = os.path.join(out_dir, "test_metrics_buckets.json")
        with open(bucket_path, 'w') as f:
            json.dump(results_ext, f, indent=2)

    # ============================================================================
    # SAMPLE PREDICTIONS
    # ============================================================================
    if is_main_process():
        log.info("\n" + "=" * 80)
        log.info("SAMPLE PREDICTIONS")
        log.info("=" * 80)
        n_show = min(max(args.print_samples, 5), len(generations))
        for i in range(n_show):
            log.info(f"\nSample {i+1}/{n_show}")
            log.info(f"PROMPT: {prompts[i][:150]}...")
            log.info(f"GENERATED: {generations[i][:150]}...")
            log.info(f"PREDICTED: {', '.join(sorted(pred_codes[i]))}")
            log.info(f"GOLD: {', '.join(sorted(gold_codes[i]))}")

    # ============================================================================
    # PRINT METRICS
    # ============================================================================
    if is_main_process():
        log.info("\n" + "=" * 80)
        log.info("OVERALL METRICS")
        log.info("=" * 80)
        
        micro = {k: v for k, v in metrics.items() if 'micro' in k}
        macro = {k: v for k, v in metrics.items() if 'macro' in k}
        sample = {k: v for k, v in metrics.items() if 'sample' in k}
        stats = {k: v for k, v in metrics.items() if k in ['num_samples', 'num_unique_codes', 'avg_gold_codes_per_sample', 'avg_pred_codes_per_sample', 'mean_terms_per_visit', 'mean_mapped_terms_per_visit', 'unmappable_term_rate']}
        
        _pretty_print_block("MICRO METRICS", micro)
        _pretty_print_block("MACRO METRICS", macro)
        _pretty_print_block("SAMPLE METRICS", sample)
        _pretty_print_block("STATISTICS", stats)
        
        if "TOP_50_CODES" in results_ext:
            _pretty_print_block("TOP_50_CODES", results_ext["TOP_50_CODES"])
        if "BOTTOM_50_CODES" in results_ext:
            _pretty_print_block("BOTTOM_50_CODES", results_ext["BOTTOM_50_CODES"])
        if "TOP_50_PARENTS" in results_ext:
            _pretty_print_block("TOP_50_PARENTS", results_ext["TOP_50_PARENTS"])
        
        log.info("=" * 80)

    # ============================================================================
    # SAVE OUTPUT
    # ============================================================================
    out_data = []
    for i, (prompt, target, gen, gold, pred) in enumerate(zip(prompts, targets, generations, gold_codes, pred_codes)):
        out_data.append({
            "idx": i,
            "prompt": prompt,
            "target": target,
            "generation": gen,
            "target_codes": gold,
            "predicted_codes": pred
        })
    
    with open(args.out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    
    output_metrics = {
        "config": {
            "test_tsv": args.test_tsv,
            "test_jsonl": args.test_jsonl,
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "subset_size": args.subset_size,
        },
        "metrics": metrics,
    }
    
    with open(args.out_metrics, "w") as f:
        json.dump(output_metrics, f, indent=2)
    
    if is_main_process():
        log.info(f"\n Results saved to {args.out_metrics}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())