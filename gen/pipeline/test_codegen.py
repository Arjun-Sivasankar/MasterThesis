# -*- coding: utf-8 -*-
import os, sys, json, time, argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from util_codegen_core import (
    set_seed, rank0_print, build_input_text,
    format_icd9_properly, is_valid_icd9, normalize_code, get_icd9_parent,
    codes_to_multihot, eval_sets, hierarchical_eval
)
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

# --- stdout: force line-buffered/unbuffered printing so logs show up live ---
try:
    sys.stdout.reconfigure(line_buffering=True)  # py3.7+
except Exception:
    pass

# -------- Hygiene & perf toggles --------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _read_first_col_codes(path):
    """Read codes from first column of CSV."""
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return []
        col = df.columns[0]
        vals = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        return sorted({v for v in vals if is_valid_icd9(v)})
    except Exception as e:
        rank0_print(f"[WARN] Could not read codes from {path}: {e}")
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
        raw = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        parents = sorted({get_icd9_parent(v) for v in raw if v})
        return [p for p in parents if p]
    except Exception as e:
        rank0_print(f"[WARN] Could not read parent codes from {path}: {e}")
        return []


def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists, labels_vocab):
    """Add comprehensive parent-level metrics."""
    g_parents = [[get_icd9_parent(c) for c in lst] for lst in gold_lists]
    p_parents = [[get_icd9_parent(c) for c in lst] for lst in pred_lists]
    
    # Get all unique parent labels
    all_parents = sorted({p for lst in g_parents for p in lst if p})
    
    if not all_parents:
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
        return all_parents, None, None
    
    # Create multihot encoding
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
        "recall_macro_parent":    float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent":        float(f1_score(Yg, Yp, average="macro", zero_division=0)),
        "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
        "recall_micro_parent":    float(recall_score(Yg, Yp, average="micro", zero_division=0)),
        "f1_micro_parent":        float(f1_score(Yg, Yp, average="micro", zero_division=0)),
        "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
        "recall_samples_parent":    float(recall_score(Yg, Yp, average="samples", zero_division=0)),
        "f1_samples_parent":        float(f1_score(Yg, Yp, average="samples", zero_division=0)),
    })
    return all_parents, Yg, Yp


def sample_set_prf(gold_lists, pred_lists):
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
    return (
        float(arr[:, 0].mean() if arr.size else 0.0),
        float(arr[:, 1].mean() if arr.size else 0.0),
        float(arr[:, 2].mean() if arr.size else 0.0)
    )


def per_label_table(y_true, y_pred, labels, out_csv_path=None):
    """Generate per-label metrics table."""
    if len(labels) == 0:
        return pd.DataFrame()
    
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Ensure arrays match label length
    if len(p) != len(labels):
        rank0_print(f"[WARN] Mismatch: {len(labels)} labels but {len(p)} metrics. Truncating.")
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


def _pretty_print_block(title: str, metrics: dict):
    """Pretty print metrics block."""
    rank0_print(f"\n--- {title} ---")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            rank0_print(f"  {k:>28s}: {v:.6f}")
        else:
            rank0_print(f"  {k:>28s}: {v}")


def show_predictions_with_metrics(test_df, preds, gold_lists, label_col, n_show=5, seed=42):
    """Show sample predictions with per-sample metrics."""
    if n_show <= 0:
        return
    
    np.random.seed(seed)
    indices = np.random.choice(len(test_df), size=min(n_show, len(test_df)), replace=False)
    
    rank0_print("\n" + "=" * 80)
    rank0_print("=== Sample predictions (with per-sample metrics) ===")
    rank0_print("=" * 80)
    for idx_num, i in enumerate(indices, 1):
        gold_set = set(gold_lists[i])
        pred_set = set(preds[i])
        
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        rank0_print(f"\n[Sample {idx_num}]")
        rank0_print(f"  GOLD codes: {', '.join(sorted(gold_set)) if gold_set else '(none)'}")
        rank0_print(f"  PREDICTED codes: {', '.join(sorted(pred_set)) if pred_set else '(none)'}")
        rank0_print(f"  Sample metrics -> P={prec:.3f} R={rec:.3f} F1={f1:.3f}")


@torch.no_grad()
def generate_codes(model, tok, prompts, labels_vocab, max_new=96, batch_size=16, max_len=3072):
    """
    Efficient batched generation with:
    - left padding (set on tokenizer)
    - KV-cache enabled
    - periodic progress logging + ETA
    """
    model.eval()
    device = next(model.parameters()).device
    allowed = set(labels_vocab)
    preds = []
    import re, time as _time

    total = len(prompts)
    rank0_print(f"[GEN] total={total}, batch_size={batch_size}, max_len={max_len}, max_new={max_new}")
    start = _time.time()

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = prompts[start_idx:end_idx]

        # tokenizer already set to left padding
        inputs = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.amp.autocast(
            "cuda",
            enabled=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
        ):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=2,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True,
                use_cache=True,
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
                    seen.add(c)
                    keep.append(c)
            preds.append(keep)

        # progress every ~10 batches or at end
        done = end_idx
        bidx = (start_idx // batch_size) + 1
        if bidx % 10 == 0 or done == total:
            elapsed = _time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
            rank0_print(f"[GEN] batch={bidx} | {done}/{total} ({done/total:.1%}) — {rate:.2f} samp/s — ETA {eta_min:.1f} min")

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--adapter_dir", default="", help="Adapter directory (optional for base_model_only)")
    ap.add_argument("--base_model_only", action="store_true", help="Evaluate base model without adapter (ablation)")
    ap.add_argument("--labels_json", required=True, help="JSON with {'labels': [...]} saved at train time")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--test_batch_size", type=int, default=16)
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")
    ap.add_argument("--out_dir", default="runs_codegen/eval")
    ap.add_argument("--test_examples", type=int, default=5)
    args = ap.parse_args()

    try:
        # Validate arguments
        if not args.base_model_only and not args.adapter_dir:
            rank0_print("[ERROR] --adapter_dir is required unless --base_model_only is set")
            return 1

        set_seed(args.seed)
        os.makedirs(args.out_dir, exist_ok=True)

        # -------- Header --------
        rank0_print("=" * 80)
        if args.base_model_only:
            rank0_print("CODEGEN - BASE MODEL ONLY (ABLATION)")
        else:
            rank0_print("CODEGEN - TESTING WITH ADAPTER")
        rank0_print("=" * 80)

        # -------- args & data preflight --------
        rank0_print(f"[ARGS] test_pickle={args.test_pickle}")
        rank0_print(f"[ARGS] base_model={args.llama_model}")
        if args.base_model_only:
            rank0_print(f"[ARGS] 🔍 ABLATION MODE: Using base model only (no adapter)")
        else:
            rank0_print(f"[ARGS] adapter_dir={args.adapter_dir}")
        rank0_print(f"[ARGS] labels_json={args.labels_json}")
        rank0_print(f"[ARGS] max_len={args.max_len}, gen_max_new={args.gen_max_new}, batch_size={args.test_batch_size}")

        test_df = pd.read_pickle(args.test_pickle)
        rank0_print(f"[DATA] Test size: {len(test_df)}")

        labels = json.load(open(args.labels_json))["labels"]
        labels_vocab = [format_icd9_properly(c) for c in labels]
        rank0_print(f"[DATA] Eval label space: {len(labels_vocab)} codes (FULL)")

        # -------- tokenizer & model --------
        if args.base_model_only:
            # Use tokenizer from base model
            tok_src = args.llama_model
            rank0_print(f"[LOAD] tokenizer from base model: {tok_src}")
        else:
            # Use tokenizer from adapter if available
            tok_src = os.path.join(args.adapter_dir, "tokenizer") \
                if os.path.exists(os.path.join(args.adapter_dir, "tokenizer")) else args.llama_model
            rank0_print(f"[LOAD] tokenizer from: {tok_src}")
        
        tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # IMPORTANT for fast generation with KV cache

        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            use_bf16 = cc[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            dev = torch.device("cuda")
            rank0_print(f"[CUDA] {torch.cuda.get_device_name(0)} cap={cc}, bf16={use_bf16}")
        else:
            dtype = torch.float32
            dev = torch.device("cpu")
            rank0_print("[WARN] CUDA not available; running on CPU (slow).")

        rank0_print("[LOAD] base model ...")
        base = AutoModelForCausalLM.from_pretrained(
            args.llama_model, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        base.config.pad_token_id = tok.pad_token_id
        base.config.use_cache = True  # enable KV cache

        if args.base_model_only:
            # Use base model directly without adapter
            model = base
            rank0_print("[LOAD] 🔍 Using BASE MODEL ONLY (no adapter) for ablation")
        else:
            # Load adapter
            rank0_print("[LOAD] applying PEFT adapter ...")
            model = PeftModel.from_pretrained(base, args.adapter_dir)
            model.generation_config.pad_token_id = tok.pad_token_id
            model.generation_config.eos_token_id = tok.eos_token_id
            model.generation_config.use_cache = True
        
        model.to(dev).eval()
        rank0_print("[LOAD] model ready")

        # -------- build prompts --------
        rank0_print("[BUILD] constructing input_text for all rows ...")
        test_df["input_text"] = test_df.apply(
            lambda r: build_input_text(r, args.use_structured == 1, args.use_notes == 1, args.subject_col), axis=1
        )
        prompts = test_df["input_text"].astype(str).tolist()
        rank0_print(f"[BUILD] prompts ready: {len(prompts)} samples")

        # -------- sort by length to reduce padding --------
        rank0_print("[SCHEDULE] sorting by input length to reduce padding waste ...")
        lens = [len(tok.encode(p, add_special_tokens=True)) for p in prompts]
        order = np.argsort(lens)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        prompts_sorted = [prompts[i] for i in order]

        # -------- gold labels --------
        gold_lists = []
        for codes in test_df[args.label_col]:
            cur = []
            for c in codes:
                z = format_icd9_properly(str(c))
                if is_valid_icd9(z):
                    cur.append(z)
            gold_lists.append(cur)

        # -------- generate --------
        t0 = time.perf_counter()
        preds_sorted = generate_codes(
            model, tok, prompts_sorted, labels_vocab,
            max_new=args.gen_max_new, batch_size=args.test_batch_size, max_len=args.max_len
        )
        preds = [preds_sorted[i] for i in inv_order]
        gen_secs = time.perf_counter() - t0
        rank0_print(f"[GEN] Generation done ({gen_secs/len(test_df):.2f}s/sample).")

        # -------- Overall metrics --------
        Y_true = codes_to_multihot(gold_lists, labels_vocab)
        Y_pred = codes_to_multihot(preds, labels_vocab)
        metrics = eval_sets(Y_true, Y_pred)
        
        # Add hierarchical/parent metrics
        metrics.update(hierarchical_eval(Y_true, Y_pred, labels_vocab))
        
        # Add comprehensive parent metrics
        parent_labels, Yg_par, Yp_par = add_parent_metrics_full(metrics, gold_lists, preds, labels_vocab)
        
        # Add sample-level set metrics
        ps, rs, fs = sample_set_prf(gold_lists, preds)
        metrics["precision_samples_set"] = ps
        metrics["recall_samples_set"] = rs
        metrics["f1_samples_set"] = fs
        
        # Meta info
        metrics["test_samples"] = len(test_df)
        metrics["test_batch_size"] = args.test_batch_size
        metrics["test_generate_seconds"] = gen_secs
        metrics["base_model_only"] = args.base_model_only

        # Save main metrics
        json.dump(metrics, open(os.path.join(args.out_dir, "test_metrics.json"), "w"), indent=2)
        
        # Per-label table: FULL
        per_label_table(Y_true, Y_pred, labels_vocab, os.path.join(args.out_dir, "per_label_FULL.csv"))
        rank0_print(f"[INFO] Metrics saved to {os.path.join(args.out_dir, 'test_metrics.json')}")

        # -------- Bucket evaluations --------
        top_codes = _read_first_col_codes(args.top_codes_csv)
        bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
        top_parents = _read_first_col_parents(args.top_parent_csv)
        results_ext = {}

        def restrict_and_eval(bucket_codes, bucket_name):
            """Only evaluate codes that exist in test set."""
            valid_bucket_codes = [c for c in bucket_codes if c in labels_vocab]
            
            if not valid_bucket_codes:
                rank0_print(f"[BUCKET] {bucket_name}: No codes from bucket found in test set. Skipping.")
                return None
            
            idx = {c: i for i, c in enumerate(labels_vocab)}
            keep = [idx[c] for c in valid_bucket_codes]
            
            yt = Y_true[:, keep]
            yp = Y_pred[:, keep]
            result = eval_sets(yt, yp)
            
            # Per-label table for bucket
            per_label_table(yt, yp, valid_bucket_codes, 
                          os.path.join(args.out_dir, f"per_label_{bucket_name}.csv"))
            
            rank0_print(f"[BUCKET] {bucket_name}: Evaluated {len(valid_bucket_codes)}/{len(bucket_codes)} codes present in test set")
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

            # Filter parent codes to only those present in gold data
            gold_parents_all = set()
            for lst in gold_lists:
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

                YgP = mh(to_parents(gold_lists), valid_top_parents)
                YpP = mh(to_parents(preds), valid_top_parents)
                results_ext["TOP_50_PARENTS"] = eval_sets(YgP, YpP)
                per_label_table(YgP, YpP, valid_top_parents, 
                              os.path.join(args.out_dir, "per_label_TOP_50_PARENTS.csv"))
                rank0_print(f"[BUCKET] TOP_50_PARENTS: Evaluated {len(valid_top_parents)}/{len(top_parents)} parents present in test set")
            else:
                rank0_print("[BUCKET] TOP_50_PARENTS: No parent codes from bucket found in test set. Skipping.")

        # Save bucket metrics
        if results_ext:
            bucket_path = os.path.join(args.out_dir, "test_metrics_buckets.json")
            json.dump(results_ext, open(bucket_path, "w"), indent=2)
            rank0_print(f"[INFO] Bucket metrics saved to {bucket_path}")

        # -------- Print metrics at the end --------
        _pretty_print_block("OVERALL (code-level)", metrics)
        
        if "TOP_50_CODES" in results_ext:
            _pretty_print_block("TOP_50_CODES (code-level)", results_ext["TOP_50_CODES"])
        
        if "BOTTOM_50_CODES" in results_ext:
            _pretty_print_block("BOTTOM_50_CODES (code-level)", results_ext["BOTTOM_50_CODES"])
        
        if "TOP_50_PARENTS" in results_ext:
            _pretty_print_block("TOP_50_PARENTS (parent-level)", results_ext["TOP_50_PARENTS"])

        # -------- Show sample predictions --------
        show_predictions_with_metrics(
            test_df, preds, gold_lists, args.label_col, 
            n_show=args.test_examples, seed=args.seed
        )

        rank0_print("\n" + "=" * 80)
        rank0_print("Testing completed successfully!")
        rank0_print("=" * 80)

        return 0

    except Exception as e:
        import traceback
        rank0_print(f"[FATAL] Exception: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

def _read_first_col_codes(path):
    if not path:
        return []
    try:
        df = pd.read_csv(path)
        col = df.columns[0]
        vals = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        return sorted({v for v in vals if is_valid_icd9(v)})
    except Exception:
        return []


def _read_first_col_parents(path):
    if not path:
        return []
    try:
        df = pd.read_csv(path)
        col = df.columns[0]
        raw = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        return sorted({get_icd9_parent(v) for v in raw if v})
    except Exception:
        return []


@torch.no_grad()
def generate_codes(model, tok, prompts, labels_vocab, max_new=96, batch_size=16, max_len=3072):
    """
    Efficient batched generation with:
    - left padding (set on tokenizer)
    - KV-cache enabled
    - periodic progress logging + ETA
    """
    model.eval()
    device = next(model.parameters()).device
    allowed = set(labels_vocab)
    preds = []
    import re, time as _time

    total = len(prompts)
    rank0_print(f"[GEN] total={total}, batch_size={batch_size}, max_len={max_len}, max_new={max_new}")
    start = _time.time()

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = prompts[start_idx:end_idx]

        # tokenizer already set to left padding
        inputs = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.amp.autocast(
            "cuda",
            enabled=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
        ):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=2,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True,
                use_cache=True,
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
                    seen.add(c)
                    keep.append(c)
            preds.append(keep)

        # progress every ~10 batches or at end (more frequent)
        done = end_idx
        bidx = (start_idx // batch_size) + 1
        if bidx % 10 == 0 or done == total:
            elapsed = _time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
            rank0_print(f"[GEN] batch={bidx} | {done}/{total} ({done/total:.1%}) — {rate:.2f} samp/s — ETA {eta_min:.1f} min")

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--labels_json", required=True, help="JSON with {'labels': [...]} saved at train time")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--test_batch_size", type=int, default=16)
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")
    ap.add_argument("--out_dir", default="runs_codegen/eval")
    ap.add_argument("--test_examples", type=int, default=5)
    args = ap.parse_args()

    try:
        set_seed(args.seed)
        os.makedirs(args.out_dir, exist_ok=True)

        # -------- args & data preflight --------
        rank0_print(f"[ARGS] test_pickle={args.test_pickle}")
        rank0_print(f"[ARGS] adapter_dir={args.adapter_dir}")
        rank0_print(f"[ARGS] labels_json={args.labels_json}")
        rank0_print(f"[ARGS] max_len={args.max_len}, gen_max_new={args.gen_max_new}, batch_size={args.test_batch_size}")

        test_df = pd.read_pickle(args.test_pickle)
        rank0_print(f"[DATA] test rows: {len(test_df)}")

        labels = json.load(open(args.labels_json))["labels"]
        labels_vocab = [format_icd9_properly(c) for c in labels]
        rank0_print(f"[LABELS] size={len(labels_vocab)}")

        # -------- tokenizer & model --------
        tok_src = os.path.join(args.adapter_dir, "tokenizer") \
            if os.path.exists(os.path.join(args.adapter_dir, "tokenizer")) else args.llama_model
        tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # IMPORTANT for fast generation with KV cache

        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            use_bf16 = cc[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            dev = torch.device("cuda")
            rank0_print(f"[CUDA] {torch.cuda.get_device_name(0)} cap={cc}, bf16={use_bf16}")
        else:
            dtype = torch.float32
            dev = torch.device("cpu")
            rank0_print("[WARN] CUDA not available; running on CPU (slow).")

        rank0_print("[LOAD] base model ...")
        base = AutoModelForCausalLM.from_pretrained(
            args.llama_model, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        base.config.pad_token_id = tok.pad_token_id
        base.config.use_cache = True  # enable KV cache

        rank0_print("[LOAD] applying PEFT adapter ...")
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.use_cache = True
        model.to(dev).eval()
        rank0_print("[LOAD] model ready")

        # -------- build prompts (with logs) --------
        rank0_print("[BUILD] constructing input_text for all rows ...")
        test_df["input_text"] = test_df.apply(
            lambda r: build_input_text(r, args.use_structured == 1, args.use_notes == 1, args.subject_col), axis=1
        )
        prompts = test_df["input_text"].astype(str).tolist()
        rank0_print(f"[BUILD] prompts ready: {len(prompts)} samples")

        # -------- sort by length to reduce padding --------
        rank0_print("[SCHEDULE] sorting by input length to reduce padding waste ...")
        lens = [len(tok.encode(p, add_special_tokens=True)) for p in prompts]
        order = np.argsort(lens)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        prompts_sorted = [prompts[i] for i in order]

        # -------- gold labels --------
        gold_lists = []
        for codes in test_df[args.label_col]:
            cur = []
            for c in codes:
                z = format_icd9_properly(str(c))
                if is_valid_icd9(z):
                    cur.append(z)
            gold_lists.append(cur)

        # -------- generate --------
        t0 = time.perf_counter()
        preds_sorted = generate_codes(
            model, tok, prompts_sorted, labels_vocab,
            max_new=args.gen_max_new, batch_size=args.test_batch_size, max_len=args.max_len
        )
        preds = [preds_sorted[i] for i in inv_order]
        gen_secs = time.perf_counter() - t0
        rank0_print(f"[GEN] finished in {gen_secs/60:.2f} minutes")

        # -------- metrics --------
        Y_true = codes_to_multihot(gold_lists, labels_vocab)
        Y_pred = codes_to_multihot(preds,       labels_vocab)
        metrics = eval_sets(Y_true, Y_pred)
        metrics.update(hierarchical_eval(Y_true, Y_pred, labels_vocab))
        metrics["test_samples"] = len(test_df)
        metrics["test_batch_size"] = args.test_batch_size
        metrics["test_generate_seconds"] = gen_secs
        json.dump(metrics, open(os.path.join(args.out_dir, "test_metrics.json"), "w"), indent=2)

        # -------- buckets --------
        top_codes    = _read_first_col_codes(args.top_codes_csv)
        bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
        top_parents  = _read_first_col_parents(args.top_parent_csv)
        results_ext = {}

        def restrict_and_eval(bucket_codes):
            idx = {c: i for i, c in enumerate(labels_vocab)}
            keep = [idx[c] for c in bucket_codes if c in idx]
            if not keep:
                return None
            yt = Y_true[:, keep]
            yp = Y_pred[:, keep]
            return eval_sets(yt, yp)

        if top_codes:
            r = restrict_and_eval(top_codes)
            if r:
                results_ext["TOP_50_CODES"] = r
        if bottom_codes:
            r = restrict_and_eval(bottom_codes)
            if r:
                results_ext["BOTTOM_50_CODES"] = r
        if top_parents:
            def to_parents(lists):
                return [[get_icd9_parent(c) for c in row] for row in lists]

            parents = sorted(set(top_parents))

            def mh(L, labels):
                idx = {c: i for i, c in enumerate(labels)}
                Y = np.zeros((len(L), len(labels)), dtype=np.int32)
                for i, lst in enumerate(L):
                    for c in lst:
                        j = idx.get(c)
                        if j is not None:
                            Y[i, j] = 1
                return Y

            YgP = mh(to_parents(gold_lists), parents)
            YpP = mh(to_parents(preds),      parents)
            results_ext["TOP_50_PARENTS"] = eval_sets(YgP, YpP)

        json.dump(results_ext, open(os.path.join(args.out_dir, "test_metrics_buckets.json"), "w"), indent=2)

        # -------- prints --------
        rank0_print("\n=== MAIN TEST METRICS ===")
        for k in sorted(metrics.keys()):
            v = metrics[k]
            rank0_print(f"{k:>28s}: {v:.6f}" if isinstance(v, float) else f"{k:>28s}: {v}")

        if results_ext:
            rank0_print("\n=== BUCKETS ===")
            for name, d in results_ext.items():
                rank0_print(f"[{name}]")
                for k in sorted(d.keys()):
                    v = d[k]
                    rank0_print(f"  {k:>26s}: {v:.6f}")

        show_test_predictions(test_df, preds, args.label_col, labels_vocab, n_show=args.test_examples, seed=args.seed)
        return 0

    except Exception as e:
        import traceback
        rank0_print(f"[FATAL] Exception: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())