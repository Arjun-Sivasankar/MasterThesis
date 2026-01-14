# eval_icd9_codes.py
import os, re, json, time, argparse, logging, pickle
import numpy as np
import pandas as pd
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

# ---- import helpers from your training script (adjust the module path if needed)
# If this file sits next to your training script, Python can import by module name.
# Otherwise add sys.path insertion here to point to it.
from finetune_llama_gen_ddp import (  
    format_icd9_properly, is_valid_icd9, get_icd9_parent,
    build_input_text, generate_codes, codes_to_multihot, eval_sets,
    show_test_predictions  # optional, we also print our own samples
)

log = logging.getLogger("eval_icd9")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------- CSV readers -------------------------
def _read_first_col_codes(path: str) -> List[str]:
    if not path:
        return []
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return []
        col = df.columns[0]
        vals = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        vals = [v for v in vals if is_valid_icd9(v)]
        return sorted(set(vals))
    except Exception as e:
        log.warning(f"[subset] Could not read codes from {path}: {e}")
        return []

def _read_first_col_parents(path: str) -> List[str]:
    if not path:
        return []
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return []
        col = df.columns[0]
        raw = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        parents = sorted(set(get_icd9_parent(x) for x in raw if x))
        return parents
    except Exception as e:
        log.warning(f"[subset] Could not read parent codes from {path}: {e}")
        return []


# ------------------------- Parent metrics -------------------------
def _to_parent_lists(code_lists: List[List[str]]) -> List[List[str]]:
    return [[get_icd9_parent(c) for c in lst] for lst in code_lists]

def _multihot(lists: List[List[str]], labels: List[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(labels)}
    Y = np.zeros((len(lists), len(labels)), dtype=np.int32)
    for i, lst in enumerate(lists):
        for c in lst:
            j = idx.get(c)
            if j is not None:
                Y[i, j] = 1
    return Y

def add_parent_metrics_full(gold_lists: List[List[str]],
                            pred_lists: List[List[str]]) -> Dict[str, float]:
    """Parent-level micro/macro/samples P/R/F1 using your get_icd9_parent rule."""
    g = _to_parent_lists(gold_lists)
    p = _to_parent_lists(pred_lists)
    labels = sorted({x for lst in g for x in lst})
    Yg = _multihot(g, labels)
    Yp = _multihot(p, labels)
    return {
        "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
        "recall_micro_parent":    float(recall_score(Yg, Yp, average="micro", zero_division=0)),
        "f1_micro_parent":        float(f1_score(Yg, Yp, average="micro", zero_division=0)),
        "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
        "recall_macro_parent":    float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent":        float(f1_score(Yg, Yp, average="macro", zero_division=0)),
        "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
        "recall_samples_parent":    float(recall_score(Yg, Yp, average="samples", zero_division=0)),
        "f1_samples_parent":        float(f1_score(Yg, Yp, average="samples", zero_division=0)),
    }


# ------------------------- Per-label tables -------------------------
def per_label_table(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_csv_path: str) -> None:
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    df = pd.DataFrame({
        "code": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support
    })
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)


# ------------------------- Helpers -------------------------
def _restrict_lists(lists: List[List[str]], allowed: List[str]) -> List[List[str]]:
    S = set(allowed)
    return [[c for c in lst if c in S] for lst in lists]

def _build_label_space_from_json_or_test(label_space_json: str, test_gold: List[List[str]]) -> List[str]:
    if label_space_json and os.path.exists(label_space_json):
        with open(label_space_json, "r") as f:
            obj = json.load(f)
        labels = obj.get("labels") or obj.get("classes") or obj
        labels = [format_icd9_properly(str(x)) for x in labels]
        labels = [c for c in labels if is_valid_icd9(c)]
        log.info(f"[labels] Loaded {len(labels)} labels from {label_space_json}")
        return labels
    # fallback: from test gold (FULL)
    labels = sorted({c for lst in test_gold for c in lst})
    log.info(f"[labels] Built {len(labels)} labels from test gold")
    return labels

def _load_model_tokenizer(base_model: str,
                          adapter_dir: str = "",
                          merged_model_dir: str = "",
                          tokenizer_dir: str = ""):
    # dtype
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    # tokenizer
    tok_src = tokenizer_dir or (merged_model_dir if merged_model_dir else base_model)
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # model
    if merged_model_dir:
        model = AutoModelForCausalLM.from_pretrained(merged_model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
        if not adapter_dir:
            raise ValueError("Provide --adapter_dir for PEFT adapter or --merged_model_dir for a merged model.")
        model = PeftModel.from_pretrained(base, adapter_dir)

    # small safety tweaks
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tok


# ------------------------- CLI -------------------------
def get_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--test_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)

    # model
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", default="", help="PEFT adapter dir (if not using merged model)")
    ap.add_argument("--merged_model_dir", default="", help="Use merged model instead of base+adapter")
    ap.add_argument("--tokenizer_dir", default="", help="Optional, else inferred from merged/base")

    # generation
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=16)

    # label space
    ap.add_argument("--label_space_json", default="", help="e.g., RUN_DIR/label_space.json; else built from test gold")

    # subsets
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")

    # output
    ap.add_argument("--out_dir", default="runs_gen/eval_out")
    ap.add_argument("--print_samples", type=int, default=5)
    return ap.parse_args()


# ------------------------- Main -------------------------
def main():
    args = get_args()

    # ---- Load test data
    test_df = pickle.load(open(args.test_pickle, "rb"))
    assert args.label_col in test_df.columns, f"Missing {args.label_col} in test dataframe"

    # gold lists (normalized & validated)
    gold_lists = []
    for codes in test_df[args.label_col].tolist():
        cleaned = [format_icd9_properly(str(c)) for c in codes]
        cleaned = [c for c in cleaned if is_valid_icd9(c)]
        gold_lists.append(sorted(set(cleaned)))

    # Build inputs (prompts)
    test_df["input_text"] = test_df.apply(
        lambda r: build_input_text(r, args.use_structured == 1, args.use_notes == 1, args.subject_col),
        axis=1
    )
    prompts = test_df["input_text"].astype(str).tolist()

    # Label space
    labels_vocab = _build_label_space_from_json_or_test(args.label_space_json, gold_lists)

    # Model & tokenizer
    model, tok = _load_model_tokenizer(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        merged_model_dir=args.merged_model_dir,
        tokenizer_dir=args.tokenizer_dir
    )

    # ---- Generate predictions (already filtered to label space by your generate_codes)
    t0 = time.time()
    preds = generate_codes(
        model, tok, prompts, labels_vocab,
        max_new=args.gen_max_new, batch_size=args.batch_size, max_len=args.max_len
    )
    gen_secs = time.time() - t0
    log.info(f"[gen] done in {gen_secs:.1f}s ({gen_secs/len(test_df):.3f}s/sample, {len(test_df)/gen_secs:.2f} samples/s)")

    # ---- FULL metrics
    Y_true = codes_to_multihot(gold_lists, labels_vocab)
    Y_pred = codes_to_multihot(preds,      labels_vocab)
    metrics = {
        "precision_micro": float(precision_score(Y_true, Y_pred, average="micro",   zero_division=0)),
        "recall_micro":    float(recall_score(Y_true, Y_pred, average="micro",      zero_division=0)),
        "f1_micro":        float(f1_score(Y_true, Y_pred, average="micro",          zero_division=0)),
        "precision_macro": float(precision_score(Y_true, Y_pred, average="macro",   zero_division=0)),
        "recall_macro":    float(recall_score(Y_true, Y_pred, average="macro",      zero_division=0)),
        "f1_macro":        float(f1_score(Y_true, Y_pred, average="macro",          zero_division=0)),
        "precision_samples": float(precision_score(Y_true, Y_pred, average="samples", zero_division=0)),
        "recall_samples":    float(recall_score(Y_true, Y_pred, average="samples",    zero_division=0)),
        "f1_samples":        float(f1_score(Y_true, Y_pred, average="samples",        zero_division=0)),
    }

    # Parent metrics (micro/macro/samples)
    metrics.update(add_parent_metrics_full(gold_lists, preds))

    # ---- Per-label CSV (FULL)
    os.makedirs(args.out_dir, exist_ok=True)
    per_label_table(Y_true, Y_pred, labels_vocab, os.path.join(args.out_dir, "per_label_FULL.csv"))

    # ---- Subset metrics (TOP/BOTTOM codes, TOP parents)
    results_ext = {}

    top_codes = _read_first_col_codes(args.top_codes_csv)
    if top_codes:
        g = _restrict_lists(gold_lists, top_codes)
        p = _restrict_lists(preds,      top_codes)
        Yg = _multihot(g, top_codes); Yp2 = _multihot(p, top_codes)
        results_ext["TOP_CODES"] = {
            "precision_micro": float(precision_score(Yg, Yp2, average="micro", zero_division=0)),
            "recall_micro":    float(recall_score(Yg, Yp2, average="micro",    zero_division=0)),
            "f1_micro":        float(f1_score(Yg, Yp2, average="micro",        zero_division=0)),
            "precision_macro": float(precision_score(Yg, Yp2, average="macro", zero_division=0)),
            "recall_macro":    float(recall_score(Yg, Yp2, average="macro",    zero_division=0)),
            "f1_macro":        float(f1_score(Yg, Yp2, average="macro",        zero_division=0)),
            "precision_samples": float(precision_score(Yg, Yp2, average="samples", zero_division=0)),
            "recall_samples":    float(recall_score(Yg, Yp2, average="samples",    zero_division=0)),
            "f1_samples":        float(f1_score(Yg, Yp2, average="samples",        zero_division=0)),
        }
        per_label_table(Yg, Yp2, top_codes, os.path.join(args.out_dir, "per_label_TOP_CODES.csv"))

    bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
    if bottom_codes:
        g = _restrict_lists(gold_lists, bottom_codes)
        p = _restrict_lists(preds,      bottom_codes)
        Yg = _multihot(g, bottom_codes); Yp2 = _multihot(p, bottom_codes)
        results_ext["BOTTOM_CODES"] = {
            "precision_micro": float(precision_score(Yg, Yp2, average="micro", zero_division=0)),
            "recall_micro":    float(recall_score(Yg, Yp2, average="micro",    zero_division=0)),
            "f1_micro":        float(f1_score(Yg, Yp2, average="micro",        zero_division=0)),
            "precision_macro": float(precision_score(Yg, Yp2, average="macro", zero_division=0)),
            "recall_macro":    float(recall_score(Yg, Yp2, average="macro",    zero_division=0)),
            "f1_macro":        float(f1_score(Yg, Yp2, average="macro",        zero_division=0)),
            "precision_samples": float(precision_score(Yg, Yp2, average="samples", zero_division=0)),
            "recall_samples":    float(recall_score(Yg, Yp2, average="samples",    zero_division=0)),
            "f1_samples":        float(f1_score(Yg, Yp2, average="samples",        zero_division=0)),
        }
        per_label_table(Yg, Yp2, bottom_codes, os.path.join(args.out_dir, "per_label_BOTTOM_CODES.csv"))

    top_parents = _read_first_col_parents(args.top_parent_csv)
    if top_parents:
        g_par = _to_parent_lists(gold_lists)
        p_par = _to_parent_lists(preds)
        g_par_r = _restrict_lists(g_par, top_parents)
        p_par_r = _restrict_lists(p_par, top_parents)
        YgP = _multihot(g_par_r, top_parents); YpP = _multihot(p_par_r, top_parents)
        results_ext["TOP_PARENTS"] = {
            "precision_micro_parent": float(precision_score(YgP, YpP, average="micro", zero_division=0)),
            "recall_micro_parent":    float(recall_score(YgP, YpP, average="micro",    zero_division=0)),
            "f1_micro_parent":        float(f1_score(YgP, YpP, average="micro",        zero_division=0)),
            "precision_macro_parent": float(precision_score(YgP, YpP, average="macro", zero_division=0)),
            "recall_macro_parent":    float(recall_score(YgP, YpP, average="macro",    zero_division=0)),
            "f1_macro_parent":        float(f1_score(YgP, YpP, average="macro",        zero_division=0)),
            "precision_samples_parent": float(precision_score(YgP, YpP, average="samples", zero_division=0)),
            "recall_samples_parent":    float(recall_score(YgP, YpP, average="samples",    zero_division=0)),
            "f1_samples_parent":        float(f1_score(YgP, YpP, average="samples",        zero_division=0)),
        }
        per_label_table(YgP, YpP, top_parents, os.path.join(args.out_dir, "per_label_TOP_PARENTS.csv"))

    # ---- Sample prints with per-sample metrics
    n_show = min(args.print_samples, len(test_df))
    if n_show > 0:
        log.info("=== Sample predictions (per-sample metrics) ===")
        idxs = list(range(n_show))
        for i in idxs:
            G = set(gold_lists[i]); P = set(preds[i])
            tp = len(G & P); fp = len(P - G); fn = len(G - P)
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            Gp = {get_icd9_parent(c) for c in G}
            Pp = {get_icd9_parent(c) for c in P}
            pr = len(Gp & Pp) / len(Gp) if len(Gp) > 0 else 0.0
            log.info(f"[Sample {i+1}] P={prec:.3f} R={rec:.3f} F1={f1:.3f} | parent_recall={pr:.3f}")
            log.info(f"  GOLD: {' '.join(sorted(G)) if G else '(none)'}")
            log.info(f"  PRED: {' '.join(sorted(P)) if P else '(none)'}")

    # ---- Write JSON + Pretty print summary
    payload = {
        "num_samples": int(len(test_df)),
        "label_space_size": int(len(labels_vocab)),
        "gen_seconds": float(gen_secs),
        "gen_sec_per_sample": float(gen_secs/len(test_df)) if len(test_df)>0 else 0.0,
        "metrics_full": metrics,
        **results_ext
    }
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # --- Pretty print summary (replace your current print block with this) ---
    print("\n=== FINAL EVALUATION SUMMARY ===")
    print(f"samples: {payload['num_samples']} | labels: {payload['label_space_size']}")
    print(f"gen: {payload['gen_seconds']:.1f}s | {payload['gen_sec_per_sample']:.3f}s/sample")

    # FULL (code-level)
    print("\nFULL (codes):")
    print(f"  micro   P={metrics['precision_micro']:.4f}  R={metrics['recall_micro']:.4f}  F1={metrics['f1_micro']:.4f}")
    print(f"  macro   P={metrics['precision_macro']:.4f}  R={metrics['recall_macro']:.4f}  F1={metrics['f1_macro']:.4f}")
    print(f"  samples P={metrics['precision_samples']:.4f} R={metrics['recall_samples']:.4f} F1={metrics['f1_samples']:.4f}")

    # PARENT (aggregated to parent categories)
    pm = payload["metrics_full"]  # contains *_parent metrics we added above
    print("\nPARENT (using get_icd9_parent):")
    print(f"  micro   P={pm['precision_micro_parent']:.4f}  R={pm['recall_micro_parent']:.4f}  F1={pm['f1_micro_parent']:.4f}")
    print(f"  macro   P={pm['precision_macro_parent']:.4f}  R={pm['recall_macro_parent']:.4f}  F1={pm['f1_macro_parent']:.4f}")
    print(f"  samples P={pm['precision_samples_parent']:.4f} R={pm['recall_samples_parent']:.4f} F1={pm['f1_samples_parent']:.4f}")

    # ---- NEW: subset prints (Top/Bottom codes & Top parents) ----
    def _fmt_subset(name, d):
        print(f"\n{name}:")
        print(f"  micro   P={d.get('precision_micro', 0.0):.4f}  R={d.get('recall_micro', 0.0):.4f}  F1={d.get('f1_micro', 0.0):.4f}")
        print(f"  macro   P={d.get('precision_macro', 0.0):.4f}  R={d.get('recall_macro', 0.0):.4f}  F1={d.get('f1_macro', 0.0):.4f}")
        print(f"  samples P={d.get('precision_samples', 0.0):.4f} R={d.get('recall_samples', 0.0):.4f} F1={d.get('f1_samples', 0.0):.4f}")

    def _fmt_subset_parent(name, d):
        print(f"\n{name} (parent-level):")
        print(f"  micro   P={d.get('precision_micro_parent', 0.0):.4f}  R={d.get('recall_micro_parent', 0.0):.4f}  F1={d.get('f1_micro_parent', 0.0):.4f}")
        print(f"  macro   P={d.get('precision_macro_parent', 0.0):.4f}  R={d.get('recall_macro_parent', 0.0):.4f}  F1={d.get('f1_macro_parent', 0.0):.4f}")
        print(f"  samples P={d.get('precision_samples_parent', 0.0):.4f} R={d.get('recall_samples_parent', 0.0):.4f} F1={d.get('f1_samples_parent', 0.0):.4f}")

    if "TOP_CODES" in results_ext:
        _fmt_subset("TOP 50 CODES", results_ext["TOP_CODES"])
    else:
        print("\nTOP 50 CODES: (no CSV provided / no overlap)")

    if "BOTTOM_CODES" in results_ext:
        _fmt_subset("BOTTOM 50 CODES", results_ext["BOTTOM_CODES"])
    else:
        print("\nBOTTOM 50 CODES: (no CSV provided / no overlap)")

    if "TOP_PARENTS" in results_ext:
        _fmt_subset_parent("TOP 50 PARENTS", results_ext["TOP_PARENTS"])
    else:
        print("\nTOP 50 PARENTS: (no CSV provided / no overlap)")

    print("\n(Per-label CSVs written to:", args.out_dir, ")")


if __name__ == "__main__":
    main()
