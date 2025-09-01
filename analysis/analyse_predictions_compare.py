# -*- coding: utf-8 -*-
"""
analyse_predictions_compare.py

Compare decoding strategies for generative ICD coding.
- Consistent with your finetune/training script: same subject split (sklearn), same label normalization,
  same label space (loaded from run_dir/label_space.json if present; otherwise computed from data).
- Works with files produced by `dump_preds_modes.py` (post-filtered and/or raw).
- Produces side-by-side metrics, OOV/dup diagnostics (to demonstrate post-filter benefits),
  head/torso/tail breakdowns, and per-label scores.

Outputs (in <run_dir>/analysis/):
  - compare_summary.csv / .json
  - per_label_{mode}.csv
  - oov_dup_{mode}.json
  - head_torso_tail_{mode}.json
  - parent3_{mode}.json
  - plots/*.png
"""

import os, re, json, argparse, logging, pickle, collections
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # <-- match training script

# ---------- Config (must match training script) ----------
SUBJECT_COL = "subject_id_x"
LABEL_COL   = "icd_code"
TEST_SIZE   = 0.10
VAL_SIZE    = 0.10
SPLIT_SEED  = 42

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Utils (consistent with training script) ----------
def set_seed(seed=42):
    np.random.seed(seed)

def normalize_code(c: str) -> str:
    c = str(c).strip().upper()
    c = re.sub(r"\s+", "", c)
    return c[:-1] if c.endswith(".") else c

def subject_splits(df: pd.DataFrame, subject_col=SUBJECT_COL,
                   test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SPLIT_SEED):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    logging.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

def load_full_df(data_pickle: str) -> pd.DataFrame:
    df = pickle.load(open(data_pickle, "rb"))

    def _tolist(x):
        # If it's already iterable of codes
        if isinstance(x, (list, tuple, set)):
            return [normalize_code(z) for z in x]
        if isinstance(x, np.ndarray):
            return [normalize_code(z) for z in x.tolist()]

        # None or explicit NaN-ish
        if x is None:
            return []
        # Convert to string for parsing
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return []

        # If looks like a python list, try literal_eval
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, set)):
                    return [normalize_code(z) for z in v]
                if isinstance(v, np.ndarray):
                    return [normalize_code(z) for z in v.tolist()]
            except Exception:
                pass

        # Fallback: split on commas/whitespace
        toks = [t for t in re.split(r"[,\s]+", s) if t]
        return [normalize_code(t) for t in toks]

    df[LABEL_COL] = df[LABEL_COL].apply(_tolist)
    return df

def load_label_space(run_dir: str, df_all: pd.DataFrame) -> List[str]:
    path = os.path.join(run_dir, "label_space.json")
    if os.path.isfile(path):
        obj = json.load(open(path, "r"))
        labels = [normalize_code(c) for c in obj.get("labels", [])]
        logging.info(f"Loaded label space from {path} ({len(labels)} codes).")
        return labels
    # fallback: compute from full df
    labels = sorted({normalize_code(c) for codes in df_all[LABEL_COL] for c in codes})
    logging.info(f"[WARN] label_space.json not found; computed labels from data ({len(labels)}).")
    return labels

# ---------- Prediction IO ----------
def discover_pred_files(run_dir: str) -> Dict[str, str]:
    """Find all preds_*.jsonl files and map by short name."""
    out = {}
    for fn in os.listdir(run_dir):
        if fn.startswith("preds_") and fn.endswith(".jsonl"):
            key = fn[len("preds_"):-len(".jsonl")]  # e.g., "greedy", "greedy_nopf"
            out[key] = os.path.join(run_dir, fn)
    return out

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def attach_preds_to_test(test_df: pd.DataFrame,
                         recs: List[Dict[str, Any]],
                         label_vocab: List[str]) -> pd.DataFrame:
    """Return a DF aligned on hadm_id with columns: pred(list), pred_raw(list|None), gold(list)."""
    by_hadm = {}
    for r in recs:
        hadm = r.get("hadm_id")
        if hadm is None: continue
        pred = [normalize_code(x) for x in (r.get("pred") or [])]
        pred_raw = r.get("pred_raw")
        if pred_raw is not None:
            pred_raw = [normalize_code(x) for x in pred_raw]
        gold = [normalize_code(x) for x in (r.get("gold") or [])]
        by_hadm[hadm] = {"pred": pred, "pred_raw": pred_raw, "gold": gold}

    rows = []
    miss = 0
    for _, row in test_df.iterrows():
        hadm = row.get("hadm_id")
        gold = [normalize_code(x) for x in row[LABEL_COL]]
        rec = by_hadm.get(hadm)
        if rec is None:
            miss += 1
            rows.append({"hadm_id": hadm, "pred": [], "pred_raw": None, "gold": gold})
        else:
            rows.append({"hadm_id": hadm,
                         "pred": rec["pred"],
                         "pred_raw": rec["pred_raw"],
                         "gold": gold})
    if miss:
        logging.warning(f"{miss} test rows had no matching prediction entry in file.")
    return pd.DataFrame(rows)

# ---------- Metrics ----------
def multihot_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "micro_precision":   float(precision_score(y_true, y_pred, average="micro",   zero_division=0)),
        "micro_recall":      float(recall_score(y_true, y_pred, average="micro",      zero_division=0)),
        "micro_f1":          float(f1_score(y_true, y_pred, average="micro",          zero_division=0)),
        "macro_precision":   float(precision_score(y_true, y_pred, average="macro",   zero_division=0)),
        "macro_recall":      float(recall_score(y_true, y_pred, average="macro",      zero_division=0)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro",          zero_division=0)),
        "samples_precision": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "samples_recall":    float(recall_score(y_true, y_pred, average="samples",    zero_division=0)),
        "samples_f1":        float(f1_score(y_true, y_pred, average="samples",        zero_division=0)),
    }

def to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(label_vocab)}
    Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
    for i, lst in enumerate(code_lists):
        for c in set(lst):
            j = idx.get(c)
            if j is not None:
                Y[i, j] = 1
    return Y

def jaccard_and_exact(preds: List[List[str]], golds: List[List[str]]) -> Tuple[float, float]:
    js, ex = [], 0
    for p, g in zip(preds, golds):
        ps, gs = set(p), set(g)
        inter = len(ps & gs)
        union = len(ps | gs)
        js.append(inter/union if union else 1.0)
        if ps == gs: ex += 1
    return float(np.mean(js)), float(ex/len(preds) if preds else 0.0)

def per_label_table(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> pd.DataFrame:
    tp = (y_true & y_pred).sum(0).astype(np.int64)
    fp = ((1 - y_true) & y_pred).sum(0).astype(np.int64)
    fn = (y_true & (1 - y_pred)).sum(0).astype(np.int64)
    support = y_true.sum(0).astype(np.int64)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)>0)
    rec  = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)>0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec, dtype=float), where=(prec+rec)>0)
    df = pd.DataFrame({
        "code": label_vocab,
        "support": support,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    })
    return df.sort_values("support", ascending=False).reset_index(drop=True)

def parent3(code: str) -> str:
    c = normalize_code(code)
    return c[:3]

def parent3_micro(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> Dict[str, float]:
    parents = [parent3(c) for c in label_vocab]
    uniq_parents = sorted(set(parents))
    map_parent_idx = {p:i for i,p in enumerate(uniq_parents)}
    Yt = np.zeros((y_true.shape[0], len(uniq_parents)), dtype=np.int32)
    Yp = np.zeros_like(Yt)
    for j, p in enumerate(parents):
        pj = map_parent_idx[p]
        Yt[:, pj] |= y_true[:, j]
        Yp[:, pj] |= y_pred[:, j]
    return multihot_metrics(Yt, Yp)

def head_torso_tail_metrics(label_freq: Dict[str,int],
                            label_vocab: List[str],
                            y_true: np.ndarray, y_pred: np.ndarray,
                            head_k=50, torso_k=450) -> Dict[str, Any]:
    freq_sorted = sorted(label_vocab, key=lambda c: -label_freq.get(c, 0))
    head = set(freq_sorted[:head_k])
    torso = set(freq_sorted[head_k:head_k+torso_k])
    tail = set(freq_sorted[head_k+torso_k:])
    idx = {c:i for i,c in enumerate(label_vocab)}
    def select(cols: List[str]):
        j = [idx[c] for c in cols if c in idx]
        return (y_true[:, j], y_pred[:, j])
    def pack(name, cols):
        yt, yp = select(list(cols))
        m = multihot_metrics(yt, yp)
        m["codes"] = len(cols)
        m["support_total"] = int(yt.sum())
        return name, m
    out = dict([pack("head", head), pack("torso", torso), pack("tail", tail)])
    return out

def oov_dup_stats(df: pd.DataFrame, label_vocab: List[str]) -> Dict[str, float]:
    allowed = set(label_vocab)
    total_raw = total_oov = total_dup = 0
    total_samples = len(df)
    kept_after_filter = kept_total = 0
    for _, r in df.iterrows():
        raw = r["pred_raw"]
        if raw is None:
            raw = r["pred"]
        raw = [normalize_code(x) for x in (raw or [])]
        total_raw += len(raw)
        # OOV
        total_oov += sum(1 for x in raw if x not in allowed)
        # duplicates
        seen = set(); dup = 0
        for x in raw:
            if x in seen: dup += 1
            else: seen.add(x)
        total_dup += dup
        # kept after strict filter
        filtered_seen = set(); filtered = []
        for x in raw:
            if x in allowed and x not in filtered_seen:
                filtered_seen.add(x); filtered.append(x)
        kept_after_filter += len(filtered)
        kept_total += len(raw)
    oov_rate = (total_oov / total_raw) if total_raw else 0.0
    dup_rate = (total_dup / total_raw) if total_raw else 0.0
    kept_rate = (kept_after_filter / kept_total) if kept_total else 0.0
    return {
        "samples_n": int(total_samples),
        "raw_tokens_total": int(total_raw),
        "oov_total": int(total_oov),
        "dup_total": int(total_dup),
        "oov_rate": float(oov_rate),
        "dup_rate": float(dup_rate),
        "filtered_kept_rate": float(kept_rate),
        "filtered_mean_len": float(kept_after_filter / total_samples if total_samples else 0.0),
        "raw_mean_len": float(total_raw / total_samples if total_samples else 0.0),
    }

def bag_micro_with_oov_penalty(df: pd.DataFrame, label_vocab: List[str]) -> Dict[str, float]:
    tp = fp = fn = 0
    for _, r in df.iterrows():
        raw = r["pred_raw"]
        if raw is None:
            raw = r["pred"]
        raw = [normalize_code(x) for x in (raw or [])]
        gold = set([normalize_code(x) for x in (r["gold"] or [])])
        raw_codes_unique = set([x for x in raw])
        tp_i = len(raw_codes_unique & gold)
        fp_i = max(0, len(raw) - tp_i)         # penalize OOV + duplicates + wrong codes
        fn_i = max(0, len(gold - raw_codes_unique))
        tp += tp_i; fp += fp_i; fn += fn_i
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return {
        "bag_micro_precision_oovpen": float(prec),
        "bag_micro_recall_oovpen": float(rec),
        "bag_micro_f1_oovpen": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn)
    }

# ---------- Plots ----------
def plot_metrics_bars(summary_df: pd.DataFrame, out_png: str):
    cols = ["micro_f1", "micro_precision", "micro_recall"]
    x = np.arange(len(summary_df))
    width = 0.25
    fig = plt.figure(figsize=(8, 5))
    for i, col in enumerate(cols):
        plt.bar(x + i*width, summary_df[col].values, width, label=col)
    plt.xticks(x + width, summary_df["mode"].tolist(), rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Micro metrics by decoding mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_predlen_hist(df_mode: pd.DataFrame, mode: str, out_png: str):
    lens = [len(p) for p in df_mode["pred"]]
    fig = plt.figure(figsize=(6,4))
    plt.hist(lens, bins=30)
    plt.xlabel("Predicted code count (post-filtered)")
    plt.ylabel("Frequency")
    plt.title(f"Pred length distribution — {mode}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_top_codes(per_label_df: pd.DataFrame, mode: str, out_png: str, topn=30):
    df = per_label_df.sort_values("support", ascending=False).head(topn)
    fig = plt.figure(figsize=(10, 6))
    plt.barh(df["code"][::-1], df["f1"][::-1])
    plt.xlabel("F1 (per label)")
    plt.title(f"Top-{topn} frequent labels — per-label F1 — {mode}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_oov_dup_bars(stats_by_mode: Dict[str, Dict[str, float]], out_png: str):
    modes = list(stats_by_mode.keys())
    oov = [stats_by_mode[m]["oov_rate"] for m in modes]
    dup = [stats_by_mode[m]["dup_rate"] for m in modes]
    kept= [stats_by_mode[m]["filtered_kept_rate"] for m in modes]
    x = np.arange(len(modes))
    width = 0.25
    fig = plt.figure(figsize=(8,5))
    plt.bar(x, oov, width, label="OOV rate")
    plt.bar(x + width, dup, width, label="Dup rate")
    plt.bar(x + 2*width, kept, width, label="Kept rate (post-filter)")
    plt.xticks(x + width, modes, rotation=30, ha="right")
    plt.ylabel("Rate")
    plt.title("Raw-token diagnostics (effect of post-filter)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_bucket_bars(bucket_by_mode: Dict[str, Dict[str, Any]], out_png: str):
    modes = list(bucket_by_mode.keys())
    head = [bucket_by_mode[m]["head"]["micro_f1"] for m in modes]
    torso= [bucket_by_mode[m]["torso"]["micro_f1"] for m in modes]
    tail = [bucket_by_mode[m]["tail"]["micro_f1"] for m in modes]
    x = np.arange(len(modes))
    width = 0.25
    fig = plt.figure(figsize=(8,5))
    plt.bar(x, head, width, label="head")
    plt.bar(x + width, torso, width, label="torso")
    plt.bar(x + 2*width, tail, width, label="tail")
    plt.xticks(x + width, modes, rotation=30, ha="right")
    plt.ylabel("Micro F1")
    plt.title("Head/Torso/Tail micro-F1 by mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_per_label_f1_hist(per_label_df: pd.DataFrame, mode: str, out_png: str):
    fig = plt.figure(figsize=(6,4))
    plt.hist(per_label_df["f1"].values, bins=40)
    plt.xlabel("Per-label F1")
    plt.ylabel("Count (labels)")
    plt.title(f"Per-label F1 distribution — {mode}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# ---------- Main pipeline ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_pickle", required=True)
    parser.add_argument("--pred_files", nargs="*", default=None,
                        help="Optional explicit prediction files (under run_dir). Default: discover all preds_*.jsonl.")
    parser.add_argument("--compare_all", action="store_true",
                        help="If set, auto-discover all preds_*.jsonl in run_dir.")
    parser.add_argument("--head_k", type=int, default=50)
    parser.add_argument("--torso_k", type=int, default=450)
    parser.add_argument("--out_dir", default=None,
                        help="Where to write analysis; default <run_dir>/analysis")
    args = parser.parse_args()

    set_seed(42)

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Data + split + vocab (consistent with training)
    df_all = load_full_df(args.data_pickle)
    train_df, val_df, test_df = subject_splits(df_all, subject_col=SUBJECT_COL,
                                               test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SPLIT_SEED)
    labels_vocab = load_label_space(run_dir, df_all)

    # frequency for head/torso/tail (global)
    label_freq = collections.Counter()
    for codes in df_all[LABEL_COL]:
        label_freq.update([normalize_code(c) for c in codes])

    # 2) Collect prediction files
    pred_map = {}
    if args.compare_all or (args.pred_files is None):
        found = discover_pred_files(run_dir)
        if not found:
            raise SystemExit(f"No preds_*.jsonl files found in {run_dir}.")
        pred_map.update(found)
    if args.pred_files:
        for fn in args.pred_files:
            path = os.path.join(run_dir, fn) if not os.path.isabs(fn) else fn
            if not os.path.isfile(path):
                raise SystemExit(f"Prediction file not found: {path}")
            base = os.path.basename(path)
            key = base[len("preds_"):-len(".jsonl")] if base.startswith("preds_") and base.endswith(".jsonl") else base
            pred_map[key] = path

    logging.info(f"Analysing {len(pred_map)} prediction file(s): {list(pred_map.keys())}")

    # 3) For each file: align, metrics, bucketed, parent3, oov/dup, tables, plots
    compare_rows = []
    oov_dup_by_mode = {}
    buckets_by_mode = {}
    parent_by_mode = {}

    for mode, path in pred_map.items():
        recs = read_jsonl(path)
        dfp = attach_preds_to_test(test_df, recs, labels_vocab)

        # post-filtered predictions are in dfp['pred']; build Y
        Y_true = to_multihot(dfp["gold"].tolist(), labels_vocab)
        Y_pred = to_multihot(dfp["pred"].tolist(), labels_vocab)
        met = multihot_metrics(Y_true, Y_pred)
        jaccard_mean, exact_match = jaccard_and_exact(dfp["pred"].tolist(), dfp["gold"].tolist())

        # per label table (for plots)
        per_lbl = per_label_table(Y_true, Y_pred, labels_vocab)
        per_lbl_path = os.path.join(out_dir, f"per_label_{mode}.csv")
        per_lbl.to_csv(per_lbl_path, index=False)

        # parent3
        p3 = parent3_micro(Y_true, Y_pred, labels_vocab)
        parent_by_mode[mode] = p3
        json.dump(p3, open(os.path.join(out_dir, f"parent3_{mode}.json"), "w"), indent=2)

        # head/torso/tail
        buckets = head_torso_tail_metrics(label_freq, labels_vocab, Y_true, Y_pred,
                                          head_k=args.head_k, torso_k=args.torso_k)
        buckets_by_mode[mode] = buckets
        json.dump(buckets, open(os.path.join(out_dir, f"head_torso_tail_{mode}.json"), "w"), indent=2)

        # OOV + dup (uses pred_raw if present, else pred as fallback)
        oovdup = oov_dup_stats(dfp, labels_vocab)
        oov_dup_by_mode[mode] = oovdup
        json.dump(oovdup, open(os.path.join(out_dir, f"oov_dup_{mode}.json"), "w"), indent=2)

        # bag-micro metrics with OOV penalty (on raw bags)
        bag_oov = bag_micro_with_oov_penalty(dfp, labels_vocab)

        # summary row
        compare_rows.append({
            "mode": mode,
            "samples_n": len(dfp),
            "pred_count_mean": float(np.mean([len(p) for p in dfp["pred"]])),
            "gold_count_mean": float(np.mean([len(g) for g in dfp["gold"]])),
            "exact_match_rate": exact_match,
            "jaccard_mean": jaccard_mean,
            **met,
            **{f"parent3_{k}": v for k, v in p3.items()},
            "oov_rate": oovdup["oov_rate"],
            "dup_rate": oovdup["dup_rate"],
            "filtered_kept_rate": oovdup["filtered_kept_rate"],
            **bag_oov
        })

        # plots per mode
        plot_predlen_hist(dfp, mode, os.path.join(out_dir, f"predlen_hist_{mode}.png"))
        plot_top_codes(per_lbl, mode, os.path.join(out_dir, f"top_codes_{mode}.png"))
        plot_per_label_f1_hist(per_lbl, mode, os.path.join(out_dir, f"per_label_f1_hist_{mode}.png"))

    # 4) Combined tables + plots
    summary_df = pd.DataFrame(compare_rows).sort_values("mode").reset_index(drop=True)
    summary_df.to_csv(os.path.join(out_dir, "compare_summary.csv"), index=False)
    json.dump(json.loads(summary_df.to_json(orient="records")),
              open(os.path.join(out_dir, "compare_summary.json"), "w"), indent=2)

    # bars across modes
    plot_metrics_bars(summary_df, os.path.join(out_dir, "metrics_bars.png"))
    plot_oov_dup_bars(oov_dup_by_mode, os.path.join(out_dir, "oov_dup_bars.png"))
    plot_bucket_bars(buckets_by_mode, os.path.join(out_dir, "head_torso_tail_bars.png"))

    # Print a tight summary to stdout
    print("\n=== COMPARISON (key metrics) ===")
    print(summary_df[[
        "mode","micro_f1","micro_precision","micro_recall",
        "samples_f1","oov_rate","dup_rate","filtered_kept_rate",
        "bag_micro_f1_oovpen"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
