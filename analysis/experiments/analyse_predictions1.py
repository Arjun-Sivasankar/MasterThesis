# -*- coding: utf-8 -*-
"""
Analysis of predictions dumped by dump_preds.py
- Metrics computed in the SAME label space as training (uses label_space.json).
- Deep dives: per-code metrics, head/torso/tail, parent-3 grouping,
  sample-level diagnostics, length effects, calibration-ish views.
- Plots and CSVs are written to: <run_dir>/analysis/

Example:
  python analyse_preds.py \
    --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
    --preds_jsonl predictions.jsonl \
    --head_n 50 \
    --torso_n 450 \
    --save_plots 1
"""

import os, re, json, argparse, logging, math, pickle
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SUBJECT_COL = "subject_id_x"
LABEL_COL   = "icd_code"
SEED        = 42

# ----------------- utils -----------------
def safe_div(num, den):
    return float(num) / float(den) if den else 0.0

def normalize_code(c: str) -> str:
    c = c.strip().upper()
    c = re.sub(r"\s+", "", c)
    return c[:-1] if c.endswith(".") else c

def parent3(code: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", code.upper())
    return s[:3] if s else ""

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def codes_to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(label_vocab)}
    Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
    for i, lst in enumerate(code_lists):
        for c in lst:
            j = idx.get(c)
            if j is not None:
                Y[i, j] = 1
    return Y

def per_sample_scores(gold: List[List[str]], pred: List[List[str]]) -> pd.DataFrame:
    """Compute sample-level Jaccard and F1 (set-F1) and counts."""
    rows = []
    for g, p in zip(gold, pred):
        G, P = set(g), set(p)
        inter = len(G & P)
        union = len(G | P)
        # per-sample jaccard & f1 (set formulation)
        jacc = safe_div(inter, union) if union > 0 else 1.0
        denom = (len(G) + len(P))
        f1 = safe_div(2.0 * inter, denom) if denom > 0 else 1.0
        rows.append({
            "gold_count": len(G),
            "pred_count": len(P),
            "jaccard": jacc,
            "sample_f1": f1,
        })
    return pd.DataFrame(rows)

def scikit_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1":   f1_score(Y_true, Y_pred, average="micro",   zero_division=0),
        "macro_f1":   f1_score(Y_true, Y_pred, average="macro",   zero_division=0),
        "samples_f1": f1_score(Y_true, Y_pred, average="samples", zero_division=0),
        "micro_precision":   precision_score(Y_true, Y_pred, average="micro",   zero_division=0),
        "macro_precision":   precision_score(Y_true, Y_pred, average="macro",   zero_division=0),
        "samples_precision": precision_score(Y_true, Y_pred, average="samples", zero_division=0),
        "micro_recall":      recall_score(Y_true, Y_pred, average="micro",   zero_division=0),
        "macro_recall":      recall_score(Y_true, Y_pred, average="macro",   zero_division=0),
        "samples_recall":    recall_score(Y_true, Y_pred, average="samples", zero_division=0),
    }

def per_code_stats(Y_true: np.ndarray, Y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    tp = (Y_true * Y_pred).sum(axis=0)
    fp = ((1 - Y_true) * Y_pred).sum(axis=0)
    fn = (Y_true * (1 - Y_pred)).sum(axis=0)
    support = Y_true.sum(axis=0)

    prec = np.array([safe_div(tp[i], tp[i] + fp[i]) for i in range(len(labels))])
    rec  = np.array([safe_div(tp[i], tp[i] + fn[i]) for i in range(len(labels))])
    f1   = np.array([safe_div(2*prec[i]*rec[i], prec[i]+rec[i]) if (prec[i]+rec[i])>0 else 0.0 for i in range(len(labels))])

    df = pd.DataFrame({
        "code": labels,
        "support": support.astype(int),
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "fn": fn.astype(int),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })
    return df.sort_values(["support","f1"], ascending=[False, False]).reset_index(drop=True)

def group_micro(df_codes: pd.DataFrame, idxs: np.ndarray) -> Dict[str, float]:
    sub = df_codes.iloc[idxs]
    TP = int(sub["tp"].sum())
    FP = int(sub["fp"].sum())
    FN = int(sub["fn"].sum())
    prec = safe_div(TP, TP+FP)
    rec  = safe_div(TP, TP+FN)
    f1   = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0.0
    return {"micro_precision": prec, "micro_recall": rec, "micro_f1": f1}

def parent_group_metrics(golds: List[List[str]], preds: List[List[str]]) -> Dict[str, float]:
    Gp = [sorted({parent3(c) for c in g if c}) for g in golds]
    Pp = [sorted({parent3(c) for c in p if c}) for p in preds]
    # Build vocab of parents present (could also load from label_space, but parents are derived anyway)
    parents = sorted({p for lst in (Gp + Pp) for p in lst})
    idx = {p:i for i,p in enumerate(parents)}
    Yg = np.zeros((len(Gp), len(parents)), dtype=np.int32)
    Yp = np.zeros_like(Yg)
    for i, lst in enumerate(Gp):
        for p in lst: Yg[i, idx[p]] = 1
    for i, lst in enumerate(Pp):
        for p in lst: Yp[i, idx[p]] = 1
    return {
        "parent3_micro_precision": precision_score(Yg, Yp, average="micro", zero_division=0),
        "parent3_micro_recall": recall_score(Yg, Yp, average="micro", zero_division=0),
        "parent3_micro_f1": f1_score(Yg, Yp, average="micro", zero_division=0),
        "parent3_labels": len(parents),
    }

# ----------------- plotting helpers -----------------
def plot_hist_counts(df_samples: pd.DataFrame, out_png: str):
    plt.figure(figsize=(7.5,4.5))
    bins = np.arange(0, max(df_samples["gold_count"].max(), df_samples["pred_count"].max()) + 2) - 0.5
    plt.hist(df_samples["gold_count"], bins=bins, alpha=0.6, label="gold")
    plt.hist(df_samples["pred_count"], bins=bins, alpha=0.6, label="pred")
    plt.xlabel("Count per sample"); plt.ylabel("Frequency"); plt.title("Gold vs Pred counts per sample"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def plot_scatter_len_vs_f1(df_samples: pd.DataFrame, prompts_len: List[int], out_png: str):
    plt.figure(figsize=(7,4.5))
    x = np.array(prompts_len)
    y = np.array(df_samples["sample_f1"].values)
    plt.scatter(x, y, s=3, alpha=0.4)
    plt.xlabel("Prompt length (chars)")
    plt.ylabel("Per-sample F1")
    plt.title("Prompt length vs sample F1")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def plot_support_vs_f1(df_codes: pd.DataFrame, out_png: str):
    plt.figure(figsize=(7,4.5))
    x = df_codes["support"].values
    y = df_codes["f"].values if "f" in df_codes.columns else df_codes["f1"].values
    plt.scatter(x, y, s=6, alpha=0.4)
    plt.xscale("log"); plt.xlabel("Support (log scale)")
    plt.ylabel("Per-code F1"); plt.title("Per-code F1 vs support")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def plot_head_torso_tail_bars(head_metrics: Dict[str,float],
                              torso_metrics: Dict[str,float],
                              tail_metrics: Dict[str,float],
                              out_png: str):
    labels = ["micro_precision", "micro_recall", "micro_f1"]
    vals = [
        [head_metrics[k] for k in labels],
        [torso_metrics[k] for k in labels],
        [tail_metrics[k] for k in labels],
    ]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(7,4.5))
    plt.bar(x - width, vals[0], width, label="Head")
    plt.bar(x,         vals[1], width, label="Torso")
    plt.bar(x + width, vals[2], width, label="Tail")
    plt.xticks(x, ["Prec","Recall","F1"])
    plt.ylim(0, max(0.01, max(v for row in vals for v in row))*1.15)
    plt.title("Head / Torso / Tail (micro)")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def plot_pred_vs_gold_scatter(df_samples: pd.DataFrame, out_png: str):
    plt.figure(figsize=(6.2,6))
    plt.scatter(df_samples["gold_count"], df_samples["pred_count"], s=4, alpha=0.35)
    lim = max(df_samples["gold_count"].max(), df_samples["pred_count"].max())
    plt.plot([0,lim], [0,lim])
    plt.xlabel("Gold count"); plt.ylabel("Pred count"); plt.title("Pred count vs Gold count")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def plot_bin_bars(bins: List[str], values: List[float], ylabel: str, title: str, out_png: str):
    plt.figure(figsize=(7,4.5))
    xs = np.arange(len(bins))
    plt.bar(xs, values)
    plt.xticks(xs, bins, rotation=15)
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory with label_space.json")
    ap.add_argument("--preds_jsonl", default=None, help="predictions.jsonl from dump_preds.py (default: <run_dir>/predictions.jsonl)")
    ap.add_argument("--data_pickle", default=None, help="If provided, use global label support (entire dataset) for head/torso/tail selection.")
    ap.add_argument("--head_n", type=int, default=50)
    ap.add_argument("--torso_n", type=int, default=450)
    ap.add_argument("--save_plots", type=int, default=1)
    args = ap.parse_args()

    out_dir = ensure_dir(os.path.join(args.run_dir, "analysis"))

    # Label space from run dir (ensures consistency with training)
    with open(os.path.join(args.run_dir, "label_space.json"), "r") as f:
        labels_vocab = json.load(f)["labels"]
    label_index = {c:i for i,c in enumerate(labels_vocab)}
    logging.info(f"Label space size: {len(labels_vocab)}")

    preds_path = os.path.join(args.run_dir, "predictions.jsonl")
    rows = read_jsonl(preds_path)
    if not rows:
        raise RuntimeError(f"No rows found in {preds_path}")

    # Load lists
    gold_lists = [ [normalize_code(x) for x in r["gold"]] for r in rows ]
    pred_lists = [ [normalize_code(x) for x in r["pred"]] for r in rows ]
    prompt_len = [ int(r.get("prompt_len_chars", 0)) for r in rows ]

    # Build Y matrices in training label space (matches training/test metrics)
    Y_true = codes_to_multihot(gold_lists, labels_vocab)
    Y_pred = codes_to_multihot(pred_lists, labels_vocab)

    # Global metrics (scikit) — identical definition to training
    metrics = scikit_metrics(Y_true, Y_pred)

    # Sample-level metrics
    df_samples = per_sample_scores(gold_lists, pred_lists)
    df_samples["prompt_len"] = prompt_len
    df_samples.to_csv(os.path.join(out_dir, "per_sample_metrics.csv"), index=False)

    # Per-code metrics
    df_codes = per_code_stats(Y_true, Y_pred, labels_vocab)
    # keep a 'f' alias to avoid old scripts breaking
    df_codes["f"] = df_codes["f1"]
    df_codes.to_csv(os.path.join(out_dir, "per_code_metrics.csv"), index=False)

    # Head/Torso/Tail selection
    if args.data_pickle and os.path.isfile(args.data_pickle):
        # Use GLOBAL support from the full dataset (gold occurrence by label)
        logging.info("Computing global support from full dataset for head/torso/tail …")
        full_df = pickle.load(open(args.data_pickle, "rb"))
        # Count support per label across ENTIRE dataset
        support_global = {c:0 for c in labels_vocab}
        for codes in full_df[LABEL_COL].tolist():
            for c in codes:
                cc = normalize_code(str(c))
                if cc in support_global:
                    support_global[cc] += 1
        support_series = pd.Series(support_global).sort_values(ascending=False)
        sorted_codes = support_series.index.tolist()
    else:
        # Use TEST-set support
        logging.info("Using TEST-set support for head/torso/tail …")
        sorted_codes = df_codes.sort_values("support", ascending=False)["code"].tolist()

    head_n  = min(args.head_n, len(sorted_codes))
    torso_n = min(args.torso_n, max(0, len(sorted_codes) - head_n))
    head_codes  = set(sorted_codes[:head_n])
    torso_codes = set(sorted_codes[head_n:head_n+torso_n])
    tail_codes  = set(sorted_codes[head_n+torso_n:])

    # Index arrays for micro on subsets
    idx_map = {c:i for i,c in enumerate(df_codes["code"].tolist())}
    head_idx  = np.array([idx_map[c] for c in df_codes["code"] if c in head_codes])
    torso_idx = np.array([idx_map[c] for c in df_codes["code"] if c in torso_codes])
    tail_idx  = np.array([idx_map[c] for c in df_codes["code"] if c in tail_codes])

    head_metrics  = group_micro(df_codes, head_idx)  if len(head_idx)  else {"micro_precision":0,"micro_recall":0,"micro_f1":0}
    torso_metrics = group_micro(df_codes, torso_idx) if len(torso_idx) else {"micro_precision":0,"micro_recall":0,"micro_f1":0}
    tail_metrics  = group_micro(df_codes, tail_idx)  if len(tail_idx)  else {"micro_precision":0,"micro_recall":0,"micro_f1":0}

    # Parent-3 grouped micro metrics
    parent_metrics = parent_group_metrics(gold_lists, pred_lists)

    # Calibration-ish binning by gold size
    bins = [(0,0),(1,2),(3,5),(6,10),(11,20),(21,9999)]
    bin_names = [f"{a}-{b}" for a,b in bins]
    bin_vals_f1 = []
    for a,b in bins:
        mask = (df_samples["gold_count"]>=a) & (df_samples["gold_count"]<=b)
        bin_vals_f1.append(float(df_samples.loc[mask, "sample_f1"].mean() if mask.any() else 0.0))
    df_bins = pd.DataFrame({"bin":bin_names, "mean_sample_f1":bin_vals_f1})
    df_bins.to_csv(os.path.join(out_dir, "per_goldcount_bins.csv"), index=False)

    # Prompt length binning (quartiles)
    if df_samples["prompt_len"].max() > 0:
        qs = df_samples["prompt_len"].quantile([0.25,0.5,0.75]).values.tolist()
        edges = [df_samples["prompt_len"].min()-1] + qs + [df_samples["prompt_len"].max()+1]
        names = [f"{int(math.floor(edges[i]+1))}-{int(math.floor(edges[i+1]))}" for i in range(len(edges)-1)]
        vals  = []
        for i in range(len(edges)-1):
            a,b = edges[i], edges[i+1]
            m = (df_samples["prompt_len"]>a) & (df_samples["prompt_len"]<=b)
            vals.append(float(df_samples.loc[m, "sample_f1"].mean() if m.any() else 0.0))
        df_lenbins = pd.DataFrame({"prompt_len_bin":names, "mean_sample_f1":vals})
        df_lenbins.to_csv(os.path.join(out_dir, "per_promptlen_bins.csv"), index=False)
    else:
        df_lenbins = pd.DataFrame({"prompt_len_bin":[],"mean_sample_f1":[]})

    # Save summary JSON
    summary = {
        "samples_n": int(len(rows)),
        "labels_n": int(len(labels_vocab)),
        "pred_count_mean": float(np.mean([len(x) for x in pred_lists])),
        "gold_count_mean": float(np.mean([len(x) for x in gold_lists])),
        "exact_match_rate": float(np.mean([
            1.0 if set(g) == set(p) else 0.0 for g,p in zip(gold_lists,pred_lists)
        ])),
        "jaccard_mean": float(df_samples["jaccard"].mean()),
        **metrics,
        "head_summary": {
            "codes": int(len(head_idx)),
            "support_total": int(df_codes.iloc[head_idx]["support"].sum()) if len(head_idx) else 0,
            "micro_precision": head_metrics["micro_precision"],
            "micro_recall": head_metrics["micro_recall"],
            "micro_f1": head_metrics["micro_f1"],
        },
        "torso_summary": {
            "codes": int(len(torso_idx)),
            "support_total": int(df_codes.iloc[torso_idx]["support"].sum()) if len(torso_idx) else 0,
            "micro_precision": torso_metrics["micro_precision"],
            "micro_recall": torso_metrics["micro_recall"],
            "micro_f1": torso_metrics["micro_f1"],
        },
        "tail_summary": {
            "codes": int(len(tail_idx)),
            "support_total": int(df_codes.iloc[tail_idx]["support"].sum()) if len(tail_idx) else 0,
            "micro_precision": tail_metrics["micro_precision"],
            "micro_recall": tail_metrics["micro_recall"],
            "micro_f1": tail_metrics["micro_f1"],
        },
        **parent_metrics,
    }
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --------- plots ----------
    if args.save_plots:
        plot_hist_counts(df_samples, os.path.join(out_dir, "counts_hist.png"))
        plot_pred_vs_gold_scatter(df_samples, os.path.join(out_dir, "pred_vs_gold_scatter.png"))
        if df_samples["prompt_len"].max() > 0:
            plot_scatter_len_vs_f1(df_samples, prompt_len, os.path.join(out_dir, "len_vs_sample_f1.png"))
        plot_support_vs_f1(df_codes, os.path.join(out_dir, "per_code_support_vs_f1.png"))
        plot_head_torso_tail_bars(head_metrics, torso_metrics, tail_metrics, os.path.join(out_dir, "htt_micro_bars.png"))

    # Top-K table for quick reading
    topk = df_codes.head(30).copy()
    topk.to_csv(os.path.join(out_dir, "top30_codes_by_support.csv"), index=False)

    # Console summary
    print("\n=== SUMMARY ===")
    for k in ["samples_n","labels_n","pred_count_mean","gold_count_mean","exact_match_rate","jaccard_mean",
              "micro_precision","micro_recall","micro_f1","macro_precision","macro_recall","macro_f1",
              "samples_precision","samples_recall","samples_f1"]:
        print(f"{k}: {summary[k]}")
    print("head_summary:", json.dumps(summary["head_summary"]))
    print("torso_summary:", json.dumps(summary["torso_summary"]))
    print("tail_summary:", json.dumps(summary["tail_summary"]))
    print(f'parent3_micro_precision: {summary["parent3_micro_precision"]}')
    print(f'parent3_micro_recall: {summary["parent3_micro_recall"]}')
    print(f'parent3_micro_f1: {summary["parent3_micro_f1"]}')
    print(f"\nWrote outputs to: {out_dir}")

if __name__ == "__main__":
    main()
