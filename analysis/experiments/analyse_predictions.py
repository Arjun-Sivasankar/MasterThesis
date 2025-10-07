## analyze_predictions.py
## Read predictions JSONL + label_space.json; produce metrics, plots, and CSVs.
#
#import os, re, json, argparse, collections, statistics
#from typing import List, Dict, Any
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
#def load_jsonl(path) -> List[Dict[str, Any]]:
#    out = []
#    with open(path, "r") as f:
#        for line in f:
#            if line.strip():
#                out.append(json.loads(line))
#    return out
#
#def normalize_code(c: str) -> str:
#    c = (c or "").strip().upper()
#    c = re.sub(r"\s+", "", c)
#    if c.endswith("."): c = c[:-1]
#    return c
#
#def to_parent_3digit(c: str) -> str:
#    # ICD-9 parent at 3-digit level; handles V/E cases by trimming after root.
#    c = normalize_code(c)
#    if not c: return c
#    if c[0] in ("V","E"):
#        # Keep the root (e.g., V10 or E934); for E-codes the root is 4 chars, V 2–3.
#        # A simple rule: strip everything after the first '.'; if none, keep as-is.
#        return c.split(".")[0]
#    # numeric
#    return c.split(".")[0][:3] if len(c) >= 3 else c.split(".")[0]
#
#def jaccard(a: set, b: set) -> float:
#    if not a and not b: return 1.0
#    if not a and b: return 0.0
#    if a and not b: return 0.0
#    return len(a & b) / len(a | b)
#
#def f1_from_pr(p: float, r: float) -> float:
#    if p + r == 0: return 0.0
#    return 2*p*r/(p+r)
#
#def safe_div(n, d): return (n / d) if d else 0.0
#
#def compute_sample_metrics(recs):
#    rows = []
#    for r in recs:
#        g = set(map(normalize_code, r["gold"]))
#        p = set(map(normalize_code, r["pred"]))
#        tp = len(g & p); fp = len(p - g); fn = len(g - p)
#        prec = safe_div(tp, tp+fp)
#        rec  = safe_div(tp, tp+fn)
#        f1   = f1_from_pr(prec, rec)
#        rows.append({
#            "idx": r["idx"],
#            "hadm_id": r.get("hadm_id"),
#            "subject_id": r.get("subject_id"),
#            "gold_count": len(g),
#            "pred_count": len(p),
#            "tp": tp, "fp": fp, "fn": fn,
#            "precision": prec, "recall": rec, "f1": f1,
#            "jaccard": jaccard(g, p),
#            "exact_match": int(g == p),
#            "prompt_len_chars": r.get("prompt_len_chars", None),
#        })
#    df = pd.DataFrame(rows)
#    return df
#
#def compute_label_metrics(recs, label_vocab: List[str]):
#    # per-label TP/FP/FN
#    idx = {c:i for i,c in enumerate(label_vocab)}
#    TP = np.zeros(len(label_vocab), dtype=np.int64)
#    FP = np.zeros(len(label_vocab), dtype=np.int64)
#    FN = np.zeros(len(label_vocab), dtype=np.int64)
#    SUP = np.zeros(len(label_vocab), dtype=np.int64)
#
#    for r in recs:
#        g = set(map(normalize_code, r["gold"]))
#        p = set(map(normalize_code, r["pred"]))
#        for c in g:
#            j = idx.get(c)
#            if j is not None:
#                SUP[j] += 1
#        for c in (g & p):
#            j = idx.get(c)
#            if j is not None:
#                TP[j] += 1
#        for c in (p - g):
#            j = idx.get(c)
#            if j is not None:
#                FP[j] += 1
#        for c in (g - p):
#            j = idx.get(c)
#            if j is not None:
#                FN[j] += 1
#
#    rows = []
#    for i, c in enumerate(label_vocab):
#        tp, fp, fn, sup = int(TP[i]), int(FP[i]), int(FN[i]), int(SUP[i])
#        prec = safe_div(tp, tp+fp)
#        rec  = safe_div(tp, tp+fn)
#        f1   = f1_from_pr(prec, rec)
#        rows.append({"code": c, "support": sup, "tp": tp, "fp": fp, "fn": fn,
#                     "precision": prec, "recall": rec, "f1": f1})
#    df = pd.DataFrame(rows).sort_values(["support","f1"], ascending=[False, False]).reset_index(drop=True)
#    return df
#
#def head_torso_tail(df_label, head_n=50, torso_n=500):
#    df = df_label.sort_values("support", ascending=False)
#    head  = df.head(head_n)
#    torso = df.iloc[head_n:torso_n]
#    tail  = df.iloc[torso_n:]
#    def summarize(x):
#        return {
#            "codes": len(x),
#            "support_total": int(x["support"].sum()),
#            "mean_f1": float(x["f1"].mean()),
#            "mean_recall": float(x["recall"].mean()),
#            "mean_precision": float(x["precision"].mean()),
#        }
#    return summarize(head), summarize(torso), summarize(tail)
#
#def per_group_by_gold_size(df_sample):
#    # buckets by gold_count
#    bins = [0,1,2,3,5,10,1000]
#    labels = ["0","1","2","3-5","6-10",">10"]
#    cats = pd.cut(df_sample["gold_count"], bins=bins, labels=labels, right=True, include_lowest=True)
#    grp = df_sample.groupby(cats).agg(
#        n=("idx","count"),
#        prec=("precision","mean"),
#        rec=("recall","mean"),
#        f1=("f1","mean"),
#        jacc=("jaccard","mean"),
#        exact=("exact_match","mean")
#    ).reset_index().rename(columns={"gold_count":"bucket","idx":"n"})
#    grp["bucket"] = grp["bucket"].astype(str)
#    return grp
#
#def per_parent3_metrics(recs):
#    # compute micro-F1 at 3-digit parent level
#    tp=fp=fn=0
#    for r in recs:
#        g = set(map(to_parent_3digit, r["gold"]))
#        p = set(map(to_parent_3digit, r["pred"]))
#        g.discard(""); p.discard("")
#        tp += len(g & p)
#        fp += len(p - g)
#        fn += len(g - p)
#    micro_p = safe_div(tp, tp+fp); micro_r = safe_div(tp, tp+fn); micro_f1 = f1_from_pr(micro_p, micro_r)
#    return {"parent3_micro_precision": micro_p, "parent3_micro_recall": micro_r, "parent3_micro_f1": micro_f1}
#
#def confusion_pairs(recs, min_support=25, top_k=50):
#    # For each gold code, count which *incorrect* codes appear with it ? likely confusions
#    co = collections.Counter()
#    gold_sup = collections.Counter()
#    for r in recs:
#        g = set(map(normalize_code, r["gold"]))
#        p = set(map(normalize_code, r["pred"]))
#        wrong = p - g
#        for c in g:
#            gold_sup[c] += 1
#            for w in wrong:
#                co[(c,w)] += 1
#    rows = []
#    for (g,w), cnt in co.most_common():
#        if gold_sup[g] >= min_support:
#            rows.append({"gold": g, "wrong": w, "count": cnt, "gold_support": gold_sup[g], "rate": cnt / gold_sup[g]})
#            if len(rows) >= 10000: break
#    df = pd.DataFrame(rows).sort_values(["rate","count"], ascending=[False,False])
#    return df.head(top_k)
#
#def top_fn_fp(df_label, top_k=50):
#    top_fn = df_label.sort_values("fn", ascending=False).head(top_k).copy()
#    top_fp = df_label.sort_values("fp", ascending=False).head(top_k).copy()
#    return top_fn, top_fp
#
#def plot_and_save(fig, path):
#    fig.tight_layout()
#    fig.savefig(path, bbox_inches="tight")
#    plt.close(fig)
#
#def main():
#    ap = argparse.ArgumentParser()
#    ap.add_argument("--run_dir", required=True, help="Run directory with label_space.json")
#    ap.add_argument("--pred_file", default=None, help="predictions_test.jsonl; default: <run_dir>/predictions_test.jsonl")
#    ap.add_argument("--out_prefix", default=None, help="prefix for outputs; default: <run_dir>/analysis/*")
#    args = ap.parse_args()
#
#    pred_path = args.pred_file or os.path.join(args.run_dir, "predictions_test.jsonl")
#    lab_path  = os.path.join(args.run_dir, "label_space.json")
#    out_dir   = args.out_prefix or os.path.join(args.run_dir, "analysis")
#    os.makedirs(out_dir, exist_ok=True)
#
#    recs = load_jsonl(pred_path)
#    with open(lab_path, "r") as f: labels_vocab = json.load(f)["labels"]
#
#    # --- SAMPLE-LEVEL ---
#    df_sample = compute_sample_metrics(recs)
#    df_sample.to_csv(os.path.join(out_dir, "per_sample_metrics.csv"), index=False)
#
#    # hist: precision/recall/f1
#    for col in ["precision","recall","f1","jaccard","pred_count","gold_count"]:
#        fig = plt.figure(figsize=(6,4))
#        df_sample[col].hist(bins=40)
#        plt.title(f"Histogram: {col}")
#        plt.xlabel(col); plt.ylabel("count")
#        plot_and_save(fig, os.path.join(out_dir, f"hist_{col}.png"))
#
#    # scatter: gold_count vs pred_count
#    fig = plt.figure(figsize=(5,5))
#    plt.scatter(df_sample["gold_count"], df_sample["pred_count"], s=6, alpha=0.4)
#    plt.xlabel("gold_count"); plt.ylabel("pred_count")
#    plt.title("Pred vs Gold counts")
#    plot_and_save(fig, os.path.join(out_dir, "scatter_pred_vs_gold.png"))
#
#    # F1 vs prompt length (chars)
#    if "prompt_len_chars" in df_sample.columns and df_sample["prompt_len_chars"].notna().any():
#      fig = plt.figure(figsize=(6,4))
#      plt.scatter(df_sample["prompt_len_chars"], df_sample["f1"], s=6, alpha=0.35)
#      plt.xlabel("prompt_len_chars"); plt.ylabel("sample F1")
#      plt.title("Sample F1 vs input length")
#      plot_and_save(fig, os.path.join(out_dir, "scatter_f1_vs_promptlen.png"))
#
#    # group by gold size
#    df_bucket = per_group_by_gold_size(df_sample)
#    df_bucket.to_csv(os.path.join(out_dir, "by_gold_size.csv"), index=False)
#    fig = plt.figure(figsize=(7,4))
#    for m in ["prec","rec","f1","exact","jacc"]:
#        plt.plot(df_bucket["bucket"], df_bucket[m], marker="o", label=m)
#    plt.legend(); plt.title("Performance by gold label count"); plt.ylabel("score"); plt.xlabel("gold_count bucket")
#    plot_and_save(fig, os.path.join(out_dir, "perf_by_gold_size.png"))
#
#    # --- LABEL-LEVEL ---
#    df_label = compute_label_metrics(recs, labels_vocab)
#    df_label.to_csv(os.path.join(out_dir, "per_label_metrics.csv"), index=False)
#
#    # recall vs support
#    fig = plt.figure(figsize=(6,4))
#    plt.scatter(df_label["support"], df_label["recall"], s=6, alpha=0.35)
#    plt.xscale("log"); plt.xlabel("support (log)"); plt.ylabel("recall")
#    plt.title("Per-label recall vs frequency")
#    plot_and_save(fig, os.path.join(out_dir, "recall_vs_support.png"))
#
#    # head/torso/tail summary
#    head, torso, tail = head_torso_tail(df_label, head_n=50, torso_n=500)
#    pd.DataFrame([head,torso,tail], index=["head_top50","torso_51to500","tail_501plus"]).to_csv(
#        os.path.join(out_dir, "head_torso_tail_summary.csv")
#    )
#
#    # top-K by F1 among supported labels (support >= 25)
#    df_s25 = df_label[df_label["support"] >= 25].copy()
#    df_s25.sort_values("f1", ascending=False).head(50).to_csv(os.path.join(out_dir, "top50_labels_by_f1_support>=25.csv"), index=False)
#
#    # top false negatives / false positives
#    top_fn, top_fp = top_fn_fp(df_label, top_k=50)
#    top_fn.to_csv(os.path.join(out_dir, "top50_false_negatives.csv"), index=False)
#    top_fp.to_csv(os.path.join(out_dir, "top50_false_positives.csv"), index=False)
#
#    # confusion pairs
#    df_conf = confusion_pairs(recs, min_support=25, top_k=50)
#    df_conf.to_csv(os.path.join(out_dir, "top50_confusion_pairs.csv"), index=False)
#
#    # parent 3-digit metrics
#    parent3 = per_parent3_metrics(recs)
#
#    # macro/micro/samples (recompute from df_sample)
#    micro_p = df_sample["tp"].sum() / max(1, (df_sample["tp"].sum() + df_sample["fp"].sum()))
#    micro_r = df_sample["tp"].sum() / max(1, (df_sample["tp"].sum() + df_sample["fn"].sum()))
#    micro_f = f1_from_pr(micro_p, micro_r)
#    exact_match = df_sample["exact_match"].mean()
#    jacc_mean   = df_sample["jaccard"].mean()
#
#    # Save a one-shot JSON report
#    report = {
#        "samples_n": int(len(df_sample)),
#        "labels_n": int(len(df_label)),
#        "pred_count_mean": float(df_sample["pred_count"].mean()),
#        "gold_count_mean": float(df_sample["gold_count"].mean()),
#        "exact_match_rate": float(exact_match),
#        "jaccard_mean": float(jacc_mean),
#        "micro_precision": float(micro_p),
#        "micro_recall": float(micro_r),
#        "micro_f1": float(micro_f),
#        "head_summary": head,
#        "torso_summary": torso,
#        "tail_summary": tail,
#        **parent3
#    }
#    with open(os.path.join(out_dir, "analysis_report.json"), "w") as f:
#        json.dump(report, f, indent=2)
#
#    # Quick console summary
#    print("\n=== SUMMARY ===")
#    for k, v in report.items():
#        if isinstance(v, dict):
#            print(f"{k}: {json.dumps(v)}")
#        else:
#            print(f"{k}: {v}")
#
#    # small bar: mean P/R/F1 head vs torso vs tail
#    fig = plt.figure(figsize=(6,4))
#    x = ["head","torso","tail"]
#    mf1 = [head["mean_f1"], torso["mean_f1"], tail["mean_f1"]]
#    mr  = [head["mean_recall"], torso["mean_recall"], tail["mean_recall"]]
#    mp  = [head["mean_precision"], torso["mean_precision"], tail["mean_precision"]]
#    X = np.arange(3)
#    w = 0.25
#    plt.bar(X - w, mp, width=w, label="precision")
#    plt.bar(X,     mr, width=w, label="recall")
#    plt.bar(X + w, mf1, width=w, label="f1")
#    plt.xticks(X, x); plt.ylabel("score"); plt.title("Head / Torso / Tail")
#    plt.legend()
#    plot_and_save(fig, os.path.join(out_dir, "head_torso_tail_bars.png"))
#
#if __name__ == "__main__":
#    main()

# -*- coding: utf-8 -*-
"""
Deep analysis of a predictions.jsonl produced by dump_predictions_consistent.py.
Writes figures and CSVs under <RUN_DIR>/analysis/.

Usage:
  python analyse_predictions_consistent.py --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002
"""

import os, re, json, argparse, logging, math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parent3(code: str) -> str:
    # ICD-9 numeric: first 3 digits before decimal
    # ICD-9 V/E: first 3 (Vxx) or 4 (Exxx) chars (ignore decimal)
    # ICD-10: first 3 alnum chars (ignore decimal)
    c = code.replace(".", "").upper()
    if not c: return c
    if c[0] == "E" and len(c) >= 4:   # Exxx
        return c[:4]
    if c[0] == "V" and len(c) >= 3:   # Vxx
        return c[:3]
    if c[0].isalpha():                # ICD-10 like
        return c[:3]
    # numeric ICD-9
    return c[:3]

def onehot(code_lists: List[List[str]], vocab: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(vocab)}
    Y = np.zeros((len(code_lists), len(vocab)), dtype=np.int32)
    for i, lst in enumerate(code_lists):
        for c in lst:
            j = idx.get(c)
            if j is not None: Y[i, j] = 1
    return Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pred_file", default="predictions.jsonl")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # label space for filtering and consistent metrics
    with open(os.path.join(run_dir, "label_space.json"), "r") as f:
        labels_vocab = json.load(f)["labels"]
    label_set = set(labels_vocab)

    # load predictions
    preds_path = os.path.join(run_dir, args.pred_file)
    rows = []
    with open(preds_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # ensure lists
    df["pred"] = df["pred"].apply(lambda x: [c for c in (x or []) if c in label_set])
    df["gold"] = df["gold"].apply(lambda x: [str(c) for c in (x or []) if str(c) in label_set])

    # global metrics (micro/macro/samples)
    Y_true = onehot(df["gold"].tolist(), labels_vocab)
    Y_pred = onehot(df["pred"].tolist(), labels_vocab)
    metrics = {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(Y_true, Y_pred, average="samples", zero_division=0),
        "micro_precision": precision_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(Y_true, Y_pred, average="macro", zero_division=0),
        "samples_precision": precision_score(Y_true, Y_pred, average="samples", zero_division=0),
        "micro_recall": recall_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_recall": recall_score(Y_true, Y_pred, average="macro", zero_division=0),
        "samples_recall": recall_score(Y_true, Y_pred, average="samples", zero_division=0),
    }

    # sample-level stats
    def jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb: return 1.0
        if not sa and sb: return 0.0
        if sa and not sb: return 0.0
        return len(sa & sb) / len(sa | sb)

    df["pred_count"] = df["pred"].apply(len)
    df["gold_count"] = df["gold"].apply(len)
    df["exact_match"] = (df["pred"] == df["gold"]).astype(int)
    df["jaccard"] = [jaccard(a,b) for a,b in zip(df["gold"], df["pred"])]

    # per-code metrics (head/torso/tail)
    gold_support = Counter([c for lst in df["gold"] for c in lst])
    head_codes = [c for c,_ in gold_support.most_common(50)]
    torso_codes = [c for c,_ in gold_support.most_common(500)][50:500]
    tail_codes = [c for c in labels_vocab if c not in set(head_codes) | set(torso_codes)]

    # per-code PRF1
    per_code = []
    for c in labels_vocab:
        y_t = np.array([int(c in lst) for lst in df["gold"]], dtype=np.int32)
        y_p = np.array([int(c in lst) for lst in df["pred"]], dtype=np.int32)
        p, r, f, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
        per_code.append({"code": c, "support": int(gold_support.get(c,0)), "precision": p, "recall": r, "f1": f})
    per_code_df = pd.DataFrame(per_code).sort_values("support", ascending=False)
    per_code_df.to_csv(os.path.join(out_dir, "per_code_metrics.csv"), index=False)

    def bucket_summary(codes):
        sub = per_code_df[per_code_df["code"].isin(codes)]
        return {
            "codes": int(len(codes)),
            "support_total": int(sub["support"].sum()),
            "mean_f1": float(sub["f1"].mean() if len(sub) else 0.0),
            "mean_recall": float(sub["recall"].mean() if len(sub) else 0.0),
            "mean_precision": float(sub["precision"].mean() if len(sub) else 0.0),
        }

    head_sum = bucket_summary(head_codes)
    torso_sum = bucket_summary(torso_codes)
    tail_sum = bucket_summary(tail_codes)

    # parent-3 aggregation
    parents = sorted({parent3(c) for c in labels_vocab})
    # map child->parent
    pmap = {c: parent3(c) for c in labels_vocab}
    parent_rows = []
    for p in parents:
        childs = [c for c in labels_vocab if pmap[c] == p]
        if not childs: continue
        y_t = np.array([[int(c in lst) for c in childs] for lst in df["gold"]], dtype=np.int32).max(axis=1)
        y_p = np.array([[int(c in lst) for c in childs] for lst in df["pred"]], dtype=np.int32).max(axis=1)
        p_, r_, f_, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
        parent_rows.append({"parent": p, "precision": p_, "recall": r_, "f1": f_, "support": int(y_t.sum())})
    parent_df = pd.DataFrame(parent_rows).sort_values("support", ascending=False)
    parent_df.to_csv(os.path.join(out_dir, "parent3_metrics.csv"), index=False)

    # plots
    plt.figure(figsize=(6,4))
    plt.hist(df["pred_count"], bins=30)
    plt.title("Pred codes per sample"); plt.xlabel("count"); plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pred_count_hist.png")); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(df["gold_count"], bins=30)
    plt.title("Gold codes per sample"); plt.xlabel("count"); plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "gold_count_hist.png")); plt.close()

    # Jaccard vs lengths
    plt.figure(figsize=(6,4))
    plt.scatter(df["gold_count"], df["jaccard"], s=2, alpha=0.25)
    plt.xlabel("gold_count"); plt.ylabel("jaccard")
    plt.title("Jaccard vs gold_count")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "jaccard_vs_gold_count.png")); plt.close()

    # precision/recall bars for top 30 head codes
    top30 = per_code_df.head(30).copy()
    plt.figure(figsize=(10,6))
    x = np.arange(len(top30))
    plt.bar(x-0.2, top30["precision"].values, width=0.4, label="precision")
    plt.bar(x+0.2, top30["recall"].values,    width=0.4, label="recall")
    plt.xticks(x, top30["code"].values, rotation=90)
    plt.legend()
    plt.title("Top-30 head codes: precision/recall")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "head_top30_pr.png")); plt.close()

    # save summary JSON
    summary = {
        "samples_n": int(len(df)),
        "labels_n": int(len(labels_vocab)),
        "pred_count_mean": float(np.mean(df["pred_count"])),
        "gold_count_mean": float(np.mean(df["gold_count"])),
        "exact_match_rate": float(np.mean(df["exact_match"])),
        "jaccard_mean": float(np.mean(df["jaccard"])),
        **metrics,
        "head_summary": head_sum,
        "torso_summary": torso_sum,
        "tail_summary": tail_sum,
        "parent3_micro_precision": float(parent_df["precision"].mean() if len(parent_df) else 0.0),
        "parent3_micro_recall": float(parent_df["recall"].mean() if len(parent_df) else 0.0),
        "parent3_micro_f1": float(parent_df["f1"].mean() if len(parent_df) else 0.0),
    }
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote analysis to: {out_dir}")

if __name__ == "__main__":
    main()

