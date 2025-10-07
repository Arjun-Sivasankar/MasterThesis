# pipelines/test_codegen.py
import os, json, time, argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .util_codegen_core import (
    set_seed, is_main_process, rank0_print, build_input_text,
    format_icd9_properly, is_valid_icd9, normalize_code, get_icd9_parent,
    codes_to_multihot, eval_sets, hierarchical_eval, show_test_predictions
)

def _read_first_col_codes(path):
    if not path: return []
    try:
        df = pd.read_csv(path)
        col = df.columns[0]
        vals = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        return sorted({v for v in vals if is_valid_icd9(v)})
    except Exception:
        return []

def _read_first_col_parents(path):
    if not path: return []
    try:
        df = pd.read_csv(path)
        col = df.columns[0]
        raw = [format_icd9_properly(str(x)) for x in df[col].tolist()]
        return sorted({get_icd9_parent(v) for v in raw if v})
    except Exception:
        return []

@torch.no_grad()
def generate_codes(model, tok, prompts, labels_vocab, max_new=96, batch_size=16, max_len=3072):
    model.eval()
    device = next(model.parameters()).device
    allowed = set(labels_vocab)
    preds = []
    import re
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)):
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, num_beams=1,
                no_repeat_ngram_size=2, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True
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
                    seen.add(c); keep.append(c)
            preds.append(keep)
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

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    test_df = pd.read_pickle(args.test_pickle)
    test_df["input_text"] = test_df.apply(lambda r: build_input_text(r, args.use_structured==1, args.use_notes==1, args.subject_col), axis=1)

    labels = json.load(open(args.labels_json))["labels"]
    labels_vocab = [format_icd9_properly(c) for c in labels]

    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    tok_src = os.path.join(args.adapter_dir, "tokenizer") if os.path.exists(os.path.join(args.adapter_dir,"tokenizer")) else args.llama_model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.llama_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    prompts = test_df["input_text"].astype(str).tolist()

    gold_lists = []
    for codes in test_df[args.label_col]:
        cur=[]
        for c in codes:
            z = format_icd9_properly(str(c))
            if is_valid_icd9(z): cur.append(z)
        gold_lists.append(cur)

    t0 = time.perf_counter()
    preds = generate_codes(model, tok, prompts, labels_vocab, max_new=args.gen_max_new, batch_size=args.test_batch_size, max_len=args.max_len)
    gen_secs = time.perf_counter() - t0

    Y_true = codes_to_multihot(gold_lists, labels_vocab)
    Y_pred = codes_to_multihot(preds,       labels_vocab)
    metrics = eval_sets(Y_true, Y_pred)
    metrics.update(hierarchical_eval(Y_true, Y_pred, labels_vocab))
    metrics["test_samples"] = len(test_df)
    metrics["test_batch_size"] = args.test_batch_size
    metrics["test_generate_seconds"] = gen_secs
    json.dump(metrics, open(os.path.join(args.out_dir, "test_metrics.json"), "w"), indent=2)

    # buckets
    top_codes   = _read_first_col_codes(args.top_codes_csv)
    bottom_codes= _read_first_col_codes(args.bottom_codes_csv)
    top_parents = _read_first_col_parents(args.top_parent_csv)
    results_ext = {}

    def restrict_and_eval(bucket_codes):
        idx = {c:i for i,c in enumerate(labels_vocab)}
        keep = [idx[c] for c in bucket_codes if c in idx]
        if not keep: return None
        yt = Y_true[:, keep]; yp = Y_pred[:, keep]
        return eval_sets(yt, yp)

    if top_codes:
        r = restrict_and_eval(top_codes)
        if r: results_ext["TOP_50_CODES"] = r
    if bottom_codes:
        r = restrict_and_eval(bottom_codes)
        if r: results_ext["BOTTOM_50_CODES"] = r
    if top_parents:
        def to_parents(lists):
            return [[get_icd9_parent(c) for c in row] for row in lists]
        parents = sorted(set(top_parents))
        def mh(L, labels):
            idx = {c:i for i,c in enumerate(labels)}
            Y = np.zeros((len(L), len(labels)), dtype=np.int32)
            for i, lst in enumerate(L):
                for c in lst:
                    j = idx.get(c)
                    if j is not None: Y[i, j]=1
            return Y
        YgP = mh(to_parents(gold_lists), parents)
        YpP = mh(to_parents(preds),      parents)
        results_ext["TOP_50_PARENTS"] = eval_sets(YgP, YpP)

    json.dump(results_ext, open(os.path.join(args.out_dir, "test_metrics_buckets.json"), "w"), indent=2)

    # prints
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

if __name__ == "__main__":
    raise SystemExit(main())
