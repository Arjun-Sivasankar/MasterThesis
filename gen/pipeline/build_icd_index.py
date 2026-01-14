# build_icd_index.py
import argparse, re, json, os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import faiss

def icd9_with_decimal(c):
    s = re.sub(r'[^0-9A-Za-z]', '', str(c))
    if not s: return s
    if s[0] in 'VEve':
        s = s.upper()
        if len(s) > 3 and '.' not in s: return s[:3] + '.' + s[3:]
        return s
    if len(s) > 3 and '.' not in s: return s[:3] + '.' + s[3:]
    return s

def norm_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def title_variants(title):
    v = set([title])
    v.add(title.replace(", unspecified", ""))
    v.add(title.replace(" due to ", " ").replace("  ", " "))
    v.add(title.replace(" el tor", " el-tor"))
    return [x for x in v if x and x != title]

@torch.no_grad()
def hf_encode(texts, model_name, batch=256, max_length=64, device=None, dtype=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    if dtype is None and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        model.to(device, dtype=torch.bfloat16)
    else:
        model.to(device)
    model.eval()

    all_embs = []
    for i in tqdm(range(0, len(texts), batch), desc=f"Encoding with {model_name}"):
        batch_text = texts[i:i+batch]
        enc = tok(batch_text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc)  # last_hidden_state
        last = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).expand_as(last).float()
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts  # [B, H]
        all_embs.append(mean_pooled.detach().cpu())

    X = torch.cat(all_embs, dim=0).numpy().astype("float32")
    # L2 normalize for cosine similarity with IndexFlatIP
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with icd_code, icd_version, long_title")
    ap.add_argument("--icd_version", type=int, default=9, choices=[9,10])
    ap.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                    help="HF Transformers model to embed titles (SapBERT works)")
    ap.add_argument("--out_dir", default="gen/pipeline/icd_index_v9")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["icd_version"] == args.icd_version].copy()

    if args.icd_version == 9:
        df["code"] = df["icd_code"].apply(icd9_with_decimal)
    else:
        df["code"] = df["icd_code"].astype(str).str.upper()

    df["title_norm"] = df["long_title"].apply(norm_text)
    df = df.dropna(subset=["code", "title_norm"])

    rows = []
    for _, r in df.iterrows():
        base = r["title_norm"]
        rows.append({"code": r["code"], "canonical": base, "text": base})
        for v in title_variants(base):
            rows.append({"code": r["code"], "canonical": base, "text": v})

    texts = [r["text"] for r in rows]
    print(f"Indexing {len(texts)} title/variant strings for ICD-{args.icd_version} using {args.model}...")

    E = hf_encode(texts, model_name=args.model, batch=args.batch, max_length=args.max_length)

    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)

    faiss.write_index(index, os.path.join(args.out_dir, "icd.faiss"))
    np.save(os.path.join(args.out_dir, "embeddings.npy"), E)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows, f)

    code2title = {r["code"]: r["title_norm"] for _, r in df.iterrows()}
    with open(os.path.join(args.out_dir, "code2title.json"), "w") as f:
        json.dump(code2title, f)

    meta = {
        "icd_version": args.icd_version,
        "model": args.model,
        "backend": "hf",
        "n_rows": len(rows),
        "n_codes": df["code"].nunique()
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done. Wrote FAISS index + metadata to:", args.out_dir)

if __name__ == "__main__":
    main()
