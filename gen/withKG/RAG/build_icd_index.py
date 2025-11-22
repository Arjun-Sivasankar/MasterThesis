#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_icd_index.py
Embed ICD-9 diagnosis profiles and build a FAISS ANN index.

Outputs in --out_dir:
  - faiss.index
  - codes.pkl
  - meta.json (records embedding model)
  - emb.npy (optional, for diagnostics)
"""

import os, json, pickle, argparse, numpy as np, faiss
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles_json", default="kg_recommender/icd9_profiles.json")
    ap.add_argument("--out_dir", default="kg_recommender")
    ap.add_argument("--model", default="gen/withKG/RAG/biobert-mnli-snli-scinli-scitail-mednli-stsb")
    ap.add_argument("--nlist", type=int, default=2048, help="IVF cluster count")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prof = json.load(open(args.profiles_json))
    codes, texts = zip(*sorted(prof.items()))
    enc = SentenceTransformer(args.model)
    X = enc.encode(list(texts), show_progress_bar=True, normalize_embeddings=False).astype("float32")
    faiss.normalize_L2(X)

    dim = X.shape[1]
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quant, dim, args.nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(X)
    index.add(X)

    faiss.write_index(index, os.path.join(args.out_dir, "faiss.index"))
    with open(os.path.join(args.out_dir, "codes.pkl"), "wb") as f:
        pickle.dump(list(codes), f)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"model": args.model}, f, indent=2)

    np.save(os.path.join(args.out_dir, "emb.npy"), X)
    print(f"Indexed {len(codes):,} codes → {args.out_dir}")

if __name__ == "__main__":
    main()
