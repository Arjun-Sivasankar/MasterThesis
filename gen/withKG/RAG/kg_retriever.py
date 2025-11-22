#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_retriever.py
Tiny client to query the FAISS index of ICD-9 diagnosis profiles.
"""

import os, json, pickle, numpy as np, faiss
from sentence_transformers import SentenceTransformer

class KGRecommender:
    def __init__(self, index_dir="kg_recommender"):
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.codes = pickle.load(open(os.path.join(index_dir, "codes.pkl"), "rb"))
        meta = json.load(open(os.path.join(index_dir, "meta.json"), "r"))
        self.model = SentenceTransformer(meta["model"])

    def query(self, text: str, topk=200) -> list[str]:
        if not text:
            return []
        q = self.model.encode([text], show_progress_bar=False, normalize_embeddings=False).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, topk)
        # Filter out -1 (if any)
        return [self.codes[i] for i in I[0] if i >= 0]
