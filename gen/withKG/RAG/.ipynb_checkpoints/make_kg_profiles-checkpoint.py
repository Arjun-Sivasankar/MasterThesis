#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_kg_profiles.py
Build compact **ICD-9 diagnosis** item profiles enriched with small KG neighborhoods.

Profile format (example):
410.71 :: Acute subendocardial myocardial infarction | Non-Q wave MI
; [H1] isa=Acute myocardial infarction
; [H1] associated_with=Troponin level abnormal
; [H1] finding_site_of=Heart structure
; [H2] due_to=Atherosclerotic heart disease

Outputs: JSON {code -> profile_text}
"""

from __future__ import annotations
import argparse, json
from typing import Dict, Set, Iterable, Optional
import pandas as pd
import networkx as nx

from kg_utils import load_nodes, load_edges, build_graph, icd9_dx_title_and_cuis

DEFAULT_RELA_KEEP = {
    "isa","inverse_isa","same_as","mapped_to",
    "associated_with","has_associated_finding",
    "finding_site_of","associated_morphology_of",
    "due_to","causative_agent_of"
}

def neighbor_snippets(G: nx.DiGraph,
                      seed_cuis: Set[str],
                      hops: int = 1,
                      rela_keep: Optional[Set[str]] = None,
                      max_per_hop: int = 20) -> str:
    out = []
    frontier = set(seed_cuis); seen = set(frontier)
    for h in range(1, hops+1):
        nxt = set(); cnt = 0
        for u in list(frontier):
            if u not in G: continue
            for v in G.successors(u):
                d = G[u][v]
                rela = (d.get("rela") or "").strip()
                if rela_keep and rela and rela not in rela_keep: 
                    continue
                if v in seen: continue
                sub = (G.nodes[u].get("name") or "").strip()
                obj = (G.nodes[v].get("name") or "").strip()
                if sub and obj and cnt < max_per_hop:
                    out.append(f"[H{h}] {rela or 'related_to'}={obj}")
                    cnt += 1
                nxt.add(v)
        frontier = nxt - seen
        seen |= nxt
        if not frontier: break
    # compact single string
    return " ; ".join(out[: (max_per_hop * max(1, hops)) ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_nodes_csv", required=True)
    ap.add_argument("--kg_edges_csv", required=True)
    ap.add_argument("--out_json", default="kg_recommender/icd9_profiles.json")
    ap.add_argument("--hops", type=int, default=1, choices=[0,1,2])
    ap.add_argument("--max_per_hop", type=int, default=20)
    ap.add_argument("--rela_keep", default=",".join(sorted(DEFAULT_RELA_KEEP)))
    args = ap.parse_args()

    nodes = load_nodes(args.kg_nodes_csv)
    edges = load_edges(args.kg_edges_csv)
    G = build_graph(nodes, edges)

    code_to_title, code_to_cuis = icd9_dx_title_and_cuis(nodes)
    rela_keep = {r.strip() for r in args.rela_keep.split(",") if r.strip()}

    profiles: Dict[str,str] = {}
    for code, cuiset in code_to_cuis.items():
        title = code_to_title.get(code, "")
        neigh = neighbor_snippets(G, cuiset, hops=args.hops, rela_keep=rela_keep, max_per_hop=args.max_per_hop)
        text = f"{code} :: {title}"
        if neigh:
            text += " ; " + neigh
        profiles[code] = text

    # write
    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(profiles):,} ICD-9 diagnosis profiles → {args.out_json}")

if __name__ == "__main__":
    main()
