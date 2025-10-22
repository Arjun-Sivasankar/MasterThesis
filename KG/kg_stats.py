#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kg_stats.py — compute comprehensive stats from KG CSVs.

Inputs
  nodes CSV: kg_nodes.csv (columns: cui, name, sab, semantic_type)
  edges CSV: kg_edges.csv (columns include: cui_start, cui_target, rel, rela, sab_relation, sab_start, sab_target)

Outputs (written to --out-dir):
  - global_stats.json
  - degrees.csv
  - top_hubs.csv
  - degree_distribution.csv
  - nodes_sab_combo_counts.csv
  - nodes_sab_membership_counts.csv
  - nodes_sab_exclusive_counts.csv
  - sab_degree_aggregates.csv
  - semantic_type_counts.csv
  - edges_rel_counts.csv
  - edges_rela_counts.csv
  - edges_sab_relation_counts.csv
  - edges_sab_combo_pair_counts.csv
  - edges_sab_pair_counts.csv       (only if --expand-sab-pairs)
  - top_nodes_by_sab.csv
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------- Utilities ---------------------
def norm_sab_tokens(s: Optional[str]) -> List[str]:
    """Split SAB string on '|' and trim/normalize; drop empties."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    toks = [t.strip() for t in str(s).split('|')]
    return [t for t in toks if t]

def canonical_combo(tokens: Iterable[str]) -> str:
    """Sorted, de-duplicated, joined by '|'. Empty -> ''."""
    toks = sorted(set(norm_sab_tokens('|'.join(tokens) if isinstance(tokens, list) else str(tokens))))
    return '|'.join(toks) if toks else ''

def is_cui(x: str) -> bool:
    return isinstance(x, str) and re.fullmatch(r'C\d+', x) is not None

def safe_str(x) -> str:
    if pd.isna(x):
        return ''
    return str(x)

def quantiles(series: pd.Series, qs=(0.25, 0.5, 0.75, 0.9, 0.99)) -> Dict[str, float]:
    qv = series.quantile(qs)
    return {f"p{int(q*100)}": float(qv.loc[q]) for q in qs}


# --------------------- Core ---------------------
def load_nodes(nodes_path: str) -> pd.DataFrame:
    df = pd.read_csv(nodes_path, dtype=str)
    # Normalize
    for col in ('cui', 'name', 'sab', 'semantic_type'):
        if col in df.columns:
            df[col] = df[col].apply(safe_str)
    df['sab_tokens'] = df['sab'].apply(norm_sab_tokens)
    df['sab_combo'] = df['sab_tokens'].apply(lambda t: '|'.join(sorted(set(t))))
    df['semantic_tokens'] = df['semantic_type'].apply(lambda s: [t.strip() for t in str(s).split('|')] if s else [])
    return df

def stream_degree_and_edge_stats(edges_path: str,
                                 chunk_size: int = 500_000,
                                 expand_sab_pairs: bool = False) -> dict:
    """Stream edges CSV to compute degrees + edge distributions."""
    # Degrees
    outdeg = Counter()
    indeg = Counter()

    # Edge distributions
    rel_counts = Counter()
    rela_counts = Counter()
    sabrel_counts = Counter()
    sab_combo_pair_counts = Counter()
    sab_pair_counts = Counter()  # optional (expanded)

    usecols = ['cui_start','cui_target','rel','rela','sab_relation','sab_start','sab_target']
    for chunk in pd.read_csv(edges_path, usecols=usecols, dtype=str, chunksize=chunk_size):
        # Fill/clean
        for c in usecols:
            if c not in chunk.columns:
                raise ValueError(f"Missing required column in edges CSV: {c}")
            chunk[c] = chunk[c].apply(safe_str)

        # Degrees
        if len(chunk):
            outdeg.update(chunk['cui_start'].values)
            indeg.update(chunk['cui_target'].values)

        # Edge distributions
        rel_counts.update(chunk['rel'].values)
        rela_counts.update(chunk['rela'].values)
        sabrel_counts.update(chunk['sab_relation'].values)

        # SAB combos (exact combos)
        start_combo = chunk['sab_start'].apply(lambda s: canonical_combo(norm_sab_tokens(s)))
        target_combo = chunk['sab_target'].apply(lambda s: canonical_combo(norm_sab_tokens(s)))
        combo_df = pd.DataFrame({'start_combo': start_combo, 'target_combo': target_combo})
        sab_combo_pair_counts.update(combo_df.value_counts().to_dict())

        # Expanded SAB pairs (optional): for each edge, all (start SAB x target SAB) token pairs
        if expand_sab_pairs:
            for s, t in zip(chunk['sab_start'].values, chunk['sab_target'].values):
                s_toks = norm_sab_tokens(s)
                t_toks = norm_sab_tokens(t)
                if not s_toks or not t_toks:
                    continue
                for a in s_toks:
                    for b in t_toks:
                        sab_pair_counts[(a, b)] += 1

    return {
        'outdeg': outdeg,
        'indeg': indeg,
        'rel_counts': rel_counts,
        'rela_counts': rela_counts,
        'sabrel_counts': sabrel_counts,
        'sab_combo_pair_counts': sab_combo_pair_counts,
        'sab_pair_counts': sab_pair_counts
    }


def sab_membership_counts(nodes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (combo_counts, membership_counts, exclusive_counts)."""
    # Exact combos
    combo_counts = (nodes['sab_combo']
                    .value_counts(dropna=False)
                    .rename_axis('sab_combo')
                    .reset_index(name='count'))

    # Membership (node counted in each SAB it carries)
    mem_counter = Counter()
    for toks in nodes['sab_tokens']:
        for t in set(toks):
            mem_counter[t] += 1
    membership_counts = pd.DataFrame(
        [{'sab': k, 'count': v} for k, v in sorted(mem_counter.items())]
    ).sort_values('count', ascending=False)

    # Exclusive (nodes that have exactly one SAB, counted to that SAB)
    excl_counter = Counter()
    for toks in nodes['sab_tokens']:
        if len(set(toks)) == 1:
            excl_counter[list(set(toks))[0]] += 1
    exclusive_counts = pd.DataFrame(
        [{'sab': k, 'count_exclusive': v} for k, v in sorted(excl_counter.items())]
    ).sort_values('count_exclusive', ascending=False)

    return combo_counts, membership_counts, exclusive_counts


def semantic_type_counts(nodes: pd.DataFrame) -> pd.DataFrame:
    cnt = Counter()
    for toks in nodes['semantic_tokens']:
        for t in toks:
            if t:
                cnt[t] += 1
    return pd.DataFrame(
        [{'semantic_type': k, 'count': v} for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
    )


def per_sab_degree_aggregates(nodes: pd.DataFrame, degrees_df: pd.DataFrame) -> pd.DataFrame:
    """Compute degree aggregates per SAB (membership-based)."""
    merged = nodes[['cui','sab_tokens']].merge(degrees_df[['cui','in_degree','out_degree','degree']], on='cui', how='left')
    rows = []
    for _, r in merged.iterrows():
        toks = set(r['sab_tokens'])
        for t in toks:
            rows.append({'sab': t, 'in_degree': r['in_degree'], 'out_degree': r['out_degree'], 'degree': r['degree']})
    if not rows:
        return pd.DataFrame(columns=['sab','count','in_mean','out_mean','deg_mean','deg_median','deg_max','deg_min'])
    df = pd.DataFrame(rows)
    agg = df.groupby('sab').agg(
        count=('degree','count'),
        in_mean=('in_degree','mean'),
        out_mean=('out_degree','mean'),
        deg_mean=('degree','mean'),
        deg_median=('degree','median'),
        deg_max=('degree','max'),
        deg_min=('degree','min')
    ).reset_index()
    # tidy floats
    for c in ['in_mean','out_mean','deg_mean','deg_median']:
        agg[c] = agg[c].astype(float)
    return agg.sort_values('deg_mean', ascending=False)


def save_counter_as_df(counter: Counter, columns: Tuple[str, str], out_path: str):
    df = pd.DataFrame(counter.items(), columns=list(columns))
    df = df.sort_values(columns[1], ascending=False)
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Compute comprehensive KG statistics from CSVs.")
    ap.add_argument('--nodes', required=True, help='Path to kg_nodes.csv')
    ap.add_argument('--edges', required=True, help='Path to kg_edges.csv')
    ap.add_argument('--out-dir', required=True, help='Directory to write outputs')
    ap.add_argument('--chunk-size', type=int, default=500_000, help='Chunk size when streaming edges (default 500k)')
    ap.add_argument('--top-k', type=int, default=100, help='Top-K hubs to save (default 100)')
    ap.add_argument('--expand-sab-pairs', action='store_true',
                    help='Also compute expanded per-SAB edge pair counts (slower)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load nodes ----------
    nodes = load_nodes(args.nodes)
    num_nodes = len(nodes)
    sab_vocab_set = sorted(set(s for toks in nodes['sab_tokens'] for s in toks))

    # ---------- Stream edges for degrees and distributions ----------
    estats = stream_degree_and_edge_stats(
        edges_path=args.edges,
        chunk_size=args.chunk_size,
        expand_sab_pairs=args.expand_sab_pairs
    )

    # Degrees per node
    in_deg_series = pd.Series(estats['indeg'], dtype='int64')
    out_deg_series = pd.Series(estats['outdeg'], dtype='int64')

    degrees_df = pd.DataFrame({
        'cui': nodes['cui'],
        'in_degree': nodes['cui'].map(in_deg_series).fillna(0).astype('int64'),
        'out_degree': nodes['cui'].map(out_deg_series).fillna(0).astype('int64'),
    })
    degrees_df['degree'] = degrees_df['in_degree'] + degrees_df['out_degree']

    # Join back useful node attrs
    degrees_df = degrees_df.merge(nodes[['cui','name','sab','semantic_type']], on='cui', how='left')
    degrees_df.to_csv(os.path.join(args.out_dir, 'degrees.csv'), index=False)

    # Top hubs
    top_hubs = degrees_df.sort_values('degree', ascending=False).head(args.top_k)
    top_hubs.to_csv(os.path.join(args.out_dir, 'top_hubs.csv'), index=False)

    # Degree distribution
    deg = degrees_df['degree']
    deg_dist = pd.DataFrame({
        'degree': deg.value_counts().sort_index().index,
        'count': deg.value_counts().sort_index().values
    })
    deg_dist.to_csv(os.path.join(args.out_dir, 'degree_distribution.csv'), index=False)

    # ---------- SAB counts ----------
    combo_counts, membership_counts, exclusive_counts = sab_membership_counts(nodes)
    combo_counts.to_csv(os.path.join(args.out_dir, 'nodes_sab_combo_counts.csv'), index=False)
    membership_counts.to_csv(os.path.join(args.out_dir, 'nodes_sab_membership_counts.csv'), index=False)
    exclusive_counts.to_csv(os.path.join(args.out_dir, 'nodes_sab_exclusive_counts.csv'), index=False)

    # SAB degree aggregates
    sab_deg_agg = per_sab_degree_aggregates(nodes, degrees_df)
    sab_deg_agg.to_csv(os.path.join(args.out_dir, 'sab_degree_aggregates.csv'), index=False)

    # ---------- Semantic types ----------
    sty_counts = semantic_type_counts(nodes)
    sty_counts.to_csv(os.path.join(args.out_dir, 'semantic_type_counts.csv'), index=False)

    # ---------- Relationship & SAB-relation counts ----------
    save_counter_as_df(estats['rel_counts'], ('rel','count'),
                       os.path.join(args.out_dir, 'edges_rel_counts.csv'))
    save_counter_as_df(estats['rela_counts'], ('rela','count'),
                       os.path.join(args.out_dir, 'edges_rela_counts.csv'))
    save_counter_as_df(estats['sabrel_counts'], ('sab_relation','count'),
                       os.path.join(args.out_dir, 'edges_sab_relation_counts.csv'))

    # SAB combo pair counts (start combo -> target combo)
    combo_pairs_df = (pd.Series(estats['sab_combo_pair_counts'])
                      .rename('count').reset_index()
                      .rename(columns={'level_0':'sab_start_combo','level_1':'sab_target_combo'}))
    combo_pairs_df = combo_pairs_df.sort_values('count', ascending=False)
    combo_pairs_df.to_csv(os.path.join(args.out_dir, 'edges_sab_combo_pair_counts.csv'), index=False)

    # Expanded per-SAB pair counts (optional)
    if args.expand_sab_pairs:
        sab_pair_df = (pd.Series(estats['sab_pair_counts'])
                       .rename('count').reset_index()
                       .rename(columns={'level_0':'sab_start','level_1':'sab_target'}))
        sab_pair_df = sab_pair_df.sort_values('count', ascending=False)
        sab_pair_df.to_csv(os.path.join(args.out_dir, 'edges_sab_pair_counts.csv'), index=False)

    # ---------- Per-SAB Top nodes ----------
    # expand membership and rank per SAB
    rows = []
    for _, r in degrees_df[['cui','name','sab','degree']].iterrows():
        toks = norm_sab_tokens(r['sab'])
        for t in set(toks):
            rows.append({'sab': t, 'cui': r['cui'], 'name': r['name'], 'degree': r['degree']})
    per_sab = pd.DataFrame(rows)
    if not per_sab.empty:
        per_sab['rank'] = per_sab.groupby('sab')['degree'].rank(method='first', ascending=False)
        top_nodes_by_sab = per_sab[per_sab['rank'] <= args.top_k].sort_values(['sab','rank'])
        top_nodes_by_sab.to_csv(os.path.join(args.out_dir, 'top_nodes_by_sab.csv'), index=False)
    else:
        top_nodes_by_sab = pd.DataFrame(columns=['sab','cui','name','degree','rank'])

    # ---------- Global JSON ----------
    global_stats = {
        'num_nodes': int(num_nodes),
        'num_edges_out_degree_sum': int(degrees_df['out_degree'].sum()),
        'num_edges_in_degree_sum': int(degrees_df['in_degree'].sum()),
        # Degree summary
        'degree': {
            'min': int(deg.min()),
            'max': int(deg.max()),
            'mean': float(deg.mean()),
            'median': float(deg.median()),
            **quantiles(deg, qs=(0.9,0.95,0.99))
        },
        # SAB vocabularies discovered
        'sab_vocabs': sab_vocab_set,
        # Node SAB counts (membership and combos)
        'nodes_sab_membership_total': int(membership_counts['count'].sum()),  # counts with overlap
        'nodes_sab_combo_distinct': int(len(combo_counts)),
        # Top REL / RELA / SAB-relation
        'top_rel': estats['rel_counts'].most_common(10),
        'top_rela': estats['rela_counts'].most_common(10),
        'top_sab_relation': estats['sabrel_counts'].most_common(10),
        # Files written
        'outputs': sorted(os.listdir(args.out_dir))
    }
    with open(os.path.join(args.out_dir, 'global_stats.json'), 'w') as f:
        json.dump(global_stats, f, indent=2)

    print(f"[OK] Wrote stats to: {args.out_dir}")


if __name__ == "__main__":
    main()