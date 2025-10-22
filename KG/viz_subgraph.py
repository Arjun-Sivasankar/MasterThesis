#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
viz_subgraph.py — visualize a local CUI-centered subgraph from your KG CSVs.

Default source files:
  nodes: kg_nodes.csv  (columns: cui, name, sab, semantic_type)
  edges: kg_edges.csv  (columns: cui_start, name_start, sab_start, rel, rela, sab_relation, cui_target, name_target, sab_target)

Examples:
  python viz_subgraph.py --seed C0011849 --radius 1 --direction both \
      --nodes /path/to/kg_nodes.csv --edges /path/to/kg_edges.csv \
      --out subgraph_C0011849_r1.png

  python viz_subgraph.py --seed C0011849 --radius 2 --direction out \
      --label-type name --rel-allow CHD,PAR \
      --nodes /path/to/kg_nodes.csv --edges /path/to/kg_edges.csv \
      --out subgraph_C0011849_r2_out.png
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Optional

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# -------- Helpers --------
def is_cui(x: str) -> bool:
    return isinstance(x, str) and re.fullmatch(r'C\d+', x) is not None

def parse_list_opt(s: Optional[str]):
    if not s:
        return None
    return [x.strip() for x in s.split(',') if x.strip()]

def to_str(x) -> str:
    """Robust coercion: NaN/None -> '', else str(x)."""
    if pd.isna(x):
        return ''
    return str(x)

def color_for_sab_list(sab_str: str) -> str:
    """
    Color by node SAB membership (union-of-sources).
    Priority order:
      ICD9CM -> red
      ATC    -> blue   (meds/drugs)
      LNC    -> green  (labs)
      SNOMEDCT_US -> purple
      else -> gray
    """
    if not isinstance(sab_str, str):
        return 'gray'
    sabs = set(sab_str.split('|'))
    if 'ICD9CM' in sabs:       return 'red'
    if 'ATC' in sabs:          return 'blue'
    if 'LNC' in sabs:          return 'green'
    if 'SNOMEDCT_US' in sabs:  return 'purple'
    return 'gray'


# -------- BFS neighborhood over CSV (no full graph load) --------
def bfs_khop_from_csv(edges_csv: str,
                      seed: str,
                      radius: int = 1,
                      direction: str = 'both',
                      rel_allow=None,
                      rela_allow=None,
                      chunk_size: int = 500_000,
                      max_nodes: Optional[int] = None):
    """
    Returns (nodes_set, edges_records) where:
      nodes_set: set of CUIs
      edges_records: list of dicts for edges to draw (u, v, rel, rela, sab_relation)
    """
    assert direction in ('out', 'in', 'both')
    if not is_cui(seed):
        raise ValueError(f"Seed must be a valid CUI, got: {seed}")

    visited = set([seed])
    frontier = set([seed])
    collected_edges = []

    usecols = ['cui_start', 'cui_target', 'rel', 'rela', 'sab_relation']

    for _depth in range(1, radius + 1):
        next_frontier = set()

        for chunk in pd.read_csv(edges_csv, usecols=usecols, dtype=str, chunksize=chunk_size):
            # Ensure all required columns exist
            for col in usecols:
                if col not in chunk.columns:
                    raise ValueError(f"Edges CSV missing required column: {col}")

            # Clean NaNs on ALL label columns
            chunk['cui_start'] = chunk['cui_start'].astype(str)
            chunk['cui_target'] = chunk['cui_target'].astype(str)
            for col in ['rel', 'rela', 'sab_relation']:
                chunk[col] = chunk[col].apply(to_str)

            # Optional filters
            if rel_allow is not None:
                chunk = chunk[chunk['rel'].isin(rel_allow)]
            if rela_allow is not None:
                chunk = chunk[chunk['rela'].isin(rela_allow)]

            if direction in ('out', 'both'):
                out_rows = chunk[chunk['cui_start'].isin(frontier)]
                for _, r in out_rows.iterrows():
                    u, v = to_str(r['cui_start']).strip(), to_str(r['cui_target']).strip()
                    if not is_cui(u) or not is_cui(v):
                        continue
                    collected_edges.append({
                        'u': u, 'v': v,
                        'rel': to_str(r['rel']).strip(),
                        'rela': to_str(r['rela']).strip(),
                        'sab_relation': to_str(r['sab_relation']).strip()
                    })
                    if v not in visited:
                        next_frontier.add(v)

            if direction in ('in', 'both'):
                in_rows = chunk[chunk['cui_target'].isin(frontier)]
                for _, r in in_rows.iterrows():
                    u, v = to_str(r['cui_start']).strip(), to_str(r['cui_target']).strip()
                    if not is_cui(u) or not is_cui(v):
                        continue
                    collected_edges.append({
                        'u': u, 'v': v,
                        'rel': to_str(r['rel']).strip(),
                        'rela': to_str(r['rela']).strip(),
                        'sab_relation': to_str(r['sab_relation']).strip()
                    })
                    if u not in visited:
                        next_frontier.add(u)

        # Update
        for n in next_frontier:
            visited.add(n)
            if max_nodes is not None and len(visited) >= max_nodes:
                break
        frontier = next_frontier
        if not frontier or (max_nodes is not None and len(visited) >= max_nodes):
            break

    return visited, collected_edges


# -------- Build a NetworkX DiGraph for the subgraph --------
def assemble_subgraph(nodes_csv: str, node_ids: set[str], edges_records: list[dict]):
    nodes_df = pd.read_csv(nodes_csv, dtype=str)
    # Coerce/clean
    nodes_df['cui'] = nodes_df['cui'].apply(to_str).str.strip()
    nodes_df['name'] = nodes_df['name'].apply(to_str)
    if 'sab' in nodes_df.columns:
        nodes_df['sab'] = nodes_df['sab'].apply(to_str)
    else:
        nodes_df['sab'] = ''

    nodes_map = nodes_df.set_index('cui').to_dict(orient='index')

    H = nx.DiGraph()
    for cui in node_ids:
        meta = nodes_map.get(cui, {})
        H.add_node(cui,
                   name=to_str(meta.get('name', 'Unknown')),
                   sab=to_str(meta.get('sab', '')))

    for e in edges_records:
        u, v = e['u'], e['v']
        if u in H and v in H:
            H.add_edge(u, v,
                       rel=to_str(e.get('rel','')),
                       rela=to_str(e.get('rela','')),
                       sab_relation=to_str(e.get('sab_relation','')))

    return H


# -------- Draw function --------
def draw_subgraph(H: nx.DiGraph,
                  out_png: str,
                  label_type: str = 'cui',
                  edge_label: str = 'auto',
                  layout: str = 'spring',
                  k: float = 0.4,
                  dpi: int = 300):
    """
    label_type: cui | name | both
    edge_label: auto | rel | rela | none
    layout: spring | kamada | random
    """
    assert label_type in ('cui', 'name', 'both')
    assert edge_label in ('auto', 'rel', 'rela', 'none')
    assert layout in ('spring', 'kamada', 'random')

    if layout == 'spring':
        pos = nx.spring_layout(H, seed=42, k=k)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(H)
    else:
        pos = nx.random_layout(H, seed=42)

    # Node colors from SAB
    node_colors = [color_for_sab_list(to_str(H.nodes[n].get('sab', ''))) for n in H.nodes()]
    plt.figure(figsize=(12, 12), dpi=dpi)

    nx.draw_networkx_nodes(H, pos, node_size=260, alpha=0.9, node_color=node_colors)
    nx.draw_networkx_edges(H, pos, alpha=0.5, arrows=True, arrowstyle='-|>', arrowsize=12)

    # Node labels
    if label_type == 'cui':
        labels = {n: n for n in H.nodes()}
    elif label_type == 'name':
        labels = {n: to_str(H.nodes[n].get('name', n)) for n in H.nodes()}
    else:  # both
        labels = {n: f"{n}\n{to_str(H.nodes[n].get('name',''))}" for n in H.nodes()}

    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight='bold')

    # Edge labels
    if edge_label != 'none':
        e_lbls = {}
        for u, v, data in H.edges(data=True):
            rel = to_str(data.get('rel'))
            rela = to_str(data.get('rela'))
            if edge_label == 'rel':
                lbl = rel
            elif edge_label == 'rela':
                lbl = rela
            else:  # auto
                lbl = rela if rela else rel
            e_lbls[(u, v)] = lbl
        if any(e_lbls.values()):
            nx.draw_networkx_edge_labels(
                H, pos, edge_labels=e_lbls, font_size=7, label_pos=0.5, rotate=False,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7)
            )

    # Legend for colors
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',    markersize=10, label='ICD9CM'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',   markersize=10, label='ATC (Meds)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='green',  markersize=10, label='LNC (Labs)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='SNOMEDCT_US'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',   markersize=10, label='Other'),
    ]
    plt.legend(handles=legend, loc='upper right')
    plt.axis('off')
    plt.title('CUI-centered Subgraph')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Visualize a subgraph around a CUI from KG CSVs.")
    ap.add_argument('--seed', required=True, help='Seed CUI (e.g., C0011849)')
    ap.add_argument('--nodes', default='kg_nodes.csv', help='Path to kg_nodes.csv')
    ap.add_argument('--edges', default='kg_edges.csv', help='Path to kg_edges.csv')
    ap.add_argument('--radius', type=int, default=1, help='k-hop radius (default: 1)')
    ap.add_argument('--direction', choices=['out','in','both'], default='both', help='Edge direction for BFS (default: both)')
    ap.add_argument('--rel-allow', default='', help='Comma-separated list of REL codes to include (optional)')
    ap.add_argument('--rela-allow', default='', help='Comma-separated list of RELA strings to include (optional)')
    ap.add_argument('--max-nodes', type=int, default=None, help='Stop expanding if node count reaches this number (optional)')
    ap.add_argument('--chunk-size', type=int, default=500_000, help='CSV chunk size for scanning edges (default: 500k)')
    ap.add_argument('--label-type', choices=['cui','name','both'], default='cui', help='Node label style (default: cui)')
    ap.add_argument('--edge-label', choices=['auto','rel','rela','none'], default='auto', help='Edge label style (default: auto)')
    ap.add_argument('--layout', choices=['spring','kamada','random'], default='spring', help='Graph layout (default: spring)')
    ap.add_argument('--k', type=float, default=0.4, help='spring_layout k (node spacing), only for --layout spring')
    ap.add_argument('--dpi', type=int, default=300, help='Output DPI (default: 300)')
    ap.add_argument('--out', default=None, help='Output PNG path (default: subgraph_<seed>_r<radius>.png)')
    args = ap.parse_args()

    seed = to_str(args.seed).strip().upper()
    if not is_cui(seed):
        raise SystemExit(f"--seed must be a valid CUI like C0011849, got: {args.seed}")

    rel_allow = parse_list_opt(args.rel_allow)
    rela_allow = parse_list_opt(args.rela_allow)

    out_png = args.out or f"subgraph_{seed}_r{args.radius}.png"

    # 1) Discover nodes & edges by scanning edges CSV (memory-friendly)
    node_ids, edges_records = bfs_khop_from_csv(
        edges_csv=args.edges,
        seed=seed,
        radius=args.radius,
        direction=args.direction,
        rel_allow=rel_allow,
        rela_allow=rela_allow,
        chunk_size=args.chunk_size,
        max_nodes=args.max_nodes
    )

    if seed not in node_ids:
        node_ids.add(seed)

    # 2) Assemble a tiny DiGraph with node attributes (from nodes.csv)
    H = assemble_subgraph(args.nodes, node_ids, edges_records)

    if seed not in H:
        raise SystemExit(f"Seed {seed} not found in nodes CSV. Check --nodes path or the CUI value.")

    # 3) Draw + save
    draw_subgraph(
        H, out_png,
        label_type=args.label_type,
        edge_label=args.edge_label,
        layout=args.layout,
        k=args.k,
        dpi=args.dpi
    )

    print(f"[OK] Saved subgraph around {seed} to: {out_png}")


if __name__ == "__main__":
    main()