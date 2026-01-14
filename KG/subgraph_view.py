#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    python subgraph_view.py --seed C0152602 --radius 2 \
        --pkl medical_knowledge_graph2.pkl --out subgraph.png
"""

import argparse
import pickle
import re
from typing import Set, List, Dict, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import random

# Set seaborn style for professional aesthetics
sns.set_theme(style="whitegrid", context="paper", palette="muted")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

def is_cui(x: str) -> bool:
    return isinstance(x, str) and re.fullmatch(r'C\d+', x) is not None

def to_str(x) -> str:
    if x is None:
        return ''
    try:
        import pandas as pd
        if pd.isna(x):
            return ''
    except ImportError:
        pass
    return str(x)

VOCAB_COLORS = {
    'ICD9CM': '#E74C3C',
    'ATC': '#3498DB',
    'LNC': '#2ECC71',
    'SNOMEDCT_US': '#9B59B6',
    'OTHER': '#95A5A6'
}

def get_node_color(sab_str: str) -> str:
    if not isinstance(sab_str, str):
        return VOCAB_COLORS['OTHER']
    sabs = set(sab_str.split('|'))
    if 'ICD9CM' in sabs:
        return VOCAB_COLORS['ICD9CM']
    if 'ATC' in sabs:
        return VOCAB_COLORS['ATC']
    if 'LNC' in sabs:
        return VOCAB_COLORS['LNC']
    if 'SNOMEDCT_US' in sabs:
        return VOCAB_COLORS['SNOMEDCT_US']
    return VOCAB_COLORS['OTHER']

def get_edge_color(rela: str) -> str:
    rela = to_str(rela).upper()
    if any(x in rela for x in ['ISA', 'IS_A', 'CHILD', 'CHD']):
        return '#34495E'
    if any(x in rela for x in ['PARENT', 'PAR', 'INVERSE_ISA']):
        return '#7F8C8D'
    if any(x in rela for x in ['ASSOCIATED_WITH', 'OCCURS_IN', 'MAY_BE']):
        return '#16A085'
    if any(x in rela for x in ['TREATS', 'PREVENTED_BY', 'CAUSES']):
        return '#E67E22'
    if any(x in rela for x in ['PART_OF', 'HAS_PART', 'CONSISTS_OF']):
        return '#8E44AD'
    return '#BDC3C7'

def extract_khop_subgraph(
    G: nx.DiGraph,
    seed: str,
    radius: int = 1,
    direction: str = 'both',
    max_nodes: Optional[int] = None,
    rela_allow: Optional[List[str]] = None,
    max_edges: Optional[int] = None,
    edge_sample_method: str = "random"
) -> nx.DiGraph:
    """
    Extract k-hop subgraph from the full graph centered on seed CUI.
    Optionally restrict the number of edges (paths) for viewing.
    """
    if seed not in G:
        raise ValueError(f"Seed CUI {seed} not found in graph")
    visited = {seed}
    frontier = {seed}
    edges_to_include = []
    for hop in range(radius):
        next_frontier = set()
        for node in frontier:
            neighbors = []
            if direction in ('out', 'both'):
                neighbors += [(node, succ) for succ in G.successors(node)]
            if direction in ('in', 'both'):
                neighbors += [(pred, node) for pred in G.predecessors(node)]
            # Filter by RELA if specified
            if rela_allow:
                neighbors = [
                    (u, v) for (u, v) in neighbors
                    if any(allowed.upper() in to_str(G[u][v].get('rela', '')).upper() for allowed in rela_allow)
                ]
            # Restrict number of edges if max_edges is set
            if max_edges is not None and len(neighbors) > max_edges:
                if edge_sample_method == "random":
                    neighbors = random.sample(neighbors, max_edges)
                elif edge_sample_method == "first":
                    neighbors = neighbors[:max_edges]
            for u, v in neighbors:
                next_frontier.add(v if u == node else u)
                edges_to_include.append((u, v))
        next_frontier = next_frontier - visited
        visited.update(next_frontier)
        frontier = next_frontier
        if max_nodes and len(visited) >= max_nodes:
            visited = set(list(visited)[:max_nodes])
            break
        if not frontier:
            break
    H = G.subgraph(visited).copy()
    if rela_allow or max_edges is not None:
        all_edges = set(H.edges())
        edges_to_remove = all_edges - set(edges_to_include)
        H.remove_edges_from(edges_to_remove)
    return H

def compute_layout(H: nx.DiGraph, layout: str = 'spring', 
                  seed_node: Optional[str] = None, k: float = 0.5) -> Dict:
    if layout == 'spring':
        pos = nx.spring_layout(H, k=k, iterations=50, seed=42)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(H)
    elif layout == 'circular':
        pos = nx.circular_layout(H)
    elif layout == 'hierarchical':
        try:
            pos = nx.nx_agraph.graphviz_layout(H, prog='dot')
        except:
            print("[WARN] Graphviz not available, falling back to spring layout")
            pos = nx.spring_layout(H, k=k, seed=42)
    else:
        pos = nx.spring_layout(H, k=k, seed=42)
    if seed_node and seed_node in pos:
        seed_pos = np.array(pos[seed_node])
        center = np.array([0.5, 0.5])
        offset = center - seed_pos
        pos = {node: tuple(np.array(p) + offset) for node, p in pos.items()}
    return pos

def draw_professional_subgraph(H: nx.DiGraph, seed: str, out_path: str,
                              label_type: str = 'cui',
                              edge_label: str = 'rela',
                              layout: str = 'spring',
                              k: float = 0.5,
                              dpi: int = 300,
                              figsize: Tuple[int, int] = (14, 10)):
    pos = compute_layout(H, layout=layout, seed_node=seed, k=k)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')
    node_colors = [get_node_color(H.nodes[n].get('sab', '')) for n in H.nodes()]
    node_sizes = [800 if n == seed else 600 for n in H.nodes()]
    node_alpha = [1.0 if n == seed else 0.85 for n in H.nodes()]
    edge_colors = [get_edge_color(H[u][v].get('rela', '')) for u, v in H.edges()]
    nx.draw_networkx_edges(
        H, pos,
        edge_color=edge_colors,
        width=1.5,
        alpha=0.6,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    for i, node in enumerate(H.nodes()):
        x, y = pos[node]
        circle = plt.Circle(
            (x, y),
            radius=0.02 if node == seed else 0.015,
            color=node_colors[i],
            alpha=node_alpha[i],
            zorder=3,
            edgecolor='white',
            linewidth=2 if node == seed else 1
        )
        ax.add_patch(circle)
    # Node labels: only CUI for all except seed, which gets CUI + name
    labels = {}
    for node in H.nodes():
        name = to_str(H.nodes[node].get('name', ''))
        if node == seed:
            short_name = name[:40] + '...' if len(name) > 40 else name
            labels[node] = f"{node}\n{short_name}"
        else:
            labels[node] = node
    for node, label in labels.items():
        x, y = pos[node]
        bbox_props = dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='gray',
            alpha=0.9 if node == seed else 0.8,
            linewidth=1.5 if node == seed else 0.8
        )
        ax.text(
            x, y,
            label,
            fontsize=10 if node == seed else 8,
            fontweight='bold' if node == seed else 'normal',
            ha='center',
            va='center',
            bbox=bbox_props,
            zorder=4
        )
    if edge_label != 'none':
        edge_labels = {}
        for u, v in H.edges():
            data = H[u][v]
            if edge_label == 'rela':
                lbl = to_str(data.get('rela', ''))
            elif edge_label == 'rel':
                lbl = to_str(data.get('rel', ''))
            else:
                lbl = to_str(data.get('rela', ''))
            if lbl:
                edge_labels[(u, v)] = lbl
        if edge_labels:
            nx.draw_networkx_edge_labels(
                H, pos,
                edge_labels=edge_labels,
                font_size=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='none', alpha=0.7),
                ax=ax
            )
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=VOCAB_COLORS['ICD9CM'], markersize=10,
               label='ICD-9-CM (Diagnoses)', markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VOCAB_COLORS['ATC'], markersize=10,
               label='ATC (Medications)', markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VOCAB_COLORS['LNC'], markersize=10,
               label='LOINC (Lab Tests)', markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VOCAB_COLORS['SNOMEDCT_US'], markersize=10,
               label='SNOMED CT', markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VOCAB_COLORS['OTHER'], markersize=10,
               label='Other Sources', markeredgewidth=0),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
        title='Vocabulary Sources',
        title_fontsize=10
    )
    legend.get_frame().set_alpha(0.95)
    seed_name = to_str(H.nodes[seed].get('name', 'Unknown'))[:50]
    title = f'Knowledge Subgraph: {seed} - {seed_name}\n'
    title += f'Nodes: {H.number_of_nodes()} | Edges: {H.number_of_edges()}'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved professional subgraph to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Professional KG subgraph visualization for thesis/publications"
    )
    parser.add_argument('--pkl', required=True,
                       help='Path to medical_knowledge_graph2.pkl')
    parser.add_argument('--seed', required=True,
                       help='Seed CUI (e.g., C0152602)')
    parser.add_argument('--radius', type=int, default=1,
                       help='k-hop radius (default: 1)')
    parser.add_argument('--direction', choices=['out', 'in', 'both'],
                       default='both', help='Edge direction (default: both)')
    parser.add_argument('--max-nodes', type=int, default=None,
                       help='Maximum nodes to include (optional)')
    parser.add_argument('--rela-allow', type=str, default=None,
                       help='Comma-separated list of allowed RELA values (e.g., "isa,part_of")')
    parser.add_argument('--label-type', choices=['cui', 'name', 'both'],
                       default='cui', help='Node label type (default: cui)')
    parser.add_argument('--edge-label', choices=['rela', 'rel', 'none'],
                       default='rela', help='Edge label type (default: rela)')
    parser.add_argument('--layout', 
                       choices=['spring', 'kamada', 'circular', 'hierarchical'],
                       default='spring', help='Graph layout (default: spring)')
    parser.add_argument('--k', type=float, default=0.5,
                       help='Spring layout spacing (default: 0.5)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output DPI (default: 300)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10],
                       help='Figure size in inches (default: 14 10)')
    parser.add_argument('--out', required=True,
                       help='Output PNG path')
    parser.add_argument('--max-edges', type=int, default=None,
                       help='Maximum number of edges (paths) to show per hop (for aesthetics)')
    parser.add_argument('--edge-sample-method', type=str, choices=['random', 'first'], default='random',
                       help='How to sample edges if more than max-edges (random or first)')
    args = parser.parse_args()
    seed = args.seed.strip().upper()
    if not is_cui(seed):
        raise ValueError(f"Invalid CUI format: {args.seed}")
    rela_allow = None
    if args.rela_allow:
        rela_allow = [x.strip() for x in args.rela_allow.split(',') if x.strip()]
        print(f"[INFO] Filtering by RELA: {rela_allow}")
    print(f"[INFO] Loading knowledge graph from: {args.pkl}")
    with open(args.pkl, 'rb') as f:
        G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        raise TypeError(f"Expected NetworkX DiGraph, got {type(G)}")
    print(f"[INFO] Full graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    print(f"[INFO] Extracting {args.radius}-hop subgraph around {seed}...")
    H = extract_khop_subgraph(
        G, seed,
        radius=args.radius,
        direction=args.direction,
        max_nodes=args.max_nodes,
        rela_allow=rela_allow,
        max_edges=args.max_edges,
        edge_sample_method=args.edge_sample_method
    )
    print(f"[INFO] Subgraph: {H.number_of_nodes():,} nodes, "
          f"{H.number_of_edges():,} edges")
    print(f"[INFO] Creating professional visualization...")
    draw_professional_subgraph(
        H, seed, args.out,
        label_type=args.label_type,
        edge_label=args.edge_label,
        layout=args.layout,
        k=args.k,
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )
    print(f"\n[STATS] Subgraph Statistics:")
    print(f"  - Nodes: {H.number_of_nodes():,}")
    print(f"  - Edges: {H.number_of_edges():,}")
    if H.number_of_nodes() > 1:
        print(f"  - Density: {nx.density(H):.4f}")
    sab_counts = defaultdict(int)
    for node in H.nodes():
        sab = to_str(H.nodes[node].get('sab', ''))
        for s in sab.split('|'):
            if s:
                sab_counts[s] += 1
    if sab_counts:
        print(f"  - Vocabulary distribution:")
        for vocab, count in sorted(sab_counts.items(), key=lambda x: -x[1]):
            print(f"    • {vocab}: {count}")
    rela_counts = defaultdict(int)
    for u, v in H.edges():
        rela = to_str(H[u][v].get('rela', ''))
        if rela:
            rela_counts[rela] += 1
    if rela_counts:
        print(f"  - RELA distribution:")
        for rela, count in sorted(rela_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    • {rela}: {count}")

if __name__ == "__main__":
    main()