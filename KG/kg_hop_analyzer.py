# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# kg_hop_analyzer.py — Analyse 1-hop / 2-hop neighborhoods for a given seed CUI.

# Inputs
#   nodes: kg_nodes.csv  (cui, name, sab, semantic_type)
#   edges: kg_edges.csv  (cui_start, name_start, sab_start, rel, rela, sab_relation, cui_target, name_target, sab_target)

# Outputs (to --out-dir/SEED):
#   seed_info.json
#   hop1_edges.csv                    # one row per edge incident to seed (after filters)
#   hop1_nodes.csv                    # unique 1-hop neighbors with counts & SABs
#   hop1_summary.json                 # relation, SAB, direction counts
#   hop2_nodes.csv                    # unique 2-hop neighbors with #paths, #midpoints, sample path
#   hop2_summary.json                 # relation + SAB summaries for 2nd hop
# """

# import argparse
# import json
# import os
# import re
# from collections import Counter, defaultdict
# from typing import Dict, Iterable, List, Optional, Tuple

# import pandas as pd


# # ------------------ helpers ------------------
# def is_cui(x: str) -> bool:
#     return isinstance(x, str) and re.fullmatch(r'C\d+', x) is not None

# def to_str(x) -> str:
#     return "" if pd.isna(x) else str(x)

# def norm_sab_tokens(s: Optional[str]) -> List[str]:
#     if s is None or (isinstance(s, float) and pd.isna(s)): return []
#     toks = [t.strip() for t in str(s).split('|')]
#     return [t for t in toks if t]

# def sab_membership_counts(cuis: Iterable[str], node_meta: Dict[str, dict]) -> Dict[str, int]:
#     c = Counter()
#     for cui in cuis:
#         sab = node_meta.get(cui, {}).get('sab', '')
#         for t in set(norm_sab_tokens(sab)):
#             c[t] += 1
#     return dict(sorted(c.items(), key=lambda kv: -kv[1]))

# def ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)


# # ------------------ load nodes ------------------
# def load_nodes(nodes_csv: str) -> Dict[str, dict]:
#     df = pd.read_csv(nodes_csv, dtype=str)
#     for col in ('cui','name','sab','semantic_type'):
#         if col in df.columns: df[col] = df[col].apply(to_str)
#     return df.set_index('cui').to_dict(orient='index')


# # ------------------ hop scanners ------------------
# def scan_hop1(edges_csv: str,
#               seed: str,
#               direction: str,
#               rel_allow: Optional[List[str]],
#               rela_allow: Optional[List[str]],
#               chunk_size: int = 500_000):
#     """
#     Return:
#       edges1: list of dicts (one per edge touching seed)
#       neigh1: set of 1-hop neighbor CUIs
#       rel_cnt, rela_cnt, dir_cnt: Counters for summaries
#     """
#     assert direction in ('out','in','both')
#     usecols = ['cui_start','cui_target','rel','rela','sab_relation']
#     edges1 = []
#     neigh1 = set()
#     rel_cnt = Counter(); rela_cnt = Counter(); dir_cnt = Counter()

#     for ch in pd.read_csv(edges_csv, usecols=usecols, dtype=str, chunksize=chunk_size):
#         for c in usecols: ch[c] = ch[c].apply(to_str)

#         if direction in ('out','both'):
#             part = ch[ch['cui_start'] == seed]
#             if rel_allow is not None:
#                 part = part[part['rel'].isin(rel_allow)]
#             if rela_allow is not None:
#                 part = part[part['rela'].isin(rela_allow)]
#             for _, r in part.iterrows():
#                 v = r['cui_target']
#                 if not is_cui(v): continue
#                 edges1.append({
#                     'dir': 'out', 'u': seed, 'v': v,
#                     'rel': to_str(r['rel']), 'rela': to_str(r['rela']), 'sab_relation': to_str(r['sab_relation'])
#                 })
#                 neigh1.add(v)
#                 rel_cnt.update([to_str(r['rel'])]); rela_cnt.update([to_str(r['rela'])]); dir_cnt.update(['out'])

#         if direction in ('in','both'):
#             part = ch[ch['cui_target'] == seed]
#             if rel_allow is not None:
#                 part = part[part['rel'].isin(rel_allow)]
#             if rela_allow is not None:
#                 part = part[part['rela'].isin(rela_allow)]
#             for _, r in part.iterrows():
#                 u = r['cui_start']
#                 if not is_cui(u): continue
#                 edges1.append({
#                     'dir': 'in', 'u': u, 'v': seed,
#                     'rel': to_str(r['rel']), 'rela': to_str(r['rela']), 'sab_relation': to_str(r['sab_relation'])
#                 })
#                 neigh1.add(u)
#                 rel_cnt.update([to_str(r['rel'])]); rela_cnt.update([to_str(r['rela'])]); dir_cnt.update(['in'])

#     return edges1, neigh1, rel_cnt, rela_cnt, dir_cnt


# def scan_hop2(edges_csv: str,
#               seed: str,
#               hop1_nodes: set,
#               direction: str,
#               rel_allow: Optional[List[str]],
#               rela_allow: Optional[List[str]],
#               chunk_size: int = 500_000):
#     """
#     Aggregate 2-hop neighbors WITHOUT exploding all paths.
#     Return:
#       hop2_info: dict[cui] -> {
#           'count_paths': int,       # number of edges from any hop1 node to this cui (or from this cui to hop1 when direction allows)
#           'midpoints': set[str],    # distinct intermediate CUIs
#           'sample': (mid, rel, rela, rel2, rela2, direction_str)  # one sample path
#       }
#       rel2_cnt, rela2_cnt: Counters for second hop edges
#     """
#     assert direction in ('out','in','both')
#     usecols = ['cui_start','cui_target','rel','rela','sab_relation']

#     # Build quick maps of how seed connects to midpoints (direction-aware)
#     # For a midpoint M, record seed-M edge roles to craft sample paths.
#     to_mid  = defaultdict(list)  # edges M -> seed (dir 'in' at hop1)
#     from_mid= defaultdict(list)  # edges seed -> M (dir 'out' at hop1)

#     # We'll fill these from hop1 later (the caller can provide edges if needed).
#     # For now we just track set membership; sample will be filled by first seen 2nd-hop edge.

#     hop2_info = {}
#     rel2_cnt = Counter(); rela2_cnt = Counter()

#     for ch in pd.read_csv(edges_csv, usecols=usecols, dtype=str, chunksize=chunk_size):
#         for c in usecols: ch[c] = ch[c].apply(to_str)

#         if rel_allow is not None:
#             ch = ch[ch['rel'].isin(rel_allow)]
#         if rela_allow is not None:
#             ch = ch[ch['rela'].isin(rela_allow)]

#         if direction in ('out','both'):
#             part = ch[ch['cui_start'].isin(hop1_nodes)]
#             for _, r in part.iterrows():
#                 mid = r['cui_start']; tgt = r['cui_target']
#                 if tgt == seed or not is_cui(tgt): continue
#                 d = hop2_info.setdefault(tgt, {'count_paths':0, 'midpoints':set(), 'sample':None})
#                 d['count_paths'] += 1
#                 d['midpoints'].add(mid)
#                 rel2_cnt.update([to_str(r['rel'])]); rela2_cnt.update([to_str(r['rela'])])
#                 if d['sample'] is None:
#                     d['sample'] = (mid, 'seed→mid', '', f"mid→nbr:{to_str(r['rel'])}", to_str(r['rela']), 'out')

#         if direction in ('in','both'):
#             part = ch[ch['cui_target'].isin(hop1_nodes)]
#             for _, r in part.iterrows():
#                 src = r['cui_start']; mid = r['cui_target']
#                 if src == seed or not is_cui(src): continue
#                 d = hop2_info.setdefault(src, {'count_paths':0, 'midpoints':set(), 'sample':None})
#                 d['count_paths'] += 1
#                 d['midpoints'].add(mid)
#                 rel2_cnt.update([to_str(r['rel'])]); rela2_cnt.update([to_str(r['rela'])])
#                 if d['sample'] is None:
#                     d['sample'] = (mid, 'mid←seed', '', f"nbr→mid:{to_str(r['rel'])}", to_str(r['rela']), 'in')

#     return hop2_info, rel2_cnt, rela2_cnt


# # ------------------ reporting ------------------
# def write_hop1_outputs(seed: str,
#                        out_dir: str,
#                        edges1: List[dict],
#                        neigh1: set,
#                        nodes_meta: Dict[str, dict],
#                        rel_cnt: Counter,
#                        rela_cnt: Counter,
#                        dir_cnt: Counter):
#     # Per-edge table
#     rows = []
#     for e in edges1:
#         u, v = e['u'], e['v']
#         if e['dir'] == 'out':
#             nbr = v
#         else:
#             nbr = u
#         rows.append({
#             'seed': seed,
#             'neighbor_cui': nbr,
#             'neighbor_name': nodes_meta.get(nbr, {}).get('name', ''),
#             'neighbor_sab': nodes_meta.get(nbr, {}).get('sab', ''),
#             'direction': e['dir'],
#             'rel': e['rel'],
#             'rela': e['rela'],
#             'sab_relation': e['sab_relation'],
#             'edge_u': u,
#             'edge_v': v
#         })
#     hop1_edges_path = os.path.join(out_dir, 'hop1_edges.csv')
#     pd.DataFrame(rows).to_csv(hop1_edges_path, index=False)

#     # Unique neighbors table (with counts by direction)
#     dir_count_map = defaultdict(lambda: {'out':0,'in':0})
#     for e in edges1:
#         if e['dir']=='out':
#             dir_count_map[e['v']]['out'] += 1
#         else:
#             dir_count_map[e['u']]['in'] += 1

#     uniq_rows = []
#     for n in sorted(neigh1):
#         meta = nodes_meta.get(n, {})
#         outc = dir_count_map[n]['out']; inc = dir_count_map[n]['in']
#         uniq_rows.append({
#             'neighbor_cui': n,
#             'neighbor_name': meta.get('name',''),
#             'neighbor_sab': meta.get('sab',''),
#             'edges_to_seed_out': outc,
#             'edges_to_seed_in': inc,
#             'edges_total': outc + inc
#         })
#     hop1_nodes_path = os.path.join(out_dir, 'hop1_nodes.csv')
#     pd.DataFrame(uniq_rows).sort_values('edges_total', ascending=False).to_csv(hop1_nodes_path, index=False)

#     # Summary JSON
#     sab_counts = sab_membership_counts(neigh1, nodes_meta)
#     summary = {
#         'seed': seed,
#         'num_neighbors': len(neigh1),
#         'direction_counts': dict(dir_cnt),
#         'rel_counts': dict(rel_cnt),
#         'rela_counts': dict(rela_cnt),
#         'neighbor_sab_counts': sab_counts
#     }
#     with open(os.path.join(out_dir, 'hop1_summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)


# def write_hop2_outputs(seed: str,
#                        out_dir: str,
#                        hop2_info: Dict[str, dict],
#                        nodes_meta: Dict[str, dict],
#                        rel2_cnt: Counter,
#                        rela2_cnt: Counter,
#                        exclude_hop1_from_hop2: bool,
#                        hop1_nodes: set):
#     # Unique neighbors (2-hop)
#     rows = []
#     for nbr, info in hop2_info.items():
#         if exclude_hop1_from_hop2 and nbr in hop1_nodes:
#             continue
#         meta = nodes_meta.get(nbr, {})
#         sample_mid, step1, rela1, step2, rela2, which = info['sample'] if info['sample'] else ('','','','','','')
#         rows.append({
#             'neighbor2_cui': nbr,
#             'neighbor2_name': meta.get('name',''),
#             'neighbor2_sab': meta.get('sab',''),
#             'paths_count': info['count_paths'],
#             'num_midpoints': len(info['midpoints']),
#             'sample_midpoint': sample_mid,
#             'sample_step1': step1,
#             'sample_step2': step2,
#             'sample_rela1': rela1,
#             'sample_rela2': rela2,
#             'which_direction_on_2nd_hop': which
#         })
#     hop2_nodes_path = os.path.join(out_dir, 'hop2_nodes.csv')
#     pd.DataFrame(rows).sort_values('paths_count', ascending=False).to_csv(hop2_nodes_path, index=False)

#     summary = {
#         'seed': seed,
#         'num_2hop_neighbors': len([r for r in rows]),
#         'rel_counts_second_hop': dict(rel2_cnt),
#         'rela_counts_second_hop': dict(rela2_cnt),
#         'neighbor2_sab_counts': sab_membership_counts([r['neighbor2_cui'] for r in rows], nodes_meta)
#     }
#     with open(os.path.join(out_dir, 'hop2_summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)


# # ------------------ CLI ------------------
# def main():
#     ap = argparse.ArgumentParser(description="Analyse 1-hop / 2-hop neighborhoods for a seed CUI.")
#     ap.add_argument('--seed', required=True, help='Seed CUI (e.g., C0011849)')
#     ap.add_argument('--nodes', required=True, help='Path to kg_nodes.csv')
#     ap.add_argument('--edges', required=True, help='Path to kg_edges.csv')
#     ap.add_argument('--radius', type=int, choices=[1,2], default=2, help='Compute up to this hop (default: 2)')
#     ap.add_argument('--direction', choices=['out','in','both'], default='both', help='Edge direction to consider (default: both)')
#     ap.add_argument('--rel-allow', default='', help='Comma-separated REL codes to include (optional)')
#     ap.add_argument('--rela-allow', default='', help='Comma-separated RELA strings to include (optional)')
#     ap.add_argument('--chunk-size', type=int, default=500_000, help='CSV chunk size (default: 500k)')
#     ap.add_argument('--exclude-hop1-from-hop2', action='store_true', help='Show only new nodes at hop-2 (exclude hop-1 set)')
#     ap.add_argument('--out-dir', required=True, help='Directory to write outputs (a subfolder with the seed will be created)')
#     args = ap.parse_args()

#     seed = to_str(args.seed).strip().upper()
#     if not is_cui(seed):
#         raise SystemExit(f"--seed must be a valid CUI like C0011849; got {args.seed}")

#     rel_allow = [s.strip() for s in args.rel_allow.split(',') if s.strip()] or None
#     rela_allow = [s.strip() for s in args.rela_allow.split(',') if s.strip()] or None

#     # Output paths
#     base_out = os.path.join(args.out_dir, seed)
#     ensure_dir(base_out)

#     # Load node metadata
#     nodes_meta = load_nodes(args.nodes)
#     seed_meta = nodes_meta.get(seed, {'name':'', 'sab':'', 'semantic_type':''})
#     with open(os.path.join(base_out, 'seed_info.json'), 'w') as f:
#         json.dump({
#             'seed': seed,
#             'name': seed_meta.get('name',''),
#             'sab': seed_meta.get('sab',''),
#             'semantic_type': seed_meta.get('semantic_type','')
#         }, f, indent=2)

#     # ---- hop 1 ----
#     edges1, neigh1, rel_cnt, rela_cnt, dir_cnt = scan_hop1(
#         edges_csv=args.edges,
#         seed=seed,
#         direction=args.direction,
#         rel_allow=rel_allow,
#         rela_allow=rela_allow,
#         chunk_size=args.chunk_size
#     )
#     write_hop1_outputs(seed, base_out, edges1, neigh1, nodes_meta, rel_cnt, rela_cnt, dir_cnt)

#     # ---- hop 2 ----
#     if args.radius == 2 and neigh1:
#         hop2_info, rel2_cnt, rela2_cnt = scan_hop2(
#             edges_csv=args.edges,
#             seed=seed,
#             hop1_nodes=neigh1,
#             direction=args.direction,
#             rel_allow=rel_allow,
#             rela_allow=rela_allow,
#             chunk_size=args.chunk_size
#         )
#         write_hop2_outputs(seed, base_out, hop2_info, nodes_meta, rel2_cnt, rela2_cnt,
#                            exclude_hop1_from_hop2=args.exclude_hop1_from_hop2,
#                            hop1_nodes=neigh1)

#     print(f"[OK] Wrote hop analysis for {seed} to: {base_out}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kg_hop_analyzer.py — Analyse 1-hop / 2-hop neighborhoods for a given seed CUI.

Inputs
  nodes: kg_nodes.csv  (cui, name, sab, semantic_type)
  edges: kg_edges.csv  (cui_start, name_start, sab_start, rel, rela, sab_relation, cui_target, name_target, sab_target)

Outputs (to --out-dir/SEED):
  seed_info.json
  hop1_edges.csv                    # per edge touching seed
  hop1_nodes.csv                    # unique 1-hop neighbors
  hop1_summary.json
  hop2_edges.csv                    # per second-hop edge (mid <-> nbr2) with path context
  hop2_nodes.csv                    # unique 2-hop neighbors with path counts
  hop2_summary.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ------------------ helpers ------------------
def is_cui(x: str) -> bool:
    return isinstance(x, str) and re.fullmatch(r'C\d+', x) is not None

def to_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def norm_sab_tokens(s: Optional[str]) -> List[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)): return []
    toks = [t.strip() for t in str(s).split('|')]
    return [t for t in toks if t]

def sab_membership_counts(cuis: Iterable[str], node_meta: Dict[str, dict]) -> Dict[str, int]:
    c = Counter()
    for cui in cuis:
        sab = node_meta.get(cui, {}).get('sab', '')
        for t in set(norm_sab_tokens(sab)):
            c[t] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------ load nodes ------------------
def load_nodes(nodes_csv: str) -> Dict[str, dict]:
    df = pd.read_csv(nodes_csv, dtype=str)
    for col in ('cui','name','sab','semantic_type'):
        if col in df.columns: df[col] = df[col].apply(to_str)
    return df.set_index('cui').to_dict(orient='index')


# ------------------ hop scanners ------------------
def scan_hop1(edges_csv: str,
              seed: str,
              direction: str,
              rel_allow: Optional[List[str]],
              rela_allow: Optional[List[str]],
              chunk_size: int = 500_000):
    """
    Return:
      edges1: list of dicts (one per edge touching seed)
      neigh1: set of 1-hop neighbor CUIs
      rel_cnt, rela_cnt, dir_cnt: Counters for summaries
      hop1_maps: {
        'seed_to_mid': dict[mid] -> set[(rel, rela)],
        'mid_to_seed': dict[mid] -> set[(rel, rela)]
      }
    """
    assert direction in ('out','in','both')
    usecols = ['cui_start','cui_target','rel','rela','sab_relation']
    edges1 = []
    neigh1 = set()
    rel_cnt = Counter(); rela_cnt = Counter(); dir_cnt = Counter()

    seed_to_mid = defaultdict(set)  # seed -> mid (out)
    mid_to_seed = defaultdict(set)  # mid -> seed (in)

    for ch in pd.read_csv(edges_csv, usecols=usecols, dtype=str, chunksize=chunk_size):
        for c in usecols: ch[c] = ch[c].apply(to_str)

        if direction in ('out','both'):
            part = ch[ch['cui_start'] == seed]
            if rel_allow is not None:
                part = part[part['rel'].isin(rel_allow)]
            if rela_allow is not None:
                part = part[part['rela'].isin(rela_allow)]
            for _, r in part.iterrows():
                v = r['cui_target']
                if not is_cui(v): continue
                edges1.append({
                    'dir': 'out', 'u': seed, 'v': v,
                    'rel': to_str(r['rel']), 'rela': to_str(r['rela']), 'sab_relation': to_str(r['sab_relation'])
                })
                neigh1.add(v)
                rel_cnt.update([to_str(r['rel'])]); rela_cnt.update([to_str(r['rela'])]); dir_cnt.update(['out'])
                seed_to_mid[v].add((to_str(r['rel']), to_str(r['rela'])))

        if direction in ('in','both'):
            part = ch[ch['cui_target'] == seed]
            if rel_allow is not None:
                part = part[part['rel'].isin(rel_allow)]
            if rela_allow is not None:
                part = part[part['rela'].isin(rela_allow)]
            for _, r in part.iterrows():
                u = r['cui_start']
                if not is_cui(u): continue
                edges1.append({
                    'dir': 'in', 'u': u, 'v': seed,
                    'rel': to_str(r['rel']), 'rela': to_str(r['rela']), 'sab_relation': to_str(r['sab_relation'])
                })
                neigh1.add(u)
                rel_cnt.update([to_str(r['rel'])]); rela_cnt.update([to_str(r['rela'])]); dir_cnt.update(['in'])
                mid_to_seed[u].add((to_str(r['rel']), to_str(r['rela'])))

    hop1_maps = {'seed_to_mid': seed_to_mid, 'mid_to_seed': mid_to_seed}
    return edges1, neigh1, rel_cnt, rela_cnt, dir_cnt, hop1_maps


def scan_hop2(edges_csv: str,
              seed: str,
              hop1_nodes: set,
              hop1_maps: dict,
              direction: str,
              rel_allow: Optional[List[str]],
              rela_allow: Optional[List[str]],
              chunk_size: int = 500_000):
    """
    Aggregate 2-hop neighbors and collect second-hop edges.

    Return:
      hop2_info: dict[cui] -> { 'count_paths': int, 'midpoints': set[str], 'sample': tuple }
      edges2: list of dicts describing each second-hop edge with path context:
        {
          'path_pattern': 'seed→mid→nbr2' | 'nbr2→mid→seed',
          'mid_cui', 'mid_name', 'mid_sab',
          'neighbor2_cui', 'neighbor2_name', 'neighbor2_sab',
          'step1_rel', 'step1_rela',                 # (summary from hop-1, seed↔mid)
          'step2_rel', 'step2_rela', 'step2_sab_relation',  # from the second-hop edge
          'edge_u', 'edge_v'                         # second-hop edge endpoints
        }
      rel2_cnt, rela2_cnt: Counters for second hop
    """
    assert direction in ('out','in','both')
    usecols = ['cui_start','cui_target','rel','rela','sab_relation']

    seed_to_mid = hop1_maps.get('seed_to_mid', {})
    mid_to_seed = hop1_maps.get('mid_to_seed', {})

    hop2_info = {}
    edges2 = []
    rel2_cnt = Counter(); rela2_cnt = Counter()

    for ch in pd.read_csv(edges_csv, usecols=usecols, dtype=str, chunksize=chunk_size):
        for c in usecols: ch[c] = ch[c].apply(to_str)

        if rel_allow is not None:
            ch = ch[ch['rel'].isin(rel_allow)]
        if rela_allow is not None:
            ch = ch[ch['rela'].isin(rela_allow)]

        # OUT: mid -> nbr2 (seed -> mid -> nbr2)
        if direction in ('out','both'):
            part = ch[ch['cui_start'].isin(hop1_nodes)]
            for _, r in part.iterrows():
                mid = r['cui_start']; nbr2 = r['cui_target']
                if nbr2 == seed or not is_cui(nbr2): continue

                d = hop2_info.setdefault(nbr2, {'count_paths':0, 'midpoints':set(), 'sample':None})
                d['count_paths'] += 1
                d['midpoints'].add(mid)
                rel2_cnt.update([to_str(r['rel'])]); rela2_cnt.update([to_str(r['rela'])])

                # Summarize step-1 rel/rela from seed_to_mid[mid]
                s1_pairs = seed_to_mid.get(mid, set())
                s1_rel  = '|'.join(sorted({p[0] for p in s1_pairs if p[0]})) if s1_pairs else ''
                s1_rela = '|'.join(sorted({p[1] for p in s1_pairs if p[1]})) if s1_pairs else ''

                edges2.append({
                    'path_pattern': 'seed→mid→nbr2',
                    'mid_cui': mid, 'neighbor2_cui': nbr2,
                    'step1_rel': s1_rel, 'step1_rela': s1_rela,
                    'step2_rel': to_str(r['rel']), 'step2_rela': to_str(r['rela']),
                    'step2_sab_relation': to_str(r['sab_relation']),
                    'edge_u': to_str(r['cui_start']), 'edge_v': to_str(r['cui_target'])
                })

                if d['sample'] is None:
                    d['sample'] = (mid, 'seed→mid', s1_rela, f"mid→nbr:{to_str(r['rel'])}", to_str(r['rela']), 'out')

        # IN: nbr2 -> mid (nbr2 -> mid -> seed)
        if direction in ('in','both'):
            part = ch[ch['cui_target'].isin(hop1_nodes)]
            for _, r in part.iterrows():
                nbr2 = r['cui_start']; mid = r['cui_target']
                if nbr2 == seed or not is_cui(nbr2): continue

                d = hop2_info.setdefault(nbr2, {'count_paths':0, 'midpoints':set(), 'sample':None})
                d['count_paths'] += 1
                d['midpoints'].add(mid)
                rel2_cnt.update([to_str(r['rel'])]); rela2_cnt.update([to_str(r['rela'])])

                # Summarize step-1 rel/rela from mid_to_seed[mid]
                s1_pairs = mid_to_seed.get(mid, set())
                s1_rel  = '|'.join(sorted({p[0] for p in s1_pairs if p[0]})) if s1_pairs else ''
                s1_rela = '|'.join(sorted({p[1] for p in s1_pairs if p[1]})) if s1_pairs else ''

                edges2.append({
                    'path_pattern': 'nbr2→mid→seed',
                    'mid_cui': mid, 'neighbor2_cui': nbr2,
                    'step1_rel': s1_rel, 'step1_rela': s1_rela,
                    'step2_rel': to_str(r['rel']), 'step2_rela': to_str(r['rela']),
                    'step2_sab_relation': to_str(r['sab_relation']),
                    'edge_u': to_str(r['cui_start']), 'edge_v': to_str(r['cui_target'])
                })

                if d['sample'] is None:
                    d['sample'] = (mid, 'mid←seed', s1_rela, f"nbr→mid:{to_str(r['rel'])}", to_str(r['rela']), 'in')

    return hop2_info, edges2, rel2_cnt, rela2_cnt


# ------------------ reporting ------------------
def write_hop1_outputs(seed: str,
                       out_dir: str,
                       edges1: List[dict],
                       neigh1: set,
                       nodes_meta: Dict[str, dict],
                       rel_cnt: Counter,
                       rela_cnt: Counter,
                       dir_cnt: Counter):
    # Per-edge table
    rows = []
    for e in edges1:
        u, v = e['u'], e['v']
        if e['dir'] == 'out':
            nbr = v
        else:
            nbr = u
        rows.append({
            'seed': seed,
            'neighbor_cui': nbr,
            'neighbor_name': nodes_meta.get(nbr, {}).get('name', ''),
            'neighbor_sab': nodes_meta.get(nbr, {}).get('sab', ''),
            'direction': e['dir'],
            'rel': e['rel'],
            'rela': e['rela'],
            'sab_relation': e['sab_relation'],
            'edge_u': u,
            'edge_v': v
        })
    hop1_edges_path = os.path.join(out_dir, 'hop1_edges.csv')
    pd.DataFrame(rows).to_csv(hop1_edges_path, index=False)

    # Unique neighbors table (with counts by direction)
    dir_count_map = defaultdict(lambda: {'out':0,'in':0})
    for e in edges1:
        if e['dir']=='out':
            dir_count_map[e['v']]['out'] += 1
        else:
            dir_count_map[e['u']]['in'] += 1

    uniq_rows = []
    for n in sorted(neigh1):
        meta = nodes_meta.get(n, {})
        outc = dir_count_map[n]['out']; inc = dir_count_map[n]['in']
        uniq_rows.append({
            'neighbor_cui': n,
            'neighbor_name': meta.get('name',''),
            'neighbor_sab': meta.get('sab',''),
            'edges_to_seed_out': outc,
            'edges_to_seed_in': inc,
            'edges_total': outc + inc
        })
    hop1_nodes_path = os.path.join(out_dir, 'hop1_nodes.csv')
    pd.DataFrame(uniq_rows).sort_values('edges_total', ascending=False).to_csv(hop1_nodes_path, index=False)

    # Summary JSON
    sab_counts = sab_membership_counts(neigh1, nodes_meta)
    summary = {
        'seed': seed,
        'num_neighbors': len(neigh1),
        'direction_counts': dict(dir_cnt),
        'rel_counts': dict(rel_cnt),
        'rela_counts': dict(rela_cnt),
        'neighbor_sab_counts': sab_counts
    }
    with open(os.path.join(out_dir, 'hop1_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def write_hop2_outputs(seed: str,
                       out_dir: str,
                       hop2_info: Dict[str, dict],
                       edges2: List[dict],
                       nodes_meta: Dict[str, dict],
                       rel2_cnt: Counter,
                       rela2_cnt: Counter,
                       exclude_hop1_from_hop2: bool,
                       hop1_nodes: set):
    # Per-edge (second hop) table with node metadata
    rows_edges2 = []
    for e in edges2:
        mid = e['mid_cui']; nbr2 = e['neighbor2_cui']
        rows_edges2.append({
            'seed': seed,
            'path_pattern': e['path_pattern'],     # 'seed→mid→nbr2' or 'nbr2→mid→seed'
            'mid_cui': mid,
            'mid_name': nodes_meta.get(mid, {}).get('name', ''),
            'mid_sab': nodes_meta.get(mid, {}).get('sab', ''),
            'neighbor2_cui': nbr2,
            'neighbor2_name': nodes_meta.get(nbr2, {}).get('name', ''),
            'neighbor2_sab': nodes_meta.get(nbr2, {}).get('sab', ''),
            'step1_rel': e['step1_rel'],
            'step1_rela': e['step1_rela'],
            'step2_rel': e['step2_rel'],
            'step2_rela': e['step2_rela'],
            'step2_sab_relation': e['step2_sab_relation'],
            'edge_u': e['edge_u'],
            'edge_v': e['edge_v']
        })
    hop2_edges_path = os.path.join(out_dir, 'hop2_edges.csv')
    pd.DataFrame(rows_edges2).to_csv(hop2_edges_path, index=False)

    # Unique neighbors (2-hop)
    rows_nodes2 = []
    for nbr, info in hop2_info.items():
        if exclude_hop1_from_hop2 and nbr in hop1_nodes:
            continue
        meta = nodes_meta.get(nbr, {})
        sample_mid, step1, rela1, step2, rela2, which = info['sample'] if info['sample'] else ('','','','','','')
        rows_nodes2.append({
            'neighbor2_cui': nbr,
            'neighbor2_name': meta.get('name',''),
            'neighbor2_sab': meta.get('sab',''),
            'paths_count': info['count_paths'],
            'num_midpoints': len(info['midpoints']),
            'sample_midpoint': sample_mid,
            'sample_step1': step1,
            'sample_step2': step2,
            'sample_rela1': rela1,
            'sample_rela2': rela2,
            'which_direction_on_2nd_hop': which
        })
    hop2_nodes_path = os.path.join(out_dir, 'hop2_nodes.csv')
    pd.DataFrame(rows_nodes2).sort_values('paths_count', ascending=False).to_csv(hop2_nodes_path, index=False)

    # Summary JSON
    summary2 = {
        'seed': seed,
        'num_2hop_neighbors': len([r for r in rows_nodes2]),
        'rel_counts_second_hop': dict(rel2_cnt),
        'rela_counts_second_hop': dict(rela2_cnt),
        'neighbor2_sab_counts': sab_membership_counts([r['neighbor2_cui'] for r in rows_nodes2], nodes_meta)
    }
    with open(os.path.join(out_dir, 'hop2_summary.json'), 'w') as f:
        json.dump(summary2, f, indent=2)


# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(description="Analyse 1-hop / 2-hop neighborhoods for a seed CUI.")
    ap.add_argument('--seed', required=True, help='Seed CUI (e.g., C0011849)')
    ap.add_argument('--nodes', required=True, help='Path to kg_nodes.csv')
    ap.add_argument('--edges', required=True, help='Path to kg_edges.csv')
    ap.add_argument('--radius', type=int, choices=[1,2], default=2, help='Compute up to this hop (default: 2)')
    ap.add_argument('--direction', choices=['out','in','both'], default='both', help='Edge direction to consider (default: both)')
    ap.add_argument('--rel-allow', default='', help='Comma-separated REL codes to include (optional)')
    ap.add_argument('--rela-allow', default='', help='Comma-separated RELA strings to include (optional)')
    ap.add_argument('--chunk-size', type=int, default=500_000, help='CSV chunk size (default: 500k)')
    ap.add_argument('--exclude-hop1-from-hop2', action='store_true', help='Show only new nodes at hop-2 (exclude hop-1 set)')
    ap.add_argument('--out-dir', required=True, help='Directory to write outputs (a subfolder with the seed will be created)')
    args = ap.parse_args()

    seed = to_str(args.seed).strip().upper()
    if not is_cui(seed):
        raise SystemExit(f"--seed must be a valid CUI like C0011849; got {args.seed}")

    rel_allow = [s.strip() for s in args.rel_allow.split(',') if s.strip()] or None
    rela_allow = [s.strip() for s in args.rela_allow.split(',') if s.strip()] or None

    # Output paths
    base_out = os.path.join(args.out_dir, seed)
    ensure_dir(base_out)

    # Load node metadata
    nodes_meta = load_nodes(args.nodes)
    seed_meta = nodes_meta.get(seed, {'name':'', 'sab':'', 'semantic_type':''})
    with open(os.path.join(base_out, 'seed_info.json'), 'w') as f:
        json.dump({
            'seed': seed,
            'name': seed_meta.get('name',''),
            'sab': seed_meta.get('sab',''),
            'semantic_type': seed_meta.get('semantic_type','')
        }, f, indent=2)

    # ---- hop 1 ----
    edges1, neigh1, rel_cnt, rela_cnt, dir_cnt, hop1_maps = scan_hop1(
        edges_csv=args.edges,
        seed=seed,
        direction=args.direction,
        rel_allow=rel_allow,
        rela_allow=rela_allow,
        chunk_size=args.chunk_size
    )
    write_hop1_outputs(seed, base_out, edges1, neigh1, nodes_meta, rel_cnt, rela_cnt, dir_cnt)

    # ---- hop 2 ----
    if args.radius == 2 and neigh1:
        hop2_info, edges2, rel2_cnt, rela2_cnt = scan_hop2(
            edges_csv=args.edges,
            seed=seed,
            hop1_nodes=neigh1,
            hop1_maps=hop1_maps,
            direction=args.direction,
            rel_allow=rel_allow,
            rela_allow=rela_allow,
            chunk_size=args.chunk_size
        )
        write_hop2_outputs(seed, base_out, hop2_info, edges2, nodes_meta, rel2_cnt, rela2_cnt,
                           exclude_hop1_from_hop2=args.exclude_hop1_from_hop2,
                           hop1_nodes=neigh1)

    print(f"[OK] Wrote hop analysis for {seed} to: {base_out}")


if __name__ == "__main__":
    main()
