# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Build a UMLS-based medical KG with CUI→CUI edges only.

# Nodes CSV schema:
#   cui, name, sab, semantic_type

# Edges CSV schema:
#   cui_start, name_start, sab_start, rel, rela, sab_relation, cui_target, name_target, sab_target
# """

# import os
# import json
# import pickle
# from collections import defaultdict
# from datetime import datetime

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import networkx as nx

# import matplotlib.pyplot as plt
# import seaborn as sns

# # -------------------- CONFIG --------------------
# UMLS_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META'
# MRCONSO = os.path.join(UMLS_DIR, 'MRCONSO.RRF')
# MRREL   = os.path.join(UMLS_DIR, 'MRREL.RRF')
# MRSTY   = os.path.join(UMLS_DIR, 'MRSTY.RRF')

# ANALYSIS_DIR   = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG'
# OUTPUT_DIR     = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output3'

# ICD9_ANALYSIS  = os.path.join(ANALYSIS_DIR, 'mapping_analysis', 'simple_icd9_cui_sab_mapping.csv')
# LAB_ANALYSIS   = os.path.join(ANALYSIS_DIR, 'lab_test_analysis', 'simple_loinc_cui_sab_mapping.csv')
# MED_ANALYSIS   = os.path.join(ANALYSIS_DIR, 'med_analysis', 'simple_med_cui_sab_mapping.csv')
# PROC_ANALYSIS  = os.path.join(ANALYSIS_DIR, 'procedure_analysis', 'simple_proc_cui_sab_mapping.csv')

# TARGET_VOCABS  = ['ICD9CM', 'LNC', 'ATC', 'SNOMEDCT_US']  # tweak as needed

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_DIR, 'caches'), exist_ok=True)

# log_file = os.path.join(OUTPUT_DIR, f'kg_build_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# def log(message: str):
#     ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     with open(log_file, 'a') as f:
#         f.write(f'[{ts}] {message}\n')
#     print(message)


# # -------------------- HELPERS --------------------
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer): return int(obj)
#         if isinstance(obj, np.floating): return float(obj)
#         if isinstance(obj, np.ndarray): return obj.tolist()
#         return super().default(obj)

# def safe_pipe_join(value):
#     if isinstance(value, list):
#         return '|'.join(map(str, value))
#     return value if isinstance(value, str) else ''


# # -------------------- LOADERS (read by integer indices, rename later) --------------------
# def load_mrconso_subset(target_vocabs=TARGET_VOCABS, langs=['ENG']):
#     """
#     Load preferred English name per CUI and union SAB list per CUI
#     for the chosen vocabularies.

#     MRCONSO field indices (2025AA):
#       0:CUI, 1:LAT, 6:ISPREF, 11:SAB, 12:TTY, 13:CODE, 14:STR
#     """
#     log(f"Loading MRCONSO for vocabs: {', '.join(target_vocabs)}")
#     cache_file = os.path.join(OUTPUT_DIR, 'caches', 'mrconso_subset.pkl')
#     if os.path.exists(cache_file):
#         log(f"MRCONSO cache: {cache_file}")
#         return pd.read_pickle(cache_file)

#     usecols_idx = [0, 1, 6, 11, 12, 13, 14]  # keep order aligned with names below
#     colnames    = ['CUI','LAT','ISPREF','SAB','TTY','CODE','STR']

#     it = pd.read_csv(
#         MRCONSO,
#         sep='|',
#         header=None,
#         usecols=usecols_idx,
#         dtype=str,
#         chunksize=1_000_000
#     )

#     chunks = []
#     for ch in it:
#         ch.columns = colnames  # rename after reading the subset
#         ch = ch[(ch['LAT'].isin(langs)) & (ch['SAB'].isin(target_vocabs))]
#         if not ch.empty:
#             chunks.append(ch)

#     conso = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=colnames)

#     # Preferred name strategy: ISPREF=Y first, then TTY ranking
#     tty_priority = {'PT': 0, 'PN': 1, 'FN': 2}
#     conso['is_pref']  = (conso['ISPREF'] == 'Y').astype(int)
#     conso['tty_rank'] = conso['TTY'].map(lambda t: tty_priority.get(t, 99))

#     conso_sorted = conso.sort_values(['CUI','is_pref','tty_rank']).drop_duplicates('CUI', keep='first')
#     conso_names  = conso_sorted[['CUI','STR']].rename(columns={'STR':'name'})

#     sab_map = (conso.groupby('CUI')['SAB']
#                .apply(lambda s: sorted(set(x for x in s if isinstance(x, str))))
#                .reset_index()
#                .rename(columns={'SAB':'sab_list'}))

#     # New: collect CODE values per CUI (may be multiple codes per CUI across SABs)
#     code_map = (conso.groupby('CUI')['CODE']
#                 .apply(lambda s: sorted(set(x for x in s if isinstance(x, str) and x != '')))
#                 .reset_index()
#                 .rename(columns={'CODE':'code_list'}))

#     # Merge names + sab list + code list
#     conso_final = conso_names.merge(sab_map, on='CUI', how='left').merge(code_map, on='CUI', how='left')
#     conso_final.to_pickle(cache_file)
#     log(f"MRCONSO prepared: {len(conso_final):,} CUIs with names")
#     return conso_final


# def load_mrrel_subset(target_vocabs=TARGET_VOCABS, cuis=None):
#     """
#     Load MRREL relations filtered by SAB vocabularies; optionally restricted to CUIs.

#     MRREL field indices (2025AA):
#       0:CUI1, 3:REL, 4:CUI2, 7:RELA, 10:SAB
#     """
#     log("Loading MRREL (int-indexed, then renamed)")
#     cache_file = os.path.join(OUTPUT_DIR, 'caches', 'mrrel_subset.pkl')
#     if os.path.exists(cache_file) and cuis is None:
#         log(f"MRREL cache: {cache_file}")
#         return pd.read_pickle(cache_file)

#     usecols_idx = [0, 3, 4, 7, 10]  # important: sorted order as pandas returns
#     colnames    = ['CUI1','REL','CUI2','RELA','SAB']

#     it = pd.read_csv(
#         MRREL,
#         sep='|',
#         header=None,
#         usecols=usecols_idx,
#         dtype=str,
#         chunksize=1_000_000
#     )

#     out_chunks = []
#     for ch in it:
#         ch.columns = colnames
#         ch = ch[ch['SAB'].isin(target_vocabs)]
#         if cuis is not None:
#             ch = ch[ch['CUI1'].isin(cuis) | ch['CUI2'].isin(cuis)]
#         if not ch.empty:
#             out_chunks.append(ch)

#     rel = pd.concat(out_chunks, ignore_index=True) if out_chunks else pd.DataFrame(columns=colnames)
#     if cuis is None:
#         rel.to_pickle(cache_file)
#     log(f"MRREL filtered: {len(rel):,} rows")
#     return rel


# def load_mrsty_subset(cuis=None):
#     """
#     Load MRSTY semantic types; optionally restricted to CUIs.

#     MRSTY field indices (2025AA):
#       0:CUI, 1:TUI, 3:STY
#     """
#     log("Loading MRSTY (int-indexed, then renamed)")
#     cache_file = os.path.join(OUTPUT_DIR, 'caches', 'mrsty_subset.pkl')
#     if os.path.exists(cache_file) and cuis is None:
#         log(f"MRSTY cache: {cache_file}")
#         return pd.read_pickle(cache_file)

#     usecols_idx = [0, 1, 3]
#     colnames    = ['CUI','TUI','STY']

#     it = pd.read_csv(
#         MRSTY,
#         sep='|',
#         header=None,
#         usecols=usecols_idx,
#         dtype=str,
#         chunksize=500_000
#     )

#     chunks = []
#     for ch in it:
#         ch.columns = colnames
#         if cuis is not None:
#             ch = ch[ch['CUI'].isin(cuis)]
#         if not ch.empty:
#             chunks.append(ch)

#     sty = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=colnames)
#     if cuis is None:
#         sty.to_pickle(cache_file)
#     log(f"MRSTY: {len(sty):,} rows")
#     return sty


# # -------------------- ANALYSIS INPUTS --------------------
# def load_analysis_files():
#     """
#     Load precomputed domain CUI lists (optional); returns dict of sources -> set(CUI), and union set.
#     """
#     log("Loading analysis-domain CUI files")
#     cui_sources = {}

#     def read_cuis(path, key):
#         if os.path.exists(path):
#             df = pd.read_csv(path)
#             s = set(df['cui'].astype(str))
#             log(f"  {key}: {len(s):,} CUIs")
#             return s
#         else:
#             log(f"  WARN: Missing file: {path}")
#             return set()

#     cui_sources['ICD9'] = read_cuis(ICD9_ANALYSIS, 'ICD9')
#     cui_sources['LAB']  = read_cuis(LAB_ANALYSIS,  'LAB')
#     cui_sources['MED']  = read_cuis(MED_ANALYSIS,  'MED')
#     cui_sources['PROC'] = read_cuis(PROC_ANALYSIS, 'PROC')

#     all_cuis = set().union(*cui_sources.values()) if cui_sources else set()
#     log(f"Total seed CUIs from analyses: {len(all_cuis):,}")
#     return cui_sources, all_cuis


# # -------------------- BUILD KG --------------------
# def build_knowledge_graph(cui_sources, all_cuis):
#     log("Building KG (CUI-only)")
#     conso_df   = load_mrconso_subset()
#     umls_cuis  = set(conso_df['CUI'])
#     combined   = set(all_cuis) | umls_cuis

#     rel_df     = load_mrrel_subset(target_vocabs=TARGET_VOCABS)
#     rel_cuis   = set(rel_df['CUI1']) | set(rel_df['CUI2'])
#     combined  |= rel_cuis

#     sty_df     = load_mrsty_subset(combined)
#     sty_map    = sty_df.groupby('CUI')['STY'].apply(lambda s: sorted(set(s))).to_dict()

#     name_map   = dict(zip(conso_df['CUI'], conso_df['name']))
#     sab_map    = dict(zip(conso_df['CUI'], conso_df['sab_list']))
#     code_map   = dict(zip(conso_df['CUI'], conso_df.get('code_list', pd.Series([[]]*len(conso_df)))))  # new

#     # Build graph
#     G = nx.DiGraph()

#     # Node sources from analyses
#     cui_to_sources = defaultdict(list)
#     for source, cuiset in cui_sources.items():
#         for cui in cuiset:
#             cui_to_sources[cui].append(source)

#     log("Adding nodes…")
#     for cui in tqdm(combined, desc="Nodes"):
#         G.add_node(
#             cui,
#             name=name_map.get(cui, "Unknown"),
#             sab=sab_map.get(cui, []),
#             code=code_map.get(cui, []),               # new attribute
#             semantic_type=sty_map.get(cui, []),
#             source=cui_to_sources.get(cui, [])
#         )

#     log("Adding edges…")
#     edges_added = 0
#     for _, r in tqdm(rel_df.iterrows(), total=len(rel_df), desc="Edges"):
#         c1, c2 = r['CUI1'], r['CUI2']
#         if c1 in G and c2 in G:
#             G.add_edge(
#                 c1, c2,
#                 rel=r.get('REL','') or '',
#                 rela=r.get('RELA','') if pd.notna(r.get('RELA')) else '',
#                 sab_relation=r.get('SAB','') or ''
#             )
#             edges_added += 1

#     log(f"Graph summary: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges (added {edges_added:,})")

#     # -------- Save node/edge CSVs in requested schema --------
#     log("Saving nodes/edges CSV…")

#     nodes_list = []
#     for cui, data in G.nodes(data=True):
#         nodes_list.append({
#             'cui': cui,
#             'name': data.get('name',''),
#             'sab': safe_pipe_join(data.get('sab', [])),
#             'code': safe_pipe_join(data.get('code', [])),                      # new column 'code'
#             'semantic_type': safe_pipe_join(data.get('semantic_type', [])),  # keep; delete if not needed
#         })
#     nodes_df = pd.DataFrame(nodes_list)
#     nodes_path = os.path.join(OUTPUT_DIR, 'kg_nodes.csv')
#     nodes_df.to_csv(nodes_path, index=False)

#     edges_list = []
#     for u, v, data in G.edges(data=True):
#         edges_list.append({
#             'cui_start':   u,
#             'name_start':  G.nodes[u].get('name',''),
#             'sab_start':   safe_pipe_join(G.nodes[u].get('sab', [])),
#             'codes_start': safe_pipe_join(G.nodes[u].get('code', [])),          # new
#             'rel':         data.get('rel',''),
#             'rela':        data.get('rela',''),
#             'sab_relation':data.get('sab_relation',''),
#             'cui_target':  v,
#             'name_target': G.nodes[v].get('name',''),
#             'sab_target':  safe_pipe_join(G.nodes[v].get('sab', [])),
#             'codes_target': safe_pipe_join(G.nodes[v].get('code', [])),         # new
#         })
#     edges_df = pd.DataFrame(edges_list)
#     edges_path = os.path.join(OUTPUT_DIR, 'kg_edges.csv')
#     edges_df.to_csv(edges_path, index=False)

#     log(f"Saved nodes: {len(nodes_df):,} -> {nodes_path}")
#     log(f"Saved edges: {len(edges_df):,} -> {edges_path}")

#     # -------- Persist graph (GraphML requires strings) --------
#     log("Saving GraphML and pickle…")
#     G_for_graphml = nx.DiGraph()
#     for n, data in G.nodes(data=True):
#         data_str = {}
#         for k, v in data.items():
#             if isinstance(v, list):
#                 data_str[k] = '|'.join(map(str, v))
#             else:
#                 data_str[k] = '' if v is None else str(v)
#         G_for_graphml.add_node(n, **data_str)

#     for u, v, data in G.edges(data=True):
#         data_str = {k: ('' if v is None else str(v)) for k, v in data.items()}
#         G_for_graphml.add_edge(u, v, **data_str)

#     nx.write_graphml(G_for_graphml, os.path.join(OUTPUT_DIR, 'medical_knowledge_graph.graphml'))
#     with open(os.path.join(OUTPUT_DIR, 'medical_knowledge_graph.pkl'), 'wb') as f:
#         pickle.dump(G, f)

#     return G, nodes_df, edges_df


# # -------------------- STATS + VIZ --------------------
# def generate_summary_statistics(G):
#     log("Generating summary statistics…")
#     num_nodes = G.number_of_nodes()
#     num_edges = G.number_of_edges()

#     degrees = [d for _, d in G.degree()]
#     stats = {
#         'num_nodes': num_nodes,
#         'num_edges': num_edges,
#         'avg_degree': float(np.mean(degrees)) if degrees else 0.0,
#         'median_degree': float(np.median(degrees)) if degrees else 0.0,
#         'max_degree': int(np.max(degrees)) if degrees else 0,
#         'min_degree': int(np.min(degrees)) if degrees else 0,
#     }

#     # Source distribution (from your analysis tags)
#     source_counts = defaultdict(int)
#     for _, data in G.nodes(data=True):
#         srcs = data.get('source', [])
#         if isinstance(srcs, str): srcs = srcs.split('|')
#         for s in srcs:
#             if s: source_counts[s] += 1
#     stats['source_counts'] = dict(source_counts)

#     # Semantic types distribution
#     sty_counts = defaultdict(int)
#     for _, data in G.nodes(data=True):
#         stys = data.get('semantic_type', [])
#         if isinstance(stys, str): stys = stys.split('|')
#         for sty in stys:
#             if sty: sty_counts[sty] += 1
#     stats['sty_counts'] = dict(sty_counts)

#     # Relationships
#     rel_counts = defaultdict(int)
#     for _, _, data in G.edges(data=True):
#         rel_counts[data.get('rel','Unknown') or 'Unknown'] += 1
#     stats['rel_counts'] = dict(rel_counts)

#     # SAB on edges
#     sab_counts = defaultdict(int)
#     for _, _, data in G.edges(data=True):
#         sab_counts[data.get('sab_relation','Unknown') or 'Unknown'] += 1
#     stats['sab_relation_counts'] = dict(sab_counts)

#     # Components
#     components = list(nx.weakly_connected_components(G)) if G.is_directed() else list(nx.connected_components(G))
#     stats['num_components'] = len(components)
#     stats['largest_component_size'] = len(max(components, key=len)) if components else 0

#     with open(os.path.join(OUTPUT_DIR, 'kg_statistics.json'), 'w') as f:
#         json.dump(stats, f, indent=2, cls=NumpyEncoder)

#     log("Summary statistics saved.")


# def create_visualizations(G):
#     log("Creating visualizations…")
#     viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')

#     # Degree distribution
#     degrees = [d for _, d in G.degree()]
#     if degrees:
#         plt.figure(figsize=(10,6))
#         sns.histplot(degrees, bins=50, kde=True)
#         plt.title('Node Degree Distribution')
#         plt.xlabel('Degree'); plt.ylabel('Frequency'); plt.yscale('log')
#         plt.tight_layout()
#         plt.savefig(os.path.join(viz_dir, 'node_degree_distribution.png'), dpi=300)
#         plt.close()

#     # Source distribution (from analyses)
#     source_counts = defaultdict(int)
#     for _, data in G.nodes(data=True):
#         srcs = data.get('source', [])
#         if isinstance(srcs, str): srcs = srcs.split('|')
#         for s in srcs:
#             if s: source_counts[s] += 1
#     if source_counts:
#         sdf = pd.DataFrame([{'source': k, 'count': v} for k, v in source_counts.items()]).sort_values('count', ascending=False)
#         plt.figure(figsize=(10,6))
#         sns.barplot(data=sdf, x='source', y='count')
#         plt.title('Node Distribution by Source'); plt.xlabel('Source'); plt.ylabel('Count')
#         plt.xticks(rotation=45); plt.tight_layout()
#         plt.savefig(os.path.join(viz_dir, 'source_distribution.png'), dpi=300)
#         plt.close()

#     # Semantic types (top 15)
#     sty_counts = defaultdict(int)
#     for _, data in G.nodes(data=True):
#         stys = data.get('semantic_type', [])
#         if isinstance(stys, str): stys = stys.split('|')
#         for sty in stys:
#             if sty: sty_counts[sty] += 1
#     if sty_counts:
#         top_sty = sorted(sty_counts.items(), key=lambda x: x[1], reverse=True)[:15]
#         sty_df = pd.DataFrame([{'semantic_type': k, 'count': v} for k, v in top_sty])
#         plt.figure(figsize=(12,6))
#         sns.barplot(data=sty_df, x='semantic_type', y='count')
#         plt.title('Top 15 Semantic Types'); plt.xlabel('Semantic Type'); plt.ylabel('Count')
#         plt.xticks(rotation=45, ha='right'); plt.tight_layout()
#         plt.savefig(os.path.join(viz_dir, 'semantic_type_distribution.png'), dpi=300)
#         plt.close()

#     # Relationship types (top 15)
#     rel_counts = defaultdict(int)
#     for _, _, data in G.edges(data=True):
#         rel_counts[data.get('rel','Unknown') or 'Unknown'] += 1
#     if rel_counts:
#         top_rel = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:15]
#         rel_df = pd.DataFrame([{'relationship': k, 'count': v} for k, v in top_rel])
#         plt.figure(figsize=(12,6))
#         sns.barplot(data=rel_df, x='relationship', y='count')
#         plt.title('Top 15 Relationship Types'); plt.xlabel('Relationship'); plt.ylabel('Count')
#         plt.xticks(rotation=45, ha='right'); plt.tight_layout()
#         plt.savefig(os.path.join(viz_dir, 'relationship_distribution.png'), dpi=300)
#         plt.close()

#     log("Visualizations saved.")


# def create_sample_subgraph_visualization(G):
#     log("Creating sample subgraph…")
#     viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')

#     degs = sorted([(n, d) for n, d in G.degree()], key=lambda x: x[1])
#     candidates = [n for n, d in degs if 5 <= d <= 15]
#     seed = candidates[0] if candidates else (next(iter(G.nodes())) if G.nodes else None)
#     if seed is None:
#         log("Graph empty; skip subgraph viz.")
#         return

#     neigh = list(G.successors(seed)) + list(G.predecessors(seed))
#     sub_nodes = [seed] + list(set(neigh))
#     H = G.subgraph(sub_nodes).copy()

#     from matplotlib.lines import Line2D

#     plt.figure(figsize=(12,12))
#     pos = nx.spring_layout(H, seed=42)

#     # Color by 'source' tag if present
#     colors = []
#     for n in H.nodes():
#         srcs = G.nodes[n].get('source', [])
#         if isinstance(srcs, str): srcs = srcs.split('|')
#         if 'ICD9' in srcs:   colors.append('red')
#         elif 'MED' in srcs:  colors.append('blue')
#         elif 'LAB' in srcs:  colors.append('green')
#         elif 'PROC' in srcs: colors.append('purple')
#         else:                colors.append('gray')

#     nx.draw_networkx_nodes(H, pos, node_size=220, alpha=0.85, node_color=colors)
#     nx.draw_networkx_edges(H, pos, alpha=0.5, arrows=True)

#     labels = {}
#     for n in H.nodes():
#         nm = G.nodes[n].get('name', n)
#         labels[n] = nm if len(nm) <= 20 else (nm[:17] + '…')
#     nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

#     legend = [
#         Line2D([0],[0], marker='o', color='w', markerfacecolor='red',    markersize=10, label='ICD9'),
#         Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',   markersize=10, label='Medication'),
#         Line2D([0],[0], marker='o', color='w', markerfacecolor='green',  markersize=10, label='Lab Test'),
#         Line2D([0],[0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Procedure'),
#         Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',   markersize=10, label='Other'),
#     ]
#     plt.legend(handles=legend, loc='upper right')
#     plt.axis('off')
#     title = G.nodes[seed].get('name', seed)
#     plt.title(f'Sample Subgraph: {title if len(title)<=60 else title[:57]+"…"}')
#     plt.tight_layout()
#     plt.savefig(os.path.join(viz_dir, 'sample_subgraph.png'), dpi=300)
#     plt.close()
#     log("Sample subgraph saved.")


# # -------------------- MAIN --------------------
# if __name__ == "__main__":
#     try:
#         log("Starting Knowledge Graph construction")

#         cui_sources, all_cuis = load_analysis_files()
#         G, nodes_df, edges_df = build_knowledge_graph(cui_sources, all_cuis)

#         # Sanity checks
#         # assert nodes_df['cui'].str.match(r'^C\\d+$').all(), "Non-CUI values found in nodes.cui"
#         # assert edges_df['cui_start'].str.match(r'^C\\d+$').all(), "Non-CUI in edges.cui_start"
#         # assert edges_df['cui_target'].str.match(r'^C\\d+$').all(), "Non-CUI in edges.cui_target"

#         # Stats + Viz
#         generate_summary_statistics(G)
#         create_visualizations(G)
#         create_sample_subgraph_visualization(G)

#         log("Knowledge graph construction complete!")
#     except Exception as e:
#         import traceback
#         log(f"ERROR: {str(e)}")
#         log(traceback.format_exc())
#         print(f"ERROR: {str(e)}")
#         print("See log file for details.")


# import os, re, json, pickle, argparse
# from typing import Dict, List, Set, Tuple
# from collections import defaultdict
# from datetime import datetime
# import csv

# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import networkx as nx

# # -------------------- Logging --------------------
# def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# def make_logger(out_dir: str):
#     ensure_dir(out_dir)
#     log_file = os.path.join(out_dir, f'kg_build_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
#     def _log(msg: str):
#         ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         with open(log_file, 'a') as f: f.write(f'[{ts}] {msg}\n')
#         print(msg)
#     return _log

# # -------------------- ICD-9 utils --------------------
# def _strip(x: str) -> str:
#     return re.sub(r"\s+","", str(x or "")).upper().rstrip(".")

# def _clean_str(x) -> str:
#     if x is None: return ""
#     if isinstance(x, float) and np.isnan(x): return ""
#     s = str(x)
#     if s.lower() == "nan": return ""
#     return s

# def format_icd9_dx(code: str) -> str:
#     c = _strip(code)
#     if not c: return ""
#     if c[0].isdigit(): return c[:3]+"."+c[3:] if len(c)>3 and "." not in c else c
#     if c[0] == "V":   return c[:3]+"."+c[3:] if len(c)>3 and "." not in c else c
#     if c[0] == "E":   return c[:4]+"."+c[4:] if len(c)>4 and "." not in c else c
#     return c

# def format_icd9_proc(code: str) -> str:
#     c = _strip(code)
#     if c.startswith("PRO_"): c = c[4:]
#     if not c: return ""
#     if c[0].isdigit(): return c[:2]+"."+c[2:] if len(c)>2 and "." not in c else c
#     return c

# _proc_pat = re.compile(r"^\d{2}(?:\.\d{1,2})?$")
# def is_proc(code: str) -> bool: return bool(_proc_pat.match(_strip(code)))

# def is_dx(code: str) -> bool:
#     c = _strip(code)
#     if is_proc(c): return False
#     return bool(re.match(r"^(?:\d{3}(?:\.\d{1,2})?|V\d{2}(?:\.\d{1,2})?|E\d{3}(?:\.\d)?)$", c))

# # numeric keys for ranges
# def parse_icd9_key(code: str) -> Tuple[str,int,int]:
#     c = _strip(code)
#     if is_proc(c):
#         whole, frac = (c.split('.')+[ ""])[0], (c.split('.')+[ ""])[1]
#         frac = (frac + "00")[:2]
#         return ("PROC", int(whole)*100 + int(frac or "0"), 100)
#     if c.startswith("E"):
#         base = c[1:]; parts = base.split('.')
#         whole, frac = parts[0], (parts[1] if len(parts)>1 else "")
#         frac = (frac + "0")[:1]
#         return ("DX_E", int(whole)*10 + int(frac or "0"), 10)
#     # numeric or V
#     base = c[1:] if c.startswith("V") else c
#     parts = base.split('.')
#     whole, frac = parts[0], (parts[1] if len(parts)>1 else "")
#     frac = (frac + "00")[:2]
#     return ("DX_NUMV", int(whole)*100 + int(frac or "0"), 100)

# def expand_json_spec_to_matcher(spec: str):
#     s = _strip(spec)
#     if "-" in s:
#         left, right = s.split("-", 1)
#         def norm(ep):
#             dx = format_icd9_dx(ep); pr = format_icd9_proc(ep)
#             if is_dx(dx): return parse_icd9_key(dx)
#             if is_proc(pr): return parse_icd9_key(pr)
#             return parse_icd9_key(ep)
#         return ('range', norm(left), norm(right))
#     else:
#         dx = format_icd9_dx(s); pr = format_icd9_proc(s)
#         if is_dx(dx):  return ('exact', dx)
#         if is_proc(pr):return ('exact', pr)
#         return ('exact', s)

# def code_in_range(code: str, start_key, end_key) -> bool:
#     cat_c, val_c, _ = parse_icd9_key(code)
#     cat_s, val_s, _ = start_key
#     cat_e, val_e, _ = end_key
#     if cat_s != cat_e or cat_c != cat_s: return False
#     return val_s <= val_c <= val_e

# # -------------------- Load master/dataset --------------------
# def load_icd9_master_list(icd9_pkl: str) -> Set[str]:
#     obj = pickle.load(open(icd9_pkl, "rb"))
#     if isinstance(obj, pd.DataFrame):
#         col = 'icd_code' if 'icd_code' in obj.columns else obj.columns[0]
#         values = obj[col].astype(str).tolist()
#     elif isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
#         values = list(map(str, obj))
#     else:
#         raise ValueError("icd9.pkl must be a DataFrame or a list/array/Series of codes.")
#     out = set()
#     for c in set(values):
#         fdx = format_icd9_dx(c)
#         if is_dx(fdx): out.add(fdx)
#     return out

# def load_icd9_proc_master_list(icd9proc_pkl: str) -> Set[str]:
#     obj = pickle.load(open(icd9proc_pkl, "rb"))
#     if isinstance(obj, pd.DataFrame):
#         col = 'icd_code' if 'icd_code' in obj.columns else obj.columns[0]
#         values = obj[col].astype(str).tolist()
#     elif isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
#         values = list(map(str, obj))
#     else:
#         raise ValueError("icd9proc.pkl must be a DataFrame or a list/array/Series of codes.")
#     out = set()
#     for c in set(values):
#         fpr = format_icd9_proc(c)
#         if is_proc(fpr): out.add(fpr)
#     return out

# def load_dataset_codes(dataset_pkl: str) -> Tuple[Set[str], Set[str]]:
#     if not dataset_pkl: return set(), set()
#     df = pickle.load(open(dataset_pkl, "rb"))
#     if not isinstance(df, pd.DataFrame):
#         raise ValueError("dataset PKL must be a DataFrame.")
#     dx_raw, pr_raw = [], []
#     if 'icd_code' in df.columns:
#         for x in df['icd_code']:
#             if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
#                 dx_raw.extend(map(str, x))
#     if 'pro_code' in df.columns:
#         for x in df['pro_code']:
#             if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
#                 pr_raw.extend(map(str, x))
#     dx = set(format_icd9_dx(c) for c in dx_raw if format_icd9_dx(c))
#     pr = set(format_icd9_proc(c) for c in pr_raw if format_icd9_proc(c))
#     return dx, pr

# # -------------------- Reverse CUI->ICD9 --------------------
# def reverse_cui_to_icd9_separate(cui_to_codes: Dict[str, List[str]],
#                                  master_dx: Set[str],
#                                  master_pr: Set[str]):
#     dx_map = defaultdict(set); pr_map = defaultdict(set)
#     compiled = {cui: [expand_json_spec_to_matcher(s) for s in specs]
#                 for cui, specs in cui_to_codes.items()}

#     for cui, specs in tqdm(compiled.items(), desc="Reversing CUI->ICD9 (DX/PROC)"):
#         exact_dx, exact_pr, ranges = [], [], []
#         for kind, *rest in specs:
#             if kind == 'exact':
#                 code = rest[0]
#                 if is_proc(code): exact_pr.append(code)
#                 elif is_dx(code): exact_dx.append(code)
#             else:
#                 ranges.append(tuple(rest))
#         # exacts
#         for code in exact_dx:
#             if code in master_dx: dx_map[code].add(cui)
#         for code in exact_pr:
#             if code in master_pr: pr_map[code].add(cui)
#         # ranges
#         for (start_key, end_key) in ranges:
#             cat = start_key[0]
#             pool = master_pr if cat == "PROC" else master_dx
#             for code in pool:
#                 if code_in_range(code, start_key, end_key):
#                     (pr_map if cat == "PROC" else dx_map)[code].add(cui)

#     return {k: sorted(v) for k,v in dx_map.items()}, {k: sorted(v) for k,v in pr_map.items()}

# # -------------------- ATC / LOINC (MRCONSO) --------------------
# def read_conso_subset(path: str, usecols_idx: List[int], chunksize: int):
#     # robust parser: python engine, no quoting, skip bad lines
#     return pd.read_csv(
#         path,
#         sep='|',
#         header=None,
#         usecols=usecols_idx,    # numeric indices
#         dtype=str,
#         chunksize=chunksize,
#         quoting=csv.QUOTE_NONE,
#         on_bad_lines='skip',
#         low_memory=False
#     )

# USECOLS_MIN = ['CUI','LAT','TS','SAB','TTY','CODE','STR','SUPPRESS']
# USECOLS_IDX = [0, 1, 2, 11, 12, 13, 14, 16]

# def build_code2cui_generic(mrconso_path: str, target_sab: str, log, keep_ts_p_only=True, chunksize=1_000_000):
#     code2cuis = defaultdict(set)
#     code2name = {}
#     for ch in tqdm(read_conso_subset(mrconso_path, USECOLS_IDX, chunksize), desc=f"SAB={target_sab}", unit="chunk"):
#         ch.columns = USECOLS_MIN
#         ch = ch[(ch['LAT'] == 'ENG') & (ch['SAB'] == target_sab)]
#         ch = ch[(ch['SUPPRESS'] != 'O')]
#         if keep_ts_p_only:
#             chp = ch[ch['TS'] == 'P']
#             if not chp.empty: ch = chp
#         if ch.empty: continue
#         for _, r in ch.iterrows():
#             code = _strip(_clean_str(r.get('CODE')))
#             if not code: continue
#             cui  = _clean_str(r.get('CUI'))
#             name = _clean_str(r.get('STR'))
#             if not cui: continue
#             code2cuis[code].add(cui)
#             if code not in code2name:
#                 code2name[code] = name
#     code2cuis = {k: sorted(v) for k, v in code2cuis.items()}
#     log(f"[{target_sab}] codes: {len(code2cuis):,}")
#     return code2cuis, code2name

# # -------------------- MRREL / MRSTY + names/sabs/codes --------------------
# def load_mrrel_filtered(mrrel_path: str, allowed_sabs: List[str], restrict_to_cuis: Set[str], log):
#     usecols_idx = [0, 3, 4, 7, 10]
#     colnames    = ['CUI1','REL','CUI2','RELA','SAB']
#     out = []
#     it = pd.read_csv(
#         mrrel_path,
#         sep='|',
#         header=None,
#         usecols=usecols_idx,
#         dtype=str,
#         chunksize=1_000_000,
#         quoting=csv.QUOTE_NONE,
#         on_bad_lines='skip',
#         low_memory=False
#     )
#     for ch in it:
#         ch.columns = colnames
#         ch = ch[ch['SAB'].isin(allowed_sabs)]
#         if restrict_to_cuis:
#             ch = ch[ch['CUI1'].isin(restrict_to_cuis) | ch['CUI2'].isin(restrict_to_cuis)]
#         if not ch.empty: out.append(ch)
#     rel = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=colnames)
#     log(f"MRREL filtered: {len(rel):,} (SAB∈{allowed_sabs})")
#     return rel

# def load_mrconso_names_sabs_codes(mrconso_path: str, cuis: Set[str], log):
#     # indices: 0 CUI, 1 LAT, 6 ISPREF, 11 SAB, 12 TTY, 13 CODE, 14 STR
#     usecols_idx = [0, 1, 6, 11, 12, 13, 14]
#     colnames    = ['CUI','LAT','ISPREF','SAB','TTY','CODE','STR']
#     name_map, sab_map = {}, defaultdict(set)
#     code_map = defaultdict(set)
#     tty_priority = {'PT':0,'PN':1,'FN':2}
#     it = pd.read_csv(
#         mrconso_path,
#         sep='|',
#         header=None,
#         usecols=usecols_idx,
#         dtype=str,
#         chunksize=1_000_000,
#         quoting=csv.QUOTE_NONE,
#         on_bad_lines='skip',
#         low_memory=False
#     )
#     for ch in it:
#         ch.columns = colnames
#         ch = ch[(ch['LAT'] == 'ENG') & (ch['CUI'].isin(cuis))]
#         if ch.empty: continue
#         # collect SABs and CODEs
#         for _, r in ch.iterrows():
#             cui  = _clean_str(r.get('CUI'))
#             sab  = _clean_str(r.get('SAB'))
#             code = _clean_str(r.get('CODE'))
#             if cui:
#                 if sab:  sab_map[cui].add(sab)
#                 if code: code_map[cui].add(code)
#         # best name per CUI
#         ch['is_pref']  = ch['ISPREF'].apply(lambda x: 1 if _clean_str(x) == 'Y' else 0)
#         ch['tty_rank'] = ch['TTY'].apply(lambda t: { 'PT':0,'PN':1,'FN':2 }.get(_clean_str(t), 99))
#         # ensure STR is string
#         ch['STR'] = ch['STR'].apply(_clean_str)
#         best = ch.sort_values(['CUI','is_pref','tty_rank']).drop_duplicates('CUI', keep='first')
#         for cui, nm in zip(best['CUI'], best['STR']):
#             if cui and nm and cui not in name_map:
#                 name_map[cui] = nm
#     # finalize as sorted string lists
#     sab_map  = {k: sorted(map(str, v)) for k,v in sab_map.items()}
#     code_map = {k: sorted(map(str, v)) for k,v in code_map.items()}
#     log(f"Names/SABs/CODEs resolved for {len(name_map):,}/{len(cuis):,}")
#     return name_map, sab_map, code_map

# def load_mrsty_for_cuis(mrsty_path: str, cuis: Set[str], log):
#     usecols_idx = [0, 3]  # CUI, STY
#     colnames    = ['CUI','STY']
#     it = pd.read_csv(
#         mrsty_path,
#         sep='|',
#         header=None,
#         usecols=usecols_idx,
#         dtype=str,
#         chunksize=500_000,
#         quoting=csv.QUOTE_NONE,
#         on_bad_lines='skip',
#         low_memory=False
#     )
#     rows = []
#     for ch in it:
#         ch.columns = colnames
#         ch = ch[ch['CUI'].isin(cuis)]
#         if not ch.empty: rows.append(ch)
#     sty = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=colnames)
#     return sty.groupby('CUI')['STY'].apply(lambda s: sorted(set(_clean_str(x) for x in s if _clean_str(x)))).to_dict()

# # -------------------- KG build --------------------
# def safe_pipe_join(v):
#     if isinstance(v, list): return '|'.join(map(str, v))
#     return v if isinstance(v, str) else ''

# def build_kg(umls_dir: str,
#              out_dir: str,
#              icd9_dx_map: Dict[str, List[str]],
#              icd9_pr_map: Dict[str, List[str]],
#              atc_map: Dict[str, List[str]],
#              loinc_map: Dict[str, List[str]],
#              target_vocabs: List[str],
#              log):
#     mrconso = os.path.join(umls_dir, "MRCONSO.RRF")
#     mrrel   = os.path.join(umls_dir, "MRREL.RRF")
#     mrsty   = os.path.join(umls_dir, "MRSTY.RRF")
#     for p in (mrconso, mrrel, mrsty):
#         if not os.path.exists(p): raise FileNotFoundError(f"Missing {p}")

#     seed_cuis = set()
#     for m in (icd9_dx_map, icd9_pr_map, atc_map, loinc_map):
#         for lst in m.values(): seed_cuis.update(lst)
#     log(f"Seed CUIs: {len(seed_cuis):,}")

#     # names/sabs/codes include all SABs the CUI appears under (incl. SNOMEDCT_US)
#     name_map, sab_map, code_map = load_mrconso_names_sabs_codes(mrconso, seed_cuis, log)

#     # edges only from allowed SABs (includes SNOMEDCT_US, as requested)
#     rel_df = load_mrrel_filtered(mrrel, allowed_sabs=target_vocabs, restrict_to_cuis=seed_cuis, log=log)
#     rel_df = rel_df[(rel_df['CUI1'].isin(seed_cuis)) & (rel_df['CUI2'].isin(seed_cuis))].reset_index(drop=True)

#     sty_map = load_mrsty_for_cuis(mrsty, seed_cuis, log)

#     G = nx.DiGraph()
#     for cui in tqdm(seed_cuis, desc="Nodes"):
#         G.add_node(cui,
#                    name=name_map.get(cui,"Unknown"),
#                    sab=sab_map.get(cui,[]),
#                    code=code_map.get(cui,[]),              # list; serialize later
#                    semantic_type=sty_map.get(cui,[]))
#     for _, r in tqdm(rel_df.iterrows(), total=len(rel_df), desc="Edges"):
#         u, v = _clean_str(r.get('CUI1')), _clean_str(r.get('CUI2'))
#         if u in G and v in G:
#             G.add_edge(u, v,
#                        rel=_clean_str(r.get('REL')),
#                        rela=_clean_str(r.get('RELA')),
#                        sab_relation=_clean_str(r.get('SAB')))

#     # write CSVs
#     nodes = [{
#         'cui': n,
#         'name': G.nodes[n].get('name',''),
#         'sab':  safe_pipe_join(G.nodes[n].get('sab',[])),
#         'code': safe_pipe_join(G.nodes[n].get('code',[])),               # include CODE list
#         'semantic_type': safe_pipe_join(G.nodes[n].get('semantic_type',[]))
#     } for n in G.nodes()]
#     pd.DataFrame(nodes).to_csv(os.path.join(out_dir,'kg_nodes.csv'), index=False)

#     edges = [{
#         'cui_start':   u,
#         'name_start':  G.nodes[u].get('name',''),
#         'sab_start':   safe_pipe_join(G.nodes[u].get('sab',[])),
#         'code_start':  safe_pipe_join(G.nodes[u].get('code',[])),        # representative codes (pipe-joined)
#         'rel':         d.get('rel',''),
#         'rela':        d.get('rela',''),
#         'sab_relation':d.get('sab_relation',''),
#         'cui_target':  v,
#         'name_target': G.nodes[v].get('name',''),
#         'sab_target':  safe_pipe_join(G.nodes[v].get('sab',[])),
#         'code_target': safe_pipe_join(G.nodes[v].get('code',[]))
#     } for u, v, d in G.edges(data=True)]
#     pd.DataFrame(edges).to_csv(os.path.join(out_dir,'kg_edges.csv'), index=False)

#     # graphml + pickle (serialize lists to pipes)
#     Gml = nx.DiGraph()
#     for n,data in G.nodes(data=True):
#         data_str = {}
#         for k,v in data.items():
#             if isinstance(v, list):
#                 data_str[k] = '|'.join(map(str, v))
#             else:
#                 data_str[k] = '' if v is None else str(v)
#         Gml.add_node(n, **data_str)
#     for u,v,data in G.edges(data=True):
#         Gml.add_edge(u,v, **{k: ('' if v is None else str(v)) for k,v in data.items()})
#     nx.write_graphml(Gml, os.path.join(out_dir,'medical_knowledge_graph.graphml'))
#     with open(os.path.join(out_dir,'medical_knowledge_graph.pkl'),'wb') as f: pickle.dump(G,f)
#     return G

# # -------------------- Coverage --------------------
# def coverage(all_codes: Set[str], mapping: Dict[str,List[str]]):
#     have = {c for c in all_codes if c in mapping}
#     miss = sorted(all_codes - have)
#     pct  = 100.0 * len(have) / max(1,len(all_codes))
#     return {"total":len(all_codes), "mapped":len(have), "missing":len(miss), "coverage_pct":pct, "missing_sample":miss[:25]}

# # -------------------- MAIN --------------------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description="ICD9 (from JSON + master lists), ATC/LNC (from MRCONSO) → CUI + KG")
#     ap.add_argument("--umls-dir", required=True)
#     ap.add_argument("--out-dir", required=True)
#     ap.add_argument("--cui-to-icd9-json", required=True)  # CUI -> [codes and/or ranges]
#     ap.add_argument("--icd9-dx-pkl", required=True)       # master diagnosis list (icd9.pkl)
#     ap.add_argument("--icd9-proc-pkl", required=True)     # master procedure list (icd9proc.pkl)
#     ap.add_argument("--dataset-pkl", default="")          # merged_icd9.pkl (optional)
#     ap.add_argument("--with_names", action="store_true")  # corrected flag name
#     ap.add_argument("--target-vocabs", default="ICD9CM,LNC,ATC,SNOMEDCT_US")
#     ap.add_argument("--chunksize", type=int, default=1_000_000)
#     args = ap.parse_args()

#     log = make_logger(args.out_dir)
#     ensure_dir(os.path.join(args.out_dir,"visualizations"))

#     # Master universes
#     log("Loading master ICD-9 DX list…")
#     master_dx = load_icd9_master_list(args.icd9_dx_pkl)
#     log(f"Master DX codes: {len(master_dx):,}")

#     log("Loading master ICD-9 PROC list…")
#     master_pr = load_icd9_proc_master_list(args.icd9_proc_pkl)
#     log(f"Master PROC codes: {len(master_pr):,}")

#     # Dataset (optional)
#     ds_dx, ds_pr = set(), set()
#     if args.dataset_pkl:
#         log("Loading dataset per-visit codes…")
#         ds_dx, ds_pr = load_dataset_codes(args.dataset_pkl)
#         log(f"Dataset DX codes: {len(ds_dx):,} | PROC codes: {len(ds_pr):,}")

#     # Reverse mapping from your JSON
#     log(f"Reading {args.cui_to_icd9_json}")
#     cui_to_codes = json.load(open(args.cui_to_icd9_json, "r"))
#     icd9_dx_map, icd9_pr_map = reverse_cui_to_icd9_separate(cui_to_codes, master_dx, master_pr)
#     pickle.dump(icd9_dx_map, open(os.path.join(args.out_dir,"code2cui_icd9_dx.pkl"),"wb"))
#     pickle.dump(icd9_pr_map, open(os.path.join(args.out_dir,"code2cui_icd9_proc.pkl"),"wb"))
#     log(f"DX→CUI codes: {len(icd9_dx_map):,} | PROC→CUI codes: {len(icd9_pr_map):,}")

#     # ATC & LOINC
#     mrconso = os.path.join(args.umls_dir,"MRCONSO.RRF")
#     log("Building ATC → CUI…")
#     atc_map, atc_name = build_code2cui_generic(mrconso, "ATC", log, keep_ts_p_only=True, chunksize=args.chunksize)
#     pickle.dump(atc_map, open(os.path.join(args.out_dir,"code2cui_atc.pkl"),"wb"))
#     if args.with_names: pickle.dump(atc_name, open(os.path.join(args.out_dir,"code2name_atc.pkl"),"wb"))

#     log("Building LOINC (LNC) → CUI…")
#     loinc_map, loinc_name = build_code2cui_generic(mrconso, "LNC", log, keep_ts_p_only=True, chunksize=args.chunksize)
#     pickle.dump(loinc_map, open(os.path.join(args.out_dir,"code2cui_loinc.pkl"),"wb"))
#     if args.with_names: pickle.dump(loinc_name, open(os.path.join(args.out_dir,"code2name_loinc.pkl"),"wb"))

#     # Coverage
#     cov = {
#         "master_dx": coverage(master_dx, icd9_dx_map),
#         "master_proc": coverage(master_pr, icd9_pr_map),
#         "dataset_dx": coverage(ds_dx, icd9_dx_map) if ds_dx else {},
#         "dataset_proc": coverage(ds_pr, icd9_pr_map) if ds_pr else {}
#     }
#     json.dump(cov, open(os.path.join(args.out_dir,"icd9_coverage.json"),"w"), indent=2)
#     log("Coverage written to icd9_coverage.json")

#     # Build KG
#     TARGETS = [s.strip() for s in args.target_vocabs.split(",") if s.strip()]
#     log(f"Building KG with MRREL SABs: {TARGETS}")
#     G = build_kg(
#         umls_dir=args.umls_dir,
#         out_dir=args.out_dir,
#         icd9_dx_map=icd9_dx_map,
#         icd9_pr_map=icd9_pr_map,
#         atc_map=atc_map,
#         loinc_map=loinc_map,
#         target_vocabs=TARGETS,
#         log=log
#     )

#     # Summary
#     stats = {
#         "counts": {
#             "icd9_dx_codes": len(icd9_dx_map),
#             "icd9_proc_codes": len(icd9_pr_map),
#             "atc_codes": len(atc_map),
#             "loinc_codes": len(loinc_map),
#             "kg_nodes": G.number_of_nodes(),
#             "kg_edges": G.number_of_edges()
#         },
#         "mrrel_allowed_sabs": TARGETS
#     }
#     json.dump(stats, open(os.path.join(args.out_dir,"build_summary.json"),"w"), indent=2)
#     log("Done.")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a UMLS-based medical KG with CUI→CUI edges only.

Nodes CSV schema:
  cui, name, sab, code, semantic_type

Edges CSV schema:
  cui_start, name_start, sab_start, codes_start, rel, rela, sab_relation, cui_target, name_target, sab_target, codes_target
"""

import os
import json
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- CONFIG --------------------
UMLS_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META'
MRCONSO = os.path.join(UMLS_DIR, 'MRCONSO.RRF')
MRREL   = os.path.join(UMLS_DIR, 'MRREL.RRF')
MRSTY   = os.path.join(UMLS_DIR, 'MRSTY.RRF')

ANALYSIS_DIR   = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG'
OUTPUT_DIR     = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output3'

ICD9_ANALYSIS  = os.path.join(ANALYSIS_DIR, 'mapping_analysis', 'simple_icd9_cui_sab_mapping.csv')
LAB_ANALYSIS   = os.path.join(ANALYSIS_DIR, 'lab_test_analysis', 'simple_loinc_cui_sab_mapping.csv')
MED_ANALYSIS   = os.path.join(ANALYSIS_DIR, 'med_analysis', 'simple_med_cui_sab_mapping.csv')
PROC_ANALYSIS  = os.path.join(ANALYSIS_DIR, 'procedure_analysis', 'simple_proc_cui_sab_mapping.csv')

TARGET_VOCABS  = ['ICD9CM', 'LNC', 'ATC', 'SNOMEDCT_US']  # tweak as needed

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'caches'), exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, f'kg_build_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

def log(message: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(f'[{ts}] {message}\n')
    print(message)


# -------------------- HELPERS --------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def safe_pipe_join(value):
    if isinstance(value, list):
        return '|'.join(map(str, value))
    return value if isinstance(value, str) else ''


# -------------------- LOADERS (read by integer indices, rename later) --------------------
def load_mrconso_subset(target_vocabs=TARGET_VOCABS, langs=['ENG']):
    """
    Load preferred English name per CUI and individual SAB-code combinations
    for the chosen vocabularies. Returns expanded rows per CUI-SAB-CODE combination.

    MRCONSO field indices (2025AA):
      0:CUI, 1:LAT, 6:ISPREF, 11:SAB, 12:TTY, 13:CODE, 14:STR
    """
    log(f"Loading MRCONSO for vocabs: {', '.join(target_vocabs)}")
    cache_file = os.path.join(OUTPUT_DIR, 'caches', 'mrconso_expanded.pkl')
    if os.path.exists(cache_file):
        log(f"MRCONSO cache: {cache_file}")
        return pd.read_pickle(cache_file)

    usecols_idx = [0, 1, 6, 11, 12, 13, 14]  # keep order aligned with names below
    colnames    = ['CUI','LAT','ISPREF','SAB','TTY','CODE','STR']

    it = pd.read_csv(
        MRCONSO,
        sep='|',
        header=None,
        usecols=usecols_idx,
        dtype=str,
        chunksize=1_000_000
    )

    chunks = []
    for ch in it:
        ch.columns = colnames  # rename after reading the subset
        ch = ch[(ch['LAT'].isin(langs)) & (ch['SAB'].isin(target_vocabs))]
        if not ch.empty:
            chunks.append(ch)

    conso = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=colnames)

    # Get preferred name per CUI (will be replicated across rows)
    tty_priority = {'PT': 0, 'PN': 1, 'FN': 2}
    conso['is_pref']  = (conso['ISPREF'] == 'Y').astype(int)
    conso['tty_rank'] = conso['TTY'].map(lambda t: tty_priority.get(t, 99))

    # Get best name per CUI
    conso_sorted = conso.sort_values(['CUI','is_pref','tty_rank']).drop_duplicates('CUI', keep='first')
    cui_to_name = dict(zip(conso_sorted['CUI'], conso_sorted['STR']))

    # Create expanded dataset: one row per CUI-SAB-CODE combination
    # Keep only rows with valid CODE values
    conso_filtered = conso[conso['CODE'].notna() & (conso['CODE'] != '') & (conso['CODE'] != 'nan')].copy()
    
    # Add the preferred name to each row
    conso_filtered['name'] = conso_filtered['CUI'].map(cui_to_name)
    
    # Select final columns and remove duplicates
    conso_final = conso_filtered[['CUI', 'name', 'SAB', 'CODE']].drop_duplicates().reset_index(drop=True)
    
    conso_final.to_pickle(cache_file)
    log(f"MRCONSO expanded: {len(conso_final):,} CUI-SAB-CODE combinations")
    return conso_final


def load_mrrel_subset(target_vocabs=TARGET_VOCABS, cuis=None):
    """
    Load MRREL relations filtered by SAB vocabularies; optionally restricted to CUIs.

    MRREL field indices (2025AA):
      0:CUI1, 3:REL, 4:CUI2, 7:RELA, 10:SAB
    """
    log("Loading MRREL (int-indexed, then renamed)")
    cache_file = os.path.join(OUTPUT_DIR, 'caches', 'mrrel_subset.pkl')
    if os.path.exists(cache_file) and cuis is None:
        log(f"MRREL cache: {cache_file}")
        return pd.read_pickle(cache_file)

    usecols_idx = [0, 3, 4, 7, 10]  # important: sorted order as pandas returns
    colnames    = ['CUI1','REL','CUI2','RELA','SAB']

    it = pd.read_csv(
        MRREL,
        sep='|',
        header=None,
        usecols=usecols_idx,
        dtype=str,
        chunksize=1_000_000
    )

    out_chunks = []
    for ch in it:
        ch.columns = colnames
        ch = ch[ch['SAB'].isin(target_vocabs)]
        if cuis is not None:
            ch = ch[ch['CUI1'].isin(cuis) | ch['CUI2'].isin(cuis)]
        if not ch.empty:
            out_chunks.append(ch)

    rel = pd.concat(out_chunks, ignore_index=True) if out_chunks else pd.DataFrame(columns=colnames)
    if cuis is None:
        rel.to_pickle(cache_file)
    log(f"MRREL filtered: {len(rel):,} rows")
    return rel


def load_mrsty_subset(cuis=None):
    """
    Load MRSTY semantic types; optionally restricted to CUIs.

    MRSTY field indices (2025AA):
      0:CUI, 1:TUI, 3:STY
    """
    log("Loading MRSTY (int-indexed, then renamed)")
    cache_file = os.path.join(OUTPUT_DIR, 'caches', 'mrsty_subset.pkl')
    if os.path.exists(cache_file) and cuis is None:
        log(f"MRSTY cache: {cache_file}")
        return pd.read_pickle(cache_file)

    usecols_idx = [0, 1, 3]
    colnames    = ['CUI','TUI','STY']

    it = pd.read_csv(
        MRSTY,
        sep='|',
        header=None,
        usecols=usecols_idx,
        dtype=str,
        chunksize=500_000
    )

    chunks = []
    for ch in it:
        ch.columns = colnames
        if cuis is not None:
            ch = ch[ch['CUI'].isin(cuis)]
        if not ch.empty:
            chunks.append(ch)

    sty = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=colnames)
    if cuis is None:
        sty.to_pickle(cache_file)
    log(f"MRSTY: {len(sty):,} rows")
    return sty


# -------------------- ANALYSIS INPUTS --------------------
def load_analysis_files():
    """
    Load precomputed domain CUI lists (optional); returns dict of sources -> set(CUI), and union set.
    """
    log("Loading analysis-domain CUI files")
    cui_sources = {}

    def read_cuis(path, key):
        if os.path.exists(path):
            df = pd.read_csv(path)
            s = set(df['cui'].astype(str))
            log(f"  {key}: {len(s):,} CUIs")
            return s
        else:
            log(f"  WARN: Missing file: {path}")
            return set()

    cui_sources['ICD9'] = read_cuis(ICD9_ANALYSIS, 'ICD9')
    cui_sources['LAB']  = read_cuis(LAB_ANALYSIS,  'LAB')
    cui_sources['MED']  = read_cuis(MED_ANALYSIS,  'MED')
    cui_sources['PROC'] = read_cuis(PROC_ANALYSIS, 'PROC')

    all_cuis = set().union(*cui_sources.values()) if cui_sources else set()
    log(f"Total seed CUIs from analyses: {len(all_cuis):,}")
    return cui_sources, all_cuis


# -------------------- BUILD KG --------------------
def build_knowledge_graph(cui_sources, all_cuis):
    log("Building KG (CUI-only)")
    conso_df   = load_mrconso_subset()
    umls_cuis  = set(conso_df['CUI'])
    combined   = set(all_cuis) | umls_cuis

    rel_df     = load_mrrel_subset(target_vocabs=TARGET_VOCABS)
    rel_cuis   = set(rel_df['CUI1']) | set(rel_df['CUI2'])
    combined  |= rel_cuis

    sty_df     = load_mrsty_subset(combined)
    sty_map    = sty_df.groupby('CUI')['STY'].apply(lambda s: sorted(set(s))).to_dict()

    # Create mappings from the expanded MRCONSO data
    name_map   = dict(zip(conso_df['CUI'], conso_df['name']))

    # For NetworkX graph, we still need aggregated attributes per CUI
    sab_map = conso_df.groupby('CUI')['SAB'].apply(lambda s: sorted(set(s))).to_dict()
    code_map = conso_df.groupby('CUI')['CODE'].apply(lambda s: sorted(set(s))).to_dict()

    # Build graph
    G = nx.DiGraph()

    # Node sources from analyses
    cui_to_sources = defaultdict(list)
    for source, cuiset in cui_sources.items():
        for cui in cuiset:
            cui_to_sources[cui].append(source)

    log("Adding nodes…")
    for cui in tqdm(combined, desc="Nodes"):
        G.add_node(
            cui,
            name=name_map.get(cui, "Unknown"),
            sab=sab_map.get(cui, []),
            code=code_map.get(cui, []),
            semantic_type=sty_map.get(cui, []),
            source=cui_to_sources.get(cui, [])
        )

    log("Adding edges…")
    edges_added = 0
    for _, r in tqdm(rel_df.iterrows(), total=len(rel_df), desc="Edges"):
        c1, c2 = r['CUI1'], r['CUI2']
        if c1 in G and c2 in G:
            G.add_edge(
                c1, c2,
                rel=r.get('REL','') or '',
                rela=r.get('RELA','') if pd.notna(r.get('RELA')) else '',
                sab_relation=r.get('SAB','') or ''
            )
            edges_added += 1

    log(f"Graph summary: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges (added {edges_added:,})")

    # -------- Save EXPANDED node/edge CSVs --------
    log("Saving expanded nodes/edges CSV…")

    # Nodes CSV: one row per CUI-SAB-CODE combination
    nodes_expanded = []
    for _, row in conso_df.iterrows():
        cui = row['CUI']
        if cui in G:  # Only include CUIs that are in our graph
            nodes_expanded.append({
                'cui': cui,
                'name': row['name'],
                'sab': row['SAB'],           # Single SAB per row
                'code': row['CODE'],         # Single CODE per row  
                'semantic_type': safe_pipe_join(G.nodes[cui].get('semantic_type', [])),
            })
    
    nodes_df = pd.DataFrame(nodes_expanded)
    nodes_path = os.path.join(OUTPUT_DIR, 'kg_nodes.csv')
    nodes_df.to_csv(nodes_path, index=False)

    # Edges CSV: expanded with individual SAB-CODE combinations for start/target
    edges_expanded = []
    for u, v, data in G.edges(data=True):
        # Get all SAB-CODE combinations for source and target
        u_combinations = conso_df[conso_df['CUI'] == u][['SAB', 'CODE']].drop_duplicates()
        v_combinations = conso_df[conso_df['CUI'] == v][['SAB', 'CODE']].drop_duplicates()
        
        # If no combinations found in conso_df, create default entry
        if u_combinations.empty:
            u_combinations = pd.DataFrame([{'SAB': '', 'CODE': ''}])
        if v_combinations.empty:
            v_combinations = pd.DataFrame([{'SAB': '', 'CODE': ''}])
        
        # Create cross-product of all combinations
        for _, u_combo in u_combinations.iterrows():
            for _, v_combo in v_combinations.iterrows():
                edges_expanded.append({
                    'cui_start':    u,
                    'name_start':   G.nodes[u].get('name',''),
                    'sab_start':    u_combo['SAB'],
                    'codes_start':  u_combo['CODE'],
                    'rel':          data.get('rel',''),
                    'rela':         data.get('rela',''),
                    'sab_relation': data.get('sab_relation',''),
                    'cui_target':   v,
                    'name_target':  G.nodes[v].get('name',''),
                    'sab_target':   v_combo['SAB'],
                    'codes_target': v_combo['CODE'],
                })

    edges_df = pd.DataFrame(edges_expanded)
    edges_path = os.path.join(OUTPUT_DIR, 'kg_edges.csv')
    edges_df.to_csv(edges_path, index=False)

    log(f"Saved expanded nodes: {len(nodes_df):,} -> {nodes_path}")
    log(f"Saved expanded edges: {len(edges_df):,} -> {edges_path}")

    # -------- Persist graph (GraphML requires strings) --------
    log("Saving GraphML and pickle…")
    G_for_graphml = nx.DiGraph()
    for n, data in G.nodes(data=True):
        data_str = {}
        for k, v in data.items():
            if isinstance(v, list):
                data_str[k] = '|'.join(map(str, v))
            else:
                data_str[k] = '' if v is None else str(v)
        G_for_graphml.add_node(n, **data_str)

    for u, v, data in G.edges(data=True):
        data_str = {k: ('' if v is None else str(v)) for k, v in data.items()}
        G_for_graphml.add_edge(u, v, **data_str)

    nx.write_graphml(G_for_graphml, os.path.join(OUTPUT_DIR, 'medical_knowledge_graph.graphml'))
    with open(os.path.join(OUTPUT_DIR, 'medical_knowledge_graph.pkl'), 'wb') as f:
        pickle.dump(G, f)

    return G, nodes_df, edges_df


# -------------------- STATS + VIZ --------------------
def generate_summary_statistics(G):
    log("Generating summary statistics…")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    degrees = [d for _, d in G.degree()]
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': float(np.mean(degrees)) if degrees else 0.0,
        'median_degree': float(np.median(degrees)) if degrees else 0.0,
        'max_degree': int(np.max(degrees)) if degrees else 0,
        'min_degree': int(np.min(degrees)) if degrees else 0,
    }

    # Source distribution (from your analysis tags)
    source_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        srcs = data.get('source', [])
        if isinstance(srcs, str): srcs = srcs.split('|')
        for s in srcs:
            if s: source_counts[s] += 1
    stats['source_counts'] = dict(source_counts)

    # Semantic types distribution
    sty_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        stys = data.get('semantic_type', [])
        if isinstance(stys, str): stys = stys.split('|')
        for sty in stys:
            if sty: sty_counts[sty] += 1
    stats['sty_counts'] = dict(sty_counts)

    # Relationships
    rel_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        rel_counts[data.get('rel','Unknown') or 'Unknown'] += 1
    stats['rel_counts'] = dict(rel_counts)

    # SAB on edges
    sab_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        sab_counts[data.get('sab_relation','Unknown') or 'Unknown'] += 1
    stats['sab_relation_counts'] = dict(sab_counts)

    # Components
    components = list(nx.weakly_connected_components(G)) if G.is_directed() else list(nx.connected_components(G))
    stats['num_components'] = len(components)
    stats['largest_component_size'] = len(max(components, key=len)) if components else 0

    with open(os.path.join(OUTPUT_DIR, 'kg_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)

    log("Summary statistics saved.")


def create_visualizations(G):
    log("Creating visualizations…")
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    if degrees:
        plt.figure(figsize=(10,6))
        sns.histplot(degrees, bins=50, kde=True)
        plt.title('Node Degree Distribution')
        plt.xlabel('Degree'); plt.ylabel('Frequency'); plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'node_degree_distribution.png'), dpi=300)
        plt.close()

    # Source distribution (from analyses)
    source_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        srcs = data.get('source', [])
        if isinstance(srcs, str): srcs = srcs.split('|')
        for s in srcs:
            if s: source_counts[s] += 1
    if source_counts:
        sdf = pd.DataFrame([{'source': k, 'count': v} for k, v in source_counts.items()]).sort_values('count', ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(data=sdf, x='source', y='count')
        plt.title('Node Distribution by Source'); plt.xlabel('Source'); plt.ylabel('Count')
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'source_distribution.png'), dpi=300)
        plt.close()

    # Semantic types (top 15)
    sty_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        stys = data.get('semantic_type', [])
        if isinstance(stys, str): stys = stys.split('|')
        for sty in stys:
            if sty: sty_counts[sty] += 1
    if sty_counts:
        top_sty = sorted(sty_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        sty_df = pd.DataFrame([{'semantic_type': k, 'count': v} for k, v in top_sty])
        plt.figure(figsize=(12,6))
        sns.barplot(data=sty_df, x='semantic_type', y='count')
        plt.title('Top 15 Semantic Types'); plt.xlabel('Semantic Type'); plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'semantic_type_distribution.png'), dpi=300)
        plt.close()

    # Relationship types (top 15)
    rel_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        rel_counts[data.get('rel','Unknown') or 'Unknown'] += 1
    if rel_counts:
        top_rel = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        rel_df = pd.DataFrame([{'relationship': k, 'count': v} for k, v in top_rel])
        plt.figure(figsize=(12,6))
        sns.barplot(data=rel_df, x='relationship', y='count')
        plt.title('Top 15 Relationship Types'); plt.xlabel('Relationship'); plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'relationship_distribution.png'), dpi=300)
        plt.close()

    log("Visualizations saved.")


def create_sample_subgraph_visualization(G):
    log("Creating sample subgraph…")
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')

    degs = sorted([(n, d) for n, d in G.degree()], key=lambda x: x[1])
    candidates = [n for n, d in degs if 5 <= d <= 15]
    seed = candidates[0] if candidates else (next(iter(G.nodes())) if G.nodes else None)
    if seed is None:
        log("Graph empty; skip subgraph viz.")
        return

    neigh = list(G.successors(seed)) + list(G.predecessors(seed))
    sub_nodes = [seed] + list(set(neigh))
    H = G.subgraph(sub_nodes).copy()

    from matplotlib.lines import Line2D

    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(H, seed=42)

    # Color by 'source' tag if present
    colors = []
    for n in H.nodes():
        srcs = G.nodes[n].get('source', [])
        if isinstance(srcs, str): srcs = srcs.split('|')
        if 'ICD9' in srcs:   colors.append('red')
        elif 'MED' in srcs:  colors.append('blue')
        elif 'LAB' in srcs:  colors.append('green')
        elif 'PROC' in srcs: colors.append('purple')
        else:                colors.append('gray')

    nx.draw_networkx_nodes(H, pos, node_size=220, alpha=0.85, node_color=colors)
    nx.draw_networkx_edges(H, pos, alpha=0.5, arrows=True)

    labels = {}
    for n in H.nodes():
        nm = G.nodes[n].get('name', n)
        labels[n] = nm if len(nm) <= 20 else (nm[:17] + '…')
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

    legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',    markersize=10, label='ICD9'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',   markersize=10, label='Medication'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='green',  markersize=10, label='Lab Test'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Procedure'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',   markersize=10, label='Other'),
    ]
    plt.legend(handles=legend, loc='upper right')
    plt.axis('off')
    title = G.nodes[seed].get('name', seed)
    plt.title(f'Sample Subgraph: {title if len(title)<=60 else title[:57]+"…"}')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sample_subgraph.png'), dpi=300)
    plt.close()
    log("Sample subgraph saved.")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    try:
        log("Starting Knowledge Graph construction")

        cui_sources, all_cuis = load_analysis_files()
        G, nodes_df, edges_df = build_knowledge_graph(cui_sources, all_cuis)

        # Sanity checks
        # assert nodes_df['cui'].str.match(r'^C\\d+$').all(), "Non-CUI values found in nodes.cui"
        # assert edges_df['cui_start'].str.match(r'^C\\d+$').all(), "Non-CUI in edges.cui_start"
        # assert edges_df['cui_target'].str.match(r'^C\\d+$').all(), "Non-CUI in edges.cui_target"

        # Stats + Viz
        generate_summary_statistics(G)
        create_visualizations(G)
        create_sample_subgraph_visualization(G)

        log("Knowledge graph construction complete!")
    except Exception as e:
        import traceback
        log(f"ERROR: {str(e)}")
        log(traceback.format_exc())
        print(f"ERROR: {str(e)}")
        print("See log file for details.")

