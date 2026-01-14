#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined UMLS code2cui mapping builder and comprehensive KG constructor.

This script:
1. Builds code->CUI maps from UMLS MRCONSO.RRF for ATC, LOINC, SNOMEDCT_US
2. Loads ICD-9 DX/PROC mappings from specific pickle files and JSON
3. Creates alias mappings for ICD-9 codes
4. Builds a comprehensive medical knowledge graph ensuring all codes have CUI mappings
5. Creates clean, non-aggregated SAB fields with only target vocabularies
6. Creates separate edge rows for each vocabulary combination

Outputs:
- code2cui_*.pkl files for each vocabulary
- alias2canon_*.pkl files for ICD-9
- Knowledge graph files (nodes.csv, edges.csv, .graphml, .pkl)
- Statistics and visualizations
"""

import os
import re
import json
import pickle
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, Tuple, List, Set, Any
from itertools import product

import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Configuration --------------------
UMLS_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META'
OUTPUT_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output3'
TARGET_VOCABS = ['ICD9CM', 'LNC', 'ATC', 'SNOMEDCT_US']

# MRCONSO columns
COLS = [
    'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI',
    'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
]
USECOLS_MIN = ['CUI', 'LAT', 'TS', 'SAB', 'TTY', 'CODE', 'STR', 'SUPPRESS']
USECOLS_IDX = [0, 1, 2, 11, 12, 13, 14, 16]

# -------------------- Utility Functions --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def make_logger(out_dir: str):
    ensure_dir(out_dir)
    log_file = os.path.join(out_dir, f'kg_build_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    def _log(msg: str):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f'[{ts}] {msg}\n')
        print(msg)
    return _log

def dump_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): 
            return int(obj)
        if isinstance(obj, np.floating): 
            return float(obj)
        if isinstance(obj, np.ndarray): 
            return obj.tolist()
        return super().default(obj)

# -------------------- String Cleaning & ICD-9 Formatting --------------------
def _strip(x: str) -> str:
    return re.sub(r"\s+", "", str(x or "")).upper().rstrip(".")

def _clean_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s

def format_icd9_dx(code: str) -> str:
    c = _strip(code)
    if not c:
        return ""
    if c[0].isdigit():
        return c[:3] + "." + c[3:] if len(c) > 3 and "." not in c else c
    if c[0] == "V":
        return c[:3] + "." + c[3:] if len(c) > 3 and "." not in c else c
    if c[0] == "E":
        return c[:4] + "." + c[4:] if len(c) > 4 and "." not in c else c
    return c

def format_icd9_proc(code: str) -> str:
    c = _strip(code)
    if c.startswith("PRO_"):
        c = c[4:]
    if not c:
        return ""
    if c[0].isdigit():
        return c[:2] + "." + c[2:] if len(c) > 2 and "." not in c else c
    return c

_dx_pat = re.compile(r"^(?:\d{3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$", re.I)
_proc_pat = re.compile(r"^\d{2}(?:\.\d{1,2})?$")

def is_icd9_dx(code: str) -> bool:
    c = _strip(code)
    return bool(_dx_pat.match(c)) and not bool(_proc_pat.match(c))

def is_icd9_proc(code: str) -> bool:
    c = _strip(code)
    return bool(_proc_pat.match(c))

def dx_aliases(k: str) -> Set[str]:
    """Generate lookup aliases for ICD-9 DX."""
    out = set()
    k = _strip(k)
    if not k:
        return out
    out.add(k)
    out.add(k.replace('.', ''))  # no-dot
    # *.00 -> *.0 (add both)
    if k[0].isdigit() and re.match(r'^\d{3}\.\d{2}$', k) and k.endswith('0'):
        out.add(k[:-1])
        out.add(k[:-1].replace('.', ''))
    # E***.x -> E**** (no-dot)
    if k.startswith('E') and '.' in k and len(k.split('.')[-1]) == 1:
        out.add(k.replace('.', ''))
    return out

def proc_aliases(k: str) -> Set[str]:
    """Lookup aliases for procedures (dot and no-dot)."""
    out = set()
    k = _strip(k)
    if not k:
        return out
    out.add(k)
    out.add(k.replace('.', ''))
    return out

# -------------------- ICD-9 Range Parsing --------------------
def parse_icd9_key(code: str) -> Tuple[str, int, int]:
    """Parse ICD-9 code into category and numeric key for range matching."""
    c = _strip(code)
    if is_icd9_proc(c):
        whole, frac = (c.split('.') + [""])[0], (c.split('.') + [""])[1]
        frac = (frac + "00")[:2]
        return ("PROC", int(whole) * 100 + int(frac or "0"), 100)
    if c.startswith("E"):
        base = c[1:]
        parts = base.split('.')
        whole, frac = parts[0], (parts[1] if len(parts) > 1 else "")
        frac = (frac + "0")[:1]
        return ("DX_E", int(whole) * 10 + int(frac or "0"), 10)
    # numeric or V
    base = c[1:] if c.startswith("V") else c
    parts = base.split('.')
    whole, frac = parts[0], (parts[1] if len(parts) > 1 else "")
    frac = (frac + "00")[:2]
    return ("DX_NUMV", int(whole) * 100 + int(frac or "0"), 100)

def expand_json_spec_to_matcher(spec: str):
    """Convert JSON specification to matcher for range or exact matching."""
    s = _strip(spec)
    if "-" in s:
        left, right = s.split("-", 1)
        def norm(ep):
            dx = format_icd9_dx(ep)
            pr = format_icd9_proc(ep)
            if is_icd9_dx(dx):
                return parse_icd9_key(dx)
            if is_icd9_proc(pr):
                return parse_icd9_key(pr)
            return parse_icd9_key(ep)
        return ('range', norm(left), norm(right))
    else:
        dx = format_icd9_dx(s)
        pr = format_icd9_proc(s)
        if is_icd9_dx(dx):
            return ('exact', dx)
        if is_icd9_proc(pr):
            return ('exact', pr)
        return ('exact', s)

def code_in_range(code: str, start_key, end_key) -> bool:
    """Check if code falls within range defined by start_key and end_key."""
    cat_c, val_c, _ = parse_icd9_key(code)
    cat_s, val_s, _ = start_key
    cat_e, val_e, _ = end_key
    if cat_s != cat_e or cat_c != cat_s:
        return False
    return val_s <= val_c <= val_e

# -------------------- Master List Loaders --------------------
def load_icd9_master_list(icd9_pkl: str, log) -> Set[str]:
    """Load ICD-9 DX master list from pickle file."""
    log(f"Loading ICD-9 DX master list: {icd9_pkl}")
    obj = pickle.load(open(icd9_pkl, "rb"))
    if isinstance(obj, pd.DataFrame):
        col = 'icd_code' if 'icd_code' in obj.columns else obj.columns[0]
        values = obj[col].astype(str).tolist()
    elif isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        values = list(map(str, obj))
    else:
        raise ValueError("icd9.pkl must be a DataFrame or a list/array/Series of codes.")
    
    out = set()
    for c in set(values):
        fdx = format_icd9_dx(c)
        if is_icd9_dx(fdx):
            out.add(fdx)
    log(f"Master DX codes loaded: {len(out):,}")
    return out

def load_icd9_proc_master_list(icd9proc_pkl: str, log) -> Set[str]:
    """Load ICD-9 PROC master list from pickle file."""
    log(f"Loading ICD-9 PROC master list: {icd9proc_pkl}")
    obj = pickle.load(open(icd9proc_pkl, "rb"))
    if isinstance(obj, pd.DataFrame):
        col = 'icd_code' if 'icd_code' in obj.columns else obj.columns[0]
        values = obj[col].astype(str).tolist()
    elif isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        values = list(map(str, obj))
    else:
        raise ValueError("icd9proc.pkl must be a DataFrame or a list/array/Series of codes.")
    
    out = set()
    for c in set(values):
        fpr = format_icd9_proc(c)
        if is_icd9_proc(fpr):
            out.add(fpr)
    log(f"Master PROC codes loaded: {len(out):,}")
    return out

def load_dataset_codes(dataset_pkl: str, log) -> Tuple[Set[str], Set[str]]:
    """Load dataset codes from pickle file."""
    if not dataset_pkl:
        return set(), set()
    
    log(f"Loading dataset codes: {dataset_pkl}")
    df = pickle.load(open(dataset_pkl, "rb"))
    if not isinstance(df, pd.DataFrame):
        raise ValueError("dataset PKL must be a DataFrame.")
    
    dx_raw, pr_raw = [], []
    if 'icd_code' in df.columns:
        for x in df['icd_code']:
            if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
                dx_raw.extend(map(str, x))
    if 'pro_code' in df.columns:
        for x in df['pro_code']:
            if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
                pr_raw.extend(map(str, x))
    
    dx = set(format_icd9_dx(c) for c in dx_raw if format_icd9_dx(c))
    pr = set(format_icd9_proc(c) for c in pr_raw if format_icd9_proc(c))
    log(f"Dataset DX codes: {len(dx):,} | PROC codes: {len(pr):,}")
    return dx, pr

# -------------------- Reverse CUI->ICD9 Mapping --------------------
def reverse_cui_to_icd9_separate(cui_to_codes: Dict[str, List[str]], 
                                master_dx: Set[str], 
                                master_pr: Set[str], 
                                log):
    """Create reverse mapping from CUI->codes JSON to code->CUI mappings."""
    log("Creating reverse mappings from CUI->ICD9 JSON")
    dx_map = defaultdict(set)
    pr_map = defaultdict(set)
    
    compiled = {cui: [expand_json_spec_to_matcher(s) for s in specs]
                for cui, specs in cui_to_codes.items()}

    for cui, specs in tqdm(compiled.items(), desc="Reversing CUI->ICD9 (DX/PROC)"):
        exact_dx, exact_pr, ranges = [], [], []
        for kind, *rest in specs:
            if kind == 'exact':
                code = rest[0]
                if is_icd9_proc(code):
                    exact_pr.append(code)
                elif is_icd9_dx(code):
                    exact_dx.append(code)
            else:
                ranges.append(tuple(rest))
        
        # Handle exact matches
        for code in exact_dx:
            if code in master_dx:
                dx_map[code].add(cui)
        for code in exact_pr:
            if code in master_pr:
                pr_map[code].add(cui)
        
        # Handle ranges
        for (start_key, end_key) in ranges:
            cat = start_key[0]
            pool = master_pr if cat == "PROC" else master_dx
            for code in pool:
                if code_in_range(code, start_key, end_key):
                    (pr_map if cat == "PROC" else dx_map)[code].add(cui)

    dx_final = {k: sorted(v) for k, v in dx_map.items()}
    pr_final = {k: sorted(v) for k, v in pr_map.items()}
    
    log(f"ICD-9 DX->CUI mappings: {len(dx_final):,}")
    log(f"ICD-9 PROC->CUI mappings: {len(pr_final):,}")
    
    return dx_final, pr_final

# -------------------- MRCONSO Streaming --------------------
def read_conso_subset(path: str, usecols_idx: List[int], chunksize: int):
    """Robust MRCONSO reader with error handling."""
    return pd.read_csv(
        path,
        sep='|',
        header=None,
        usecols=usecols_idx,
        dtype=str,
        chunksize=chunksize,
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip',
        low_memory=False
    )

def build_code2cui_generic(mrconso_path: str, target_sab: str, log, 
                          keep_ts_p_only=True, chunksize=1_000_000):
    """Build generic code->CUI mapping from MRCONSO for ATC, LOINC, SNOMEDCT_US."""
    code2cuis = defaultdict(set)
    code2name = {}
    
    for ch in tqdm(read_conso_subset(mrconso_path, USECOLS_IDX, chunksize), 
                   desc=f"SAB={target_sab}", unit="chunk"):
        ch.columns = USECOLS_MIN
        ch = ch[(ch['LAT'] == 'ENG') & (ch['SAB'] == target_sab)]
        ch = ch[(ch['SUPPRESS'] != 'O')]
        
        if keep_ts_p_only:
            chp = ch[ch['TS'] == 'P']
            if not chp.empty:
                ch = chp
        
        if ch.empty:
            continue
            
        for _, r in ch.iterrows():
            code = _strip(_clean_str(r.get('CODE')))
            if not code:
                continue
            cui = _clean_str(r.get('CUI'))
            name = _clean_str(r.get('STR'))
            if not cui:
                continue
            code2cuis[code].add(cui)
            if code not in code2name:
                code2name[code] = name
    
    code2cuis = {k: sorted(v) for k, v in code2cuis.items()}
    log(f"[{target_sab}] codes: {len(code2cuis):,}")
    return code2cuis, code2name

# -------------------- Enhanced MRCONSO Loader for KG --------------------
def load_mrconso_for_kg(mrconso_path: str, target_vocabs: List[str], 
                       restrict_to_cuis: Set[str], log):
    """
    Load MRCONSO data for KG building with separate rows for each CUI-SAB-CODE combination.
    Only includes target vocabularies to avoid noise.
    """
    log(f"Loading MRCONSO for KG (target vocabs: {target_vocabs})")
    usecols_idx = [0, 1, 6, 11, 12, 13, 14]
    colnames = ['CUI', 'LAT', 'ISPREF', 'SAB', 'TTY', 'CODE', 'STR']
    
    # First pass: collect preferred names per CUI
    name_map = {}
    tty_priority = {'PT': 0, 'PN': 1, 'FN': 2}
    
    it = pd.read_csv(
        mrconso_path,
        sep='|',
        header=None,
        usecols=usecols_idx,
        dtype=str,
        chunksize=1_000_000,
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip',
        low_memory=False
    )
    
    name_candidates = []
    expanded_rows = []
    
    for ch in it:
        ch.columns = colnames
        ch = ch[(ch['LAT'] == 'ENG') & (ch['SAB'].isin(target_vocabs))]
        if restrict_to_cuis:
            ch = ch[ch['CUI'].isin(restrict_to_cuis)]
        if ch.empty:
            continue
        
        # Clean data
        ch['CUI'] = ch['CUI'].apply(_clean_str)
        ch['SAB'] = ch['SAB'].apply(_clean_str)
        ch['CODE'] = ch['CODE'].apply(_clean_str)
        ch['STR'] = ch['STR'].apply(_clean_str)
        ch['ISPREF'] = ch['ISPREF'].apply(_clean_str)
        ch['TTY'] = ch['TTY'].apply(_clean_str)
        
        # Filter out empty codes and CUIs
        ch = ch[(ch['CUI'] != '') & (ch['CODE'] != '') & (ch['STR'] != '')]
        if ch.empty:
            continue
        
        # Collect name candidates
        ch['is_pref'] = ch['ISPREF'].apply(lambda x: 1 if x == 'Y' else 0)
        ch['tty_rank'] = ch['TTY'].apply(lambda t: tty_priority.get(t, 99))
        name_candidates.append(ch[['CUI', 'STR', 'is_pref', 'tty_rank']])
        
        # Collect expanded rows (one per CUI-SAB-CODE combination)
        for _, row in ch.iterrows():
            expanded_rows.append({
                'cui': row['CUI'],
                'sab': row['SAB'],
                'code': row['CODE'],
                'str': row['STR']
            })
    
    # Select best names
    if name_candidates:
        all_names = pd.concat(name_candidates, ignore_index=True)
        best_names = (all_names.sort_values(['CUI', 'is_pref', 'tty_rank'], ascending=[True, False, True])
                     .drop_duplicates('CUI', keep='first'))
        name_map = dict(zip(best_names['CUI'], best_names['STR']))
    
    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    if not expanded_df.empty:
        expanded_df['name'] = expanded_df['cui'].map(name_map)
        # Remove duplicates
        expanded_df = expanded_df.drop_duplicates(['cui', 'sab', 'code']).reset_index(drop=True)
    
    log(f"MRCONSO for KG: {len(expanded_df):,} CUI-SAB-CODE combinations")
    log(f"Unique CUIs: {expanded_df['cui'].nunique():,}")
    log(f"SAB distribution: {dict(expanded_df['sab'].value_counts())}")
    
    return expanded_df, name_map

# -------------------- MRREL and MRSTY Loaders --------------------
def load_mrrel_filtered(mrrel_path: str, allowed_sabs: List[str], 
                       restrict_to_cuis: Set[str], log):
    """Load MRREL relations filtered by vocabularies and CUIs."""
    usecols_idx = [0, 3, 4, 7, 10]
    colnames = ['CUI1', 'REL', 'CUI2', 'RELA', 'SAB']
    out = []
    
    it = pd.read_csv(
        mrrel_path,
        sep='|',
        header=None,
        usecols=usecols_idx,
        dtype=str,
        chunksize=1_000_000,
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip',
        low_memory=False
    )
    
    for ch in it:
        ch.columns = colnames
        ch = ch[ch['SAB'].isin(allowed_sabs)]
        if restrict_to_cuis:
            ch = ch[ch['CUI1'].isin(restrict_to_cuis) | ch['CUI2'].isin(restrict_to_cuis)]
        if not ch.empty:
            out.append(ch)
    
    rel = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=colnames)
    log(f"MRREL filtered: {len(rel):,} (SAB∈{allowed_sabs})")
    return rel

def load_mrsty_for_cuis(mrsty_path: str, cuis: Set[str], log):
    """Load semantic types for specified CUIs."""
    usecols_idx = [0, 3]  # CUI, STY
    colnames = ['CUI', 'STY']
    
    it = pd.read_csv(
        mrsty_path,
        sep='|',
        header=None,
        usecols=usecols_idx,
        dtype=str,
        chunksize=500_000,
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip',
        low_memory=False
    )
    
    rows = []
    for ch in it:
        ch.columns = colnames
        ch = ch[ch['CUI'].isin(cuis)]
        if not ch.empty:
            rows.append(ch)
    
    sty = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=colnames)
    return sty.groupby('CUI')['STY'].apply(
        lambda s: sorted(set(_clean_str(x) for x in s if _clean_str(x)))
    ).to_dict()

# -------------------- Knowledge Graph Builder --------------------
def build_knowledge_graph(umls_dir: str, out_dir: str, code_mappings: Dict[str, Dict], 
                         target_vocabs: List[str], log):
    """Build comprehensive knowledge graph from all code mappings."""
    
    mrconso = os.path.join(umls_dir, "MRCONSO.RRF")
    mrrel = os.path.join(umls_dir, "MRREL.RRF")
    mrsty = os.path.join(umls_dir, "MRSTY.RRF")
    
    for p in (mrconso, mrrel, mrsty):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")

    # Collect all CUIs from mappings
    seed_cuis = set()
    for mapping in code_mappings.values():
        for cui_list in mapping.values():
            seed_cuis.update(cui_list)
    log(f"Seed CUIs from all mappings: {len(seed_cuis):,}")

    # Load MRCONSO data for KG (expanded format)
    conso_df, name_map = load_mrconso_for_kg(mrconso, target_vocabs, seed_cuis, log)
    
    # Load relations and semantic types
    rel_df = load_mrrel_filtered(mrrel, allowed_sabs=target_vocabs, 
                                restrict_to_cuis=seed_cuis, log=log)
    rel_df = rel_df[(rel_df['CUI1'].isin(seed_cuis)) & 
                    (rel_df['CUI2'].isin(seed_cuis))].reset_index(drop=True)
    sty_map = load_mrsty_for_cuis(mrsty, seed_cuis, log)

    # Build NetworkX graph
    G = nx.DiGraph()
    log("Adding nodes to graph...")
    
    # Aggregate data per CUI for NetworkX graph
    cui_sabs = defaultdict(set)
    cui_codes = defaultdict(set)
    for _, row in conso_df.iterrows():
        cui = row['cui']
        cui_sabs[cui].add(row['sab'])
        cui_codes[cui].add(row['code'])
    
    for cui in tqdm(seed_cuis, desc="Nodes"):
        G.add_node(cui,
                   name=name_map.get(cui, "Unknown"),
                   sab=sorted(cui_sabs.get(cui, [])),
                   code=sorted(cui_codes.get(cui, [])),
                   semantic_type=sty_map.get(cui, []))

    log("Adding edges to graph...")
    for _, r in tqdm(rel_df.iterrows(), total=len(rel_df), desc="Edges"):
        u, v = _clean_str(r.get('CUI1')), _clean_str(r.get('CUI2'))
        if u in G and v in G:
            G.add_edge(u, v,
                       rel=_clean_str(r.get('REL')),
                       rela=_clean_str(r.get('RELA')),
                       sab_relation=_clean_str(r.get('SAB')))

    log(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Save CSV files (using expanded format)
    save_graph_csvs_expanded(G, conso_df, out_dir, log)
    
    # Save graph files
    save_graph_files(G, out_dir, log)
    
    return G, conso_df

def save_graph_csvs_expanded(G: nx.DiGraph, conso_df: pd.DataFrame, out_dir: str, log):
    """
    Save nodes and edges as CSV files.
    Nodes CSV uses expanded format: one row per CUI-SAB-CODE combination.
    Edges CSV uses expanded format: separate rows for each vocabulary combination.
    """
    
    # Nodes CSV - Expanded format (one row per CUI-SAB-CODE combination)
    nodes_data = []
    for _, row in conso_df.iterrows():
        cui = row['cui']
        if cui in G:  # Only include CUIs that are in our graph
            nodes_data.append({
                'cui': cui,
                'name': row['name'] if pd.notna(row['name']) else '',
                'sab': row['sab'],  # Single SAB per row
                'code': row['code'],  # Single CODE per row
                'semantic_type': '|'.join(G.nodes[cui].get('semantic_type', []))
            })
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_path = os.path.join(out_dir, 'kg_nodes.csv')
    nodes_df.to_csv(nodes_path, index=False)
    log(f"Saved nodes (expanded): {len(nodes_df):,} rows -> {nodes_path}")
    log(f"Unique SABs in nodes: {sorted(nodes_df['sab'].unique())}")
    
    # Also save aggregated nodes for reference
    nodes_agg_data = []
    for cui, data in G.nodes(data=True):
        nodes_agg_data.append({
            'cui': cui,
            'name': data.get('name', ''),
            'sab': '|'.join(data.get('sab', [])),
            'code': '|'.join(data.get('code', [])),
            'semantic_type': '|'.join(data.get('semantic_type', []))
        })
    
    nodes_agg_df = pd.DataFrame(nodes_agg_data)
    nodes_agg_path = os.path.join(out_dir, 'kg_nodes_aggregated.csv')
    nodes_agg_df.to_csv(nodes_agg_path, index=False)
    log(f"Saved nodes (aggregated): {len(nodes_agg_df):,} rows -> {nodes_agg_path}")
    
    # Edges CSV - Expanded format (separate rows for each vocabulary combination)
    log("Creating expanded edges CSV...")
    edges_data = []
    
    # Create lookup for CUI -> SABs from the expanded nodes
    cui_to_sabs = defaultdict(set)
    for _, row in conso_df.iterrows():
        cui_to_sabs[row['cui']].add(row['sab'])
    
    # Create lookup for CUI -> codes from the expanded nodes  
    cui_to_codes = defaultdict(set)
    for _, row in conso_df.iterrows():
        cui_to_codes[row['cui']].add(row['code'])
    
    for u, v, data in tqdm(G.edges(data=True), desc="Expanding edges"):
        # Get SABs for source and target CUIs
        start_sabs = sorted(cui_to_sabs.get(u, set()))
        target_sabs = sorted(cui_to_sabs.get(v, set()))
        
        # Get codes for source and target CUIs
        start_codes = sorted(cui_to_codes.get(u, set()))
        target_codes = sorted(cui_to_codes.get(v, set()))
        
        # Create cartesian product of SAB combinations
        for start_sab in start_sabs:
            for target_sab in target_sabs:
                edges_data.append({
                    'cui_start': u,
                    'name_start': G.nodes[u].get('name', ''),
                    'sab_start': start_sab,  # Single SAB per row
                    'codes_start': '|'.join(start_codes),  # All codes for this CUI
                    'rel': data.get('rel', ''),
                    'rela': data.get('rela', ''),
                    'sab_relation': data.get('sab_relation', ''),
                    'cui_target': v,
                    'name_target': G.nodes[v].get('name', ''),
                    'sab_target': target_sab,  # Single SAB per row
                    'codes_target': '|'.join(target_codes)  # All codes for this CUI
                })
    
    edges_df = pd.DataFrame(edges_data)
    edges_path = os.path.join(out_dir, 'kg_edges.csv')
    edges_df.to_csv(edges_path, index=False)
    log(f"Saved edges (expanded): {len(edges_df):,} -> {edges_path}")
    log(f"Unique SABs in edge starts: {sorted(edges_df['sab_start'].unique())}")
    log(f"Unique SABs in edge targets: {sorted(edges_df['sab_target'].unique())}")

def save_graph_files(G: nx.DiGraph, out_dir: str, log):
    """Save graph in GraphML and pickle formats."""
    
    # Convert for GraphML (requires string attributes)
    G_ml = nx.DiGraph()
    for n, data in G.nodes(data=True):
        data_str = {}
        for k, v in data.items():
            if isinstance(v, list):
                data_str[k] = '|'.join(map(str, v))
            else:
                data_str[k] = '' if v is None else str(v)
        G_ml.add_node(n, **data_str)
    
    for u, v, data in G.edges(data=True):
        data_str = {k: ('' if v is None else str(v)) for k, v in data.items()}
        G_ml.add_edge(u, v, **data_str)
    
    # Save files
    graphml_path = os.path.join(out_dir, 'medical_knowledge_graph.graphml')
    pickle_path = os.path.join(out_dir, 'medical_knowledge_graph.pkl')
    
    nx.write_graphml(G_ml, graphml_path)
    dump_pickle(pickle_path, G)
    
    log(f"Saved GraphML: {graphml_path}")
    log(f"Saved pickle: {pickle_path}")

# -------------------- Coverage Analysis --------------------
def coverage_analysis(all_codes: Set[str], mapping: Dict[str, List[str]]):
    """Analyze coverage of codes in mapping."""
    have = {c for c in all_codes if c in mapping}
    miss = sorted(all_codes - have)
    pct = 100.0 * len(have) / max(1, len(all_codes))
    return {
        "total": len(all_codes),
        "mapped": len(have),
        "missing": len(miss),
        "coverage_pct": pct,
        "missing_sample": miss[:25]
    }

# -------------------- Statistics and Visualization --------------------
def generate_statistics(G: nx.DiGraph, code_mappings: Dict, conso_df: pd.DataFrame, out_dir: str, log):
    """Generate comprehensive statistics."""
    
    degrees = [d for _, d in G.degree()]
    
    stats = {
        'graph': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'avg_degree': float(np.mean(degrees)) if degrees else 0.0,
            'median_degree': float(np.median(degrees)) if degrees else 0.0,
            'max_degree': int(np.max(degrees)) if degrees else 0,
            'min_degree': int(np.min(degrees)) if degrees else 0,
        },
        'vocabularies': {k: len(v) for k, v in code_mappings.items()},
        'components': len(list(nx.weakly_connected_components(G))),
        'sab_distribution': dict(conso_df['sab'].value_counts()) if not conso_df.empty else {}
    }
    
    # Semantic types
    sty_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        stys = data.get('semantic_type', [])
        for sty in stys:
            if sty:
                sty_counts[sty] += 1
    stats['semantic_types'] = dict(sorted(sty_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    # Relationships
    rel_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        rel_counts[data.get('rel', 'Unknown') or 'Unknown'] += 1
    stats['relationships'] = dict(sorted(rel_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Save statistics
    stats_path = os.path.join(out_dir, 'kg_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    
    log(f"Statistics saved: {stats_path}")
    return stats

# -------------------- Main Function --------------------
def main():
    parser = argparse.ArgumentParser(description="Build comprehensive UMLS KG with clean vocabulary filtering")
    parser.add_argument("--umls-dir", default=UMLS_DIR, help="UMLS META directory")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--cui-to-icd9-json", required=True, help="CUI -> [codes and/or ranges] JSON file")
    parser.add_argument("--icd9-dx-pkl", required=True, help="Master diagnosis list (icd9.pkl)")
    parser.add_argument("--icd9-proc-pkl", required=True, help="Master procedure list (icd9proc.pkl)")
    parser.add_argument("--dataset-pkl", default="", help="Dataset codes pickle (optional)")
    parser.add_argument("--with-names", action="store_true", help="Include name mappings")
    parser.add_argument("--target-vocabs", default="ICD9CM,LNC,ATC,SNOMEDCT_US", help="Target vocabularies")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Chunk size for processing")
    
    args = parser.parse_args()
    
    # Setup
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, 'visualizations'))
    log = make_logger(args.out_dir)
    
    target_vocabs = [s.strip() for s in args.target_vocabs.split(",") if s.strip()]
    mrconso_path = os.path.join(args.umls_dir, 'MRCONSO.RRF')
    
    log("Starting comprehensive UMLS processing with clean vocabulary filtering...")
    log(f"Target vocabularies: {target_vocabs}")
    
    # Step 1: Load master ICD-9 lists
    master_dx = load_icd9_master_list(args.icd9_dx_pkl, log)
    master_pr = load_icd9_proc_master_list(args.icd9_proc_pkl, log)
    
    # Step 2: Load dataset codes (optional)
    ds_dx, ds_pr = set(), set()
    if args.dataset_pkl:
        ds_dx, ds_pr = load_dataset_codes(args.dataset_pkl, log)
    
    # Step 3: Create ICD-9 mappings from JSON
    log(f"Reading CUI->ICD9 JSON: {args.cui_to_icd9_json}")
    cui_to_codes = json.load(open(args.cui_to_icd9_json, "r"))
    icd9_dx_map, icd9_pr_map = reverse_cui_to_icd9_separate(
        cui_to_codes, master_dx, master_pr, log)
    
    # Step 4: Build other vocabulary mappings
    code_mappings = {}
    
    # ATC
    log("Building ATC -> CUI mapping...")
    atc_map, atc_names = build_code2cui_generic(
        mrconso_path, "ATC", log, keep_ts_p_only=True, chunksize=args.chunksize)
    code_mappings['atc'] = atc_map
    
    # LOINC
    log("Building LOINC -> CUI mapping...")
    loinc_map, loinc_names = build_code2cui_generic(
        mrconso_path, "LNC", log, keep_ts_p_only=True, chunksize=args.chunksize)
    code_mappings['loinc'] = loinc_map
    
    # SNOMEDCT_US
    log("Building SNOMEDCT_US -> CUI mapping...")
    snomed_map, snomed_names = build_code2cui_generic(
        mrconso_path, "SNOMEDCT_US", log, keep_ts_p_only=True, chunksize=args.chunksize)
    code_mappings['snomedct_us'] = snomed_map
    
    # Add ICD-9 mappings
    code_mappings['icd9_dx'] = icd9_dx_map
    code_mappings['icd9_proc'] = icd9_pr_map
    
    # Step 5: Save all mappings
    dump_pickle(os.path.join(args.out_dir, "code2cui_icd9_dx.pkl"), icd9_dx_map)
    dump_pickle(os.path.join(args.out_dir, "code2cui_icd9_proc.pkl"), icd9_pr_map)
    dump_pickle(os.path.join(args.out_dir, "code2cui_atc.pkl"), atc_map)
    dump_pickle(os.path.join(args.out_dir, "code2cui_loinc.pkl"), loinc_map)
    dump_pickle(os.path.join(args.out_dir, "code2cui_snomedct_us.pkl"), snomed_map)
    
    # Create alias mappings for ICD-9
    alias2canon_dx = {}
    alias2canon_pr = {}
    for code in icd9_dx_map.keys():
        for alias in dx_aliases(code):
            alias2canon_dx.setdefault(alias, code)
    for code in icd9_pr_map.keys():
        for alias in proc_aliases(code):
            alias2canon_pr.setdefault(alias, code)
    
    dump_pickle(os.path.join(args.out_dir, "alias2canon_icd9_dx.pkl"), alias2canon_dx)
    dump_pickle(os.path.join(args.out_dir, "alias2canon_icd9_proc.pkl"), alias2canon_pr)
    
    # Save names if requested
    if args.with_names:
        dump_pickle(os.path.join(args.out_dir, "code2name_atc.pkl"), atc_names)
        dump_pickle(os.path.join(args.out_dir, "code2name_loinc.pkl"), loinc_names)
        dump_pickle(os.path.join(args.out_dir, "code2name_snomedct_us.pkl"), snomed_names)
    
    # Step 6: Coverage analysis
    coverage_stats = {
        "master_dx": coverage_analysis(master_dx, icd9_dx_map),
        "master_proc": coverage_analysis(master_pr, icd9_pr_map),
        "dataset_dx": coverage_analysis(ds_dx, icd9_dx_map) if ds_dx else {},
        "dataset_proc": coverage_analysis(ds_pr, icd9_pr_map) if ds_pr else {}
    }
    
    with open(os.path.join(args.out_dir, "coverage_analysis.json"), "w") as f:
        json.dump(coverage_stats, f, indent=2, cls=NumpyEncoder)
    log("Coverage analysis saved")
    
    # Step 7: Build knowledge graph
    log("Building comprehensive knowledge graph with clean vocabulary filtering...")
    G, conso_df = build_knowledge_graph(
        umls_dir=args.umls_dir,
        out_dir=args.out_dir,
        code_mappings=code_mappings,
        target_vocabs=target_vocabs,
        log=log
    )
    
    # Step 8: Generate statistics
    log("Generating comprehensive statistics...")
    stats = generate_statistics(G, code_mappings, conso_df, args.out_dir, log)
    
    # Step 9: Final summary
    summary = {
        'vocabulary_counts': {k: len(v) for k, v in code_mappings.items()},
        'graph_stats': stats['graph'],
        'total_unique_codes': sum(len(v) for v in code_mappings.values()),
        'coverage_stats': coverage_stats,
        'target_vocabularies': target_vocabs,
        'sab_distribution': stats.get('sab_distribution', {})
    }
    
    summary_path = os.path.join(args.out_dir, 'comprehensive_build_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    log("\n=== COMPREHENSIVE SUMMARY ===")
    log(f"Vocabulary mappings built:")
    for vocab, mapping in code_mappings.items():
        log(f"  {vocab.upper()}: {len(mapping):,} codes")
    log(f"Knowledge Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    log(f"Total unique codes across all vocabularies: {summary['total_unique_codes']:,}")
    log(f"SAB distribution in final KG: {stats.get('sab_distribution', {})}")
    log(f"Coverage:")
    for coverage_type, cov_data in coverage_stats.items():
        if cov_data:
            log(f"  {coverage_type}: {cov_data['coverage_pct']:.1f}% ({cov_data['mapped']}/{cov_data['total']})")
    log(f"Output directory: {args.out_dir}")
    log("Comprehensive build completed successfully!")

if __name__ == "__main__":
    main()