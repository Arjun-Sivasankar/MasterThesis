#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_medical_fact_index_dataset_aware.py

Build a searchable medical fact index from UMLS KG with MEMORY OPTIMIZATION.
Only includes paths involving CUIs present in the dataset.

NEW: Stores relationship metadata explicitly for accurate weighted retrieval.

OPTIMIZATIONS:
  - Chunked FAISS index building
  - Memory cleanup after encoding
  - CPU-only FAISS (no GPU memory issues)
  - Periodic garbage collection
  - Explicit relationship metadata storage
"""

import argparse
import pickle
import json
import logging
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import gc

import numpy as np
import pandas as pd
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Run: pip install faiss-cpu")
    sys.exit(1)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CURATED_RELATIONSHIPS = {
    'isa', 'other', 'location', 'measurement', 'assoc', 'meta',
    'proc_method', 'morphology', 'finding_site', 'equivalent',
    'etiology', 'temporal', 'pathology', 'procedure_method',
    'proc_site', 'intent', 'proc_device', 'course', 'priority',
    'severity', 'procedure_device', 'clinical_course', 'may_treat',
    'may_cause', 'associated_with'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _strip(x) -> str:
    return str(x or "").strip().upper().replace(" ", "")

def format_icd9_proc_from_pro(c: str) -> str:
    s = _strip(c)
    if s.startswith("PRO_"): s = s[4:]
    s = re.sub(r"[^0-9]", "", s)
    if not s: return ""
    if len(s) >= 3:
        return s[:2] + "." + s[2:]
    return s

def format_icd9(code: str) -> str:
    """Format ICD9 diagnosis code."""
    c = _strip(code)
    if not c or c.upper() in ("NAN", "NONE"):
        return ""
    c = c.replace(".", "")
    if len(c) > 3:
        c = c[:3] + "." + c[3:]
    return c

def to_list(x) -> List[str]:
    """Convert various types to list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        it = x.tolist() if hasattr(x, "tolist") else x
        out = []
        for v in it:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            sv = str(v).strip()
            if sv and sv.lower() not in ("nan", "none"):
                out.append(sv)
        return out
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
    return [t for t in re.split(r"[,\s]+", s) if t]

# =============================================================================
# DATASET CODE EXTRACTION WITH STATISTICS
# =============================================================================

def extract_codes_from_dataframe(df: pd.DataFrame, split_name: str) -> Dict[str, Set[str]]:
    """Extract all unique codes from a dataframe by code type."""
    codes_by_type = {
        'medications': set(),
        'labs': set(),
        'procedures': set(),
        'diagnoses': set()
    }
    
    if 'ndc' in df.columns:
        log.info(f"  Extracting medication codes (NDC/ATC) from {split_name}...")
        for ndc_list in df['ndc']:
            codes_by_type['medications'].update(to_list(ndc_list))
        log.info(f"    → Found {len(codes_by_type['medications'])} unique medication codes")
    
    if 'lab_test_loinc' in df.columns:
        log.info(f"  Extracting lab codes (LOINC) from {split_name}...")
        for loinc_list in df['lab_test_loinc']:
            codes_by_type['labs'].update(to_list(loinc_list))
        log.info(f"    → Found {len(codes_by_type['labs'])} unique lab codes")
    
    if 'pro_code' in df.columns:
        log.info(f"  Extracting procedure codes (ICD9-PROC) from {split_name}...")
        for proc_list in df['pro_code']:
            formatted = [format_icd9_proc_from_pro(p) for p in to_list(proc_list)]
            codes_by_type['procedures'].update(formatted)
        log.info(f"    → Found {len(codes_by_type['procedures'])} unique procedure codes")
    
    if 'icd_code' in df.columns:
        log.info(f"  Extracting diagnosis codes (ICD9-DIAG) from {split_name}...")
        for icd_list in df['icd_code']:
            formatted = [format_icd9(c) for c in to_list(icd_list)]
            codes_by_type['diagnoses'].update(formatted)
        log.info(f"    → Found {len(codes_by_type['diagnoses'])} unique diagnosis codes")
    
    for key in codes_by_type:
        codes_by_type[key].discard("")
    
    return codes_by_type

def map_codes_to_cuis_detailed(
    codes_by_type: Dict[str, Set[str]],
    atc_map: Dict[str, List[str]],
    loinc_map: Dict[str, List[str]],
    proc_map: Dict[str, List[str]],
    split_name: str
) -> Tuple[Set[str], Dict[str, Dict]]:
    """Map dataset codes to CUIs and track statistics."""
    log.info(f"\n  Mapping {split_name} codes to CUIs...")
    
    cuis = set()
    mapping_stats = {
        'medications': {'codes': 0, 'mapped': 0, 'cuis': set()},
        'labs': {'codes': 0, 'mapped': 0, 'cuis': set()},
        'procedures': {'codes': 0, 'mapped': 0, 'cuis': set()},
        'unmapped': {'codes': 0}
    }
    
    log.info(f"    Mapping medications (ATC)...")
    for code in codes_by_type['medications']:
        mapping_stats['medications']['codes'] += 1
        if code in atc_map:
            cui_list = atc_map[code]
            cuis.update(cui_list)
            mapping_stats['medications']['cuis'].update(cui_list)
            mapping_stats['medications']['mapped'] += 1
        else:
            mapping_stats['unmapped']['codes'] += 1
    
    log.info(f"      → {mapping_stats['medications']['mapped']}/{mapping_stats['medications']['codes']} mapped "
             f"({100*mapping_stats['medications']['mapped']/max(1, mapping_stats['medications']['codes']):.1f}%) "
             f"→ {len(mapping_stats['medications']['cuis'])} CUIs")
    
    log.info(f"    Mapping labs (LOINC)...")
    for code in codes_by_type['labs']:
        mapping_stats['labs']['codes'] += 1
        if code in loinc_map:
            cui_list = loinc_map[code]
            cuis.update(cui_list)
            mapping_stats['labs']['cuis'].update(cui_list)
            mapping_stats['labs']['mapped'] += 1
        else:
            mapping_stats['unmapped']['codes'] += 1
    
    log.info(f"      → {mapping_stats['labs']['mapped']}/{mapping_stats['labs']['codes']} mapped "
             f"({100*mapping_stats['labs']['mapped']/max(1, mapping_stats['labs']['codes']):.1f}%) "
             f"→ {len(mapping_stats['labs']['cuis'])} CUIs")
    
    log.info(f"    Mapping procedures (ICD9-PROC)...")
    for code in codes_by_type['procedures']:
        mapping_stats['procedures']['codes'] += 1
        if code in proc_map:
            cui_list = proc_map[code]
            cuis.update(cui_list)
            mapping_stats['procedures']['cuis'].update(cui_list)
            mapping_stats['procedures']['mapped'] += 1
        else:
            mapping_stats['unmapped']['codes'] += 1
    
    log.info(f"      → {mapping_stats['procedures']['mapped']}/{mapping_stats['procedures']['codes']} mapped "
             f"({100*mapping_stats['procedures']['mapped']/max(1, mapping_stats['procedures']['codes']):.1f}%) "
             f"→ {len(mapping_stats['procedures']['cuis'])} CUIs")
    
    if codes_by_type['diagnoses']:
        log.info(f"    Diagnoses ({len(codes_by_type['diagnoses'])} codes): Already in KG via UMLS/SNOMED")
    
    return cuis, mapping_stats

def extract_dataset_cuis(
    train_pkl: str, 
    val_pkl: str, 
    test_pkl: str,
    atc_map: Dict, 
    loinc_map: Dict, 
    proc_map: Dict
) -> Tuple[Set[str], Dict]:
    """Extract all CUIs present in the entire dataset with detailed statistics."""
    log.info("="*80)
    log.info("EXTRACTING DATASET CUIs WITH DETAILED STATISTICS")
    log.info("="*80)
    
    overall_stats = {
        'train': {'n_visits': 0, 'codes': {}, 'mapping': {}},
        'val': {'n_visits': 0, 'codes': {}, 'mapping': {}},
        'test': {'n_visits': 0, 'codes': {}, 'mapping': {}},
        'total': {
            'n_visits': 0,
            'codes_by_type': {
                'medications': 0, 'labs': 0, 'procedures': 0, 'diagnoses': 0
            },
            'total_codes': 0,
            'mapped_codes': 0,
            'unmapped_codes': 0,
            'total_cuis': 0
        }
    }
    
    all_cuis = set()
    train_cuis = set()
    val_cuis = set()
    test_cuis = set()
    
    # TRAIN SET
    if train_pkl and Path(train_pkl).exists():
        log.info(f"\n{'='*80}")
        log.info(f"TRAIN SET")
        log.info(f"{'='*80}")
        log.info(f"Loading train data from {train_pkl}")
        train_df = pd.read_pickle(train_pkl)
        overall_stats['train']['n_visits'] = len(train_df)
        log.info(f"  ✓ Train set: {len(train_df)} visits\n")
        
        train_codes = extract_codes_from_dataframe(train_df, "train")
        overall_stats['train']['codes'] = {k: len(v) for k, v in train_codes.items()}
        
        train_cuis, train_mapping = map_codes_to_cuis_detailed(
            train_codes, atc_map, loinc_map, proc_map, "train"
        )
        all_cuis.update(train_cuis)
        overall_stats['train']['mapping'] = {}
        for k, v in train_mapping.items():
            if k == 'unmapped':
                overall_stats['train']['mapping'][k] = {'codes': v['codes']}
            else:
                overall_stats['train']['mapping'][k] = {
                    'codes': v['codes'],
                    'mapped': v['mapped'],
                    'cuis': len(v.get('cuis', set()))
                }
        log.info(f"\n  ✓ Train CUIs extracted: {len(train_cuis)}")
        
        del train_df, train_codes
        gc.collect()
    
    # VAL SET
    if val_pkl and Path(val_pkl).exists():
        log.info(f"\n{'='*80}")
        log.info(f"VAL SET")
        log.info(f"{'='*80}")
        log.info(f"Loading val data from {val_pkl}")
        val_df = pd.read_pickle(val_pkl)
        overall_stats['val']['n_visits'] = len(val_df)
        log.info(f"  ✓ Val set: {len(val_df)} visits\n")
        
        val_codes = extract_codes_from_dataframe(val_df, "val")
        overall_stats['val']['codes'] = {k: len(v) for k, v in val_codes.items()}
        
        val_cuis, val_mapping = map_codes_to_cuis_detailed(
            val_codes, atc_map, loinc_map, proc_map, "val"
        )
        all_cuis.update(val_cuis)
        overall_stats['val']['mapping'] = {}
        for k, v in val_mapping.items():
            if k == 'unmapped':
                overall_stats['val']['mapping'][k] = {'codes': v['codes']}
            else:
                overall_stats['val']['mapping'][k] = {
                    'codes': v['codes'],
                    'mapped': v['mapped'],
                    'cuis': len(v.get('cuis', set()))
                }
        log.info(f"\n  ✓ Val CUIs extracted: {len(val_cuis)}")
        
        del val_df, val_codes
        gc.collect()
    
    # TEST SET
    if test_pkl and Path(test_pkl).exists():
        log.info(f"\n{'='*80}")
        log.info(f"TEST SET")
        log.info(f"{'='*80}")
        log.info(f"Loading test data from {test_pkl}")
        test_df = pd.read_pickle(test_pkl)
        overall_stats['test']['n_visits'] = len(test_df)
        log.info(f"  ✓ Test set: {len(test_df)} visits\n")
        
        test_codes = extract_codes_from_dataframe(test_df, "test")
        overall_stats['test']['codes'] = {k: len(v) for k, v in test_codes.items()}
        
        test_cuis, test_mapping = map_codes_to_cuis_detailed(
            test_codes, atc_map, loinc_map, proc_map, "test"
        )
        all_cuis.update(test_cuis)
        overall_stats['test']['mapping'] = {}
        for k, v in test_mapping.items():
            if k == 'unmapped':
                overall_stats['test']['mapping'][k] = {'codes': v['codes']}
            else:
                overall_stats['test']['mapping'][k] = {
                    'codes': v['codes'],
                    'mapped': v['mapped'],
                    'cuis': len(v.get('cuis', set()))
                }
        log.info(f"\n  ✓ Test CUIs extracted: {len(test_cuis)}")
        
        del test_df, test_codes
        gc.collect()
    
    # OVERALL SUMMARY
    log.info(f"\n{'='*80}")
    log.info("OVERALL DATASET STATISTICS")
    log.info(f"{'='*80}\n")
    
    overall_stats['total']['n_visits'] = (
        overall_stats['train']['n_visits'] +
        overall_stats['val']['n_visits'] +
        overall_stats['test']['n_visits']
    )
    log.info(f"Total visits: {overall_stats['total']['n_visits']}")
    log.info(f"  Train: {overall_stats['train']['n_visits']}")
    log.info(f"  Val:   {overall_stats['val']['n_visits']}")
    log.info(f"  Test:  {overall_stats['test']['n_visits']}")
    
    for code_type in ['medications', 'labs', 'procedures', 'diagnoses']:
        total = 0
        for split in ['train', 'val', 'test']:
            if 'codes' in overall_stats[split]:
                total += overall_stats[split]['codes'].get(code_type, 0)
        overall_stats['total']['codes_by_type'][code_type] = total
    
    overall_stats['total']['total_codes'] = sum(overall_stats['total']['codes_by_type'].values())
    
    log.info(f"\nTotal unique codes by type:")
    log.info(f"  Medications:  {overall_stats['total']['codes_by_type']['medications']:5d}")
    log.info(f"  Labs:         {overall_stats['total']['codes_by_type']['labs']:5d}")
    log.info(f"  Procedures:   {overall_stats['total']['codes_by_type']['procedures']:5d}")
    log.info(f"  Diagnoses:    {overall_stats['total']['codes_by_type']['diagnoses']:5d}")
    log.info(f"  ────────────────────")
    log.info(f"  TOTAL:        {overall_stats['total']['total_codes']:5d}")
    
    total_mapped = 0
    total_unmapped = 0
    for split in ['train', 'val', 'test']:
        if 'mapping' in overall_stats[split]:
            for code_type in ['medications', 'labs', 'procedures']:
                total_mapped += overall_stats[split]['mapping'].get(code_type, {}).get('mapped', 0)
            total_unmapped += overall_stats[split]['mapping'].get('unmapped', {}).get('codes', 0)
    
    overall_stats['total']['mapped_codes'] = total_mapped
    overall_stats['total']['unmapped_codes'] = total_unmapped
    
    mapping_rate = 100 * total_mapped / max(1, total_mapped + total_unmapped)
    
    log.info(f"\nCode mapping summary:")
    log.info(f"  Mapped codes:    {total_mapped:5d} ({mapping_rate:.1f}%)")
    log.info(f"  Unmapped codes:  {total_unmapped:5d} ({100-mapping_rate:.1f}%)")
    
    overall_stats['total']['total_cuis'] = len(all_cuis)
    
    log.info(f"\nCUI extraction:")
    log.info(f"  Total unique CUIs: {len(all_cuis)}")
    log.info(f"    From train: {len(train_cuis)}")
    log.info(f"    From val:   {len(val_cuis)}")
    log.info(f"    From test:  {len(test_cuis)}")
    
    log.info(f"\n{'='*80}")
    
    return all_cuis, overall_stats

# =============================================================================
# PATH MINING WITH STATISTICS
# =============================================================================

def mine_h1_paths_dataset_aware(
    G: nx.DiGraph, 
    dataset_cuis: Set[str],
    max_paths_per_source: int = None,
    require_target_in_dataset: bool = True
) -> Tuple[List[dict], Dict[str, int]]:
    """Mine H1 paths where source is in dataset CUIs."""
    log.info(f"Mining H1 paths (dataset-aware, require_target={'YES' if require_target_in_dataset else 'NO'})...")
    H1 = []
    paths_per_cui = defaultdict(int)
    paths_added_per_cui = defaultdict(int)
    
    total_edges = G.number_of_edges()
    
    with tqdm(total=total_edges, desc="H1 edges") as pbar:
        for u, v, edge_data in G.edges(data=True):
            pbar.update(1)
            
            if u not in dataset_cuis:
                continue
            
            if require_target_in_dataset and v not in dataset_cuis:
                continue
            
            rela_canon = edge_data.get("rela_canon", "")
            if rela_canon not in CURATED_RELATIONSHIPS:
                continue
            
            paths_per_cui[u] += 1
            
            if max_paths_per_source is not None:
                if paths_added_per_cui[u] >= max_paths_per_source:
                    continue
            
            u_name = G.nodes[u].get("name", u)
            v_name = G.nodes[v].get("name", v)
            
            H1.append({
                "src_cui": u,
                "nbr_cui": v,
                "src_name": u_name,
                "nbr_name": v_name,
                "rel": edge_data.get("rel", ""),
                "rela": edge_data.get("rela", ""),
                "rela_canon": rela_canon
            })
            
            paths_added_per_cui[u] += 1
    
    log.info(f"✓ Mined {len(H1)} H1 paths from {len(paths_added_per_cui)} source nodes")
    
    if paths_per_cui:
        all_counts = list(paths_per_cui.values())
        
        log.info(f"\n  H1 PATH STATISTICS (Available paths per CUI):")
        log.info(f"    CUIs with H1 paths: {len(paths_per_cui)}")
        log.info(f"    Total available paths: {sum(all_counts)}")
        log.info(f"    Mean paths per CUI: {np.mean(all_counts):.1f}")
        log.info(f"    Median paths per CUI: {np.median(all_counts):.1f}")
        log.info(f"    Min paths per CUI: {np.min(all_counts)}")
        log.info(f"    Max paths per CUI: {np.max(all_counts)}")
        log.info(f"    Std dev: {np.std(all_counts):.1f}")
        
        percentiles = [25, 50, 75, 90, 95, 99]
        log.info(f"    Percentiles:")
        for p in percentiles:
            val = np.percentile(all_counts, p)
            log.info(f"      {p}th: {val:.1f}")
        
        if max_paths_per_source is not None:
            added_counts = list(paths_added_per_cui.values())
            log.info(f"\n  H1 PATH STATISTICS (After limit of {max_paths_per_source}):")
            log.info(f"    Paths actually added: {len(H1)}")
            log.info(f"    CUIs affected by limit: {sum(1 for c in paths_per_cui if paths_per_cui[c] > max_paths_per_source)}")
            log.info(f"    Mean paths added per CUI: {np.mean(added_counts):.1f}")
    
    return H1, dict(paths_per_cui)

def mine_h2_paths_dataset_aware(
    G: nx.DiGraph,
    dataset_cuis: Set[str],
    max_paths_per_source: int = None,
    require_target_in_dataset: bool = True
) -> Tuple[List[dict], Dict[str, int]]:
    """Mine H2 paths where source is in dataset CUIs."""
    log.info(f"Mining H2 paths (dataset-aware, require_target={'YES' if require_target_in_dataset else 'NO'})...")
    H2 = []
    paths_per_cui = defaultdict(int)
    paths_added_per_cui = defaultdict(int)
    
    dataset_nodes = [n for n in G.nodes() if n in dataset_cuis]
    log.info(f"  Iterating over {len(dataset_nodes)} dataset nodes as sources...")
    
    for u in tqdm(dataset_nodes, desc="H2 paths"):
        for v in G.successors(u):
            edge_uv = G[u][v]
            rela_uv_canon = edge_uv.get("rela_canon", "")
            
            if rela_uv_canon not in CURATED_RELATIONSHIPS:
                continue
            
            for w in G.successors(v):
                if w == u:
                    continue
                
                if require_target_in_dataset and w not in dataset_cuis:
                    continue
                
                edge_vw = G[v][w]
                rela_vw_canon = edge_vw.get("rela_canon", "")
                
                if rela_vw_canon not in CURATED_RELATIONSHIPS:
                    continue
                
                paths_per_cui[u] += 1
                
                if max_paths_per_source is not None:
                    if paths_added_per_cui[u] >= max_paths_per_source:
                        continue
                
                u_name = G.nodes[u].get("name", u)
                v_name = G.nodes[v].get("name", v)
                w_name = G.nodes[w].get("name", w)
                
                H2.append({
                    "u": u,
                    "v": v,
                    "w": w,
                    "u_name": u_name,
                    "v_name": v_name,
                    "w_name": w_name,
                    "rel_uv": edge_uv.get("rel", ""),
                    "rela_uv": edge_uv.get("rela", ""),
                    "rela_uv_canon": rela_uv_canon,
                    "rel_vw": edge_vw.get("rel", ""),
                    "rela_vw": edge_vw.get("rela", ""),
                    "rela_vw_canon": rela_vw_canon
                })
                
                paths_added_per_cui[u] += 1
    
    log.info(f"✓ Mined {len(H2)} H2 paths from {len(paths_added_per_cui)} source nodes")
    
    if paths_per_cui:
        all_counts = list(paths_per_cui.values())
        
        log.info(f"\n  H2 PATH STATISTICS (Available paths per CUI):")
        log.info(f"    CUIs with H2 paths: {len(paths_per_cui)}")
        log.info(f"    Total available paths: {sum(all_counts)}")
        log.info(f"    Mean paths per CUI: {np.mean(all_counts):.1f}")
        log.info(f"    Median paths per CUI: {np.median(all_counts):.1f}")
        log.info(f"    Min paths per CUI: {np.min(all_counts)}")
        log.info(f"    Max paths per CUI: {np.max(all_counts)}")
        log.info(f"    Std dev: {np.std(all_counts):.1f}")
        
        percentiles = [25, 50, 75, 90, 95, 99]
        log.info(f"    Percentiles:")
        for p in percentiles:
            val = np.percentile(all_counts, p)
            log.info(f"      {p}th: {val:.1f}")
        
        if max_paths_per_source is not None:
            added_counts = list(paths_added_per_cui.values())
            log.info(f"\n  H2 PATH STATISTICS (After limit of {max_paths_per_source}):")
            log.info(f"    Paths actually added: {len(H2)}")
            log.info(f"    CUIs affected by limit: {sum(1 for c in paths_per_cui if paths_per_cui[c] > max_paths_per_source)}")
            log.info(f"    Mean paths added per CUI: {np.mean(added_counts):.1f}")
    
    return H2, dict(paths_per_cui)

# =============================================================================
# SAPBERT ENCODER WITH MEMORY OPTIMIZATION
# =============================================================================

class SapBERTEncoder:
    """Wrapper for SapBERT model with memory optimization."""
    
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        log.info(f"Loading SapBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        log.info(f"  SapBERT loaded on device: {self.device}")
    
    def _mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling over token embeddings."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode with memory cleanup."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # MEMORY CLEANUP
            del encoded, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Periodic garbage collection
            if i % 1000 == 0 and i > 0:
                gc.collect()
        
        result = np.vstack(all_embeddings)
        del all_embeddings
        gc.collect()
        
        return result

# =============================================================================
# PATH LINEARIZATION WITH METADATA (UPDATED)
# =============================================================================

def clean_name(name: str) -> str:
    """Clean concept name for readability."""
    if not name:
        return ""
    name = re.sub(r'^C\d+:', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def linearize_h1_path_with_metadata(path: dict) -> Tuple[str, str]:
    """
    Convert H1 path to natural language + extract relationship metadata.
    
    Returns:
        Tuple[str, str]: (fact_text, relationship_type)
    """
    src = clean_name(path.get("src_name", ""))
    tgt = clean_name(path.get("nbr_name", ""))
    rel = path.get("rela_canon") or path.get("rela") or path.get("rel") or "related_to"
    rel_text = rel.strip().replace("_", " ")
    
    if not src or not tgt:
        return "", "other"
    
    fact_text = f"{src} {rel_text} {tgt}"
    relationship_type = path.get("rela_canon", "other")  # Ground truth from KG
    
    return fact_text, relationship_type

def linearize_h2_path_with_metadata(path: dict) -> Tuple[str, str]:
    """
    Convert H2 path to natural language + extract FIRST relationship.
    
    Returns:
        Tuple[str, str]: (fact_text, relationship_type)
    """
    u = clean_name(path.get("u_name", ""))
    v = clean_name(path.get("v_name", ""))
    w = clean_name(path.get("w_name", ""))
    
    rel_uv = path.get("rela_uv_canon") or path.get("rela_uv") or "related_to"
    rel_vw = path.get("rela_vw_canon") or path.get("rela_vw") or "related_to"
    
    rel_uv_text = rel_uv.strip().replace("_", " ")
    rel_vw_text = rel_vw.strip().replace("_", " ")
    
    if not u or not v or not w:
        return "", "other"
    
    fact_text = f"{u} {rel_uv_text} {v} which {rel_vw_text} {w}"
    relationship_type = path.get("rela_uv_canon", "other")  # Use FIRST relationship
    
    return fact_text, relationship_type

# =============================================================================
# FAISS INDEX WITH CHUNKING (MEMORY OPTIMIZED)
# =============================================================================

def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """Create FAISS index with chunked adding to reduce memory pressure."""
    n, d = embeddings.shape
    log.info(f"Creating FAISS index: {n:,} vectors, {d} dimensions")
    
    # Normalize IN PLACE
    faiss.normalize_L2(embeddings)
    
    # ALWAYS use CPU to avoid GPU OOM
    log.info("  Using CPU FAISS (memory optimized)")
    index = faiss.IndexFlatIP(d)
    
    # Add in chunks
    chunk_size = 10000
    log.info(f"  Adding embeddings in chunks of {chunk_size:,}...")
    
    for i in tqdm(range(0, n, chunk_size), desc="Building index"):
        end_idx = min(i + chunk_size, n)
        chunk = embeddings[i:end_idx].astype(np.float32)
        index.add(chunk)
        
        del chunk
        if i % 50000 == 0 and i > 0:
            gc.collect()
    
    log.info(f"✓ FAISS index created with {index.ntotal:,} vectors")
    return index

# =============================================================================
# SAVE WITH RELATIONSHIP METADATA (UPDATED)
# =============================================================================

def save_fact_index(facts: List[str], 
                    embeddings: np.ndarray, 
                    index: faiss.Index,
                    relationships: List[str],  # NEW
                    output_dir: Path,
                    metadata: dict = None):
    """Save the complete fact index with relationship metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\nSaving fact index to {output_dir}")
    
    # Validate alignment
    assert len(facts) == len(relationships) == embeddings.shape[0], \
        f"Misalignment: {len(facts)} facts, {len(relationships)} rels, {embeddings.shape[0]} embeddings"
    
    # Save facts
    facts_file = output_dir / "facts.json"
    with open(facts_file, 'w', encoding='utf-8') as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    log.info(f"  ✓ Saved {len(facts)} facts to {facts_file}")
    
    # Save relationships (NEW)
    relationships_file = output_dir / "relationships.json"
    with open(relationships_file, 'w', encoding='utf-8') as f:
        json.dump(relationships, f, ensure_ascii=False, indent=2)
    log.info(f"  ✓ Saved {len(relationships)} relationships to {relationships_file}")
    
    # Relationship distribution stats
    rel_counter = Counter(relationships)
    log.info(f"\n  Relationship distribution (top 10):")
    for rel, count in rel_counter.most_common(10):
        pct = 100 * count / len(relationships)
        log.info(f"    {rel:20s}: {count:6d} ({pct:5.1f}%)")
    
    # Save embeddings
    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    log.info(f"\n  ✓ Saved embeddings ({embeddings.shape}) to {embeddings_file}")
    
    # Save FAISS index
    index_file = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_file))
    log.info(f"  ✓ Saved FAISS index to {index_file}")
    
    # Save metadata
    meta = {
        "n_facts": len(facts),
        "embedding_dim": embeddings.shape[1],
        "index_type": "IndexFlatIP",
        "metric": "cosine",
        "normalized": True,
        "has_relationship_metadata": True,
        "unique_relationships": len(rel_counter)
    }
    if metadata:
        meta.update(metadata)
    
    meta_file = output_dir / "metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  ✓ Saved metadata to {meta_file}")
    
    log.info(f"\n✓ Complete fact index saved to {output_dir}")

# =============================================================================
# MAIN (UPDATED)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Dataset-Aware Medical Fact Index with Relationship Metadata")
    
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data")
    parser.add_argument("--atc_map", required=True)
    parser.add_argument("--loinc_map", required=True)
    parser.add_argument("--proc_map", required=True)
    parser.add_argument("--kg_pkl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_h1_per_source", type=int, default=None)
    parser.add_argument("--max_h2_per_source", type=int, default=None)
    parser.add_argument("--require_target_in_dataset", action="store_true")
    parser.add_argument("--sapbert_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_gpu_faiss", action="store_true")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    start_time = time.time()
    
    # STEP 1: LOAD MAPPINGS
    log.info("="*80)
    log.info("STEP 1: LOADING CODE MAPPINGS")
    log.info("="*80)
    
    with open(args.atc_map, "rb") as f:
        atc_map = pickle.load(f)
    log.info(f"✓ ATC mappings: {len(atc_map)}")
    
    with open(args.loinc_map, "rb") as f:
        loinc_map = pickle.load(f)
    log.info(f"✓ LOINC mappings: {len(loinc_map)}")
    
    with open(args.proc_map, "rb") as f:
        proc_map = pickle.load(f)
    log.info(f"✓ ICD9-PROC mappings: {len(proc_map)}")
    
    # STEP 2: EXTRACT CUIs
    dataset_cuis, code_stats = extract_dataset_cuis(
        args.train_data, args.val_data, args.test_data,
        atc_map, loinc_map, proc_map
    )
    
    # Clean up mapping dicts
    del atc_map, loinc_map, proc_map
    gc.collect()
    
    # STEP 3: LOAD KG
    log.info("\n" + "="*80)
    log.info("STEP 3: LOADING KNOWLEDGE GRAPH")
    log.info("="*80)
    
    with open(args.kg_pkl, "rb") as f:
        G = pickle.load(f)
    log.info(f"✓ KG loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # STEP 4: MINE PATHS
    log.info("\n" + "="*80)
    log.info("STEP 4: MINING PATHS")
    log.info("="*80)
    
    mining_start = time.time()
    
    H1_paths, h1_stats = mine_h1_paths_dataset_aware(
        G, dataset_cuis, 
        args.max_h1_per_source,
        args.require_target_in_dataset
    )
    
    H2_paths, h2_stats = mine_h2_paths_dataset_aware(
        G, dataset_cuis, 
        args.max_h2_per_source,
        args.require_target_in_dataset
    )
    
    mining_time = time.time() - mining_start
    log.info(f"\n✓ Mining complete in {mining_time/60:.1f} minutes")
    
    # Clean up KG
    del G
    gc.collect()
    
    # STEP 5: LINEARIZE WITH METADATA (UPDATED)
    log.info("\n" + "="*80)
    log.info("STEP 5: LINEARIZING WITH RELATIONSHIP METADATA")
    log.info("="*80)
    
    facts = []
    relationships = []  # NEW: parallel array of relationship types
    
    log.info("Processing H1 paths...")
    for path in tqdm(H1_paths, desc="H1"):
        fact, rel = linearize_h1_path_with_metadata(path)
        if fact:
            facts.append(fact)
            relationships.append(rel)
    
    log.info("Processing H2 paths...")
    for path in tqdm(H2_paths, desc="H2"):
        fact, rel = linearize_h2_path_with_metadata(path)
        if fact:
            facts.append(fact)
            relationships.append(rel)
    
    log.info(f"✓ Linearized {len(facts):,} facts with relationship metadata")
    
    # Verify alignment
    assert len(facts) == len(relationships), \
        f"Mismatch: {len(facts)} facts vs {len(relationships)} relationships"
    
    log.info("\nExample facts with relationships:")
    for i in range(min(10, len(facts))):
        log.info(f"  {i+1}. [{relationships[i]:15s}] {facts[i]}")
    
    # Clean up paths
    del H1_paths, H2_paths
    gc.collect()
    
    # STEP 6: ENCODE
    log.info("\n" + "="*80)
    log.info("STEP 6: ENCODING WITH SAPBERT")
    log.info("="*80)
    
    encoder = SapBERTEncoder(args.sapbert_model)
    
    encoding_start = time.time()
    embeddings = encoder.encode_batch(facts, batch_size=args.batch_size)
    encoding_time = time.time() - encoding_start
    
    log.info(f"✓ Encoding complete in {encoding_time/60:.1f} minutes")
    log.info(f"  Embeddings shape: {embeddings.shape}")
    
    # Verify alignment
    assert embeddings.shape[0] == len(facts) == len(relationships), \
        f"Alignment error: {embeddings.shape[0]} embeddings, {len(facts)} facts, {len(relationships)} rels"
    
    # CLEAR ENCODER FROM MEMORY
    del encoder.model, encoder.tokenizer, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log.info("  ✓ Cleared encoder from memory")
    
    # STEP 7: CREATE FAISS INDEX
    log.info("\n" + "="*80)
    log.info("STEP 7: CREATING FAISS INDEX")
    log.info("="*80)
    
    index = create_faiss_index(embeddings, use_gpu=False)
    
    # STEP 8: SAVE WITH RELATIONSHIPS (UPDATED)
    log.info("\n" + "="*80)
    log.info("STEP 8: SAVING WITH RELATIONSHIP METADATA")
    log.info("="*80)
    
    metadata = {
        "n_dataset_cuis": len(dataset_cuis),
        "n_h1_paths": len(h1_stats),
        "n_h2_paths": len(h2_stats),
        "max_h1_per_source": args.max_h1_per_source,
        "max_h2_per_source": args.max_h2_per_source,
        "require_target_in_dataset": args.require_target_in_dataset,
        "sapbert_model": args.sapbert_model
    }
    
    save_fact_index(facts, embeddings, index, relationships, output_dir, metadata)
    
    # SAVE STATISTICS
    stats = {
        "dataset_statistics": code_stats,
        "h1_statistics": {
            "total_available_paths": sum(h1_stats.values()) if h1_stats else 0,
            "cuis_with_paths": len(h1_stats),
            "mean": float(np.mean(list(h1_stats.values()))) if h1_stats else 0,
            "median": float(np.median(list(h1_stats.values()))) if h1_stats else 0,
            "percentiles": {
                f"{p}th": float(np.percentile(list(h1_stats.values()), p))
                for p in [25, 50, 75, 90, 95, 99]
            } if h1_stats else {}
        },
        "h2_statistics": {
            "total_available_paths": sum(h2_stats.values()) if h2_stats else 0,
            "cuis_with_paths": len(h2_stats),
            "mean": float(np.mean(list(h2_stats.values()))) if h2_stats else 0,
            "median": float(np.median(list(h2_stats.values()))) if h2_stats else 0,
            "percentiles": {
                f"{p}th": float(np.percentile(list(h2_stats.values()), p))
                for p in [25, 50, 75, 90, 95, 99]
            } if h2_stats else {}
        }
    }
    
    stats_file = output_dir / "comprehensive_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    log.info(f"✓ Saved statistics to {stats_file}")
    
    total_time = time.time() - start_time
    log.info("\n" + "="*80)
    log.info("✓ FACT INDEX BUILD COMPLETE")
    log.info("="*80)
    log.info(f"Total time: {total_time/60:.1f} minutes")
    log.info(f"Output: {output_dir}")
    log.info(f"Files: facts.json, relationships.json, embeddings.npy, faiss_index.bin, metadata.json")
    log.info("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())