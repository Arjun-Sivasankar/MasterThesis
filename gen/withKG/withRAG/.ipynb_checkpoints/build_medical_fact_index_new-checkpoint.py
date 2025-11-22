# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# build_medical_fact_index_dataset_aware.py

# Build separate searchable medical fact indexes from UMLS KG with MEMORY OPTIMIZATION.
# - Separate H1 and H2 indexes for flexible retrieval
# - Diagnosis CUIs as TARGETS ONLY (not sources from dataset)
# - Source CUIs come from medications, labs, procedures only
# - Separate limits for general paths vs diagnosis-targeting paths
# - Path deduplication to avoid repeated paths
# - Dual limits: per-source AND per-target

# KEY PRINCIPLE: 
# - ICD diagnosis codes are TARGETS to predict, not evidence sources
# - Only medications, labs, procedures contribute to source CUIs
# - Diagnosis CUIs from code2cui_icd9_dx.pkl are potential targets
# - Paths are deduplicated (no path appears twice)
# - Per-source limits: control how many diagnosis paths each source can have
# - Per-target limits: control how many incoming paths each diagnosis can receive
# """

# import sys
# import os
# import gc
# import json
# import time
# import pickle
# import logging
# import argparse
# import numpy as np
# import pandas as pd
# import networkx as nx
# import torch
# import re
# from pathlib import Path
# from typing import List, Dict, Tuple, Set
# from collections import defaultdict, Counter
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm

# try:
#     import faiss
# except ImportError:
#     print("ERROR: faiss not installed. Run: pip install faiss-cpu")
#     sys.exit(1)

# # =============================================================================
# # LOGGING
# # =============================================================================

# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# log = logging.getLogger(__name__)

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# CURATED_RELATIONSHIPS = {
#     'isa', 'other', 'location', 'measurement', 'assoc', 'meta',
#     'proc_method', 'morphology', 'finding_site', 'equivalent',
#     'etiology', 'temporal', 'pathology', 'procedure_method',
#     'proc_site', 'intent', 'proc_device', 'course', 'priority',
#     'severity', 'procedure_device', 'clinical_course', 'may_treat',
#     'may_cause', 'associated_with'
# }

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def _strip(x) -> str:
#     return str(x or "").strip().upper().replace(" ", "")

# def format_icd9_proc_from_pro(c: str) -> str:
#     s = _strip(c)
#     if s.startswith("PRO_"): 
#         s = s[4:]
#     s = re.sub(r"[^0-9]", "", s)
#     if not s: 
#         return ""
#     if len(s) >= 3:
#         return f"{s[:2]}.{s[2:]}"
#     return s

# def format_icd9(code: str) -> str:
#     """Format ICD9 diagnosis code."""
#     c = _strip(code)
#     if not c or c.upper() in ("NAN", "NONE"):
#         return ""
#     c = c.replace(".", "")
#     if len(c) > 3:
#         return f"{c[:3]}.{c[3:]}"
#     return c

# def to_list(x) -> List[str]:
#     """Convert various types to list of strings."""
#     if x is None:
#         return []
#     if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
#         return [str(t).strip() for t in x if str(t).strip()]
#     s = str(x).strip()
#     if not s or s.lower() in ("nan", "none"):
#         return []
#     if s.startswith("[") and s.endswith("]"):
#         s = s[1:-1]
#     return [t for t in re.split(r"[,\s]+", s) if t]

# # =============================================================================
# # DATASET CODE EXTRACTION - NO ICD CODES AS SOURCES
# # =============================================================================

# def extract_source_codes_from_dataframe(df: pd.DataFrame, split_name: str) -> Dict[str, Set[str]]:
#     """
#     Extract SOURCE codes from dataframe - EXCLUDES ICD diagnosis codes.
    
#     ICD codes are targets to predict, not evidence sources.
#     Only medications, labs, procedures are valid sources.
#     """
#     codes_by_type = {
#         'medications': set(),
#         'labs': set(), 
#         'procedures': set()
#         # NOTE: NO 'diagnoses' - they are targets only!
#     }
    
#     # Medications (NDC codes)
#     if 'ndc' in df.columns:
#         for ndc_list in df['ndc'].dropna():
#             codes_by_type['medications'].update(to_list(ndc_list))
    
#     # Lab tests (LOINC codes)  
#     if 'lab_test_loinc' in df.columns:
#         for loinc_list in df['lab_test_loinc'].dropna():
#             codes_by_type['labs'].update(to_list(loinc_list))
    
#     # Procedures (ICD9-PROC codes)
#     if 'pro_code' in df.columns:
#         for proc_list in df['pro_code'].dropna():
#             formatted = [format_icd9_proc_from_pro(c) for c in to_list(proc_list)]
#             codes_by_type['procedures'].update([c for c in formatted if c])
    
#     # Clean empty entries
#     for key in codes_by_type:
#         codes_by_type[key] = {c for c in codes_by_type[key] if c and c.strip()}
    
#     return codes_by_type

# def map_source_codes_to_cuis(
#     codes_by_type: Dict[str, Set[str]],
#     atc_map: Dict[str, List[str]],
#     loinc_map: Dict[str, List[str]], 
#     proc_map: Dict[str, List[str]],
#     split_name: str
# ) -> Tuple[Set[str], Dict[str, Dict]]:
#     """Map SOURCE codes to CUIs - no diagnosis codes included."""
#     log.info(f"\n  Mapping {split_name} SOURCE codes to CUIs (excluding diagnosis codes)...")
    
#     source_cuis = set()
#     mapping_stats = {
#         'medications': {'codes': 0, 'mapped': 0, 'cuis': set()},
#         'labs': {'codes': 0, 'mapped': 0, 'cuis': set()},
#         'procedures': {'codes': 0, 'mapped': 0, 'cuis': set()},
#         'unmapped': {'codes': 0}
#     }
    
#     # Medications (ATC)
#     log.info(f"    Mapping medications (ATC)...")
#     for code in codes_by_type['medications']:
#         mapping_stats['medications']['codes'] += 1
#         if code in atc_map:
#             cui_list = atc_map[code]
#             mapping_stats['medications']['mapped'] += 1
#             mapping_stats['medications']['cuis'].update(cui_list)
#             source_cuis.update(cui_list)
#         else:
#             mapping_stats['unmapped']['codes'] += 1
    
#     log.info(f"      → {mapping_stats['medications']['mapped']}/{mapping_stats['medications']['codes']} mapped "
#              f"({100*mapping_stats['medications']['mapped']/max(1, mapping_stats['medications']['codes']):.1f}%) "
#              f"→ {len(mapping_stats['medications']['cuis'])} CUIs")
    
#     # Labs (LOINC)
#     log.info(f"    Mapping labs (LOINC)...")
#     for code in codes_by_type['labs']:
#         mapping_stats['labs']['codes'] += 1
#         if code in loinc_map:
#             cui_list = loinc_map[code]
#             mapping_stats['labs']['mapped'] += 1
#             mapping_stats['labs']['cuis'].update(cui_list)
#             source_cuis.update(cui_list)
#         else:
#             mapping_stats['unmapped']['codes'] += 1
    
#     log.info(f"      → {mapping_stats['labs']['mapped']}/{mapping_stats['labs']['codes']} mapped "
#              f"({100*mapping_stats['labs']['mapped']/max(1, mapping_stats['labs']['codes']):.1f}%) "
#              f"→ {len(mapping_stats['labs']['cuis'])} CUIs")
    
#     # Procedures (ICD9-PROC)
#     log.info(f"    Mapping procedures (ICD9-PROC)...")
#     for code in codes_by_type['procedures']:
#         mapping_stats['procedures']['codes'] += 1
#         if code in proc_map:
#             cui_list = proc_map[code]
#             mapping_stats['procedures']['mapped'] += 1
#             mapping_stats['procedures']['cuis'].update(cui_list)
#             source_cuis.update(cui_list)
#         else:
#             mapping_stats['unmapped']['codes'] += 1
    
#     log.info(f"      → {mapping_stats['procedures']['mapped']}/{mapping_stats['procedures']['codes']} mapped "
#              f"({100*mapping_stats['procedures']['mapped']/max(1, mapping_stats['procedures']['codes']):.1f}%) "
#              f"→ {len(mapping_stats['procedures']['cuis'])} CUIs")
    
#     return source_cuis, mapping_stats

# def extract_diagnosis_target_cuis(dx_map: Dict[str, List[str]]) -> Tuple[Set[str], Dict]:
#     """
#     Extract ALL diagnosis CUIs as potential targets from the pkl mapping file.
    
#     Args:
#         dx_map: code2cui_icd9_dx.pkl mapping
        
#     Returns:
#         diagnosis_cuis: Set of all diagnosis CUIs
#         diagnosis_stats: Statistics about diagnosis CUIs
#     """
#     log.info(f"\nExtracting diagnosis target CUIs from pkl mapping file...")
    
#     diagnosis_cuis = set()
#     for icd_code, cui_list in dx_map.items():
#         diagnosis_cuis.update(cui_list)
    
#     diagnosis_stats = {
#         'total_icd_codes_in_mapping': len(dx_map),
#         'unique_diagnosis_cuis': len(diagnosis_cuis),
#         'avg_cuis_per_code': len(diagnosis_cuis) / len(dx_map) if dx_map else 0
#     }
    
#     log.info(f"✓ Extracted {len(diagnosis_cuis)} unique diagnosis CUIs from {len(dx_map)} ICD9-DX codes")
#     log.info(f"  Average CUIs per ICD code: {diagnosis_stats['avg_cuis_per_code']:.2f}")
    
#     return diagnosis_cuis, diagnosis_stats

# def extract_dataset_source_cuis(
#     train_pkl: str, 
#     val_pkl: str, 
#     test_pkl: str,
#     atc_map: Dict, 
#     loinc_map: Dict, 
#     proc_map: Dict,
#     dx_map: Dict
# ) -> Tuple[Set[str], Dict, Set[str], Dict]:
#     """
#     Extract SOURCE CUIs from dataset (no diagnosis codes) and ALL diagnosis target CUIs from pkl.
    
#     Returns:
#         - source_cuis: CUIs from medications, labs, procedures in dataset
#         - stats: mapping statistics  
#         - diagnosis_target_cuis: ALL possible diagnosis CUIs (from dx_map pkl)
#         - diagnosis_stats: Statistics about diagnosis CUIs from pkl
#     """
#     log.info("="*80)
#     log.info("EXTRACTING SOURCE CUIs (NO DIAGNOSIS) + TARGET DIAGNOSIS CUIs")
#     log.info("="*80)
    
#     all_source_cuis = set()
    
#     # Extract source CUIs from each dataset split (no diagnosis codes)
#     for split_name, pkl_path in [("train", train_pkl), ("val", val_pkl), ("test", test_pkl)]:
#         if pkl_path and Path(pkl_path).exists():
#             log.info(f"\n📊 Processing {split_name} split: {pkl_path}")
#             df = pd.read_pickle(pkl_path)
#             log.info(f"  Loaded {len(df)} visits")
            
#             # Extract source codes (medications, labs, procedures)
#             source_codes_by_type = extract_source_codes_from_dataframe(df, split_name)
#             source_cuis, mapping_stats = map_source_codes_to_cuis(
#                 source_codes_by_type, atc_map, loinc_map, proc_map, split_name
#             )
#             all_source_cuis.update(source_cuis)
    
#     # Extract ALL diagnosis target CUIs from pkl mapping (NOT from dataset)
#     diagnosis_target_cuis, diagnosis_stats = extract_diagnosis_target_cuis(dx_map)
    
#     # Summary
#     log.info(f"\n{'='*80}")
#     log.info("OVERALL CUI EXTRACTION SUMMARY")
#     log.info(f"{'='*80}")
#     log.info(f"SOURCE CUIs (from dataset): {len(all_source_cuis)}")
#     log.info(f"  - From medications, labs, procedures only")
#     log.info(f"  - ICD diagnosis codes EXCLUDED (they are targets)")
#     log.info(f"")
#     log.info(f"DIAGNOSIS TARGET CUIs (from pkl mapping): {len(diagnosis_target_cuis)}")
#     log.info(f"  - From code2cui_icd9_dx.pkl")
#     log.info(f"  - Total ICD codes: {diagnosis_stats['total_icd_codes_in_mapping']}")
#     log.info(f"  - Unique diagnosis CUIs: {diagnosis_stats['unique_diagnosis_cuis']}")
#     log.info(f"")
#     log.info(f"OVERLAP: {len(all_source_cuis & diagnosis_target_cuis)} CUIs appear as both source and diagnosis target")
#     log.info(f"{'='*80}")
    
#     return all_source_cuis, {}, diagnosis_target_cuis, diagnosis_stats

# # =============================================================================
# # PATH MINING WITH DEDUPLICATION AND DUAL LIMITS
# # =============================================================================

# def mine_h1_paths_source_to_diagnosis(
#     G: nx.DiGraph, 
#     source_cuis: Set[str],
#     diagnosis_target_cuis: Set[str],
#     max_paths_per_source: int = None,
#     max_diagnosis_paths_per_source: int = None,  # Per source limit
#     max_diagnosis_paths_per_target: int = None   # Per target limit
# ) -> Tuple[List[dict], Dict[str, int]]:
#     """
#     Mine H1 paths with deduplication and dual limits.
    
#     Limits:
#     - max_paths_per_source: General paths per source CUI
#     - max_diagnosis_paths_per_source: Diagnosis paths per source CUI
#     - max_diagnosis_paths_per_target: Diagnosis paths per target CUI
#     """
#     log.info(f"Mining H1 paths (source→any/diagnosis targets) with deduplication...")
#     log.info(f"  Max general paths per source: {max_paths_per_source}")
#     log.info(f"  Max diagnosis paths per source: {max_diagnosis_paths_per_source}")
#     log.info(f"  Max diagnosis paths per target: {max_diagnosis_paths_per_target}")
    
#     H1 = []
#     seen_paths = set()  # For deduplication: (src, nbr) tuples
#     paths_per_cui = defaultdict(int)
#     general_paths_added = defaultdict(int)
#     diagnosis_paths_added = defaultdict(int)
#     diagnosis_paths_to_target = defaultdict(int)  # Track paths TO each diagnosis
#     duplicate_count = 0
#     skipped_by_source_limit = 0
#     skipped_by_target_limit = 0
    
#     total_edges = G.number_of_edges()
    
#     with tqdm(total=total_edges, desc="H1 edges") as pbar:
#         for u, v, data in G.edges(data=True):
#             pbar.update(1)
            
#             # Source must be from dataset (medications, labs, procedures)
#             if u not in source_cuis:
#                 continue
            
#             # Check for duplicate path
#             path_key = (u, v)
#             if path_key in seen_paths:
#                 duplicate_count += 1
#                 continue
            
#             is_diagnosis_target = v in diagnosis_target_cuis
#             paths_per_cui[u] += 1
            
#             # Apply separate limits
#             if is_diagnosis_target:
#                 # Check per-source limit
#                 if max_diagnosis_paths_per_source and diagnosis_paths_added[u] >= max_diagnosis_paths_per_source:
#                     skipped_by_source_limit += 1
#                     continue
                
#                 # Check per-target limit
#                 if max_diagnosis_paths_per_target and diagnosis_paths_to_target[v] >= max_diagnosis_paths_per_target:
#                     skipped_by_target_limit += 1
#                     continue
#             else:
#                 if max_paths_per_source and general_paths_added[u] >= max_paths_per_source:
#                     continue
                
#             # Filter relationship types
#             rela_canon = data.get('rela_canon', '').strip().lower()
#             if rela_canon and rela_canon not in CURATED_RELATIONSHIPS:
#                 continue
                
#             path = {
#                 'src': u,
#                 'nbr': v, 
#                 'src_name': data.get('u_name', ''),
#                 'nbr_name': data.get('v_name', ''),
#                 'rela_canon': rela_canon,
#                 'rela': data.get('rela', ''),
#                 'rel': data.get('rel', ''),
#                 'is_diagnosis_target': is_diagnosis_target
#             }
            
#             H1.append(path)
#             seen_paths.add(path_key)
            
#             if is_diagnosis_target:
#                 diagnosis_paths_added[u] += 1
#                 diagnosis_paths_to_target[v] += 1
#             else:
#                 general_paths_added[u] += 1
    
#     diagnosis_count = sum(1 for p in H1 if p['is_diagnosis_target'])
    
#     log.info(f"✓ Mined {len(H1)} H1 paths from {len(set(general_paths_added.keys()) | set(diagnosis_paths_added.keys()))} source nodes")
#     log.info(f"  Diagnosis target paths: {diagnosis_count} from {len(diagnosis_paths_added)} sources to {len(diagnosis_paths_to_target)} targets")
#     log.info(f"  General knowledge paths: {len(H1) - diagnosis_count} from {len(general_paths_added)} sources")
#     log.info(f"  Duplicates skipped: {duplicate_count}")
#     log.info(f"  Skipped by source limit: {skipped_by_source_limit}")
#     log.info(f"  Skipped by target limit: {skipped_by_target_limit}")
    
#     stats = {
#         'total_paths': len(H1),
#         'diagnosis_paths': diagnosis_count,
#         'general_paths': len(H1) - diagnosis_count,
#         'sources_with_diagnosis': len(diagnosis_paths_added),
#         'sources_with_general': len(general_paths_added),
#         'diagnosis_targets_reached': len(diagnosis_paths_to_target),
#         'duplicates_skipped': duplicate_count,
#         'skipped_by_source_limit': skipped_by_source_limit,
#         'skipped_by_target_limit': skipped_by_target_limit
#     }
    
#     return H1, stats

# def mine_h2_paths_source_to_diagnosis(
#     G: nx.DiGraph,
#     source_cuis: Set[str],
#     diagnosis_target_cuis: Set[str],
#     max_paths_per_source: int = None,
#     max_diagnosis_paths_per_source: int = None,  # Per source limit
#     max_diagnosis_paths_per_target: int = None   # Per target limit
# ) -> Tuple[List[dict], Dict[str, int]]:
#     """
#     Mine H2 paths with deduplication and dual limits.
    
#     Limits:
#     - max_paths_per_source: General paths per source CUI
#     - max_diagnosis_paths_per_source: Diagnosis paths per source CUI
#     - max_diagnosis_paths_per_target: Diagnosis paths per target CUI
#     """
#     log.info(f"Mining H2 paths (source→intermediate→any/diagnosis targets) with deduplication...")
#     log.info(f"  Max general paths per source: {max_paths_per_source}")
#     log.info(f"  Max diagnosis paths per source: {max_diagnosis_paths_per_source}")
#     log.info(f"  Max diagnosis paths per target: {max_diagnosis_paths_per_target}")
    
#     H2 = []
#     seen_paths = set()  # For deduplication: (u, v, w) tuples
#     paths_per_cui = defaultdict(int)
#     general_paths_added = defaultdict(int)
#     diagnosis_paths_added = defaultdict(int)
#     diagnosis_paths_to_target = defaultdict(int)  # Track paths TO each diagnosis
#     duplicate_count = 0
#     skipped_by_source_limit = 0
#     skipped_by_target_limit = 0
    
#     source_nodes = [n for n in G.nodes() if n in source_cuis]
#     log.info(f"  Iterating over {len(source_nodes)} source nodes...")
    
#     for u in tqdm(source_nodes, desc="H2 paths"):
#         for v in G.neighbors(u):
#             uv_data = G.get_edge_data(u, v, {})
#             rela_uv_canon = uv_data.get('rela_canon', '').strip().lower()
            
#             if rela_uv_canon and rela_uv_canon not in CURATED_RELATIONSHIPS:
#                 continue
                
#             for w in G.neighbors(v):
#                 if w == u:  # No cycles
#                     continue
                
#                 # Check for duplicate path
#                 path_key = (u, v, w)
#                 if path_key in seen_paths:
#                     duplicate_count += 1
#                     continue
                    
#                 vw_data = G.get_edge_data(v, w, {})
#                 rela_vw_canon = vw_data.get('rela_canon', '').strip().lower()
                
#                 if rela_vw_canon and rela_vw_canon not in CURATED_RELATIONSHIPS:
#                     continue
                
#                 is_diagnosis_target = w in diagnosis_target_cuis
#                 paths_per_cui[u] += 1
                
#                 # Apply separate limits
#                 if is_diagnosis_target:
#                     # Check per-source limit
#                     if max_diagnosis_paths_per_source and diagnosis_paths_added[u] >= max_diagnosis_paths_per_source:
#                         skipped_by_source_limit += 1
#                         continue
                    
#                     # Check per-target limit
#                     if max_diagnosis_paths_per_target and diagnosis_paths_to_target[w] >= max_diagnosis_paths_per_target:
#                         skipped_by_target_limit += 1
#                         continue
#                 else:
#                     if max_paths_per_source and general_paths_added[u] >= max_paths_per_source:
#                         continue
                    
#                 path = {
#                     'u': u, 'v': v, 'w': w,
#                     'u_name': uv_data.get('u_name', ''),
#                     'v_name': uv_data.get('v_name', ''),
#                     'w_name': vw_data.get('v_name', ''),
#                     'rela_uv_canon': rela_uv_canon,
#                     'rela_vw_canon': rela_vw_canon,
#                     'rela_uv': uv_data.get('rela', ''),
#                     'rela_vw': vw_data.get('rela', ''),
#                     'is_diagnosis_target': is_diagnosis_target
#                 }
                
#                 H2.append(path)
#                 seen_paths.add(path_key)
                
#                 if is_diagnosis_target:
#                     diagnosis_paths_added[u] += 1
#                     diagnosis_paths_to_target[w] += 1
#                 else:
#                     general_paths_added[u] += 1
    
#     diagnosis_count = sum(1 for p in H2 if p['is_diagnosis_target'])
    
#     log.info(f"✓ Mined {len(H2)} H2 paths from {len(set(general_paths_added.keys()) | set(diagnosis_paths_added.keys()))} source nodes")
#     log.info(f"  Diagnosis target paths: {diagnosis_count} from {len(diagnosis_paths_added)} sources to {len(diagnosis_paths_to_target)} targets")
#     log.info(f"  General knowledge paths: {len(H2) - diagnosis_count} from {len(general_paths_added)} sources")
#     log.info(f"  Duplicates skipped: {duplicate_count}")
#     log.info(f"  Skipped by source limit: {skipped_by_source_limit}")
#     log.info(f"  Skipped by target limit: {skipped_by_target_limit}")
    
#     stats = {
#         'total_paths': len(H2),
#         'diagnosis_paths': diagnosis_count,
#         'general_paths': len(H2) - diagnosis_count,
#         'sources_with_diagnosis': len(diagnosis_paths_added),
#         'sources_with_general': len(general_paths_added),
#         'diagnosis_targets_reached': len(diagnosis_paths_to_target),
#         'duplicates_skipped': duplicate_count,
#         'skipped_by_source_limit': skipped_by_source_limit,
#         'skipped_by_target_limit': skipped_by_target_limit
#     }
    
#     return H2, stats

# # =============================================================================
# # SAPBERT ENCODER
# # =============================================================================

# class SapBERTEncoder:
#     """Wrapper for SapBERT model with memory optimization."""
    
#     def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
#         log.info(f"Loading SapBERT model: {model_name}")
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         log.info(f"  Using device: {self.device}")
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()
        
#         log.info(f"✓ SapBERT loaded")
    
#     def _mean_pooling(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         return sum_embeddings / sum_mask
    
#     @torch.no_grad()
#     def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
#         log.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
#         all_embeddings = []
        
#         for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
#             batch_texts = texts[i:i + batch_size]
            
#             encoded = self.tokenizer(
#                 batch_texts,
#                 truncation=True,
#                 padding=True,
#                 max_length=512,
#                 return_tensors='pt'
#             ).to(self.device)
            
#             outputs = self.model(**encoded)
#             embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
#             all_embeddings.append(embeddings.cpu().numpy())
            
#             # Memory cleanup
#             del encoded, outputs, embeddings
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
        
#         result = np.vstack(all_embeddings)
#         log.info(f"✓ Encoded to shape: {result.shape}")
#         return result

# # =============================================================================
# # PATH LINEARIZATION
# # =============================================================================

# def clean_name(name: str) -> str:
#     """Clean concept name for readability."""
#     if not name:
#         return "unknown"
#     name = re.sub(r'^C\d+:', '', name)
#     name = re.sub(r'\s+', ' ', name).strip()
#     return name

# def linearize_h1_path_with_metadata(path: dict) -> Tuple[str, str, bool]:
#     """Convert H1 path to natural language + extract relationship metadata."""
#     src = clean_name(path.get("src_name", ""))
#     tgt = clean_name(path.get("nbr_name", ""))
#     rel = path.get("rela_canon") or path.get("rela") or path.get("rel") or "related_to"
#     rel_text = rel.strip().replace("_", " ")
    
#     if not src or not tgt:
#         return "", "other", False
    
#     fact_text = f"{src} {rel_text} {tgt}"
#     relationship_type = path.get("rela_canon", "other")
#     is_diagnosis_target = path.get("is_diagnosis_target", False)
    
#     return fact_text, relationship_type, is_diagnosis_target

# def linearize_h2_path_with_metadata(path: dict) -> Tuple[str, str, bool]:
#     """Convert H2 path to natural language + extract FIRST relationship."""
#     u = clean_name(path.get("u_name", ""))
#     v = clean_name(path.get("v_name", ""))
#     w = clean_name(path.get("w_name", ""))
    
#     rel_uv = path.get("rela_uv_canon") or path.get("rela_uv") or "related_to"
#     rel_vw = path.get("rela_vw_canon") or path.get("rela_vw") or "related_to"
    
#     rel_uv_text = rel_uv.strip().replace("_", " ")
#     rel_vw_text = rel_vw.strip().replace("_", " ")
    
#     if not u or not v or not w:
#         return "", "other", False
    
#     fact_text = f"{u} {rel_uv_text} {v} which {rel_vw_text} {w}"
#     relationship_type = path.get("rela_uv_canon", "other")
#     is_diagnosis_target = path.get("is_diagnosis_target", False)
    
#     return fact_text, relationship_type, is_diagnosis_target

# # =============================================================================
# # FAISS INDEX CREATION
# # =============================================================================

# def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
#     """Create FAISS index with chunked adding to reduce memory pressure."""
#     n, d = embeddings.shape
#     log.info(f"Creating FAISS index: {n:,} vectors, {d} dimensions")
    
#     # Normalize IN PLACE
#     faiss.normalize_L2(embeddings)
    
#     # Use CPU to avoid GPU OOM
#     log.info("  Using CPU FAISS (memory optimized)")
#     index = faiss.IndexFlatIP(d)
    
#     # Add in chunks
#     chunk_size = 10000
#     log.info(f"  Adding embeddings in chunks of {chunk_size:,}...")
    
#     for i in tqdm(range(0, n, chunk_size), desc="Building index"):
#         end_idx = min(i + chunk_size, n)
#         chunk = embeddings[i:end_idx]
#         index.add(chunk)
        
#         # Memory cleanup
#         del chunk
#         gc.collect()
    
#     log.info(f"✓ FAISS index created with {index.ntotal:,} vectors")
#     return index

# # =============================================================================
# # SEPARATE INDEX SAVING
# # =============================================================================

# def save_separate_indexes(
#     h1_facts: List[str], h1_embeddings: np.ndarray, h1_relationships: List[str], h1_diagnosis_flags: List[bool],
#     h2_facts: List[str], h2_embeddings: np.ndarray, h2_relationships: List[str], h2_diagnosis_flags: List[bool],
#     output_dir: Path,
#     metadata: dict = None
# ):
#     """Save separate H1 and H2 indexes."""
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     log.info(f"\nSaving separate H1 and H2 indexes to {output_dir}")
    
#     # Create H1 index
#     log.info("\n📋 Creating H1 index...")
#     h1_index = create_faiss_index(h1_embeddings, use_gpu=False)
#     h1_dir = output_dir / "h1_index"
#     h1_dir.mkdir(exist_ok=True)
    
#     # Save H1 components
#     with open(h1_dir / "facts.json", 'w', encoding='utf-8') as f:
#         json.dump(h1_facts, f, ensure_ascii=False, indent=2)
    
#     with open(h1_dir / "relationships.json", 'w', encoding='utf-8') as f:
#         json.dump(h1_relationships, f, ensure_ascii=False, indent=2)
    
#     with open(h1_dir / "diagnosis_flags.json", 'w', encoding='utf-8') as f:
#         json.dump(h1_diagnosis_flags, f, ensure_ascii=False, indent=2)
        
#     np.save(h1_dir / "embeddings.npy", h1_embeddings)
#     faiss.write_index(h1_index, str(h1_dir / "faiss_index.bin"))
    
#     h1_diagnosis_count = sum(h1_diagnosis_flags)
#     log.info(f"  ✓ H1 index: {len(h1_facts)} facts ({h1_diagnosis_count} diagnosis targets)")
    
#     # Create H2 index  
#     log.info("\n📋 Creating H2 index...")
#     h2_index = create_faiss_index(h2_embeddings, use_gpu=False)
#     h2_dir = output_dir / "h2_index"
#     h2_dir.mkdir(exist_ok=True)
    
#     # Save H2 components
#     with open(h2_dir / "facts.json", 'w', encoding='utf-8') as f:
#         json.dump(h2_facts, f, ensure_ascii=False, indent=2)
    
#     with open(h2_dir / "relationships.json", 'w', encoding='utf-8') as f:
#         json.dump(h2_relationships, f, ensure_ascii=False, indent=2)
        
#     with open(h2_dir / "diagnosis_flags.json", 'w', encoding='utf-8') as f:
#         json.dump(h2_diagnosis_flags, f, ensure_ascii=False, indent=2)
        
#     np.save(h2_dir / "embeddings.npy", h2_embeddings)
#     faiss.write_index(h2_index, str(h2_dir / "faiss_index.bin"))
    
#     h2_diagnosis_count = sum(h2_diagnosis_flags)
#     log.info(f"  ✓ H2 index: {len(h2_facts)} facts ({h2_diagnosis_count} diagnosis targets)")
    
#     # Save combined metadata
#     combined_meta = {
#         "h1_index": {
#             "n_facts": len(h1_facts),
#             "n_diagnosis_targets": h1_diagnosis_count,
#             "n_general_targets": len(h1_facts) - h1_diagnosis_count,
#             "embedding_dim": h1_embeddings.shape[1],
#             "relationships": dict(Counter(h1_relationships))
#         },
#         "h2_index": {
#             "n_facts": len(h2_facts), 
#             "n_diagnosis_targets": h2_diagnosis_count,
#             "n_general_targets": len(h2_facts) - h2_diagnosis_count,
#             "embedding_dim": h2_embeddings.shape[1],
#             "relationships": dict(Counter(h2_relationships))
#         },
#         "index_type": "IndexFlatIP",
#         "metric": "cosine", 
#         "normalized": True,
#         "separate_indexes": True,
#         "diagnosis_aware": True,
#         "deduplication_enabled": True,
#         "source_types": ["medications", "labs", "procedures"],
#         "excluded_from_sources": ["diagnosis_codes"]
#     }
    
#     if metadata:
#         combined_meta.update(metadata)
    
#     with open(output_dir / "metadata.json", 'w') as f:
#         json.dump(combined_meta, f, indent=2)
    
#     log.info(f"\n✓ Separate indexes saved to {output_dir}")
#     log.info(f"  H1: {len(h1_facts)} facts, H2: {len(h2_facts)} facts")
#     log.info(f"  Total diagnosis targets: {h1_diagnosis_count + h2_diagnosis_count}")

# # =============================================================================
# # MAIN FUNCTION
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(description="Build Separate H1/H2 Medical Fact Indexes with Deduplication and Dual Limits")
    
#     parser.add_argument("--train_data", required=True)
#     parser.add_argument("--val_data", required=True)
#     parser.add_argument("--test_data")
#     parser.add_argument("--atc_map", required=True)
#     parser.add_argument("--loinc_map", required=True)
#     parser.add_argument("--proc_map", required=True)
#     parser.add_argument("--dx_map", required=True, help="code2cui_icd9_dx.pkl - source of diagnosis target CUIs")
#     parser.add_argument("--kg_pkl", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--max_h1_per_source", type=int, default=None, help="Max general H1 paths per source")
#     parser.add_argument("--max_h2_per_source", type=int, default=None, help="Max general H2 paths per source")
#     parser.add_argument("--max_h1_diagnosis_per_source", type=int, default=None, help="Max diagnosis H1 paths per source")
#     parser.add_argument("--max_h2_diagnosis_per_source", type=int, default=None, help="Max diagnosis H2 paths per source")
#     parser.add_argument("--max_h1_diagnosis_per_target", type=int, default=None, help="Max diagnosis H1 paths per target")
#     parser.add_argument("--max_h2_diagnosis_per_target", type=int, default=None, help="Max diagnosis H2 paths per target")
#     parser.add_argument("--sapbert_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#     parser.add_argument("--batch_size", type=int, default=16)
    
#     args = parser.parse_args()
    
#     output_dir = Path(args.output_dir)
#     start_time = time.time()
    
#     # STEP 1: LOAD MAPPINGS
#     log.info("="*80)
#     log.info("STEP 1: LOADING CODE MAPPINGS")
#     log.info("="*80)
    
#     with open(args.atc_map, "rb") as f:
#         atc_map = pickle.load(f)
#     log.info(f"✓ ATC mappings: {len(atc_map)}")
    
#     with open(args.loinc_map, "rb") as f:
#         loinc_map = pickle.load(f)
#     log.info(f"✓ LOINC mappings: {len(loinc_map)}")
    
#     with open(args.proc_map, "rb") as f:
#         proc_map = pickle.load(f)
#     log.info(f"✓ ICD9-PROC mappings: {len(proc_map)}")
    
#     with open(args.dx_map, "rb") as f:
#         dx_map = pickle.load(f)
#     log.info(f"✓ ICD9-DX mappings (diagnosis targets): {len(dx_map)}")
    
#     # STEP 2: EXTRACT CUIs
#     source_cuis, source_stats, diagnosis_target_cuis, diagnosis_stats = extract_dataset_source_cuis(
#         args.train_data, args.val_data, args.test_data,
#         atc_map, loinc_map, proc_map, dx_map
#     )
    
#     # Clean up mapping dicts
#     del atc_map, loinc_map, proc_map, dx_map
#     gc.collect()
    
#     # STEP 3: LOAD KG
#     log.info("\n" + "="*80)
#     log.info("STEP 3: LOADING KNOWLEDGE GRAPH")
#     log.info("="*80)
    
#     with open(args.kg_pkl, "rb") as f:
#         G = pickle.load(f)
#     log.info(f"✓ KG loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
#     # STEP 4: MINE PATHS WITH DEDUPLICATION
#     log.info("\n" + "="*80)
#     log.info("STEP 4: MINING PATHS WITH DEDUPLICATION AND DUAL LIMITS")
#     log.info("="*80)
    
#     mining_start = time.time()
    
#     H1_paths, h1_stats = mine_h1_paths_source_to_diagnosis(
#         G, source_cuis, diagnosis_target_cuis, 
#         args.max_h1_per_source,
#         args.max_h1_diagnosis_per_source,
#         args.max_h1_diagnosis_per_target
#     )
    
#     H2_paths, h2_stats = mine_h2_paths_source_to_diagnosis(
#         G, source_cuis, diagnosis_target_cuis,
#         args.max_h2_per_source,
#         args.max_h2_diagnosis_per_source,
#         args.max_h2_diagnosis_per_target
#     )
    
#     mining_time = time.time() - mining_start
#     log.info(f"\n✓ Mining complete in {mining_time/60:.1f} minutes")
#     log.info(f"  Total duplicates skipped: {h1_stats['duplicates_skipped'] + h2_stats['duplicates_skipped']}")
#     log.info(f"  Total skipped by source limits: {h1_stats['skipped_by_source_limit'] + h2_stats['skipped_by_source_limit']}")
#     log.info(f"  Total skipped by target limits: {h1_stats['skipped_by_target_limit'] + h2_stats['skipped_by_target_limit']}")
    
#     # Clean up KG
#     del G
#     gc.collect()
    
#     # STEP 5: LINEARIZE PATHS
#     log.info("\n" + "="*80)
#     log.info("STEP 5: LINEARIZING PATHS")
#     log.info("="*80)
    
#     # Process H1 paths
#     h1_facts = []
#     h1_relationships = []
#     h1_diagnosis_flags = []
    
#     log.info("Processing H1 paths...")
#     for path in tqdm(H1_paths, desc="H1"):
#         fact, rel, is_dx = linearize_h1_path_with_metadata(path)
#         if fact:
#             h1_facts.append(fact)
#             h1_relationships.append(rel)
#             h1_diagnosis_flags.append(is_dx)
    
#     # Process H2 paths  
#     h2_facts = []
#     h2_relationships = []
#     h2_diagnosis_flags = []
    
#     log.info("Processing H2 paths...")
#     for path in tqdm(H2_paths, desc="H2"):
#         fact, rel, is_dx = linearize_h2_path_with_metadata(path)
#         if fact:
#             h2_facts.append(fact)
#             h2_relationships.append(rel)
#             h2_diagnosis_flags.append(is_dx)
    
#     log.info(f"✓ Linearized H1: {len(h1_facts)} facts, H2: {len(h2_facts)} facts")
    
#     # Clean up paths
#     del H1_paths, H2_paths
#     gc.collect()
    
#     # STEP 6: ENCODE
#     log.info("\n" + "="*80)
#     log.info("STEP 6: ENCODING H1 AND H2 FACTS")
#     log.info("="*80)
    
#     encoder = SapBERTEncoder(args.sapbert_model)
    
#     # Encode H1
#     log.info("Encoding H1 facts...")
#     h1_embeddings = encoder.encode_batch(h1_facts, batch_size=args.batch_size)
    
#     # Encode H2
#     log.info("Encoding H2 facts...")
#     h2_embeddings = encoder.encode_batch(h2_facts, batch_size=args.batch_size)
    
#     # Clean up encoder
#     del encoder
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     gc.collect()
    
#     # STEP 7: SAVE INDEXES
#     log.info("\n" + "="*80)
#     log.info("STEP 7: SAVING SEPARATE INDEXES")
#     log.info("="*80)
    
#     metadata = {
#         "n_source_cuis": len(source_cuis),
#         "n_diagnosis_target_cuis": len(diagnosis_target_cuis),
#         "diagnosis_statistics": diagnosis_stats,
#         "h1_statistics": h1_stats,
#         "h2_statistics": h2_stats,
#         "source_types": ["medications", "labs", "procedures"],
#         "excluded_from_sources": ["diagnosis_codes"],
#         "diagnosis_source": "code2cui_icd9_dx.pkl (NOT from dataset)",
#         "limits": {
#             "max_h1_per_source": args.max_h1_per_source,
#             "max_h2_per_source": args.max_h2_per_source,
#             "max_h1_diagnosis_per_source": args.max_h1_diagnosis_per_source,
#             "max_h2_diagnosis_per_source": args.max_h2_diagnosis_per_source,
#             "max_h1_diagnosis_per_target": args.max_h1_diagnosis_per_target,
#             "max_h2_diagnosis_per_target": args.max_h2_diagnosis_per_target
#         },
#         "sapbert_model": args.sapbert_model,
#         "creation_time": time.time(),
#         "principle": "Diagnosis CUIs from pkl mapping. Paths deduplicated. Dual limits: per-source AND per-target."
#     }
    
#     save_separate_indexes(
#         h1_facts, h1_embeddings, h1_relationships, h1_diagnosis_flags,
#         h2_facts, h2_embeddings, h2_relationships, h2_diagnosis_flags,
#         output_dir, metadata
#     )
    
#     total_time = time.time() - start_time
#     log.info(f"\n✓ COMPLETE in {total_time/60:.1f} minutes")
#     log.info(f"\n🎯 KEY PRINCIPLES APPLIED:")
#     log.info(f"  - SOURCE CUIs: medications, labs, procedures from dataset")
#     log.info(f"  - DIAGNOSIS TARGET CUIs: from code2cui_icd9_dx.pkl (NOT dataset)")
#     log.info(f"  - ICD codes excluded from sources")
#     log.info(f"  - Paths deduplicated")
#     log.info(f"  - Dual limits: per-source (control spreading) AND per-target (control popular diagnoses)")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_medical_fact_index_new.py

Build separate searchable medical fact indexes from UMLS KG with MEMORY OPTIMIZATION.
- Separate H1 and H2 indexes for flexible retrieval
- Diagnosis CUIs as TARGETS ONLY (not sources from dataset)
- Source CUIs come from medications, labs, procedures only
- Separate limits for general paths vs diagnosis-targeting paths
- Path deduplication to avoid repeated paths
- Dual limits: per-source AND per-target

KEY PRINCIPLE: 
- ICD diagnosis codes are TARGETS to predict, not evidence sources
- Only medications, labs, procedures contribute to source CUIs
- Diagnosis CUIs from code2cui_icd9_dx.pkl are potential targets
- Paths are deduplicated (no path appears twice)
- Per-source limits: control how many diagnosis paths each source can have
- Per-target limits: control how many incoming paths each diagnosis can receive
"""
import sys
import json
import pickle
import re
import gc
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

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
    if s.startswith("PRO_"): 
        s = s[4:]
    s = re.sub(r"[^0-9]", "", s)
    if not s: 
        return ""
    if len(s) >= 3:
        return f"{s[:2]}.{s[2:]}"
    return s

def format_icd9(code: str) -> str:
    """Format ICD9 diagnosis code."""
    c = _strip(code)
    if not c or c.upper() in ("NAN", "NONE"):
        return ""
    c = c.replace(".", "")
    if len(c) > 3:
        return f"{c[:3]}.{c[3:]}"
    return c

def to_list(x) -> List[str]:
    """Convert various types to list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    if s.startswith("[") and s.endswith("]"):
        return [t.strip() for t in s[1:-1].split(",") if t.strip()]
    return [t for t in re.split(r"[,\s]+", s) if t]

# =============================================================================
# DATASET CODE EXTRACTION - NO ICD CODES AS SOURCES
# =============================================================================

def extract_source_codes_from_dataframe(df: pd.DataFrame, split_name: str) -> Dict[str, Set[str]]:
    """
    Extract SOURCE codes from dataframe - EXCLUDES ICD diagnosis codes.
    
    ICD codes are targets to predict, not evidence sources.
    Only medications, labs, procedures are valid sources.
    """
    codes_by_type = {
        'medications': set(),
        'labs': set(), 
        'procedures': set()
    }
    
    # Medications (NDC codes)
    if 'ndc' in df.columns:
        for row_val in df['ndc']:
            for code in to_list(row_val):
                if code and code.upper() not in ('NAN', 'NONE'):
                    codes_by_type['medications'].add(code.strip())
    
    # Lab tests (LOINC codes)  
    if 'lab_test_loinc' in df.columns:
        for row_val in df['lab_test_loinc']:
            for code in to_list(row_val):
                if code and code.upper() not in ('NAN', 'NONE'):
                    codes_by_type['labs'].add(code.strip())
    
    # Procedures (ICD9-PROC codes)
    if 'pro_code' in df.columns:
        for row_val in df['pro_code']:
            for code in to_list(row_val):
                formatted = format_icd9_proc_from_pro(code)
                if formatted:
                    codes_by_type['procedures'].add(formatted)
    
    # Clean empty entries
    for key in codes_by_type:
        codes_by_type[key].discard('')
    
    return codes_by_type

def map_source_codes_to_cuis(
    codes_by_type: Dict[str, Set[str]],
    atc_map: Dict[str, List[str]],
    loinc_map: Dict[str, List[str]], 
    proc_map: Dict[str, List[str]],
    split_name: str
) -> Tuple[Set[str], Dict[str, Dict]]:
    """Map SOURCE codes to CUIs - no diagnosis codes included."""
    log.info(f"\n  Mapping {split_name} SOURCE codes to CUIs (excluding diagnosis codes)...")
    
    source_cuis = set()
    mapping_stats = {
        'medications': {'codes': 0, 'mapped': 0, 'cuis': set()},
        'labs': {'codes': 0, 'mapped': 0, 'cuis': set()},
        'procedures': {'codes': 0, 'mapped': 0, 'cuis': set()},
        'unmapped': {'codes': 0}
    }
    
    # Medications (ATC)
    log.info(f"    Mapping medications (ATC)...")
    for code in codes_by_type['medications']:
        mapping_stats['medications']['codes'] += 1
        if code in atc_map:
            cui_list = atc_map[code]
            source_cuis.update(cui_list)
            mapping_stats['medications']['cuis'].update(cui_list)
            mapping_stats['medications']['mapped'] += 1
        else:
            mapping_stats['unmapped']['codes'] += 1
    
    log.info(f"      → {mapping_stats['medications']['mapped']}/{mapping_stats['medications']['codes']} mapped "
             f"({100*mapping_stats['medications']['mapped']/max(1, mapping_stats['medications']['codes']):.1f}%) "
             f"→ {len(mapping_stats['medications']['cuis'])} CUIs")
    
    # Labs (LOINC)
    log.info(f"    Mapping labs (LOINC)...")
    for code in codes_by_type['labs']:
        mapping_stats['labs']['codes'] += 1
        if code in loinc_map:
            cui_list = loinc_map[code]
            source_cuis.update(cui_list)
            mapping_stats['labs']['cuis'].update(cui_list)
            mapping_stats['labs']['mapped'] += 1
        else:
            mapping_stats['unmapped']['codes'] += 1
    
    log.info(f"      → {mapping_stats['labs']['mapped']}/{mapping_stats['labs']['codes']} mapped "
             f"({100*mapping_stats['labs']['mapped']/max(1, mapping_stats['labs']['codes']):.1f}%) "
             f"→ {len(mapping_stats['labs']['cuis'])} CUIs")
    
    # Procedures (ICD9-PROC)
    log.info(f"    Mapping procedures (ICD9-PROC)...")
    for code in codes_by_type['procedures']:
        mapping_stats['procedures']['codes'] += 1
        if code in proc_map:
            cui_list = proc_map[code]
            source_cuis.update(cui_list)
            mapping_stats['procedures']['cuis'].update(cui_list)
            mapping_stats['procedures']['mapped'] += 1
        else:
            mapping_stats['unmapped']['codes'] += 1
    
    log.info(f"      → {mapping_stats['procedures']['mapped']}/{mapping_stats['procedures']['codes']} mapped "
             f"({100*mapping_stats['procedures']['mapped']/max(1, mapping_stats['procedures']['codes']):.1f}%) "
             f"→ {len(mapping_stats['procedures']['cuis'])} CUIs")
    
    return source_cuis, mapping_stats

def extract_diagnosis_target_cuis(dx_map: Dict[str, List[str]]) -> Tuple[Set[str], Dict]:
    """
    Extract ALL diagnosis CUIs as potential targets from the pkl mapping file.
    
    Args:
        dx_map: code2cui_icd9_dx.pkl mapping
        
    Returns:
        diagnosis_cuis: Set of all diagnosis CUIs
        diagnosis_stats: Statistics about diagnosis CUIs
    """
    log.info(f"\nExtracting diagnosis target CUIs from pkl mapping file...")
    
    diagnosis_cuis = set()
    for icd_code, cui_list in dx_map.items():
        if isinstance(cui_list, list):
            diagnosis_cuis.update(cui_list)
    
    diagnosis_stats = {
        'total_icd_codes_in_mapping': len(dx_map),
        'unique_diagnosis_cuis': len(diagnosis_cuis),
        'avg_cuis_per_code': len(diagnosis_cuis) / len(dx_map) if dx_map else 0
    }
    
    log.info(f"✓ Extracted {len(diagnosis_cuis)} unique diagnosis CUIs from {len(dx_map)} ICD9-DX codes")
    log.info(f"  Average CUIs per ICD code: {diagnosis_stats['avg_cuis_per_code']:.2f}")
    
    return diagnosis_cuis, diagnosis_stats

def extract_dataset_source_cuis(
    train_pkl: str, 
    val_pkl: str, 
    test_pkl: str,
    atc_map: Dict, 
    loinc_map: Dict, 
    proc_map: Dict,
    dx_map: Dict
) -> Tuple[Set[str], Dict, Set[str], Dict]:
    """
    Extract SOURCE CUIs from dataset (no diagnosis codes) and ALL diagnosis target CUIs from pkl.
    
    Returns:
        - source_cuis: CUIs from medications, labs, procedures in dataset
        - stats: mapping statistics  
        - diagnosis_target_cuis: ALL possible diagnosis CUIs (from dx_map pkl)
        - diagnosis_stats: Statistics about diagnosis CUIs from pkl
    """
    log.info("="*80)
    log.info("EXTRACTING SOURCE CUIs (NO DIAGNOSIS) + TARGET DIAGNOSIS CUIs")
    log.info("="*80)
    
    all_source_cuis = set()
    overall_stats = {}
    
    # Extract source CUIs from each dataset split (no diagnosis codes)
    for split_name, pkl_path in [("train", train_pkl), ("val", val_pkl), ("test", test_pkl)]:
        if pkl_path and Path(pkl_path).exists():
            log.info(f"\nProcessing {split_name} split: {pkl_path}")
            df = pd.read_pickle(pkl_path)
            log.info(f"  Loaded {len(df)} visits")
            
            codes = extract_source_codes_from_dataframe(df, split_name)
            source_cuis, mapping_stats = map_source_codes_to_cuis(codes, atc_map, loinc_map, proc_map, split_name)
            
            all_source_cuis.update(source_cuis)
            overall_stats[split_name] = {
                'n_visits': len(df),
                'codes': codes,
                'mapping': mapping_stats,
                'source_cuis': len(source_cuis)
            }
            
            log.info(f"  → {len(source_cuis)} unique source CUIs from {split_name}")
    
    # Extract ALL diagnosis target CUIs from pkl mapping (NOT from dataset)
    diagnosis_target_cuis, diagnosis_stats = extract_diagnosis_target_cuis(dx_map)
    
    # Summary
    log.info(f"\n{'='*80}")
    log.info("OVERALL CUI EXTRACTION SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"SOURCE CUIs (from dataset): {len(all_source_cuis)}")
    log.info(f"  - From medications, labs, procedures only")
    log.info(f"  - ICD diagnosis codes EXCLUDED (they are targets)")
    log.info(f"")
    log.info(f"DIAGNOSIS TARGET CUIs (from pkl mapping): {len(diagnosis_target_cuis)}")
    log.info(f"  - From code2cui_icd9_dx.pkl")
    log.info(f"  - Total ICD codes: {diagnosis_stats['total_icd_codes_in_mapping']}")
    log.info(f"  - Unique diagnosis CUIs: {diagnosis_stats['unique_diagnosis_cuis']}")
    log.info(f"")
    log.info(f"OVERLAP: {len(all_source_cuis & diagnosis_target_cuis)} CUIs appear as both source and diagnosis target")
    log.info(f"{'='*80}")
    
    return all_source_cuis, overall_stats, diagnosis_target_cuis, diagnosis_stats

# =============================================================================
# PATH MINING WITH DEDUPLICATION AND DUAL LIMITS
# =============================================================================

def mine_h1_paths_source_to_diagnosis(
    G: nx.DiGraph, 
    source_cuis: Set[str],
    diagnosis_target_cuis: Set[str],
    max_paths_per_source: int = None,
    max_diagnosis_paths_per_source: int = None,
    max_diagnosis_paths_per_target: int = None
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Mine H1 paths with deduplication and dual limits.
    
    Limits:
    - max_paths_per_source: General paths per source CUI
    - max_diagnosis_paths_per_source: Diagnosis paths per source CUI
    - max_diagnosis_paths_per_target: Diagnosis paths per target CUI
    """
    log.info(f"Mining H1 paths (source→any/diagnosis targets) with deduplication...")
    log.info(f"  Max general paths per source: {max_paths_per_source}")
    log.info(f"  Max diagnosis paths per source: {max_diagnosis_paths_per_source}")
    log.info(f"  Max diagnosis paths per target: {max_diagnosis_paths_per_target}")
    
    H1 = []
    seen_paths = set()
    paths_per_cui = defaultdict(int)
    general_paths_added = defaultdict(int)
    diagnosis_paths_added = defaultdict(int)
    diagnosis_paths_to_target = defaultdict(int)
    duplicate_count = 0
    skipped_by_source_limit = 0
    skipped_by_target_limit = 0
    
    total_edges = G.number_of_edges()
    
    with tqdm(total=total_edges, desc="H1 edges") as pbar:
        for u, v, data in G.edges(data=True):
            pbar.update(1)
            
            if u not in source_cuis:
                continue
            
            path_tuple = (u, v)
            if path_tuple in seen_paths:
                duplicate_count += 1
                continue
            
            is_diagnosis_target = v in diagnosis_target_cuis
            
            # Apply limits
            if is_diagnosis_target:
                if max_diagnosis_paths_per_source and diagnosis_paths_added[u] >= max_diagnosis_paths_per_source:
                    skipped_by_source_limit += 1
                    continue
                if max_diagnosis_paths_per_target and diagnosis_paths_to_target[v] >= max_diagnosis_paths_per_target:
                    skipped_by_target_limit += 1
                    continue
            else:
                if max_paths_per_source and general_paths_added[u] >= max_paths_per_source:
                    skipped_by_source_limit += 1
                    continue
            
            rela_canon = data.get('rela_canon', 'other')
            if rela_canon not in CURATED_RELATIONSHIPS:
                rela_canon = 'other'
            
            path = {
                'src': u,
                'nbr': v,
                'src_name': G.nodes[u].get('name', u),  # ✓ FIXED: Get from node attributes
                'nbr_name': G.nodes[v].get('name', v),  # ✓ FIXED: Get from node attributes
                'rela_canon': rela_canon,
                'rela': data.get('rela', ''),
                'rel': data.get('rel', ''),
                'is_diagnosis_target': is_diagnosis_target
            }
            
            H1.append(path)
            seen_paths.add(path_tuple)
            paths_per_cui[u] += 1
            
            if is_diagnosis_target:
                diagnosis_paths_added[u] += 1
                diagnosis_paths_to_target[v] += 1
            else:
                general_paths_added[u] += 1
    
    diagnosis_count = sum(1 for p in H1 if p['is_diagnosis_target'])
    
    log.info(f"✓ Mined {len(H1)} H1 paths from {len(set(general_paths_added.keys()) | set(diagnosis_paths_added.keys()))} source nodes")
    log.info(f"  Diagnosis target paths: {diagnosis_count} from {len(diagnosis_paths_added)} sources to {len(diagnosis_paths_to_target)} targets")
    log.info(f"  General knowledge paths: {len(H1) - diagnosis_count} from {len(general_paths_added)} sources")
    log.info(f"  Duplicates skipped: {duplicate_count}")
    log.info(f"  Skipped by source limit: {skipped_by_source_limit}")
    log.info(f"  Skipped by target limit: {skipped_by_target_limit}")
    
    stats = {
        'total_paths': len(H1),
        'diagnosis_paths': diagnosis_count,
        'general_paths': len(H1) - diagnosis_count,
        'sources_with_diagnosis': len(diagnosis_paths_added),
        'sources_with_general': len(general_paths_added),
        'diagnosis_targets_reached': len(diagnosis_paths_to_target),
        'duplicates_skipped': duplicate_count,
        'skipped_by_source_limit': skipped_by_source_limit,
        'skipped_by_target_limit': skipped_by_target_limit
    }
    
    return H1, stats

def mine_h2_paths_source_to_diagnosis(
    G: nx.DiGraph,
    source_cuis: Set[str],
    diagnosis_target_cuis: Set[str],
    max_paths_per_source: int = None,
    max_diagnosis_paths_per_source: int = None,
    max_diagnosis_paths_per_target: int = None
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Mine H2 paths with deduplication and dual limits.
    
    Limits:
    - max_paths_per_source: General paths per source CUI
    - max_diagnosis_paths_per_source: Diagnosis paths per source CUI
    - max_diagnosis_paths_per_target: Diagnosis paths per target CUI
    """
    log.info(f"Mining H2 paths (source→intermediate→any/diagnosis targets) with deduplication...")
    log.info(f"  Max general paths per source: {max_paths_per_source}")
    log.info(f"  Max diagnosis paths per source: {max_diagnosis_paths_per_source}")
    log.info(f"  Max diagnosis paths per target: {max_diagnosis_paths_per_target}")
    
    H2 = []
    seen_paths = set()
    paths_per_cui = defaultdict(int)
    general_paths_added = defaultdict(int)
    diagnosis_paths_added = defaultdict(int)
    diagnosis_paths_to_target = defaultdict(int)
    duplicate_count = 0
    skipped_by_source_limit = 0
    skipped_by_target_limit = 0
    
    source_nodes = [n for n in G.nodes() if n in source_cuis]
    log.info(f"  Iterating over {len(source_nodes)} source nodes...")
    
    for u in tqdm(source_nodes, desc="H2 paths"):
        for v in G.neighbors(u):
            if v == u:
                continue
            
            for w in G.neighbors(v):
                if w == u or w == v:
                    continue
                
                path_tuple = (u, v, w)
                if path_tuple in seen_paths:
                    duplicate_count += 1
                    continue
                
                is_diagnosis_target = w in diagnosis_target_cuis
                
                # Apply limits
                if is_diagnosis_target:
                    if max_diagnosis_paths_per_source and diagnosis_paths_added[u] >= max_diagnosis_paths_per_source:
                        skipped_by_source_limit += 1
                        continue
                    if max_diagnosis_paths_per_target and diagnosis_paths_to_target[w] >= max_diagnosis_paths_per_target:
                        skipped_by_target_limit += 1
                        continue
                else:
                    if max_paths_per_source and general_paths_added[u] >= max_paths_per_source:
                        skipped_by_source_limit += 1
                        continue
                
                uv_data = G.get_edge_data(u, v)
                vw_data = G.get_edge_data(v, w)
                
                rela_uv_canon = uv_data.get('rela_canon', 'other')
                rela_vw_canon = vw_data.get('rela_canon', 'other')
                
                if rela_uv_canon not in CURATED_RELATIONSHIPS:
                    rela_uv_canon = 'other'
                if rela_vw_canon not in CURATED_RELATIONSHIPS:
                    rela_vw_canon = 'other'
                
                path = {
                    'u': u, 'v': v, 'w': w,
                    'u_name': G.nodes[u].get('name', u),  # ✓ FIXED: Get from node attributes
                    'v_name': G.nodes[v].get('name', v),  # ✓ FIXED: Get from node attributes
                    'w_name': G.nodes[w].get('name', w),  # ✓ FIXED: Get from node attributes
                    'rela_uv_canon': rela_uv_canon,
                    'rela_vw_canon': rela_vw_canon,
                    'rela_uv': uv_data.get('rela', ''),
                    'rela_vw': vw_data.get('rela', ''),
                    'is_diagnosis_target': is_diagnosis_target
                }
                
                H2.append(path)
                seen_paths.add(path_tuple)
                paths_per_cui[u] += 1
                
                if is_diagnosis_target:
                    diagnosis_paths_added[u] += 1
                    diagnosis_paths_to_target[w] += 1
                else:
                    general_paths_added[u] += 1
    
    diagnosis_count = sum(1 for p in H2 if p['is_diagnosis_target'])
    
    log.info(f"✓ Mined {len(H2)} H2 paths from {len(set(general_paths_added.keys()) | set(diagnosis_paths_added.keys()))} source nodes")
    log.info(f"  Diagnosis target paths: {diagnosis_count} from {len(diagnosis_paths_added)} sources to {len(diagnosis_paths_to_target)} targets")
    log.info(f"  General knowledge paths: {len(H2) - diagnosis_count} from {len(general_paths_added)} sources")
    log.info(f"  Duplicates skipped: {duplicate_count}")
    log.info(f"  Skipped by source limit: {skipped_by_source_limit}")
    log.info(f"  Skipped by target limit: {skipped_by_target_limit}")
    
    stats = {
        'total_paths': len(H2),
        'diagnosis_paths': diagnosis_count,
        'general_paths': len(H2) - diagnosis_count,
        'sources_with_diagnosis': len(diagnosis_paths_added),
        'sources_with_general': len(general_paths_added),
        'diagnosis_targets_reached': len(diagnosis_paths_to_target),
        'duplicates_skipped': duplicate_count,
        'skipped_by_source_limit': skipped_by_source_limit,
        'skipped_by_target_limit': skipped_by_target_limit
    }
    
    return H2, stats

# =============================================================================
# SAPBERT ENCODER
# =============================================================================

class SapBERTEncoder:
    """Wrapper for SapBERT model with memory optimization."""
    
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        log.info(f"Loading SapBERT model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"  Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        log.info("✓ SapBERT model loaded")
    
    def _mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches with memory cleanup."""
        log.info(f"Encoding {len(texts)} texts with batch_size={batch_size}...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            del encoded, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if (i // batch_size) % 100 == 0:
                gc.collect()
        
        result = np.vstack(all_embeddings)
        log.info(f"✓ Encoded to shape {result.shape}")
        
        del all_embeddings
        gc.collect()
        
        return result

# =============================================================================
# PATH LINEARIZATION
# =============================================================================

def clean_name(name: str) -> str:
    """Clean concept name for readability."""
    if not name:
        return "unknown"
    name = re.sub(r'^C\d+:', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name if name else "unknown"

def linearize_h1_path_with_metadata(path: dict) -> Tuple[str, str, bool]:
    """
    Convert H1 path to natural language + extract relationship metadata.
    
    Returns:
        Tuple[str, str, bool]: (fact_text, relationship_type, is_diagnosis_target)
    """
    src = clean_name(path.get("src_name", ""))
    tgt = clean_name(path.get("nbr_name", ""))
    rel = path.get("rela_canon") or path.get("rela") or path.get("rel") or "related_to"
    rel_text = rel.strip().replace("_", " ")
    
    if not src or not tgt:
        return "", "other", False
    
    fact_text = f"{src} {rel_text} {tgt}"
    relationship_type = path.get("rela_canon", "other")
    is_diagnosis_target = path.get("is_diagnosis_target", False)
    
    return fact_text, relationship_type, is_diagnosis_target

def linearize_h2_path_with_metadata(path: dict) -> Tuple[str, str, bool]:
    """
    Convert H2 path to natural language + extract FIRST relationship.
    
    Returns:
        Tuple[str, str, bool]: (fact_text, relationship_type, is_diagnosis_target)
    """
    u = clean_name(path.get("u_name", ""))
    v = clean_name(path.get("v_name", ""))
    w = clean_name(path.get("w_name", ""))
    
    rel_uv = path.get("rela_uv_canon") or path.get("rela_uv") or "related_to"
    rel_vw = path.get("rela_vw_canon") or path.get("rela_vw") or "related_to"
    
    rel_uv_text = rel_uv.strip().replace("_", " ")
    rel_vw_text = rel_vw.strip().replace("_", " ")
    
    if not u or not v or not w:
        return "", "other", False
    
    fact_text = f"{u} {rel_uv_text} {v} which {rel_vw_text} {w}"
    relationship_type = path.get("rela_uv_canon", "other")
    is_diagnosis_target = path.get("is_diagnosis_target", False)
    
    return fact_text, relationship_type, is_diagnosis_target

# =============================================================================
# FAISS INDEX CREATION
# =============================================================================

def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """Create FAISS index with chunked adding to reduce memory pressure."""
    n, d = embeddings.shape
    log.info(f"Creating FAISS index: {n:,} vectors, {d} dimensions")
    
    faiss.normalize_L2(embeddings)
    
    log.info("  Using CPU FAISS (memory optimized)")
    index = faiss.IndexFlatIP(d)
    
    chunk_size = 10000
    log.info(f"  Adding embeddings in chunks of {chunk_size:,}...")
    
    for i in tqdm(range(0, n, chunk_size), desc="Building index"):
        end = min(i + chunk_size, n)
        index.add(embeddings[i:end])
    
    log.info(f"✓ FAISS index created with {index.ntotal:,} vectors")
    return index

# =============================================================================
# SEPARATE INDEX SAVING
# =============================================================================

def save_separate_indexes(
    h1_facts: List[str], h1_embeddings: np.ndarray, h1_relationships: List[str], h1_diagnosis_flags: List[bool],
    h2_facts: List[str], h2_embeddings: np.ndarray, h2_relationships: List[str], h2_diagnosis_flags: List[bool],
    output_dir: Path,
    metadata: dict = None
):
    """Save H1 and H2 indexes separately with relationship metadata and diagnosis flags."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\n{'='*80}")
    log.info(f"SAVING SEPARATE INDEXES TO: {output_dir}")
    log.info(f"{'='*80}")
    
    # H1 Index
    h1_dir = output_dir / "h1_index"
    h1_dir.mkdir(exist_ok=True)
    log.info(f"\n[H1 INDEX]")
    
    assert len(h1_facts) == len(h1_relationships) == len(h1_diagnosis_flags) == h1_embeddings.shape[0], \
        f"H1 misalignment: {len(h1_facts)} facts, {len(h1_relationships)} rels, {len(h1_diagnosis_flags)} flags, {h1_embeddings.shape[0]} embeddings"
    
    with open(h1_dir / "facts.json", 'w', encoding='utf-8') as f:
        json.dump(h1_facts, f, indent=2)
    log.info(f"  ✓ Saved {len(h1_facts)} facts")
    
    with open(h1_dir / "relationships.json", 'w', encoding='utf-8') as f:
        json.dump(h1_relationships, f, indent=2)
    log.info(f"  ✓ Saved {len(h1_relationships)} relationships")
    
    with open(h1_dir / "diagnosis_flags.json", 'w', encoding='utf-8') as f:
        json.dump(h1_diagnosis_flags, f, indent=2)
    log.info(f"  ✓ Saved {len(h1_diagnosis_flags)} diagnosis flags")
    
    h1_rel_counter = Counter(h1_relationships)
    log.info(f"  Relationship distribution (top 5):")
    for rel, count in h1_rel_counter.most_common(5):
        log.info(f"    - {rel}: {count}")
    
    np.save(h1_dir / "embeddings.npy", h1_embeddings)
    log.info(f"  ✓ Saved embeddings {h1_embeddings.shape}")
    
    h1_index = create_faiss_index(h1_embeddings.copy())
    faiss.write_index(h1_index, str(h1_dir / "faiss_index.bin"))
    log.info(f"  ✓ Saved FAISS index")
    
    # H2 Index
    h2_dir = output_dir / "h2_index"
    h2_dir.mkdir(exist_ok=True)
    log.info(f"\n[H2 INDEX]")
    
    assert len(h2_facts) == len(h2_relationships) == len(h2_diagnosis_flags) == h2_embeddings.shape[0], \
        f"H2 misalignment: {len(h2_facts)} facts, {len(h2_relationships)} rels, {len(h2_diagnosis_flags)} flags, {h2_embeddings.shape[0]} embeddings"
    
    with open(h2_dir / "facts.json", 'w', encoding='utf-8') as f:
        json.dump(h2_facts, f, indent=2)
    log.info(f"  ✓ Saved {len(h2_facts)} facts")
    
    with open(h2_dir / "relationships.json", 'w', encoding='utf-8') as f:
        json.dump(h2_relationships, f, indent=2)
    log.info(f"  ✓ Saved {len(h2_relationships)} relationships")
    
    with open(h2_dir / "diagnosis_flags.json", 'w', encoding='utf-8') as f:
        json.dump(h2_diagnosis_flags, f, indent=2)
    log.info(f"  ✓ Saved {len(h2_diagnosis_flags)} diagnosis flags")
    
    h2_rel_counter = Counter(h2_relationships)
    log.info(f"  Relationship distribution (top 5):")
    for rel, count in h2_rel_counter.most_common(5):
        log.info(f"    - {rel}: {count}")
    
    np.save(h2_dir / "embeddings.npy", h2_embeddings)
    log.info(f"  ✓ Saved embeddings {h2_embeddings.shape}")
    
    h2_index = create_faiss_index(h2_embeddings.copy())
    faiss.write_index(h2_index, str(h2_dir / "faiss_index.bin"))
    log.info(f"  ✓ Saved FAISS index")
    
    # Combined metadata
    meta = {
        "h1_index": {
            "n_facts": len(h1_facts),
            "embedding_dim": h1_embeddings.shape[1],
            "unique_relationships": len(h1_rel_counter),
            "diagnosis_targets": sum(h1_diagnosis_flags)
        },
        "h2_index": {
            "n_facts": len(h2_facts),
            "embedding_dim": h2_embeddings.shape[1],
            "unique_relationships": len(h2_rel_counter),
            "diagnosis_targets": sum(h2_diagnosis_flags)
        },
        "index_type": "IndexFlatIP",
        "metric": "cosine",
        "normalized": True
    }
    
    if metadata:
        meta.update(metadata)
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"\n✓ Saved combined metadata")
    
    log.info(f"\n{'='*80}")
    log.info("INDEX SAVING COMPLETE")
    log.info(f"{'='*80}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build separate H1/H2 fact indexes")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--atc_map", required=True)
    parser.add_argument("--loinc_map", required=True)
    parser.add_argument("--proc_map", required=True)
    parser.add_argument("--dx_map", required=True)
    parser.add_argument("--kg_pkl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sapbert_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--batch_size", type=int, default=64)
    
    # Limits
    parser.add_argument("--max_h1_per_source", type=int, default=None)
    parser.add_argument("--max_h2_per_source", type=int, default=None)
    parser.add_argument("--max_h1_diagnosis_per_source", type=int, default=None)
    parser.add_argument("--max_h2_diagnosis_per_source", type=int, default=None)
    parser.add_argument("--max_h1_diagnosis_per_target", type=int, default=None)
    parser.add_argument("--max_h2_diagnosis_per_target", type=int, default=None)
    
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("MEDICAL FACT INDEX BUILDER - SEPARATE H1/H2")
    log.info("="*80)
    
    # Load mappings
    log.info("\nLoading code mappings...")
    with open(args.atc_map, 'rb') as f:
        atc_map = pickle.load(f)
    with open(args.loinc_map, 'rb') as f:
        loinc_map = pickle.load(f)
    with open(args.proc_map, 'rb') as f:
        proc_map = pickle.load(f)
    with open(args.dx_map, 'rb') as f:
        dx_map = pickle.load(f)
    log.info("✓ Loaded all mappings")
    
    # Extract CUIs
    source_cuis, source_stats, diagnosis_target_cuis, diagnosis_stats = extract_dataset_source_cuis(
        args.train_data, args.val_data, args.test_data,
        atc_map, loinc_map, proc_map, dx_map
    )
    
    # Load KG
    log.info(f"\nLoading knowledge graph: {args.kg_pkl}")
    with open(args.kg_pkl, 'rb') as f:
        G = pickle.load(f)
    log.info(f"✓ Loaded KG: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Mine H1 paths
    log.info("\n" + "="*80)
    log.info("MINING H1 PATHS")
    log.info("="*80)
    h1_paths, h1_stats = mine_h1_paths_source_to_diagnosis(
        G, source_cuis, diagnosis_target_cuis,
        max_paths_per_source=args.max_h1_per_source,
        max_diagnosis_paths_per_source=args.max_h1_diagnosis_per_source,
        max_diagnosis_paths_per_target=args.max_h1_diagnosis_per_target
    )
    
    # Mine H2 paths
    log.info("\n" + "="*80)
    log.info("MINING H2 PATHS")
    log.info("="*80)
    h2_paths, h2_stats = mine_h2_paths_source_to_diagnosis(
        G, source_cuis, diagnosis_target_cuis,
        max_paths_per_source=args.max_h2_per_source,
        max_diagnosis_paths_per_source=args.max_h2_diagnosis_per_source,
        max_diagnosis_paths_per_target=args.max_h2_diagnosis_per_target
    )
    
    # Linearize paths
    log.info("\n" + "="*80)
    log.info("LINEARIZING PATHS")
    log.info("="*80)
    
    log.info("Linearizing H1 paths...")
    h1_facts, h1_relationships, h1_diagnosis_flags = [], [], []
    for path in tqdm(h1_paths, desc="H1 linearization"):
        fact, rel, is_diag = linearize_h1_path_with_metadata(path)
        if fact:
            h1_facts.append(fact)
            h1_relationships.append(rel)
            h1_diagnosis_flags.append(is_diag)
    
    log.info("Linearizing H2 paths...")
    h2_facts, h2_relationships, h2_diagnosis_flags = [], [], []
    for path in tqdm(h2_paths, desc="H2 linearization"):
        fact, rel, is_diag = linearize_h2_path_with_metadata(path)
        if fact:
            h2_facts.append(fact)
            h2_relationships.append(rel)
            h2_diagnosis_flags.append(is_diag)
    
    log.info(f"✓ H1: {len(h1_facts)} facts")
    log.info(f"✓ H2: {len(h2_facts)} facts")
    
    # Encode
    log.info("\n" + "="*80)
    log.info("ENCODING FACTS WITH SAPBERT")
    log.info("="*80)
    
    encoder = SapBERTEncoder(args.sapbert_model)
    
    log.info("\nEncoding H1 facts...")
    h1_embeddings = encoder.encode_batch(h1_facts, batch_size=args.batch_size)
    
    log.info("\nEncoding H2 facts...")
    h2_embeddings = encoder.encode_batch(h2_facts, batch_size=args.batch_size)
    
    del encoder
    gc.collect()
    
    # Save
    metadata = {
        "source_statistics": source_stats,
        "diagnosis_statistics": diagnosis_stats,
        "h1_statistics": h1_stats,
        "h2_statistics": h2_stats,
        "sapbert_model": args.sapbert_model
    }
    
    save_separate_indexes(
        h1_facts, h1_embeddings, h1_relationships, h1_diagnosis_flags,
        h2_facts, h2_embeddings, h2_relationships, h2_diagnosis_flags,
        Path(args.output_dir),
        metadata=metadata
    )
    
    log.info("\n" + "="*80)
    log.info("✅ COMPLETE!")
    log.info("="*80)

if __name__ == "__main__":
    main()