#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_medical_fact_index.py

OFFLINE PREPROCESSING: Build a searchable index of medical facts from UMLS KG.

This script:
1. Mines ALL H1/H2 paths from your KG
2. Linearizes them to natural language
3. Embeds them with SapBERT
4. Stores in FAISS for fast semantic retrieval

Output: A fact index that can be used for RAG-based preprocessing.
"""

import argparse
import pickle
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set
import re

import numpy as np
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# FAISS
try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Run: pip install faiss-cpu")
    sys.exit(1)

# Import your common utilities
from common_textgen import log, is_main_process, _strip

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =============================================================================
# SAPBERT ENCODER
# =============================================================================

class SapBERTEncoder:
    """Wrapper for SapBERT model to encode medical text."""
    
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        log.info(f"Loading SapBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        log.info(f"SapBERT loaded on device: {self.device}")
    
    def _mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling over token embeddings."""
        # Expand attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        
        # Count non-padding tokens
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,  # SapBERT was trained on 128 tokens
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            outputs = self.model(**encoded)
            
            # Mean pooling
            embeddings = self._mean_pooling(
                outputs.last_hidden_state, 
                encoded['attention_mask']
            )
            
            # Normalize (for cosine similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

# =============================================================================
# PATH MINING (Same as your preprocessing script)
# =============================================================================

def _arrow_label(rela: str, rel: str) -> str:
    """Format relationship label."""
    r = (rela or "").strip() or (rel or "").strip()
    return f" {r} " if r else " related_to "

def mine_all_h1_paths(G: nx.DiGraph, max_paths: int = None) -> List[dict]:
    """
    Mine ALL H1 (direct) paths from the knowledge graph.
    
    Returns:
        List of dicts with keys: src_cui, nbr_cui, src_name, nbr_name, rel, rela, rela_canon
    """
    log.info("Mining H1 paths (direct edges)...")
    H1 = []
    
    for u, v, edge_data in tqdm(G.edges(data=True), desc="H1 edges"):
        # Get node names
        u_name = G.nodes[u].get("name", u)
        v_name = G.nodes[v].get("name", v)
        
        # Get edge attributes
        rel = edge_data.get("rel", "")
        rela = edge_data.get("rela", "")
        rela_canon = edge_data.get("rela_canon", "") or rela
        
        H1.append({
            "src_cui": u,
            "nbr_cui": v,
            "src_name": u_name,
            "nbr_name": v_name,
            "rel": rel,
            "rela": rela,
            "rela_canon": rela_canon
        })
        
        if max_paths and len(H1) >= max_paths:
            break
    
    log.info(f"Mined {len(H1)} H1 paths")
    return H1

def mine_all_h2_paths(G: nx.DiGraph, max_paths: int = None, 
                      max_per_source: int = 100) -> List[dict]:
    """
    Mine ALL H2 (two-hop) paths from the knowledge graph.
    
    Args:
        G: NetworkX directed graph
        max_paths: Maximum total paths to mine (None = all)
        max_per_source: Maximum paths to mine per source node
    
    Returns:
        List of dicts with keys: u, v, w, u_name, v_name, w_name, rel_uv, rela_uv, etc.
    """
    log.info("Mining H2 paths (two-hop)...")
    H2 = []
    
    nodes_list = list(G.nodes())
    
    for u in tqdm(nodes_list, desc="H2 paths"):
        if max_paths and len(H2) >= max_paths:
            break
        
        paths_from_u = 0
        
        # First hop: u -> v
        for v in G.successors(u):
            if max_paths and len(H2) >= max_paths:
                break
            if paths_from_u >= max_per_source:
                break
            
            # Get edge u->v
            edge_uv = G[u][v]
            rel_uv = edge_uv.get("rel", "")
            rela_uv = edge_uv.get("rela", "")
            rela_uv_canon = edge_uv.get("rela_canon", "") or rela_uv
            
            # Second hop: v -> w
            for w in G.successors(v):
                if w == u:  # Skip cycles back to source
                    continue
                
                if max_paths and len(H2) >= max_paths:
                    break
                if paths_from_u >= max_per_source:
                    break
                
                # Get edge v->w
                edge_vw = G[v][w]
                rel_vw = edge_vw.get("rel", "")
                rela_vw = edge_vw.get("rela", "")
                rela_vw_canon = edge_vw.get("rela_canon", "") or rela_vw
                
                # Get names
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
                    "rel_uv": rel_uv,
                    "rela_uv": rela_uv,
                    "rela_uv_canon": rela_uv_canon,
                    "rel_vw": rel_vw,
                    "rela_vw": rela_vw,
                    "rela_vw_canon": rela_vw_canon
                })
                
                paths_from_u += 1
    
    log.info(f"Mined {len(H2)} H2 paths")
    return H2

# =============================================================================
# PATH LINEARIZATION
# =============================================================================

def clean_name(name: str) -> str:
    """Clean concept name for readability."""
    if not name:
        return ""
    # Remove CUI prefix if present
    name = re.sub(r'^C\d+:', '', name)
    # Remove excessive whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def linearize_h1_path(path: dict) -> str:
    """
    Convert H1 path to natural language sentence.
    
    Example:
        Input: {"src_name": "Lisinopril", "nbr_name": "Hypertension", "rela_canon": "may_treat"}
        Output: "Lisinopril may_treat Hypertension"
    """
    src = clean_name(path.get("src_name", ""))
    tgt = clean_name(path.get("nbr_name", ""))
    rel = path.get("rela_canon") or path.get("rela") or path.get("rel") or "related_to"
    
    # Clean relationship
    rel = rel.strip().replace("_", " ")
    
    if not src or not tgt:
        return ""
    
    return f"{src} {rel} {tgt}"

def linearize_h2_path(path: dict) -> str:
    """
    Convert H2 path to natural language sentence.
    
    Example:
        Input: {
            "u_name": "Aspirin", 
            "v_name": "Platelet aggregation", 
            "w_name": "Myocardial infarction",
            "rela_uv_canon": "inhibits",
            "rela_vw_canon": "associated_with"
        }
        Output: "Aspirin inhibits Platelet aggregation which associated with Myocardial infarction"
    """
    u = clean_name(path.get("u_name", ""))
    v = clean_name(path.get("v_name", ""))
    w = clean_name(path.get("w_name", ""))
    
    rel_uv = path.get("rela_uv_canon") or path.get("rela_uv") or path.get("rel_uv") or "related_to"
    rel_vw = path.get("rela_vw_canon") or path.get("rela_vw") or path.get("rel_vw") or "related_to"
    
    # Clean relationships
    rel_uv = rel_uv.strip().replace("_", " ")
    rel_vw = rel_vw.strip().replace("_", " ")
    
    if not u or not v or not w:
        return ""
    
    return f"{u} {rel_uv} {v} which {rel_vw} {w}"

# =============================================================================
# FAISS INDEX CREATION
# =============================================================================

def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Create FAISS index for fast similarity search.
    
    Args:
        embeddings: numpy array of shape (n_facts, embedding_dim)
        use_gpu: whether to use GPU for FAISS
    
    Returns:
        FAISS index (normalized for cosine similarity)
    """
    n, d = embeddings.shape
    log.info(f"Creating FAISS index: {n} vectors, {d} dimensions")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index
    if use_gpu and torch.cuda.is_available():
        log.info("Using GPU for FAISS index")
        try:
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(d)  # Inner product = cosine after normalization
            index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        except Exception as e:
            log.warning(f"GPU FAISS failed, falling back to CPU: {e}")
            index = faiss.IndexFlatIP(d)
    else:
        log.info("Using CPU for FAISS index")
        index = faiss.IndexFlatIP(d)  # Inner product
    
    # Add vectors
    index.add(embeddings.astype(np.float32))
    
    log.info(f"FAISS index created with {index.ntotal} vectors")
    return index

# =============================================================================
# SAVE/LOAD FUNCTIONS
# =============================================================================

def save_fact_index(facts: List[str], 
                    embeddings: np.ndarray, 
                    index: faiss.Index, 
                    output_dir: Path,
                    metadata: dict = None):
    """Save the complete fact index to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Saving fact index to {output_dir}")
    
    # 1. Save facts as JSON
    facts_file = output_dir / "facts.json"
    with open(facts_file, 'w', encoding='utf-8') as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    log.info(f"  ✓ Saved {len(facts)} facts to {facts_file}")
    
    # 2. Save embeddings as numpy
    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    log.info(f"  ✓ Saved embeddings to {embeddings_file}")
    
    # 3. Save FAISS index
    index_file = output_dir / "faiss_index.bin"
    # Convert GPU index to CPU before saving
    if hasattr(index, 'index'):  # GPU index
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, str(index_file))
    else:
        faiss.write_index(index, str(index_file))
    log.info(f"  ✓ Saved FAISS index to {index_file}")
    
    # 4. Save metadata
    meta = {
        "n_facts": len(facts),
        "embedding_dim": embeddings.shape[1],
        "index_type": "IndexFlatIP",
        "metric": "cosine",
        "normalized": True,
        "sapbert_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    }
    if metadata:
        meta.update(metadata)
    
    meta_file = output_dir / "metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  ✓ Saved metadata to {meta_file}")
    
    log.info(f"✓ Complete fact index saved to {output_dir}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Medical Fact Index from UMLS Knowledge Graph"
    )
    
    # Input
    parser.add_argument("--kg_pkl", required=True, 
                       help="Path to KG pickle file (NetworkX DiGraph)")
    
    # Output
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save the fact index")
    
    # Mining parameters
    parser.add_argument("--max_h1", type=int, default=None,
                       help="Maximum H1 paths to mine (None = all)")
    parser.add_argument("--max_h2", type=int, default=None,
                       help="Maximum H2 paths to mine (None = all)")
    parser.add_argument("--max_h2_per_source", type=int, default=100,
                       help="Maximum H2 paths per source node")
    
    # Encoding parameters
    parser.add_argument("--sapbert_model", 
                       default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                       help="SapBERT model name or path")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for encoding")
    
    # Hardware
    parser.add_argument("--use_gpu_faiss", action="store_true",
                       help="Use GPU for FAISS index")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # ==========================================================================
    # STEP 1: LOAD KNOWLEDGE GRAPH
    # ==========================================================================
    log.info("="*80)
    log.info("STEP 1: LOADING KNOWLEDGE GRAPH")
    log.info("="*80)
    
    log.info(f"Loading KG from {args.kg_pkl}")
    with open(args.kg_pkl, "rb") as f:
        G = pickle.load(f)
    
    log.info(f"✓ KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # ==========================================================================
    # STEP 2: MINE PATHS
    # ==========================================================================
    log.info("\n" + "="*80)
    log.info("STEP 2: MINING PATHS FROM KG")
    log.info("="*80)
    
    start_time = time.time()
    
    # Mine H1 paths
    H1_paths = mine_all_h1_paths(G, max_paths=args.max_h1)
    
    # Mine H2 paths
    H2_paths = mine_all_h2_paths(
        G, 
        max_paths=args.max_h2,
        max_per_source=args.max_h2_per_source
    )
    
    mining_time = time.time() - start_time
    log.info(f"✓ Mining complete in {mining_time/60:.1f} minutes")
    log.info(f"  H1 paths: {len(H1_paths)}")
    log.info(f"  H2 paths: {len(H2_paths)}")
    log.info(f"  Total: {len(H1_paths) + len(H2_paths)} paths")
    
    # ==========================================================================
    # STEP 3: LINEARIZE PATHS
    # ==========================================================================
    log.info("\n" + "="*80)
    log.info("STEP 3: LINEARIZING PATHS TO NATURAL LANGUAGE")
    log.info("="*80)
    
    facts = []
    
    # Linearize H1
    log.info("Linearizing H1 paths...")
    for path in tqdm(H1_paths, desc="H1"):
        fact = linearize_h1_path(path)
        if fact:  # Skip empty
            facts.append(fact)
    
    # Linearize H2
    log.info("Linearizing H2 paths...")
    for path in tqdm(H2_paths, desc="H2"):
        fact = linearize_h2_path(path)
        if fact:  # Skip empty
            facts.append(fact)
    
    log.info(f"✓ Linearized {len(facts)} facts")
    
    # Show examples
    log.info("\nExample facts:")
    for i, fact in enumerate(facts[:5], 1):
        log.info(f"  {i}. {fact}")
    
    # ==========================================================================
    # STEP 4: ENCODE WITH SAPBERT
    # ==========================================================================
    log.info("\n" + "="*80)
    log.info("STEP 4: ENCODING FACTS WITH SAPBERT")
    log.info("="*80)
    
    # Initialize encoder
    encoder = SapBERTEncoder(args.sapbert_model)
    
    # Encode all facts
    start_time = time.time()
    embeddings = encoder.encode_batch(facts, batch_size=args.batch_size)
    encoding_time = time.time() - start_time
    
    log.info(f"✓ Encoding complete in {encoding_time/60:.1f} minutes")
    log.info(f"  Embeddings shape: {embeddings.shape}")
    log.info(f"  Speed: {len(facts)/encoding_time:.1f} facts/second")
    
    # ==========================================================================
    # STEP 5: CREATE FAISS INDEX
    # ==========================================================================
    log.info("\n" + "="*80)
    log.info("STEP 5: CREATING FAISS INDEX")
    log.info("="*80)
    
    index = create_faiss_index(embeddings, use_gpu=args.use_gpu_faiss)
    
    log.info(f"✓ FAISS index created with {index.ntotal} vectors")
    
    # ==========================================================================
    # STEP 6: SAVE EVERYTHING
    # ==========================================================================
    log.info("\n" + "="*80)
    log.info("STEP 6: SAVING FACT INDEX")
    log.info("="*80)
    
    metadata = {
        "n_h1_paths": len(H1_paths),
        "n_h2_paths": len(H2_paths),
        "mining_time_seconds": mining_time,
        "encoding_time_seconds": encoding_time,
        "sapbert_model": args.sapbert_model
    }
    
    save_fact_index(facts, embeddings, index, output_dir, metadata)
    
    # ==========================================================================
    # DONE
    # ==========================================================================
    total_time = time.time() - start_time + mining_time
    
    log.info("\n" + "="*80)
    log.info("✓ FACT INDEX BUILD COMPLETE")
    log.info("="*80)
    log.info(f"Total time: {total_time/60:.1f} minutes")
    log.info(f"Total facts: {len(facts)}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"\nFiles created:")
    log.info(f"  - facts.json (list of fact strings)")
    log.info(f"  - embeddings.npy (numpy array)")
    log.info(f"  - faiss_index.bin (FAISS index)")
    log.info(f"  - metadata.json (index metadata)")
    log.info("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())