#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieval_utils.py

Utilities for fact retrieval with optional relationship weighting.

UPDATED: Now loads and uses relationship metadata from index for accurate weighting.
"""

import json
import logging
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger(__name__)

# =============================================================================
# RELATIONSHIP WEIGHTS (Manual curation based on diagnostic relevance)
# =============================================================================

RELATIONSHIP_WEIGHTS = {
    # Tier 1: Direct causal/diagnostic relationships (3.0)
    'etiology': 3.0,
    'may_cause': 3.0,
    'pathology': 3.0,
    
    # Tier 2: Strong associations (2.0-2.5)
    'may_treat': 2.5,
    'finding_site': 2.5,
    'clinical_course': 2.5,
    'associated_with': 2.0,
    'assoc': 2.0,
    'morphology': 2.0,
    
    # Tier 3: Moderate associations (1.0-1.5)
    'location': 1.0,
    'measurement': 1.5,
    'procedure_method': 1.5,
    'proc_method': 1.5,
    'proc_site': 1.0,
    'course': 1.0,
    
    # Tier 4: Structural relationships (0.5-0.8)
    'isa': 0.5,
    'equivalent': 0.8,
    'meta': 0.5,
    'other': 0.5,
    
    # Tier 5: Less diagnostic (0.3-1.2)
    'temporal': 0.7,
    'intent': 0.3,
    'priority': 0.3,
    'severity': 1.2,
    'proc_device': 0.5,
    'procedure_device': 0.5,
}

# =============================================================================
# SAPBERT ENCODER
# =============================================================================

class SapBERTEncoder:
    """Wrapper for SapBERT model."""
    
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        log.info(f"Loading SapBERT encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        log.info(f"  Device: {self.device}")
    
    def _mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling over token embeddings."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**encoded)
        embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()

# =============================================================================
# MEDICAL FACT INDEX (UPDATED WITH RELATIONSHIP METADATA)
# =============================================================================

class MedicalFactIndex:
    """Container for fact index with embeddings, FAISS, and relationship metadata."""
    
    def __init__(self, index_dir: str):
        index_path = Path(index_dir)
        
        log.info(f"Loading fact index from {index_dir}")
        
        # Load facts
        facts_file = index_path / "facts.json"
        with open(facts_file, 'r', encoding='utf-8') as f:
            self.facts = json.load(f)
        log.info(f"  ✓ Loaded {len(self.facts):,} facts")
        
        # Load relationships (NEW: ground truth metadata)
        relationships_file = index_path / "relationships.json"
        if relationships_file.exists():
            with open(relationships_file, 'r', encoding='utf-8') as f:
                self.relationships = json.load(f)
            log.info(f"  ✓ Loaded {len(self.relationships):,} relationship labels (GROUND TRUTH)")
            
            # Validate alignment
            if len(self.facts) != len(self.relationships):
                log.error(f"  ✗ MISALIGNMENT: {len(self.facts)} facts vs {len(self.relationships)} relationships")
                raise ValueError("Facts and relationships count mismatch!")
        else:
            log.warning("  ⚠ No relationships.json found, will extract from text (less accurate)")
            self.relationships = None
        
        # Load embeddings
        embeddings_file = index_path / "embeddings.npy"
        self.embeddings = np.load(embeddings_file)
        log.info(f"  ✓ Loaded embeddings: {self.embeddings.shape}")
        
        # Load FAISS index
        faiss_file = index_path / "faiss_index.bin"
        self.index = faiss.read_index(str(faiss_file))
        log.info(f"  ✓ Loaded FAISS index: {self.index.ntotal:,} vectors")
        
        # Load metadata
        meta_file = index_path / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Final validation
        if len(self.facts) != self.embeddings.shape[0]:
            raise ValueError(f"Facts/embeddings mismatch: {len(self.facts)} vs {self.embeddings.shape[0]}")
        
        if self.relationships and len(self.relationships) != len(self.facts):
            raise ValueError(f"Relationships/facts mismatch: {len(self.relationships)} vs {len(self.facts)}")
        
        log.info(f"✓ Fact index loaded successfully")
    
    def get_relationship(self, fact_index: int) -> str:
        """
        Get relationship for a fact by index.
        
        Uses ground truth metadata if available, otherwise extracts from text.
        """
        if self.relationships:
            return self.relationships[fact_index]
        else:
            # Fallback to text extraction
            return extract_relationship_from_fact(self.facts[fact_index])

# =============================================================================
# RELATIONSHIP EXTRACTION (FALLBACK FOR OLD INDICES)
# =============================================================================

def extract_relationship_from_fact(fact: str) -> str:
    """
    Extract relationship type from linearized fact (FALLBACK method).
    
    This is used only if relationship metadata is not available.
    
    Examples:
        "Aspirin may treat MI" → "may_treat"
        "Bacteria etiology Pneumonia" → "etiology"
        "Hysteroscopy (procedure) proc method Evaluation" → "proc_method"
    """
    fact_lower = fact.lower()
    
    # For H2 paths, take first part (before "which")
    if ' which ' in fact_lower:
        fact_lower = fact_lower.split(' which ')[0]
    
    # Try to match known relationships by scanning tokens
    # Sort by length (longest first) to match "may_treat" before "treat"
    sorted_rels = sorted(RELATIONSHIP_WEIGHTS.keys(), key=len, reverse=True)
    
    for rel in sorted_rels:
        # Try both underscore and space versions
        if rel in fact_lower or rel.replace('_', ' ') in fact_lower:
            return rel
    
    # Fallback
    return 'other'

# =============================================================================
# RETRIEVAL FUNCTIONS (UPDATED)
# =============================================================================

def retrieve_facts(
    query_text: str,
    fact_index: MedicalFactIndex,
    encoder: SapBERTEncoder,
    k: int = 20,
    use_weighting: bool = False,
    alpha: float = 0.3,
    debug: bool = False
) -> List[str]:
    """
    Retrieve top-K facts for a query.
    
    Args:
        query_text: Query string (e.g., clinical notes)
        fact_index: MedicalFactIndex object (with relationship metadata)
        encoder: SapBERTEncoder object
        k: Number of facts to retrieve
        use_weighting: Whether to use relationship weighting
        alpha: Interpolation between semantic (1-alpha) and weight (alpha)
        debug: If True, print detailed scores
    
    Returns:
        List of retrieved fact strings
    """
    # Encode query
    query_embedding = encoder.encode(query_text)
    
    if not use_weighting:
        # UNWEIGHTED: Pure semantic similarity
        return _retrieve_unweighted(
            query_embedding, fact_index, k, debug
        )
    else:
        # WEIGHTED: Relationship-aware
        return _retrieve_weighted(
            query_embedding, fact_index, k, alpha, debug
        )

def _retrieve_unweighted(
    query_embedding: np.ndarray,
    fact_index: MedicalFactIndex,
    k: int,
    debug: bool
) -> List[str]:
    """Pure semantic retrieval."""
    # Ensure correct shape
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    
    # Search
    scores, indices = fact_index.index.search(query_embedding, k)
    
    retrieved_facts = [fact_index.facts[idx] for idx in indices[0]]
    
    if debug:
        log.info("\n" + "="*80)
        log.info("UNWEIGHTED RETRIEVAL (Pure Semantic Similarity)")
        log.info("="*80)
        for i, (idx, fact, score) in enumerate(zip(indices[0], retrieved_facts, scores[0]), 1):
            rel = fact_index.get_relationship(idx)
            weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
            log.info(f"{i:2d}. [Score: {score:.4f} | Rel: {rel:20s} w={weight:.1f}] {fact}")
        log.info("="*80)
    
    return retrieved_facts

def _retrieve_weighted(
    query_embedding: np.ndarray,
    fact_index: MedicalFactIndex,
    k: int,
    alpha: float,
    debug: bool
) -> List[str]:
    """
    Weighted retrieval with relationship importance.
    
    UPDATED: Uses ground truth relationship metadata from index.
    """
    # Ensure correct shape
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    
    # Step 1: Over-retrieve candidates
    k_candidates = min(k * 5, fact_index.index.ntotal)
    
    semantic_scores, indices = fact_index.index.search(query_embedding, k_candidates)
    
    semantic_scores = semantic_scores[0]
    indices = indices[0]
    
    # Step 2: Get relationship weights using GROUND TRUTH metadata
    relationship_scores = []
    for idx in indices:
        rel = fact_index.get_relationship(idx)  # Uses metadata if available
        weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
        relationship_scores.append(weight)
    
    relationship_scores = np.array(relationship_scores)
    
    # Step 3: Normalize scores
    semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (
        semantic_scores.max() - semantic_scores.min() + 1e-8
    )
    
    relationship_scores_norm = (relationship_scores - relationship_scores.min()) / (
        relationship_scores.max() - relationship_scores.min() + 1e-8
    )
    
    # Step 4: Combine scores
    final_scores = (
        (1 - alpha) * semantic_scores_norm +
        alpha * relationship_scores_norm
    )
    
    # Step 5: Select top-K
    top_k_positions = np.argsort(final_scores)[::-1][:k]
    
    retrieved_facts = [fact_index.facts[indices[i]] for i in top_k_positions]
    
    if debug:
        log.info("\n" + "="*80)
        log.info(f"WEIGHTED RETRIEVAL (Alpha={alpha})")
        log.info("="*80)
        
        if fact_index.relationships:
            log.info("Using GROUND TRUTH relationship metadata")
        else:
            log.info("⚠ Using EXTRACTED relationships (no metadata)")
        
        log.info("="*80)
        
        for i, pos in enumerate(top_k_positions, 1):
            idx = indices[pos]
            fact = fact_index.facts[idx]
            rel = fact_index.get_relationship(idx)
            rel_weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
            
            log.info(
                f"{i:2d}. [Sem: {semantic_scores[pos]:.4f} | "
                f"Rel: {rel:20s} ({rel_weight:.1f}) | "
                f"Final: {final_scores[pos]:.4f}] {fact}"
            )
        log.info("="*80)
    
    return retrieved_facts

# =============================================================================
# ANALYSIS UTILITIES (UPDATED)
# =============================================================================

def print_relationship_distribution(
    facts: List[str], 
    fact_index: Optional[MedicalFactIndex] = None,
    top_n: int = 15
):
    """
    Print distribution of relationship types in retrieved facts.
    
    Args:
        facts: List of fact strings
        fact_index: Optional MedicalFactIndex to use metadata
        top_n: Number of top relationships to show
    """
    if fact_index and fact_index.relationships:
        # Use ground truth metadata
        relationships = []
        for fact in facts:
            try:
                idx = fact_index.facts.index(fact)
                rel = fact_index.get_relationship(idx)
                relationships.append(rel)
            except ValueError:
                relationships.append('other')
        
        log.info("Using GROUND TRUTH relationship metadata")
    else:
        # Fallback to extraction
        relationships = [extract_relationship_from_fact(fact) for fact in facts]
        log.info("Using EXTRACTED relationships (no metadata)")
    
    counter = Counter(relationships)
    
    log.info("\nRelationship Distribution:")
    log.info(f"  Total facts: {len(facts)}")
    log.info(f"  Unique relationships: {len(counter)}")
    log.info("\nTop relationships:")
    
    for rel, count in counter.most_common(top_n):
        weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
        pct = 100 * count / len(facts)
        log.info(f"  {rel:25s}: {count:5d} ({pct:5.1f}%) [weight={weight:.1f}]")

def get_relationship_weights_for_facts(
    facts: List[str],
    fact_index: MedicalFactIndex
) -> List[float]:
    """
    Get relationship weights for a list of facts.
    
    Args:
        facts: List of fact strings
        fact_index: MedicalFactIndex with relationship metadata
    
    Returns:
        List of weights corresponding to each fact
    """
    weights = []
    for fact in facts:
        try:
            idx = fact_index.facts.index(fact)
            rel = fact_index.get_relationship(idx)
            weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
            weights.append(weight)
        except ValueError:
            # Fact not found in index
            weights.append(1.0)
    
    return weights