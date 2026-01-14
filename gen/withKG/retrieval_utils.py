#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieval_utils.py

Utilities for fact retrieval with optional relationship weighting.
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

def get_relationship_weight(relationship_entry, aggregation='max'):
    """
    Get weight for a relationship entry (can be single or dual).
    
    Args:
        relationship_entry: Either a string (H1) or list of strings (H2)
        aggregation: How to combine dual relationships for H2 paths
                    'max': Take maximum weight (emphasizes strongest connection)
                    'mean': Average the weights
                    'product': Multiply weights (compound path strength)
                    'sum': Add weights (cumulative evidence)
    
    Returns:
        float: Relationship weight
    """
    # Handle single relationship (H1 format)
    if isinstance(relationship_entry, str):
        return RELATIONSHIP_WEIGHTS.get(relationship_entry, 1.0)
    
    # Handle dual relationships (H2 format)
    if isinstance(relationship_entry, list):
        if len(relationship_entry) == 0:
            return 1.0
        
        weights = [RELATIONSHIP_WEIGHTS.get(rel, 1.0) for rel in relationship_entry]
        
        if aggregation == 'max':
            return max(weights)
        elif aggregation == 'mean':
            return np.mean(weights)
        elif aggregation == 'product':
            return np.prod(weights)
        elif aggregation == 'sum':
            return sum(weights)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Fallback
    return 1.0

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
# MEDICAL FACT INDEX
# =============================================================================

class MedicalFactIndex:
    """Container for fact index with embeddings, FAISS, and relationship metadata."""
    
    def __init__(self, index_dir: str, index_type: str = "combined"):
        """
        Args:
            index_dir: Directory containing the fact index
            index_type: Type of index ("h1", "h2", or "combined")
        """
        index_path = Path(index_dir)
        self.index_type = index_type
        
        log.info(f"Loading {index_type.upper()} fact index from {index_dir}")
        
        # Load facts
        facts_file = index_path / "facts.json"
        with open(facts_file, 'r', encoding='utf-8') as f:
            self.facts = json.load(f)
        log.info(f"  ✓ Loaded {len(self.facts):,} facts")
        
        # Load relationships
        relationships_file = index_path / "relationships.json"
        if relationships_file.exists():
            with open(relationships_file, 'r', encoding='utf-8') as f:
                self.relationships = json.load(f)
            log.info(f"  ✓ Loaded {len(self.relationships):,} relationship labels")
            
            # ✓ Detect format based on first entry
            if self.relationships:
                first_rel = self.relationships[0]
                if isinstance(first_rel, list):
                    log.info(f"  ✓ Detected DUAL relationship format (H2)")
                    self.has_dual_rels = True
                else:
                    log.info(f"  ✓ Detected SINGLE relationship format (H1)")
                    self.has_dual_rels = False
            
            if len(self.facts) != len(self.relationships):
                log.error(f"  ✗ MISALIGNMENT: {len(self.facts)} facts vs {len(self.relationships)} relationships")
                raise ValueError("Facts and relationships count mismatch!")
        else:
            log.warning("  ⚠ No relationships.json found, will extract from text")
            self.relationships = None
            self.has_dual_rels = False

        # Load diagnosis flags
        diagnosis_flags_file = index_path / "diagnosis_flags.json"
        if diagnosis_flags_file.exists():
            with open(diagnosis_flags_file, 'r', encoding='utf-8') as f:
                self.diagnosis_flags = json.load(f)
            log.info(f"  ✓ Loaded {len(self.diagnosis_flags):,} diagnosis flags")
            log.info(f"    Diagnosis target facts: {sum(self.diagnosis_flags)}")
            
            if len(self.diagnosis_flags) != len(self.facts):
                raise ValueError(f"Diagnosis flags/facts mismatch: {len(self.diagnosis_flags)} vs {len(self.facts)}")
        else:
            log.warning("  ⚠ No diagnosis_flags.json found, all facts treated as non-diagnosis")
            self.diagnosis_flags = [False] * len(self.facts)
        
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
        
        log.info(f"✓ {index_type.upper()} fact index loaded successfully")
    
    def get_relationship(self, fact_index: int):
        """
        Get relationship(s) for a fact by index.
        
        Returns:
            str or list: Single relationship (H1) or list of relationships (H2)
        """
        if self.relationships:
            return self.relationships[fact_index]
        else:
            return extract_relationship_from_fact(self.facts[fact_index])
    
    def get_relationship_weight(self, fact_index: int, aggregation: str = 'max') -> float:
        """
        Get relationship weight for a fact.
        
        Args:
            fact_index: Index of the fact
            aggregation: How to combine dual relationships (for H2)
        
        Returns:
            float: Relationship weight
        """
        rel_entry = self.get_relationship(fact_index)
        return get_relationship_weight(rel_entry, aggregation)
    
    def get_relationships_list(self, fact_index: int) -> List[str]:
        """
        Get relationships as a list (for both H1 and H2).
        
        Returns:
            List[str]: Always returns a list, even for single relationships
        """
        rel_entry = self.get_relationship(fact_index)
        if isinstance(rel_entry, list):
            return rel_entry
        else:
            return [rel_entry]

# =============================================================================
# COMBINED INDEX MANAGER
# =============================================================================

class CombinedFactIndex:
    """Manages H1 and H2 indices for combined retrieval."""
    
    def __init__(self, h1_dir: str, h2_dir: str):
        """
        Args:
            h1_dir: Directory for H1 fact index
            h2_dir: Directory for H2 fact index
        """
        log.info("\n" + "="*80)
        log.info("LOADING COMBINED H1 + H2 FACT INDICES")
        log.info("="*80)
        
        self.h1_index = MedicalFactIndex(h1_dir, index_type="h1")
        self.h2_index = MedicalFactIndex(h2_dir, index_type="h2")
        
        log.info(f"\n✓ Combined index ready:")
        log.info(f"  H1 facts: {len(self.h1_index.facts):,} (single relationships)")
        log.info(f"  H2 facts: {len(self.h2_index.facts):,} (dual relationships)")
        log.info(f"  Total:    {len(self.h1_index.facts) + len(self.h2_index.facts):,}")
        log.info("="*80)
    
    def get_relationship_weight(self, fact_index: int, aggregation: str = 'max') -> float:
        """Get weight - delegates to correct sub-index."""
        h1_size = len(self.h1_index.facts)
        
        if fact_index < h1_size:
            # H1 fact - uses single relationship
            return self.h1_index.get_relationship_weight(fact_index, aggregation)
        else:
            # H2 fact - uses dual relationship with aggregation
            h2_idx = fact_index - h1_size
            return self.h2_index.get_relationship_weight(h2_idx, aggregation)
    
    def get_fact(self, fact_index: int) -> str:
        """Get fact text by global index."""
        h1_size = len(self.h1_index.facts)
        
        if fact_index < h1_size:
            return self.h1_index.facts[fact_index]
        else:
            h2_idx = fact_index - h1_size
            return self.h2_index.facts[h2_idx]
    
    def get_relationship(self, fact_index: int):
        """Get relationship(s) by global index."""
        h1_size = len(self.h1_index.facts)
        
        if fact_index < h1_size:
            return self.h1_index.get_relationship(fact_index)
        else:
            h2_idx = fact_index - h1_size
            return self.h2_index.get_relationship(h2_idx)

# =============================================================================
# RELATIONSHIP EXTRACTION (FALLBACK)
# =============================================================================

def extract_relationship_from_fact(fact: str) -> str:
    """Extract relationship type from linearized fact (FALLBACK method)."""
    fact_lower = fact.lower()
    
    if ' which ' in fact_lower:
        fact_lower = fact_lower.split(' which ')[0]
    
    sorted_rels = sorted(RELATIONSHIP_WEIGHTS.keys(), key=len, reverse=True)
    
    for rel in sorted_rels:
        if rel in fact_lower or rel.replace('_', ' ') in fact_lower:
            return rel
    
    return 'other'

# =============================================================================
# RETRIEVAL FUNCTIONS
# =============================================================================

def retrieve_facts(
    query_text: str,
    fact_index,  # Can be MedicalFactIndex or CombinedFactIndex
    encoder: SapBERTEncoder,
    k: int = 20,
    use_weighting: bool = False,
    alpha: float = 0.3,
    h1_ratio: float = 0.5,  # For combined retrieval
    rel_aggregation: str = 'max',  # ✓ NEW: For H2 dual relationships
    debug: bool = False,
    diagnosis_only: bool = False
) -> List[str]:
    """
    Retrieve top-K facts for a query.
    
    Args:
        query_text: Query string
        fact_index: MedicalFactIndex or CombinedFactIndex
        encoder: SapBERTEncoder object
        k: Number of facts to retrieve
        use_weighting: Whether to use relationship weighting
        alpha: Interpolation between semantic and weight
        h1_ratio: Ratio of H1 facts when using CombinedFactIndex (0.0-1.0)
        rel_aggregation: How to aggregate dual relationships for H2
                        'max', 'mean', 'product', or 'sum'
        debug: If True, print detailed scores
    
    Returns:
        List of retrieved fact strings
    """
    query_embedding = encoder.encode(query_text)
    
    # Check if we have combined index
    if isinstance(fact_index, CombinedFactIndex):
        return _retrieve_combined(
            query_embedding, fact_index, k, use_weighting, alpha, h1_ratio, rel_aggregation, debug, diagnosis_only
        )
    
    # Single index retrieval
    if not use_weighting:
        return _retrieve_unweighted(query_embedding, fact_index, k, debug, diagnosis_only)
    else:
        return _retrieve_weighted(query_embedding, fact_index, k, alpha, debug, rel_aggregation, diagnosis_only)

def _retrieve_combined(
    query_embedding: np.ndarray,
    combined_index: CombinedFactIndex,
    k: int,
    use_weighting: bool,
    alpha: float,
    h1_ratio: float,
    rel_aggregation: str,  # ✓ NEW parameter
    debug: bool,
    diagnosis_only: bool = False
) -> List[str]:
    """Retrieve from both H1 and H2 indices using specified ratio."""
    k_h1 = int(k * h1_ratio)
    k_h2 = k - k_h1
    
    if debug:
        log.info(f"\nCombined retrieval: K={k}, H1={k_h1}, H2={k_h2}, ratio={h1_ratio:.2f}, agg={rel_aggregation}")
    
    # Retrieve from H1
    if k_h1 > 0:
        if use_weighting:
            h1_facts = _retrieve_weighted(
                query_embedding, combined_index.h1_index, k_h1, alpha, False, 'max'  # H1 always uses 'max' (single rel)
            )
        else:
            h1_facts = _retrieve_unweighted(
                query_embedding, combined_index.h1_index, k_h1, False
            )
    else:
        h1_facts = []
    
    # Retrieve from H2 (with dual relationship aggregation)
    if k_h2 > 0:
        if use_weighting:
            h2_facts = _retrieve_weighted(
                query_embedding, combined_index.h2_index, k_h2, alpha, False, rel_aggregation  # ✓ Use aggregation
            )
        else:
            h2_facts = _retrieve_unweighted(
                query_embedding, combined_index.h2_index, k_h2, False
            )
    else:
        h2_facts = []
    
    combined_facts = h1_facts + h2_facts
    
    if debug:
        log.info("\n" + "="*80)
        log.info(f"COMBINED RETRIEVAL (H1 ratio={h1_ratio:.2f}, Weighted={use_weighting}, Agg={rel_aggregation})")
        log.info("="*80)
        log.info(f"\nH1 Facts ({len(h1_facts)}):")
        for i, fact in enumerate(h1_facts, 1):
            log.info(f"  {i:2d}. [H1] {fact}")
        log.info(f"\nH2 Facts ({len(h2_facts)}):")
        for i, fact in enumerate(h2_facts, 1):
            log.info(f"  {i:2d}. [H2] {fact}")
        log.info("="*80)
    
    return combined_facts

def _retrieve_unweighted(
    query_embedding: np.ndarray,
    fact_index: MedicalFactIndex,
    k: int,
    debug: bool,
    diagnosis_only: bool = False
) -> List[str]:
    """Pure semantic retrieval."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    
    # ✓ Retrieve more candidates if filtering
    k_retrieve = k * 5 if diagnosis_only else k
    
    scores, indices = fact_index.index.search(query_embedding, k_retrieve)
    
    # ✓ Filter by diagnosis flag if requested
    if diagnosis_only:
        filtered_indices = []
        filtered_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(fact_index.diagnosis_flags) and fact_index.diagnosis_flags[idx]:
                filtered_indices.append(idx)
                filtered_scores.append(score)
                if len(filtered_indices) >= k:
                    break
        
        # ✓ HANDLE EMPTY RESULTS
        if not filtered_indices:
            log.warning(f"No diagnosis-targeting facts found for query (retrieved {len(indices[0])} candidates)")
            return []
        
        indices = [filtered_indices]
        scores = [filtered_scores]
    
    retrieved_facts = [fact_index.facts[idx] for idx in indices[0]]
    
    if debug:
        log.info("\n" + "="*80)
        mode = "DIAGNOSIS-ONLY" if diagnosis_only else "ALL FACTS"
        log.info(f"UNWEIGHTED RETRIEVAL - {fact_index.index_type.upper()} ({mode})")
        log.info("="*80)
        for i, (idx, fact, score) in enumerate(zip(indices[0], retrieved_facts, scores[0]), 1):
            rel = fact_index.get_relationship(idx)
            weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
            is_dx = "DX" if fact_index.diagnosis_flags[idx] else "GK"
            log.info(f"{i:2d}. [{is_dx}] [Score: {score:.4f} | Rel: {rel:20s} w={weight:.1f}] {fact}")
        log.info("="*80)
    
    return retrieved_facts

def _retrieve_weighted(
    query_embedding: np.ndarray,
    fact_index: MedicalFactIndex,
    k: int,
    alpha: float,
    debug: bool,
    rel_aggregation: str = 'max',
    diagnosis_only: bool = False
) -> List[str]:
    """Weighted retrieval with relationship importance."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    
    # ✓ Retrieve more candidates if filtering
    k_candidates = min(k * 10 if diagnosis_only else k * 5, fact_index.index.ntotal)
    
    semantic_scores, indices = fact_index.index.search(query_embedding, k_candidates)
    
    semantic_scores = semantic_scores[0]
    indices = indices[0]

    # ✓ Filter by diagnosis flag BEFORE scoring
    if diagnosis_only:
        filtered_indices = []
        filtered_scores = []
        for idx, score in zip(indices, semantic_scores):
            if idx >= 0 and idx < len(fact_index.diagnosis_flags) and fact_index.diagnosis_flags[idx]:
                filtered_indices.append(idx)
                filtered_scores.append(score)
        
        # ✓ HANDLE EMPTY RESULTS
        if not filtered_indices:
            log.warning(f"No diagnosis-targeting facts found for query (retrieved {k_candidates} candidates)")
            return []
        
        indices = np.array(filtered_indices)
        semantic_scores = np.array(filtered_scores)
    
    # ✓ Get relationship weights
    relationship_scores = []
    for idx in indices:
        weight = fact_index.get_relationship_weight(idx, aggregation=rel_aggregation)
        relationship_scores.append(weight)
    
    relationship_scores = np.array(relationship_scores)
    
    # ✓ SAFE NORMALIZATION - Check for empty arrays and identical values
    if len(semantic_scores) == 0:
        return []
    
    if len(semantic_scores) == 1:
        semantic_scores_norm = np.array([1.0])
        relationship_scores_norm = np.array([1.0])
    else:
        # Normalize semantic scores
        sem_min, sem_max = semantic_scores.min(), semantic_scores.max()
        if sem_max - sem_min < 1e-8:
            semantic_scores_norm = np.ones_like(semantic_scores)
        else:
            semantic_scores_norm = (semantic_scores - sem_min) / (sem_max - sem_min)
        
        # Normalize relationship scores
        rel_min, rel_max = relationship_scores.min(), relationship_scores.max()
        if rel_max - rel_min < 1e-8:
            relationship_scores_norm = np.ones_like(relationship_scores)
        else:
            relationship_scores_norm = (relationship_scores - rel_min) / (rel_max - rel_min)
    
    # Combine scores
    final_scores = (1 - alpha) * semantic_scores_norm + alpha * relationship_scores_norm
    
    # Select top-K
    k_select = min(k, len(final_scores))
    top_k_positions = np.argsort(final_scores)[::-1][:k_select]
    retrieved_facts = [fact_index.facts[indices[i]] for i in top_k_positions]
    
    if debug:
        log.info("\n" + "="*80)
        mode = "DIAGNOSIS-ONLY" if diagnosis_only else "ALL FACTS"
        log.info(f"WEIGHTED RETRIEVAL - {fact_index.index_type.upper()} (Alpha={alpha}, Agg={rel_aggregation})")
        log.info("="*80)
        
        for i, pos in enumerate(top_k_positions, 1):
            idx = indices[pos]
            fact = fact_index.facts[idx]
            rel = fact_index.get_relationship(idx)
            rel_weight = fact_index.get_relationship_weight(idx, aggregation=rel_aggregation)
            
            # ✓ Format relationship display for H1 vs H2
            if isinstance(rel, list):
                rel_str = " + ".join(rel)  # e.g., "isa + other"
                weights_str = " + ".join([f"{RELATIONSHIP_WEIGHTS.get(r, 1.0):.1f}" for r in rel])
                rel_display = f"{rel_str:30s} ({weights_str}) → {rel_weight:.1f}"
            else:
                rel_display = f"{rel:30s} ({rel_weight:.1f})"
            
            log.info(
                f"{i:2d}. [Sem: {semantic_scores[pos]:.4f} | "
                f"Rel: {rel_display} | "
                f"Final: {final_scores[pos]:.4f}] {fact}"
            )
        log.info("="*80)
    
    return retrieved_facts

# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def print_relationship_distribution(
    facts: List[str], 
    fact_index: Optional[MedicalFactIndex] = None,
    aggregation: str = 'sum',  # ✓ ADD THIS
    top_n: int = 15
):
    """Print distribution of relationship types in retrieved facts."""
    if fact_index and fact_index.relationships:
        relationships = []
        weights_map = {}  # ✓ Store actual aggregated weights
        
        for fact in facts:
            try:
                idx = fact_index.facts.index(fact)
                rel_entry = fact_index.get_relationship(idx)
                weight = fact_index.get_relationship_weight(idx, aggregation=aggregation)
                
                # ✓ Format relationship display
                if isinstance(rel_entry, list):
                    rel_str = "+".join(rel_entry)
                else:
                    rel_str = rel_entry
                
                relationships.append(rel_str)
                weights_map[rel_str] = weight  # ✓ Store aggregated weight
                
            except ValueError:
                relationships.append('other')
                weights_map['other'] = 1.0
        
        log.info("Using GROUND TRUTH relationship metadata")
    else:
        relationships = [extract_relationship_from_fact(fact) for fact in facts]
        weights_map = {rel: RELATIONSHIP_WEIGHTS.get(rel, 1.0) for rel in set(relationships)}
        log.info("Using EXTRACTED relationships")
    
    counter = Counter(relationships)
    
    log.info("\nRelationship Distribution:")
    log.info(f"  Total facts: {len(facts)}")
    log.info(f"  Unique relationships: {len(counter)}")
    log.info("\nTop relationships:")
    
    for rel, count in counter.most_common(top_n):
        weight = weights_map.get(rel, 1.0)  # ✓ Use actual aggregated weight
        pct = 100 * count / len(facts)
        
        # ✓ Better formatting for dual relationships
        if '+' in rel:
            # Parse dual relationship
            parts = rel.split('+')
            if len(parts) == 2:
                w1 = RELATIONSHIP_WEIGHTS.get(parts[0], 1.0)
                w2 = RELATIONSHIP_WEIGHTS.get(parts[1], 1.0)
                log.info(
                    f"  {rel:30s}: {count:5d} ({pct:5.1f}%) "
                    f"[{parts[0]}={w1:.1f} + {parts[1]}={w2:.1f} → {weight:.1f}]"
                )
            else:
                log.info(f"  {rel:30s}: {count:5d} ({pct:5.1f}%) [weight={weight:.1f}]")
        else:
            log.info(f"  {rel:30s}: {count:5d} ({pct:5.1f}%) [weight={weight:.1f}]")

def get_relationship_weights_for_facts(
    facts: List[str],
    fact_index,
    aggregation: str = 'max'  
) -> List[float]:
    """Get relationship weights for a list of facts."""
    weights = []
    
    # Handle both single and combined indices
    if isinstance(fact_index, CombinedFactIndex):
        for fact in facts:
            # Try H1 first, then H2
            try:
                idx = fact_index.h1_index.facts.index(fact)
                weight = fact_index.h1_index.get_relationship_weight(idx, aggregation)
                weights.append(weight)
            except ValueError:
                try:
                    idx = fact_index.h2_index.facts.index(fact)
                    weight = fact_index.h2_index.get_relationship_weight(idx, aggregation)
                    weights.append(weight)
                except ValueError:
                    weights.append(1.0)
    else:
        for fact in facts:
            try:
                idx = fact_index.facts.index(fact)
                weight = fact_index.get_relationship_weight(idx, aggregation)
                weights.append(weight)
            except ValueError:
                weights.append(1.0)
    
    return weights