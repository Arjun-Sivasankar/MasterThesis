#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_data_with_rag.py

RAG-based preprocessing: Retrieve relevant facts from the medical index
and include them in the training prompts.

UPDATED:
  - Fixed comparison mode (weighted vs unweighted)
  - Matches common_textgen.py prompt format
  - KG facts placed AFTER clinical notes (before task)
  - Comprehensive statistics tracking
"""

import argparse
import json
import logging
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

# Import from common_textgen
sys.path.insert(0, '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/withKG/withRAG/')

from common_textgen_rag import (
    to_list,
    format_icd9,
    format_icd9_proc_from_pro,
    serialize_structured_readable,
    serialize_notes,
    token_len,
    load_knowledge_graph
)

# Import from retrieval_utils
from retrieval_utils import (
    MedicalFactIndex,
    SapBERTEncoder,
    retrieve_facts,
    print_relationship_distribution,
    get_relationship_weights_for_facts,
    RELATIONSHIP_WEIGHTS
)

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
# STATISTICS TRACKER
# =============================================================================

class RetrievalStatistics:
    """Track detailed statistics for retrieval experiments."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        # Per-sample stats
        self.sample_stats = []
        
        # Aggregate stats
        self.all_retrieved_facts = []
        self.all_relationship_weights = []
        self.all_relationships = []
        
        # Token stats
        self.prompt_tokens = []
        self.kg_tokens = []
        self.notes_tokens = []
        
        # Overlap stats (for comparison mode)
        self.overlap_counts = []
        
        # Timing
        self.retrieval_times = []
    
    def add_sample(self, 
                   retrieved_facts: List[str],
                   relationships: List[str],
                   weights: List[float],
                   prompt_tokens: int,
                   kg_tokens: int = 0,
                   notes_tokens: int = 0,
                   retrieval_time: float = 0.0,
                   overlap_count: int = None):
        """Add statistics for one sample."""
        
        # Store per-sample data
        sample_data = {
            'n_facts': len(retrieved_facts),
            'avg_weight': np.mean(weights) if weights else 0.0,
            'prompt_tokens': prompt_tokens,
            'kg_tokens': kg_tokens,
            'retrieval_time': retrieval_time
        }
        
        if overlap_count is not None:
            sample_data['overlap'] = overlap_count
        
        self.sample_stats.append(sample_data)
        
        # Aggregate data
        self.all_retrieved_facts.extend(retrieved_facts)
        self.all_relationship_weights.extend(weights)
        self.all_relationships.extend(relationships)
        
        # Token stats
        self.prompt_tokens.append(prompt_tokens)
        if kg_tokens > 0:
            self.kg_tokens.append(kg_tokens)
        if notes_tokens > 0:
            self.notes_tokens.append(notes_tokens)
        
        # Timing
        if retrieval_time > 0:
            self.retrieval_times.append(retrieval_time)
        
        if overlap_count is not None:
            self.overlap_counts.append(overlap_count)
    
    def print_summary(self, title: str = "RETRIEVAL STATISTICS"):
        """Print comprehensive statistics summary."""
        
        if not self.sample_stats:
            log.info(f"\n{title}: No data")
            return
        
        log.info("\n" + "="*80)
        log.info(title)
        log.info("="*80)
        
        # Sample-level stats
        log.info(f"\nSample Statistics (n={len(self.sample_stats)}):")
        log.info(f"  Facts per sample:")
        log.info(f"    Mean:   {np.mean([s['n_facts'] for s in self.sample_stats]):.1f}")
        log.info(f"    Median: {np.median([s['n_facts'] for s in self.sample_stats]):.1f}")
        log.info(f"    Min:    {np.min([s['n_facts'] for s in self.sample_stats])}")
        log.info(f"    Max:    {np.max([s['n_facts'] for s in self.sample_stats])}")
        
        # Weight stats
        if self.all_relationship_weights:
            log.info(f"\nRelationship Weight Statistics:")
            log.info(f"  Total facts retrieved: {len(self.all_relationship_weights):,}")
            log.info(f"  Average weight: {np.mean(self.all_relationship_weights):.3f}")
            log.info(f"  Median weight:  {np.median(self.all_relationship_weights):.3f}")
            log.info(f"  Std dev:        {np.std(self.all_relationship_weights):.3f}")
            log.info(f"  Min weight:     {np.min(self.all_relationship_weights):.1f}")
            log.info(f"  Max weight:     {np.max(self.all_relationship_weights):.1f}")
            
            # Weight distribution
            weight_counter = Counter(self.all_relationship_weights)
            log.info(f"\n  Weight Distribution:")
            for weight in sorted(weight_counter.keys(), reverse=True):
                count = weight_counter[weight]
                pct = 100 * count / len(self.all_relationship_weights)
                log.info(f"    Weight {weight:.1f}: {count:7,} facts ({pct:5.1f}%)")
        
        # Relationship distribution
        if self.all_relationships:
            rel_counter = Counter(self.all_relationships)
            log.info(f"\n  Relationship Distribution (Top 15):")
            log.info(f"    Total unique relationships: {len(rel_counter)}")
            for rel, count in rel_counter.most_common(15):
                weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
                pct = 100 * count / len(self.all_relationships)
                log.info(f"    {rel:25s}: {count:7,} ({pct:5.1f}%) [weight={weight:.1f}]")
        
        # Token stats
        if self.prompt_tokens:
            log.info(f"\nToken Statistics:")
            log.info(f"  Prompt tokens:")
            log.info(f"    Mean:   {np.mean(self.prompt_tokens):,.0f}")
            log.info(f"    Median: {np.median(self.prompt_tokens):,.0f}")
            log.info(f"    Min:    {np.min(self.prompt_tokens):,.0f}")
            log.info(f"    Max:    {np.max(self.prompt_tokens):,.0f}")
            
            if self.kg_tokens:
                log.info(f"  KG facts tokens:")
                log.info(f"    Mean:   {np.mean(self.kg_tokens):,.0f}")
                log.info(f"    Median: {np.median(self.kg_tokens):,.0f}")
        
        # Overlap stats (for comparison mode)
        if self.overlap_counts:
            log.info(f"\nOverlap Statistics:")
            log.info(f"  Mean overlap:   {np.mean(self.overlap_counts):.1f} facts")
            log.info(f"  Median overlap: {np.median(self.overlap_counts):.1f} facts")
            log.info(f"  Min overlap:    {np.min(self.overlap_counts)} facts")
            log.info(f"  Max overlap:    {np.max(self.overlap_counts)} facts")
        
        # Timing
        if self.retrieval_times:
            total_time = sum(self.retrieval_times)
            log.info(f"\nRetrieval Timing:")
            log.info(f"  Total retrieval time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            log.info(f"  Avg time per sample:  {np.mean(self.retrieval_times):.3f} seconds")
            log.info(f"  Median time:          {np.median(self.retrieval_times):.3f} seconds")
        
        log.info("="*80)
    
    def save_to_json(self, output_file: Path):
        """Save statistics to JSON file."""
        stats_dict = {
            'n_samples': len(self.sample_stats),
            'total_facts': len(self.all_retrieved_facts),
            'sample_stats': self.sample_stats,
            'aggregate_stats': {
                'avg_weight': float(np.mean(self.all_relationship_weights)) if self.all_relationship_weights else 0.0,
                'median_weight': float(np.median(self.all_relationship_weights)) if self.all_relationship_weights else 0.0,
                'weight_distribution': dict(Counter(self.all_relationship_weights)),
                'relationship_distribution': dict(Counter(self.all_relationships)),
                'avg_prompt_tokens': float(np.mean(self.prompt_tokens)) if self.prompt_tokens else 0.0,
                'total_retrieval_time': float(sum(self.retrieval_times)) if self.retrieval_times else 0.0
            }
        }
        
        if self.overlap_counts:
            stats_dict['aggregate_stats']['avg_overlap'] = float(np.mean(self.overlap_counts))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        log.info(f"✓ Saved statistics to {output_file}")

# =============================================================================
# PROMPT BUILDING - UPDATED ORDER
# =============================================================================

def build_prompt_with_rag(
    visit: pd.Series,
    retrieved_facts: List[str],
    tokenizer,
    max_notes_tokens: int = 3000,
    max_facts_tokens: int = 1500,
    map_desc: bool = False,
    kg=None
) -> Tuple[str, int, int]:
    """
    Build training prompt with retrieved KG facts.
    Matches common_textgen.py format with KG facts AFTER clinical notes.
    
    Returns:
        Tuple[str, int, int]: (prompt, total_tokens, kg_tokens)
    """
    # 1. Visit header
    header_parts = []
    header_parts.append(f"[VISIT] subject_id={visit.get('subject_id_x','?')} hadm_id={visit.get('hadm_id','?')}")
    # 2. Structured data
    structured = serialize_structured_readable(visit, map_desc=map_desc, kg=kg)
    header_parts.append(structured)

    # 3. Clinical notes (trim if needed)
    notes_full = serialize_notes(visit)
    notes_tokens = token_len(tokenizer, notes_full)
    
    if notes_tokens > max_notes_tokens:
        # Trim by character length proportionally
        ratio = max_notes_tokens / notes_tokens
        target_chars = int(len(notes_full) * ratio * 0.9)  # 90% safety margin
        notes = notes_full[:target_chars]
    else:
        notes = notes_full
    
    # 4. KG Facts section (AFTER clinical notes, BEFORE task)
    if retrieved_facts:
        kg_facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(retrieved_facts)])
        
        # Trim facts if too long
        kg_facts_tokens = token_len(tokenizer, kg_facts_text)
        if kg_facts_tokens > max_facts_tokens:
            # Binary search to find how many facts fit
            left, right = 1, len(retrieved_facts)
            while left < right:
                mid = (left + right + 1) // 2
                test_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(retrieved_facts[:mid])])
                if token_len(tokenizer, test_text) <= max_facts_tokens:
                    left = mid
                else:
                    right = mid - 1
            
            retrieved_facts = retrieved_facts[:left]
            kg_facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(retrieved_facts)])
            kg_facts_tokens = token_len(tokenizer, kg_facts_text)
        
        kg_section = f"[KNOWLEDGE GRAPH FACTS]\n{kg_facts_text}"
    else:
        kg_section = "[KNOWLEDGE GRAPH FACTS]\n(No relevant knowledge graph facts found)"
        kg_facts_tokens = token_len(tokenizer, kg_section)
    
    # 5. Task instruction (matching common_textgen.py format)
    N_max_terms = 14  # Can be made configurable
    
    task_parts = [
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]"
    ]
    
    # 6. Combine all parts in order:
    # Header -> Structured -> Clinical Notes -> KG Facts -> Task
    all_parts = [*header_parts]
    if notes:
        all_parts.append(notes)
    all_parts.append(kg_section)
    all_parts.extend(task_parts)
    
    prompt = "\n".join([p for p in all_parts if p])
    
    total_tokens = token_len(tokenizer, prompt)
    
    return prompt, total_tokens, kg_facts_tokens


def build_prompt_baseline(
    visit: pd.Series,
    tokenizer,
    max_notes_tokens: int = 3000
) -> Tuple[str, int]:
    """
    Build baseline prompt WITHOUT KG facts.
    Matches common_textgen.py format exactly.
    
    Returns:
        Tuple[str, int]: (prompt, total_tokens)
    """
    # 1. Visit header
    header_parts = []
    header_parts.append(f"[VISIT] subject_id={visit.get('subject_id_x','?')} hadm_id={visit.get('hadm_id','?')}")
    
    # 2. Structured data
    structured = serialize_structured_readable(visit)
    header_parts.append(structured)
    
    # 3. Clinical notes (trim if needed)
    notes_full = serialize_notes(visit)
    notes_tokens = token_len(tokenizer, notes_full)
    
    if notes_tokens > max_notes_tokens:
        ratio = max_notes_tokens / notes_tokens
        target_chars = int(len(notes_full) * ratio * 0.9)
        notes = notes_full[:target_chars]
    else:
        notes = notes_full
    
    # 4. Task instruction (matching common_textgen.py format)
    N_max_terms = 15
    
    task_parts = [
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]"
    ]
    
    # 5. Combine all parts
    all_parts = [*header_parts]
    if notes:
        all_parts.append(notes)
    all_parts.extend(task_parts)
    
    prompt = "\n".join([p for p in all_parts if p])
    
    total_tokens = token_len(tokenizer, prompt)
    
    return prompt, total_tokens

# =============================================================================
# MAIN PREPROCESSING (FIXED COMPARISON MODE)
# =============================================================================

def preprocess_split_with_rag(
    df: pd.DataFrame,
    fact_index: MedicalFactIndex,
    encoder: SapBERTEncoder,
    tokenizer,
    output_file: Path,
    use_weighting: bool = False,
    alpha: float = 0.3,
    k: int = 20,
    use_kg: bool = True,
    subset_size: int = None,
    comparison_mode: bool = False,
    code2title: Dict[str, str] = {},
    map_desc: bool = False,
    kg=None
) -> RetrievalStatistics:
    """
    Preprocess a data split with enhanced statistics tracking.
    
    FIXED: Comparison mode now correctly compares weighted vs unweighted.
    
    Args:
        df: DataFrame with visits
        fact_index: MedicalFactIndex object (with relationship metadata)
        encoder: SapBERTEncoder object
        tokenizer: Tokenizer for token counting
        output_file: Where to save .jsonl
        use_weighting: Whether to use relationship weighting
        alpha: Weight interpolation (only if use_weighting=True)
        k: Number of facts to retrieve
        use_kg: If False, creates baseline without KG
        subset_size: If set, only process first N samples (for testing)
        comparison_mode: If True, retrieves both weighted and unweighted for comparison
    
    Returns:
        RetrievalStatistics object with all tracked stats
    """
    # Apply subset if requested
    if subset_size is not None and subset_size < len(df):
        log.info(f"\n{'!'*80}")
        log.info(f"SUBSET MODE: Processing only first {subset_size} samples (out of {len(df)})")
        log.info(f"{'!'*80}")
        df = df.iloc[:subset_size].copy()
    
    log.info(f"\nProcessing {len(df)} visits...")
    log.info(f"  Use KG: {use_kg}")
    
    if use_kg:
        log.info(f"  Use weighting: {use_weighting}")
        log.info(f"  Alpha: {alpha}")
        log.info(f"  K facts: {k}")
        log.info(f"  Comparison mode: {comparison_mode}")
        
        if fact_index.relationships:
            log.info(f"  ✓ Using GROUND TRUTH relationship metadata")
        else:
            log.info(f"  ⚠ No relationship metadata, will extract from text")
    
    log.info(f"  Output: {output_file}")
    
    # Statistics trackers
    stats = RetrievalStatistics()
    stats_comparison = RetrievalStatistics() if comparison_mode else None
    
    examples = []
    
    for idx, (_, visit) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing")):
        try:
            if use_kg:
                # Retrieve facts
                query_text = serialize_notes(visit)
                
                # Time retrieval
                retrieval_start = time.time()
                
                retrieved_facts = retrieve_facts(
                    query_text,
                    fact_index,
                    encoder,
                    k=k,
                    use_weighting=use_weighting,
                    alpha=alpha,
                    debug=False
                )
                
                retrieval_time = time.time() - retrieval_start
                
                # Get relationships and weights
                relationships = []
                weights = []
                for fact in retrieved_facts:
                    try:
                        fact_idx = fact_index.facts.index(fact)
                        rel = fact_index.get_relationship(fact_idx)
                        weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
                        relationships.append(rel)
                        weights.append(weight)
                    except ValueError:
                        relationships.append('other')
                        weights.append(1.0)
                
                # Build prompt
                prompt, total_tokens, kg_tokens = build_prompt_with_rag(
                    visit,
                    retrieved_facts,
                    tokenizer,
                    max_notes_tokens=3000,
                    max_facts_tokens=1500,
                    map_desc=False,
                    kg=None
                )
                
                # ✅ FIXED: Comparison mode - retrieve with OPPOSITE weighting
                overlap_count = None
                if comparison_mode:
                    # Retrieve with opposite weighting setting
                    retrieved_comparison = retrieve_facts(
                        query_text,
                        fact_index,
                        encoder,
                        k=k,
                        use_weighting=(not use_weighting),  # ✅ FIX: opposite of main mode
                        alpha=alpha,  # Use same alpha for consistency
                        debug=False
                    )
                    
                    # Calculate overlap
                    overlap_count = len(set(retrieved_facts) & set(retrieved_comparison))
                    
                    # Track comparison stats
                    rel_comparison = []
                    wt_comparison = []
                    for fact in retrieved_comparison:
                        try:
                            fact_idx = fact_index.facts.index(fact)
                            rel = fact_index.get_relationship(fact_idx)
                            weight = RELATIONSHIP_WEIGHTS.get(rel, 1.0)
                            rel_comparison.append(rel)
                            wt_comparison.append(weight)
                        except ValueError:
                            rel_comparison.append('other')
                            wt_comparison.append(1.0)
                    
                    stats_comparison.add_sample(
                        retrieved_comparison,
                        rel_comparison,
                        wt_comparison,
                        total_tokens,
                        kg_tokens,
                        0,
                        0.0,
                        None
                    )
                
                # Track statistics
                stats.add_sample(
                    retrieved_facts,
                    relationships,
                    weights,
                    total_tokens,
                    kg_tokens,
                    token_len(tokenizer, serialize_notes(visit)),
                    retrieval_time,
                    overlap_count
                )
                
            else:
                # Baseline: no KG
                retrieved_facts = []
                relationships = []
                weights = []
                prompt, total_tokens = build_prompt_baseline(visit, tokenizer, max_notes_tokens=3000)
                
                stats.add_sample(
                    retrieved_facts,
                    relationships,
                    weights,
                    total_tokens,
                    0,
                    token_len(tokenizer, serialize_notes(visit)),
                    0.0,
                    None
                )
            
            # Get target diagnoses
            target_codes = to_list(visit.get('icd_code', []))
            target_codes = [format_icd9(code) for code in target_codes]
            target_descs = []
            for i in target_codes:
                n = code2title.get(i)
                target_descs.append(n if n else "Unknown Diagnosis")
            
            # Format target
            if target_descs:
                target = '; '.join(target_descs)
            elif target_codes:
                target = '; '.join(target_codes)
            else:
                target = ""
            
            if not target:
                continue
            
            example = {
                'hadm_id': int(visit.get('hadm_id', idx)),
                'prompt': prompt,
                'target': target,
                'target_codes': target_codes,
                'target_descriptions': target_descs,
                'retrieved_facts': retrieved_facts if use_kg else [],
                'use_weighting': use_weighting if use_kg else False,
                'alpha': alpha if use_kg else 0.0
            }
            
            # Add per-sample stats to example
            if use_kg and weights:
                example['sample_stats'] = {
                    'n_facts': len(retrieved_facts),
                    'avg_weight': float(np.mean(weights)),
                    'relationships': relationships
                }
            
            examples.append(example)
            
        except Exception as e:
            log.error(f"Error processing visit {idx}: {e}")
            import traceback
            log.error(traceback.format_exc())
            continue
    
    # Save examples to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    log.info(f"✓ Saved {len(examples)} examples to {output_file}")
    
    # Print statistics
    if use_kg:
        mode_str = "WEIGHTED" if use_weighting else "UNWEIGHTED"
        stats.print_summary(f"{mode_str} RETRIEVAL STATISTICS")
        
        # Save stats to JSON
        stats_file = output_file.parent / f"{output_file.stem}_stats.json"
        stats.save_to_json(stats_file)
        
        # ✅ FIXED: Comparison mode with correct labels
        if comparison_mode and stats_comparison:
            comparison_mode_label = "UNWEIGHTED" if use_weighting else "WEIGHTED"
            stats_comparison.print_summary(f"{comparison_mode_label} RETRIEVAL STATISTICS (for comparison)")
            
            # Comparison summary
            log.info("\n" + "="*80)
            if use_weighting:
                log.info("WEIGHTED vs UNWEIGHTED COMPARISON")
            else:
                log.info("UNWEIGHTED vs WEIGHTED COMPARISON")
            log.info("="*80)
            
            avg_wt_main = np.mean(stats.all_relationship_weights) if stats.all_relationship_weights else 0
            avg_wt_comparison = np.mean(stats_comparison.all_relationship_weights) if stats_comparison.all_relationship_weights else 0
            
            log.info(f"\nAverage Relationship Weights:")
            main_label = "Weighted" if use_weighting else "Unweighted"
            comparison_label = "Unweighted" if use_weighting else "Weighted"
            log.info(f"  {main_label} retrieval:   {avg_wt_main:.3f}")
            log.info(f"  {comparison_label} retrieval: {avg_wt_comparison:.3f}")
            
            # Calculate difference
            if use_weighting:
                # Weighted is main, unweighted is comparison
                if avg_wt_comparison > 0:
                    boost = 100 * (avg_wt_main - avg_wt_comparison) / avg_wt_comparison
                    log.info(f"  Weight boost:         {boost:+.1f}%")
                    
                    if boost > 5:
                        log.info(f"\n  ✓ Weighted retrieval successfully boosted high-value relationships!")
                    elif abs(boost) < 2:
                        log.info(f"\n  ⚠ Minimal difference detected")
                    else:
                        log.info(f"\n  ⚠ Unexpected result")
            else:
                # Unweighted is main, weighted is comparison
                if avg_wt_main > 0:
                    boost = 100 * (avg_wt_comparison - avg_wt_main) / avg_wt_main
                    log.info(f"  Weighted would boost: {boost:+.1f}%")
            
            if stats.overlap_counts:
                avg_overlap = np.mean(stats.overlap_counts)
                overlap_pct = 100 * avg_overlap / k
                log.info(f"\nOverlap Statistics:")
                log.info(f"  Average overlap: {avg_overlap:.1f} / {k} facts ({overlap_pct:.1f}%)")
                log.info(f"  This means {100-overlap_pct:.1f}% of facts changed due to weighting")
            
            log.info("="*80)
            
            # Save comparison stats
            comparison_file = output_file.parent / f"{output_file.stem}_comparison.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                comparison_data = {
                    'main_mode': main_label.lower(),
                    'comparison_mode': comparison_label.lower(),
                    'main': {
                        'avg_weight': float(avg_wt_main),
                        'relationship_distribution': dict(Counter(stats.all_relationships))
                    },
                    'comparison': {
                        'avg_weight': float(avg_wt_comparison),
                        'relationship_distribution': dict(Counter(stats_comparison.all_relationships))
                    }
                }
                
                if use_weighting and avg_wt_comparison > 0:
                    comparison_data['weight_boost_percent'] = float(boost)
                elif not use_weighting and avg_wt_main > 0:
                    comparison_data['weighted_would_boost_percent'] = float(boost)
                
                if stats.overlap_counts:
                    comparison_data['avg_overlap'] = float(avg_overlap)
                    comparison_data['overlap_percent'] = float(overlap_pct)
                
                json.dump(comparison_data, f, indent=2)
            
            log.info(f"✓ Saved comparison statistics to {comparison_file}")
    
    return stats

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EHR data with RAG-based KG retrieval (Enhanced with Statistics)"
    )
    
    # Input data
    parser.add_argument("--train_data", required=True, help="Train .pkl file")
    parser.add_argument("--val_data", required=True, help="Val .pkl file")
    parser.add_argument("--test_data", help="Test .pkl file (optional)")
    
    # Fact index
    parser.add_argument("--fact_index_dir", required=True,
                       help="Directory with pre-built fact index (must have relationships.json)")
    parser.add_argument("--icd_index_dir", required=True,
                       help="Path to code2title.json directory")

    # Output
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save preprocessed .jsonl files")
    
    # Retrieval parameters
    parser.add_argument("--use_kg", action="store_true",
                       help="Use KG facts (if False, creates baseline)")
    parser.add_argument("--use_weighting", action="store_true",
                       help="Use relationship weighting")
    parser.add_argument("--alpha", type=float, default=0.3,
                       help="Weight interpolation (0=semantic, 1=weight-based)")
    parser.add_argument("--k", type=int, default=20,
                       help="Number of facts to retrieve per visit")
    
    # Testing & analysis
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Process only first N samples (for testing)")
    parser.add_argument("--comparison_mode", action="store_true",
                       help="Retrieve both weighted and unweighted for comparison")
    
    # Model
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name for tokenizer")
    parser.add_argument("--sapbert_model", 
                       default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                       help="SapBERT model for encoding")

    # KG
    parser.add_argument("--kg", default=None,
                       help="Path to knowledge graph (if using KG)")

    # Map codes to descriptions
    parser.add_argument("--map_desc", action="store_true",
                       help="Map ICD codes to descriptions in structured data")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    
    start_time = time.time()
    
    # ==========================================================================
    # LOAD MODELS
    # ==========================================================================
    log.info("="*80)
    log.info("LOADING MODELS")
    log.info("="*80)
    
    # Tokenizer
    log.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("  ✓ Tokenizer loaded")
    
    # SapBERT encoder (only if using KG)
    if args.use_kg:
        encoder = SapBERTEncoder(args.sapbert_model)
        
        # Fact index (WITH RELATIONSHIP METADATA)
        log.info(f"\nLoading fact index from {args.fact_index_dir}")
        fact_index = MedicalFactIndex(args.fact_index_dir)
        
        # Check for relationship metadata
        if not fact_index.relationships:
            log.warning("\n" + "!"*80)
            log.warning("⚠ WARNING: No relationship metadata found in index!")
            log.warning("  Weighted retrieval will use extracted relationships (less accurate)")
            log.warning("  Consider rebuilding index with updated build script")
            log.warning("!"*80 + "\n")
    else:
        encoder = None
        fact_index = None
        log.info("  ⊗ Skipping KG models (baseline mode)")
    
    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    log.info("\n" + "="*80)
    log.info("LOADING DATA")
    log.info("="*80)
    
    log.info(f"Loading train data from {args.train_data}")
    train_df = pd.read_pickle(args.train_data)
    log.info(f"  ✓ Train: {len(train_df)} visits")
    
    log.info(f"Loading val data from {args.val_data}")
    val_df = pd.read_pickle(args.val_data)
    log.info(f"  ✓ Val: {len(val_df)} visits")
    
    if args.test_data:
        log.info(f"Loading test data from {args.test_data}")
        test_df = pd.read_pickle(args.test_data)
        log.info(f"  ✓ Test: {len(test_df)} visits")
    else:
        test_df = None
        log.info("  ⊗ No test data provided")

    if args.icd_index_dir:
        icd_index_file = Path(args.icd_index_dir) / "code2title.json"
        log.info(f"\nLoading ICD code2title from {icd_index_file}")
        with open(icd_index_file, 'r', encoding='utf-8') as f:
            code2title = json.load(f)
        log.info(f"  ✓ Loaded {len(code2title)} ICD-9 code mappings")
    else:
        code2title = {}
        log.info("  ⊗ No ICD code2title mapping provided")

    if args.map_desc:
        log.info("\nMapping structured data codes to descriptions")
        map_desc = True
    else:
        log.info("\nSkipping mapping of structured data codes to descriptions")
        map_desc = False

    if args.kg:
        log.info(f"\nLoading knowledge graph from {args.kg}")
        kg = load_knowledge_graph(args.kg)
        log.info(f"  ✓ Loaded knowledge graph with {len(kg)} entities")
        log.info(f"Graph info:")
        log.info(f"- Nodes: {kg.number_of_nodes()}")
        log.info(f"- Edges: {kg.number_of_edges()}")

    # ==========================================================================
    # PREPROCESS SPLITS
    # ==========================================================================
    log.info("\n" + "="*80)
    log.info("PREPROCESSING DATA")
    log.info("="*80)
    
    # Determine output suffix based on settings
    if not args.use_kg:
        suffix = "baseline"
    elif not args.use_weighting:
        suffix = "rag_unweighted"
    else:
        suffix = f"rag_weighted_alpha{args.alpha:.1f}"
    
    if args.subset_size:
        suffix += f"_subset{args.subset_size}"
    
    log.info(f"\nOutput suffix: {suffix}")
    
    # Train
    log.info("\n" + "─"*80)
    log.info("PROCESSING TRAIN SET")
    log.info("─"*80)
    train_output = output_dir / f"train_{suffix}.jsonl"
    train_stats = preprocess_split_with_rag(
        train_df, fact_index, encoder, tokenizer, train_output,
        use_weighting=args.use_weighting,
        alpha=args.alpha,
        k=args.k,
        use_kg=args.use_kg,
        subset_size=args.subset_size,
        comparison_mode=args.comparison_mode,
        code2title=code2title,
        map_desc=map_desc,
        kg=kg
    )
    
    # Val
    log.info("\n" + "─"*80)
    log.info("PROCESSING VAL SET")
    log.info("─"*80)
    val_output = output_dir / f"val_{suffix}.jsonl"
    val_stats = preprocess_split_with_rag(
        val_df, fact_index, encoder, tokenizer, val_output,
        use_weighting=args.use_weighting,
        alpha=args.alpha,
        k=args.k,
        use_kg=args.use_kg,
        subset_size=args.subset_size,
        comparison_mode=args.comparison_mode,
        code2title=code2title,
        map_desc=map_desc,
        kg=kg
    )
    
    # Test (if provided)
    if test_df is not None:
        log.info("\n" + "─"*80)
        log.info("PROCESSING TEST SET")
        log.info("─"*80)
        test_output = output_dir / f"test_{suffix}.jsonl"
        test_stats = preprocess_split_with_rag(
            test_df, fact_index, encoder, tokenizer, test_output,
            use_weighting=args.use_weighting,
            alpha=args.alpha,
            k=args.k,
            use_kg=args.use_kg,
            subset_size=args.subset_size,
            comparison_mode=args.comparison_mode,
            code2title=code2title,
            map_desc=map_desc,
            kg=kg
        )
    
    # ==========================================================================
    # OVERALL SUMMARY
    # ==========================================================================
    total_time = time.time() - start_time
    
    log.info("\n" + "="*80)
    log.info("✓ PREPROCESSING COMPLETE")
    log.info("="*80)
    log.info(f"Total time: {total_time/60:.1f} minutes")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Suffix: {suffix}")
    
    # Print cross-split summary
    if args.use_kg:
        log.info(f"\nCross-Split Summary:")
        log.info(f"  Train samples: {len(train_stats.sample_stats)}")
        log.info(f"  Val samples:   {len(val_stats.sample_stats)}")
        if test_df is not None:
            log.info(f"  Test samples:  {len(test_stats.sample_stats)}")
        
        if train_stats.all_relationship_weights:
            log.info(f"\n  Average Relationship Weights:")
            log.info(f"    Train: {np.mean(train_stats.all_relationship_weights):.3f}")
            log.info(f"    Val:   {np.mean(val_stats.all_relationship_weights):.3f}")
            if test_df is not None and test_stats.all_relationship_weights:
                log.info(f"    Test:  {np.mean(test_stats.all_relationship_weights):.3f}")
    
    log.info(f"\nGenerated files:")
    for file in sorted(output_dir.glob(f"*{suffix}*")):
        log.info(f"  - {file.name}")
    
    log.info("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())