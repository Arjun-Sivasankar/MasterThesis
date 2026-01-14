#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_baseline_only.py
Creates train_baseline.jsonl, val_baseline.jsonl, and test_baseline.jsonl.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Import utilities
sys.path.insert(0, '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/pipeline/')

from common_textgen_util import (
    to_list,
    format_icd9,
    serialize_structured_readable,
    serialize_notes,
    token_len
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
# PROMPT BUILDING
# =============================================================================

def build_baseline_prompt(
    visit: pd.Series,
    tokenizer,
    max_notes_tokens: int = 3000,
    map_desc: bool = False,
    kg=None
) -> Tuple[str, int]:
    """
    Build baseline prompt WITHOUT KG facts.
    
    Returns:
        Tuple[str, int]: (prompt, total_tokens)
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
        ratio = max_notes_tokens / notes_tokens
        target_chars = int(len(notes_full) * ratio * 0.9)
        notes = notes_full[:target_chars]
    else:
        notes = notes_full
    
    # 4. Task instruction
    N_max_terms = 12
    
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
# PREPROCESSING
# =============================================================================

def preprocess_baseline_split(
    df: pd.DataFrame,
    tokenizer,
    output_file: Path,
    code2title: Dict[str, str] = {},
    map_desc: bool = False,
    kg=None,
    subset_size: int = None
):
    """
    Preprocess a data split for baseline (no KG).
    
    Args:
        df: DataFrame with visits
        tokenizer: Tokenizer for token counting
        output_file: Where to save .jsonl
        code2title: ICD-9 code to description mapping
        map_desc: Whether to map descriptions in structured data
        kg: Knowledge graph (if map_desc=True)
        subset_size: If set, only process first N samples
    """
    # Apply subset if requested
    if subset_size is not None and subset_size < len(df):
        log.info(f"Using subset: {subset_size} samples")
        df = df.iloc[:subset_size].copy()
    
    log.info(f"Processing {len(df)} visits...")
    
    examples = []
    skipped = 0
    
    for idx, (_, visit) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing")):
        try:
            # Build prompt
            prompt, total_tokens = build_baseline_prompt(
                visit,
                tokenizer,
                max_notes_tokens=3000,
                map_desc=map_desc,
                kg=kg
            )
            
            # Get target diagnoses
            target_codes = to_list(visit.get('icd_code', []))
            target_codes = [format_icd9(code) for code in target_codes]
            target_descs = []
            for code in target_codes:
                desc = code2title.get(code)
                target_descs.append(desc if desc else "Unknown Diagnosis")
            
            # Format target
            if target_descs:
                target = '; '.join(target_descs)
            elif target_codes:
                target = '; '.join(target_codes)
            else:
                target = ""
            
            if not target:
                skipped += 1
                continue
            
            example = {
                'hadm_id': int(visit.get('hadm_id', idx)),
                'prompt': prompt,
                'target': target,
                'target_codes': target_codes,
                'target_descriptions': target_descs,
                'prompt_tokens': total_tokens
            }
            
            examples.append(example)
            
        except Exception as e:
            log.error(f"Error processing visit {idx}: {e}")
            import traceback
            log.error(traceback.format_exc())
            skipped += 1
            continue
    
    # Save to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    log.info(f" Saved {len(examples)} examples to {output_file}")
    if skipped > 0:
        log.warning(f"  Skipped {skipped} visits (no target diagnoses)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline training data (no KG retrieval)"
    )
    
    # Input data
    parser.add_argument("--train_data", required=True, help="Train .pkl file")
    parser.add_argument("--val_data", required=True, help="Val .pkl file")
    parser.add_argument("--test_data", help="Test .pkl file (optional)")
    
    # Output
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for baseline JSONL files")
    
    # Optional mappings
    parser.add_argument("--icd_index_dir", help="ICD-9 code2title mapping directory")
    parser.add_argument("--kg", help="Knowledge graph .pkl file (for map_desc)")
    parser.add_argument("--map_desc", action="store_true",
                       help="Map descriptions in structured data")
    
    # Model
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name for tokenizer")
    
    # Testing
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Process only first N samples (for testing)")
    
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("BASELINE PREPROCESSING (No KG)")
    log.info("="*80)
    log.info(f"Model: {args.model_name}")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Map descriptions: {args.map_desc}")
    if args.subset_size:
        log.info(f"Subset size: {args.subset_size}")
    log.info("="*80)
    
    # Load tokenizer
    log.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("  Tokenizer loaded")
    
    # Load ICD mapping
    code2title = {}
    if args.icd_index_dir:
        icd_index_file = Path(args.icd_index_dir) / "code2title.json"
        log.info(f"\nLoading ICD code2title from {icd_index_file}")
        with open(icd_index_file, 'r', encoding='utf-8') as f:
            code2title = json.load(f)
        log.info(f"   Loaded {len(code2title)} ICD-9 code mappings")
    
    # Load KG if needed
    kg = None
    if args.map_desc and args.kg:
        log.info(f"\nLoading knowledge graph from {args.kg}")
        import pickle
        with open(args.kg, 'rb') as f:
            kg = pickle.load(f)
        log.info("   Knowledge graph loaded")
    
    # Load data
    log.info("\nLoading data...")
    train_df = pd.read_pickle(args.train_data)
    val_df = pd.read_pickle(args.val_data)
    test_df = pd.read_pickle(args.test_data) if args.test_data else None
    
    log.info(f"  Train: {len(train_df):,}")
    log.info(f"  Val:   {len(val_df):,}")
    if test_df is not None:
        log.info(f"  Test:  {len(test_df):,}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process splits
    log.info("\n" + "="*80)
    log.info("PROCESSING TRAIN SET")
    log.info("="*80)
    preprocess_baseline_split(
        train_df, tokenizer, output_dir / "train_baseline.jsonl",
        code2title=code2title, map_desc=args.map_desc, kg=kg,
        subset_size=args.subset_size
    )
    
    log.info("\n" + "="*80)
    log.info("PROCESSING VAL SET")
    log.info("="*80)
    preprocess_baseline_split(
        val_df, tokenizer, output_dir / "val_baseline.jsonl",
        code2title=code2title, map_desc=args.map_desc, kg=kg,
        subset_size=args.subset_size
    )
    
    if test_df is not None:
        log.info("\n" + "="*80)
        log.info("PROCESSING TEST SET")
        log.info("="*80)
        preprocess_baseline_split(
            test_df, tokenizer, output_dir / "test_baseline.jsonl",
            code2title=code2title, map_desc=args.map_desc, kg=kg,
            subset_size=args.subset_size
        )
    
    log.info("\n" + "="*80)
    log.info(" ALL PROCESSING COMPLETE")
    log.info("="*80)
    log.info(f"\nOutput files:")
    log.info(f"  {output_dir / 'train_baseline.jsonl'}")
    log.info(f"  {output_dir / 'val_baseline.jsonl'}")
    if test_df is not None:
        log.info(f"  {output_dir / 'test_baseline.jsonl'}")

if __name__ == "__main__":
    main()