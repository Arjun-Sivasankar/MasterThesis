#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_data_with_kg.py
OFFLINE PREPROCESSING SCRIPT - OPTIMIZED VERSION

Combines V1's prompt structure with V2's speed optimizations:
- Manual chat template formatting (faster than apply_chat_template)
- Simplified budgeting (fewer tokenization calls)
- Same prompt structure as original
"""

import argparse
import sys
import os
import re
import time
import json
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx

from typing import List, Dict, Set, Tuple
from transformers import AutoTokenizer

# ---- Import helpers from common_textgen.py ----
from common_textgen import (
    log,
    to_list, format_icd9, is_valid_icd9,
    serialize_notes,
    token_len
)

# =================================================================================
# ---- OPTIMIZED HELPER FUNCTIONS ----
# =================================================================================

def trim_to_token_budget_fast(tok, text: str, max_tokens: int) -> str:
    """
    OPTIMIZED: Faster trimming with fewer tokenization calls.
    Uses larger steps initially, then refines.
    """
    if max_tokens <= 0 or not text:
        return ""
    
    current_len = token_len(tok, text)
    if current_len <= max_tokens:
        return text
    
    # Estimate characters per token (rough heuristic: ~4 chars/token)
    estimated_chars = int(len(text) * (max_tokens / current_len) * 0.95)  # 95% safety margin
    
    # Quick first cut
    candidate = text[:estimated_chars]
    cand_len = token_len(tok, candidate)
    
    # If we're close enough, use it
    if abs(cand_len - max_tokens) <= 5:  # Within 5 tokens is fine
        return candidate
    
    # Otherwise, binary search (but with fewer iterations needed)
    if cand_len > max_tokens:
        lo, hi = 0, estimated_chars
    else:
        lo, hi = estimated_chars, len(text)
    
    best = candidate if cand_len <= max_tokens else ""
    iterations = 0
    max_iterations = 8  # Limit binary search iterations
    
    while lo <= hi and iterations < max_iterations:
        mid = (lo + hi) // 2
        cand = text[:mid]
        cand_len = token_len(tok, cand)
        
        if cand_len <= max_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
        
        iterations += 1
    
    return best

def _strip(x) -> str:
    """Strip and uppercase a code."""
    return str(x or "").strip().upper().replace(" ", "")

def format_icd9_proc_from_pro(c: str) -> str:
    """Format procedure code from pro_code column."""
    s = _strip(c)
    if s.startswith("PRO_"):
        s = s[4:]
    s = re.sub(r"[^0-9]", "", s)
    if not s:
        return ""
    if len(s) >= 3:
        return s[:2] + "." + s[2:]
    return s

def visit_evidence_cuis(
    row: pd.Series,
    icd9_proc_map: Dict[str, List[str]],
    loinc_map: Dict[str, List[str]],
    atc_map: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Extract CUIs from structured codes in the visit."""
    src2cuis = {}
    ev = set()
    
    icd9_proc_map = icd9_proc_map or {}
    loinc_map = loinc_map or {}
    atc_map = atc_map or {}
    
    # ATC codes from medications
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        cuis = atc_map.get(key, [])
        if cuis:
            src2cuis[f"ATC:{key}"] = cuis
            ev.update(cuis)
    
    # LOINC codes from lab tests
    for c in to_list(row.get("lab_test_loinc", [])):
        key = _strip(c)
        cuis = loinc_map.get(key, [])
        if cuis:
            src2cuis[f"LNC:{key}"] = cuis
            ev.update(cuis)
    
    # ICD-9 procedure codes
    for c in to_list(row.get("pro_code", [])):
        fmt = format_icd9_proc_from_pro(c)
        if not fmt:
            continue
        cuis = icd9_proc_map.get(fmt, [])
        if cuis:
            src2cuis[f"PROC:{fmt}"] = cuis
            ev.update(cuis)
    
    return src2cuis, ev

def _arrow_label(rela: str, rel: str) -> str:
    """Format relationship label for display."""
    r = (rela or "").strip() or (rel or "").strip()
    return f" --{r}--> " if r else " → "

def mine_hops_simple(
    G: nx.DiGraph,
    ev_cuis: Set[str],
    k1: int = 30,
    k2: int = 30
) -> Tuple[List[dict], List[dict]]:
    """Mine H1 and H2 paths from the knowledge graph."""
    H1, H2 = [], []
    
    if G is None:
        return H1, H2
    
    def _edge_attrs(d):
        rela_canon = d.get("rela_canon") or d.get("rela") or d.get("rel") or ""
        score = d.get("rela_score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        return rela_canon, score
    
    # H1: Direct edges between evidence nodes
    for u in ev_cuis:
        if u not in G:
            continue
        first = []
        for v in G.successors(u):
            d = G[u][v]
            rela_canon, score = _edge_attrs(d)
            first.append((v, rela_canon, -1e9 if score is None else score, G.nodes[v].get("name", "")))
        first.sort(key=lambda t: (t[2], t[0]), reverse=True)
        if k1 and k1 > 0:
            first = first[:k1]
        
        for v, rela_canon, score, vname in first:
            d = G[u][v]
            H1.append({
                "src_name": G.nodes[u].get("name", ""),
                "nbr_name": vname,
                "rela_canon": rela_canon,
                "rela": d.get("rela") or "",
                "rel": d.get("rel") or ""
            })
    
    # H2: Two-hop paths (evidence -> bridge -> evidence)
    for u in ev_cuis:
        if u not in G:
            continue
        first = []
        for v in G.successors(u):
            d = G[u][v]
            rela_canon, score = _edge_attrs(d)
            first.append((v, rela_canon, -1e9 if score is None else score, G.nodes[v].get("name", "")))
        first.sort(key=lambda t: (t[2], t[0]), reverse=True)
        if k1 and k1 > 0:
            first = first[:k1]
        
        for v, rela_uv_canon, score_uv, vname in first:
            if v not in G:
                continue
            second = []
            for w in G.successors(v):
                d_uv, d_vw = G[u][v], G[v][w]
                rela_uv_c, score_uv2 = _edge_attrs(d_uv)
                rela_vw_c, score_vw = _edge_attrs(d_vw)
                s_uv = score_uv if score_uv != -1e9 else (-1e9 if score_uv2 is None else score_uv2)
                s_vw = -1e9 if score_vw is None else score_vw
                total = (s_uv if s_uv != -1e9 else 0.0) + (s_vw if s_vw != -1e9 else 0.0)
                second.append((total, v, w, rela_uv_c, (None if s_uv == -1e9 else s_uv),
                             rela_vw_c, (None if s_vw == -1e9 else s_vw), vname, G.nodes[w].get("name", "")))
            second.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            if k2 and k2 > 0:
                second = second[:k2]
            
            for total, v, w, rela_uv_c, s_uv, rela_vw_c, s_vw, vname, wname in second:
                H2.append({
                    "u_name": G.nodes[u].get("name", ""),
                    "v_name": vname,
                    "w_name": wname,
                    "rela_uv_canon": rela_uv_c,
                    "rela_vw_canon": rela_vw_c,
                    "rela_uv": G[u][v].get("rela") or "",
                    "rela_vw": G[v][w].get("rela") or "",
                    "rel_uv": G[u][v].get("rel") or "",
                    "rel_vw": G[v][w].get("rel") or ""
                })
    
    return H1, H2

def render_h2_block(H2_rows: List[dict]) -> str:
    """Render H2 paths as text."""
    lines = ["[KG context - H2 paths]"]
    if not H2_rows:
        return lines[0] + "\n- (none)"
    for c in H2_rows:
        u = c.get("u_name") or ""
        v = c.get("v_name") or ""
        w = c.get("w_name") or ""
        r_uv = c.get("rela_uv_canon") or c.get("rela_uv") or c.get("rel_uv") or ""
        r_vw = c.get("rela_vw_canon") or c.get("rela_vw") or c.get("rel_vw") or ""
        lines.append(f"- {u}{_arrow_label(r_uv, '')}{v}{_arrow_label(r_vw, '')}{w}")
    return "\n".join(lines)

def render_h1_block(H1_rows: List[dict]) -> str:
    """Render H1 paths as text."""
    lines = ["[KG context - H1 paths]"]
    if not H1_rows:
        return lines[0] + "\n- (none)"
    for c in H1_rows:
        u = c.get("src_name") or ""
        v = c.get("nbr_name") or ""
        r = c.get("rela_canon") or c.get("rela") or c.get("rel") or ""
        lines.append(f"- {u}{_arrow_label(r, '')}{v}")
    return "\n".join(lines)

def combine_kg_blocks_with_budget(
    tok,
    h2_text: str,
    h1_text: str,
    budget: int,
    h2_ratio: float = 1.0,
    mode: str = "both"
):
    """Combine H1 and H2 blocks within a token budget."""
    mode = (mode or "both").lower()
    
    if budget is None or budget <= 0:
        if mode == "h1":
            return h1_text
        if mode == "h2":
            return h2_text
        return h2_text + ("\n" + h1_text if h1_text else "")
    
    if mode == "h1":
        return trim_to_token_budget_fast(tok, h1_text, budget)
    if mode == "h2":
        return trim_to_token_budget_fast(tok, h2_text, budget)
    
    # Both: Split budget between H2 and H1
    h2_quota = int(max(0, min(1.0, h2_ratio)) * budget)
    h1_quota = max(0, budget - h2_quota)
    
    h2_trim = trim_to_token_budget_fast(tok, h2_text, h2_quota) if h2_quota > 0 else ""
    h1_trim = trim_to_token_budget_fast(tok, h1_text, h1_quota) if h1_quota > 0 else ""
    
    used_h2 = token_len(tok, h2_trim)
    leftover = max(0, budget - used_h2 - token_len(tok, h1_trim))
    
    # Give leftover to H1
    if leftover > 0 and h1_text:
        h1_trim = trim_to_token_budget_fast(tok, h1_text, token_len(tok, h1_trim) + leftover)
    
    return (h2_trim if h2_trim else "") + (("\n" + h1_trim) if h1_trim else "")

def build_tail(N_max_terms: int) -> str:
    """Build the task instruction tail."""
    lines = [
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]",
    ]
    return "\n".join(lines)

def build_textgen_prompt_budgeted_with_kg_fast(
    header_text: str,
    row: pd.Series,
    tok,
    max_len: int,
    min_assist_tokens: int,
    N_max_terms: int,
    kg_text: str,
    notes_soft_budget: int,
    target_text: str
) -> str:
    """
    OPTIMIZED: Build the final user prompt with simplified budgeting.
    Uses direct token counting instead of chat_token_len for speed.
    """
    
    full_notes = serialize_notes(row)
    notes = trim_to_token_budget_fast(tok, full_notes, notes_soft_budget) if notes_soft_budget > 0 else full_notes
    tail = build_tail(N_max_terms)
    
    def assemble(notes_text: str, kg_text_block: str) -> str:
        parts = [header_text]
        if notes_text:
            parts.append(notes_text)
        if kg_text_block:
            parts.append(kg_text_block)
        parts.append(tail)
        return "\n".join(parts)
    
    # OPTIMIZATION: Estimate token overhead for chat template
    # Llama-3 adds roughly: system (~25 tokens) + user wrapper (~15 tokens) + assistant wrapper (~10 tokens)
    chat_template_overhead = 50  # Conservative estimate
    
    # Calculate actual assistant tokens needed
    assistant_tok_len = token_len(tok, target_text)
    assistant_tok_len = max(min_assist_tokens, assistant_tok_len) + chat_template_overhead
    
    max_prompt_len = max_len - assistant_tok_len
    
    # Base prompt (header + tail, no notes/KG)
    base_prompt = assemble("", "")
    base_prompt_len = token_len(tok, base_prompt)
    
    available_budget = max_prompt_len - base_prompt_len
    
    notes_len = token_len(tok, notes)
    kg_len = token_len(tok, kg_text)
    
    final_notes = notes
    final_kg = kg_text
    
    # If over budget, trim proportionally
    if notes_len + kg_len > available_budget:
        # Give priority to notes vs KG based on original budgets
        notes_budget = max(0, available_budget - kg_len)
        final_notes = trim_to_token_budget_fast(tok, notes, notes_budget)
        notes_len = token_len(tok, final_notes)
        
        if notes_len + kg_len > available_budget:
            kg_budget = max(0, available_budget - notes_len)
            final_kg = trim_to_token_budget_fast(tok, kg_text, kg_budget)
    
    final_prompt = assemble(final_notes, final_kg)
    
    return final_prompt

def _create_code_map(code_to_cui_map, cui_to_name_map):
    """Create a mapping from code to name via CUIs."""
    code_to_name = {}
    if code_to_cui_map is None:
        return code_to_name
    
    for code, cuis in code_to_cui_map.items():
        if not cuis:
            continue
        # Use first CUI's name
        cui = cuis[0]
        name = cui_to_name_map.get(cui)
        if name:
            code_to_name[code] = name
    
    return code_to_name

def get_target_text(row, label_col, target_mode, code2title):
    """Get the target text (ground truth diagnoses)."""
    if target_mode == "icd_titles":
        codes = [format_icd9(c) for c in to_list(row.get(label_col, [])) if c]
        codes = [c for c in codes if is_valid_icd9(c)]
        titles = [f"- {code2title.get(c, c).strip()}" for c in codes if len(code2title.get(c, "")) > 3]
        target = "\n".join(titles)
    else:  # discharge_dx
        target_raw = " ".join(to_list(row.get("discharge_diagnoses", [])))
        target_raw = re.sub(r"\s+", " ", target_raw).strip()
        target = target_raw if len(target_raw) >= 5 else ""
    
    return target

# =================================================================================
# ---- OPTIMIZED MAIN PREPROCESSING FUNCTION ----
# =================================================================================

def process_dataframe(df, tok, code2title, G, maps, args, out_path):
    """
    OPTIMIZED: Process a dataframe and save to .jsonl
    
    Key optimizations:
    1. Manual chat template formatting (no apply_chat_template calls)
    2. Faster token budget trimming with better heuristics
    3. Batch-friendly progress logging
    4. Pre-computed code-to-name mappings
    """
    
    # Pre-compute code-to-name mappers (done once)
    cui_to_name = {}
    if G is not None:
        for node, data in G.nodes(data=True):
            if 'name' in data and data['name']:
                cui_to_name[node] = data['name']
    
    atc_to_name = _create_code_map(maps["atc"], cui_to_name)
    loinc_to_name = _create_code_map(maps["loinc"], cui_to_name)
    proc_to_name = _create_code_map(maps["proc"], cui_to_name)
    
    log.info(f"Processing {len(df)} samples for {out_path}...")
    log.info(f"Format: {args.structured_format}, KG block: {args.kg_block}, KG budget: {args.kg_soft_budget}")
    
    start_time = time.time()
    count_processed = 0
    count_skipped = 0
    
    # Create output directory
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-compile system prompt (done once)
    system_prompt = "You are a medical expert. Predict the diagnoses for the current visit based on the provided clinical notes and medical facts."
    system_header = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            
            # Progress logging every 1000 samples
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(df) - idx - 1) / rate if rate > 0 else 0
                log.info(f"  ... processed {idx + 1} / {len(df)} "
                        f"({rate:.1f} samples/s, ETA: {eta/60:.1f} min, "
                        f"skipped: {count_skipped})")
            
            # 1. Get target text
            target = get_target_text(row, args.label_col, args.target_mode, code2title)
            if not target:
                count_skipped += 1
                continue
            
            # 2. Extract evidence CUIs
            src2cuis, ev_cuis = visit_evidence_cuis(
                row, maps["proc"], maps["loinc"], maps["atc"]
            )
            
            # 3. Mine KG paths
            H1_rows, H2_rows = mine_hops_simple(G, ev_cuis, k1=args.kg_k1, k2=args.kg_k2)
            h2_block = render_h2_block(H2_rows)
            h1_block = render_h1_block(H1_rows)
            kg_text = combine_kg_blocks_with_budget(
                tok, h2_block, h1_block,
                args.kg_soft_budget, args.kg_h2_ratio, mode=args.kg_block
            )
            
            # 4. Build header (structured data)
            header_parts = []
            header_parts.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
            header_parts.append(f"DEMOGRAPHICS: gender={row.get('gender','')} age_group={row.get('age','')}")
            
            if args.structured_format == "names":
                med_codes = to_list(row.get("ndc", []))[:24]
                med_names = [atc_to_name.get(_strip(c), c) for c in med_codes]
                if med_names:
                    header_parts.append(f"MEDICATIONS: {', '.join(med_names)}")
                
                proc_codes_raw = to_list(row.get("pro_code", []))[:24]
                proc_codes_clean = [format_icd9_proc_from_pro(c) for c in proc_codes_raw if c]
                proc_names = [proc_to_name.get(c, c) for c in proc_codes_clean if c]
                if proc_names:
                    header_parts.append(f"PROCEDURES: {', '.join(proc_names)}")
                
                lab_codes = to_list(row.get("lab_test_loinc", []))[:48]
                lab_names = [loinc_to_name.get(_strip(c), c) for c in lab_codes]
                if lab_names:
                    header_parts.append(f"LAB TESTS: {', '.join(lab_names)}")
            
            else:  # codes
                med_codes = to_list(row.get("ndc", []))[:24]
                proc_codes_raw = to_list(row.get("pro_code", []))[:24]
                proc_codes = [format_icd9_proc_from_pro(c) for c in proc_codes_raw if c]
                lab_codes = to_list(row.get("lab_test_loinc", []))[:48]
                
                if med_codes:
                    header_parts.append(f"MEDICATIONS: {' '.join(med_codes)}")
                if proc_codes:
                    header_parts.append(f"PROCEDURES: {' '.join(proc_codes)}")
                if lab_codes:
                    header_parts.append(f"LAB TESTS: {' '.join(lab_codes)}")
            
            header_text = "\n".join(header_parts)
            
            # 5. Build final prompt with optimized budgeting
            prompt_text = build_textgen_prompt_budgeted_with_kg_fast(
                header_text, row, tok,
                args.max_len, args.min_assistant_tokens, args.N_max_terms,
                kg_text, args.notes_soft_budget, target
            )
            
            # 6. OPTIMIZED: Manual chat template formatting (much faster!)
            final_text = (
                f"{system_header}"
                f"<|start_header_id|>user<|end_header_id|>\n{prompt_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n{target}{tok.eos_token}"
            )
            
            # 7. Save to .jsonl
            f.write(json.dumps({"text": final_text}, ensure_ascii=False) + "\n")
            count_processed += 1
    
    elapsed = time.time() - start_time
    log.info(f"✓ Saved {count_processed} samples to {out_path}")
    log.info(f"  Skipped {count_skipped} samples (no target)")
    log.info(f"  Processing rate: {count_processed / elapsed:.2f} samples/sec")
    log.info(f"  Total time: {elapsed / 60:.1f} minutes")

# =================================================================================
# ---- MAIN FUNCTION ----
# =================================================================================

def main():
    ap = argparse.ArgumentParser()
    
    # Data
    ap.add_argument("--train_data", required=True, help="Path to pre-split training data (.pkl)")
    ap.add_argument("--val_data", required=True, help="Path to pre-split validation data (.pkl)")
    ap.add_argument("--val_n", type=int, default=100, help="Process only first N validation samples")
    
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--target_mode", choices=["icd_titles", "discharge_dx"], default="icd_titles")
    ap.add_argument("--icd_index_dir", required=True)
    
    # Tokenizer & budgeting
    ap.add_argument("--llm_tokenizer", required=True, help="Path to tokenizer (e.g., model path)")
    ap.add_argument("--max_len", type=int, default=5120)
    ap.add_argument("--N_max_terms", type=int, default=18)
    ap.add_argument("--min_assistant_tokens", type=int, default=128)
    
    # Output files
    ap.add_argument("--out_train_file", required=True, help="Path to save processed train .jsonl")
    ap.add_argument("--out_val_file", required=True, help="Path to save processed val .jsonl")
    
    # Format
    ap.add_argument("--structured_format", type=str, choices=["codes", "names"], default="names")
    
    # KG parameters
    ap.add_argument("--notes_soft_budget", type=int, default=3008)
    ap.add_argument("--kg_soft_budget", type=int, default=1500)
    ap.add_argument("--kg_h2_ratio", type=float, default=0.7)
    ap.add_argument("--kg_block", choices=["both", "h1", "h2"], default="both")
    ap.add_argument("--kg_k1", type=int, default=30)
    ap.add_argument("--kg_k2", type=int, default=30)
    
    ap.add_argument("--kg_pkl", required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl", required=True)
    ap.add_argument("--atc_map_pkl", required=True)
    
    args = ap.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Load tokenizer
    log.info(f"Loading tokenizer from {args.llm_tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.llm_tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    
    # 2. Load data
    log.info(f"Loading pre-split train data: {args.train_data}")
    train_df = pd.read_pickle(args.train_data)
    log.info(f"Loading pre-split val data: {args.val_data}")
    val_df = pd.read_pickle(args.val_data)
    
    # Apply val_n
    if args.val_n and args.val_n > 0:
        log.info(f"Limiting validation set to first {args.val_n} samples")
        val_df = val_df.head(args.val_n).reset_index(drop=True)
    
    log.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # 3. Load KG & maps
    log.info(f"Loading KG from {args.kg_pkl}")
    with open(args.kg_pkl, "rb") as f:
        G_kg = pickle.load(f)
    log.info(f"KG loaded: {G_kg.number_of_nodes()} nodes, {G_kg.number_of_edges()} edges")
    
    log.info("Loading code-to-CUI mappings...")
    with open(args.icd9_proc_map_pkl, "rb") as f:
        icd9_proc_map = pickle.load(f)
    with open(args.loinc_map_pkl, "rb") as f:
        loinc_map = pickle.load(f)
    with open(args.atc_map_pkl, "rb") as f:
        atc_map = pickle.load(f)
    
    maps = {
        "proc": icd9_proc_map,
        "loinc": loinc_map,
        "atc": atc_map
    }
    
    # 4. Load ICD-9 title map
    log.info(f"Loading ICD-9 index from {args.icd_index_dir}")
    code2title = {}
    try:
        with open(os.path.join(args.icd_index_dir, "code2title.json"), "r") as f:
            code2title = json.load(f)
        log.info(f"Loaded {len(code2title)} ICD-9 titles")
    except Exception as e:
        log.warning(f"Could not load ICD index: {e}")
    
    # 5. Process train data
    log.info("\n" + "="*80)
    log.info("PROCESSING TRAIN DATA (OPTIMIZED)")
    log.info("="*80)
    process_dataframe(train_df, tok, code2title, G_kg, maps, args, args.out_train_file)
    
    # 6. Process val data
    log.info("\n" + "="*80)
    log.info("PROCESSING VALIDATION DATA (OPTIMIZED)")
    log.info("="*80)
    process_dataframe(val_df, tok, code2title, G_kg, maps, args, args.out_val_file)
    
    log.info("\n" + "="*80)
    log.info("✓ PREPROCESSING COMPLETE")
    log.info("="*80)
    log.info(f"Train file: {args.out_train_file}")
    log.info(f"Val file: {args.out_val_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())