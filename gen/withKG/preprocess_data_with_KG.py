#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_data_with_kg.py
OFFLINE SCRIPT.
Loads the full dataset and KG, performs all slow processing
(KG mining, prompt building, budgeting) and saves the final,
ready-to-train data as .jsonl files.
Logs progress every 1000 samples.
"""

import os, re, json, time, argparse, logging, pickle, sys, math
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split # Keep for fallback

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import networkx as nx

# ---- Import all helpers from your common file ----
# (Make sure common_textgen.py is in the same directory or python path)
from common_textgen import (
    log, is_main_process, local_rank,
    pad_collate, load_llm_with_lora,
    to_list, format_icd9, is_valid_icd9, build_eval_labels,
    serialize_structured_readable, serialize_notes,
    chat_token_len, token_len, get_icd9_parent
)

# =================================================================================
# ---- HELPER FUNCTIONS (Copied from train_textgen_with_kg.py) ----
# =================================================================================

def trim_to_token_budget(tok, text: str, max_tokens: int) -> str:
    """Trims a string to a maximum token count."""
    if max_tokens <= 0 or not text: return ""
    if token_len(tok, text) <= max_tokens: return text
    lo, hi = 0, len(text); best = ""
    while lo <= hi:
        mid = (lo + hi)//2
        cand = text[:mid]
        if token_len(tok, cand) <= max_tokens:
            best = cand; lo = mid + 1
        else:
            hi = mid - 1
    return best

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

def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str,List[str]], Set[str]]:
    src2cuis = {}
    ev = set()
    icd9_proc_map = icd9_proc_map or {}
    loinc_map = loinc_map or {}
    atc_map = atc_map or {}
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        cuis = atc_map.get(key, [])
        if cuis: src2cuis[f"ATC:{key}"] = cuis; ev.update(cuis)
    for c in to_list(row.get("lab_test_loinc", [])):
        key = _strip(c)
        cuis = loinc_map.get(key, [])
        if cuis: src2cuis[f"LNC:{key}"] = cuis; ev.update(cuis)
    for c in to_list(row.get("pro_code", [])):
        cc = format_icd9_proc_from_pro(c)
        if not cc: continue
        cuis = icd9_proc_map.get(cc, [])
        if cuis: src2cuis[f"PROC:{cc}"] = cuis; ev.update(cuis)
    return src2cuis, ev

def _arrow_label(rela: str, rel: str) -> str:
    r = (rela or "").strip() or (rel or "").strip()
    return f" --{r}--> " if r else " → "

def mine_hops_simple(G: nx.DiGraph,
                     ev_cuis: Set[str],
                     k1:int=30, k2:int=30) -> Tuple[List[dict], List[dict]]:
    H1, H2 = [], []
    if G is None: return H1, H2
    def _edge_attrs(d):
        rela_canon = d.get("rela_canon") or d.get("rela") or d.get("rel") or ""
        score = d.get("rela_score")
        try: score = float(score) if score is not None else None
        except Exception: score = None
        return rela_canon, score
    for u in ev_cuis:
        if u not in G: continue
        first = []
        for v in G.successors(u):
            d = G[u][v]
            rela_canon, score = _edge_attrs(d)
            first.append((v, rela_canon, -1e9 if score is None else score, G.nodes[v].get("name","")))
        first.sort(key=lambda t: (t[2], t[0]), reverse=True)
        if k1 and k1 > 0: first = first[:k1]
        for v, rela_canon, score, vname in first:
            d = G[u][v]
            H1.append({"src_cui": u, "nbr_cui": v, "src_name": G.nodes[u].get("name",""), "nbr_name": vname,
                       "rel": (d.get("rel") or ""), "rela": (d.get("rela") or ""), "rela_canon": rela_canon,
                       "rela_score": (None if score == -1e9 else score)})
        for v, rela_uv_canon, score_uv, vname in first:
            if v not in G: continue
            second = [] 
            for w in G.successors(v):
                d_uv, d_vw = G[u][v], G[v][w]
                rela_uv_c, score_uv2 = _edge_attrs(d_uv)
                rela_vw_c, score_vw  = _edge_attrs(d_vw)
                s_uv = score_uv if score_uv != -1e9 else (-1e9 if score_uv2 is None else score_uv2)
                s_vw = -1e9 if score_vw is None else score_vw
                total = (s_uv if s_uv != -1e9 else 0.0) + (s_vw if s_vw != -1e9 else 0.0)
                second.append((total, v, w, rela_uv_c, (None if s_uv == -1e9 else s_uv),
                               rela_vw_c, (None if s_vw == -1e9 else s_vw), vname, G.nodes[w].get("name","")))
            second.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            if k2 and k2 > 0: second = second[:k2]
            for total, v, w, rela_uv_c, s_uv, rela_vw_c, s_vw, vname, wname in second:
                H2.append({"u": u, "v": v, "w": w, "u_name": G.nodes[u].get("name",""), "v_name": vname, "w_name": wname,
                           "rel_uv":  (G[u][v].get("rel")  or ""), "rela_uv": (G[u][v].get("rela") or ""),
                           "rel_vw":  (G[v][w].get("rel")  or ""), "rela_vw": (G[v][w].get("rela") or ""),
                           "rela_uv_canon": rela_uv_c, "rela_vw_canon": rela_vw_c,
                           "score_uv": s_uv, "score_vw": s_vw, "score_total": ((s_uv or 0.0) + (s_vw or 0.0))})
    return H1, H2

def render_h2_block(H2_rows: List[dict]) -> str:
    lines = ["[KG context - H2 paths]"]
    if not H2_rows: return lines[0] + "\n- (none)"
    for c in H2_rows:
        u, v, w = (c.get(k) or "" for k in ["u_name", "v_name", "w_name"])
        r_uv = c.get("rela_uv_canon") or c.get("rela_uv") or c.get("rel_uv") or ""
        r_vw = c.get("rela_vw_canon") or c.get("rela_vw") or c.get("rel_vw") or ""
        lines.append(f"- {u}{_arrow_label(r_uv, '')}{v}{_arrow_label(r_vw, '')}{w}")
    return "\n".join(lines)

def render_h1_block(H1_rows: List[dict]) -> str:
    lines = ["[KG context - H1 paths]"]
    if not H1_rows: return lines[0] + "\n- (none)"
    for c in H1_rows:
        u, v = (c.get(k) or "" for k in ["src_name", "nbr_name"])
        r = c.get("rela_canon") or c.get("rela") or c.get("rel") or ""
        lines.append(f"- {u}{_arrow_label(r, '')}{v}")
    return "\n".join(lines)

def combine_kg_blocks_with_budget(tok, h2_text: str, h1_text: str, budget: int, h2_ratio: float = 1.0, mode: str = "both"):
    mode = (mode or "both").lower()
    if budget is None or budget <= 0:
        if mode == "h1": return h1_text
        if mode == "h2": return h2_text
        return h2_text + ("\n" + h1_text if h1_text else "")
    if mode == "h1": return trim_to_token_budget(tok, h1_text, budget)
    if mode == "h2": return trim_to_token_budget(tok, h2_text, budget)
    h2_quota = int(max(0, min(1.0, h2_ratio)) * budget)
    h1_quota = max(0, budget - h2_quota)
    h2_trim = trim_to_token_budget(tok, h2_text, h2_quota) if h2_quota>0 else ""
    h1_trim = trim_to_token_budget(tok, h1_text, h1_quota) if h1_quota>0 else ""
    used_h2 = token_len(tok, h2_trim)
    leftover = max(0, budget - used_h2 - token_len(tok, h1_trim))
    if leftover > 0 and h1_text:
        h1_trim = trim_to_token_budget(tok, h1_text, token_len(tok, h1_trim) + leftover)
    return (h2_trim if h2_trim else "") + (("\n"+h1_trim) if h1_trim else "")

def build_tail(N_max_terms:int) -> str:
    lines = [
        "[TASK] List the final clinical diagnoses for this admission.", "[FORMAT]",
        "- One diagnosis per line", "- Avoid abbreviations if possible",
        "- No ICD codes or explanations", f"- Maximum: {N_max_terms} lines", "[OUTPUT]",
    ]
    return "\n".join(lines)

def build_textgen_prompt_budgeted_with_kg(
    header_text: str, row: pd.Series, tok, max_len: int, min_assist_tokens: int, N_max_terms: int,
    kg_text: str, notes_soft_budget: int,
    target_text: str
) -> Tuple[str, Dict[str, int]]:
    
    full_notes = serialize_notes(row)
    notes = trim_to_token_budget(tok, full_notes, notes_soft_budget) if notes_soft_budget > 0 else full_notes
    tail = build_tail(N_max_terms)
    
    def assemble(notes_text: str, kg_text_block: str) -> str:
        parts = [header_text]
        if notes_text: parts.append(notes_text)
        if kg_text_block: parts.append(kg_text_block)
        parts.append(tail)
        return "\n".join(parts)

    def get_prompt_len(user_content: str) -> int:
        return chat_token_len(tok, [{"role":"user", "content": user_content}], add_generation_prompt=True)

    asst_msg_for_len = {"role": "assistant", "content": target_text}
    assistant_tok_len = chat_token_len(tok, [asst_msg_for_len], add_generation_prompt=False) + 2
    assistant_tok_len = max(min_assist_tokens, assistant_tok_len)
    
    max_prompt_len = max_len - assistant_tok_len

    base_prompt_len = get_prompt_len(assemble("", ""))
    available_budget = max_prompt_len - base_prompt_len
    
    notes_len = token_len(tok, notes)
    kg_len = token_len(tok, kg_text)
    
    final_notes = notes
    final_kg = kg_text
    
    if notes_len + kg_len > available_budget:
        notes_budget = max(0, available_budget - kg_len)
        final_notes = trim_to_token_budget(tok, notes, notes_budget)
        notes_len = token_len(tok, final_notes)
        
        if notes_len + kg_len > available_budget:
            kg_budget = max(0, available_budget - notes_len)
            final_kg = trim_to_token_budget(tok, kg_text, kg_budget)
    
    final_prompt = assemble(final_notes, final_kg)
    final_prompt_len = get_prompt_len(final_prompt)
    
    if final_prompt_len > max_prompt_len:
         overage = final_prompt_len - max_prompt_len
         final_notes = trim_to_token_budget(tok, final_notes, token_len(tok, final_notes) - overage - 2)
         final_prompt = assemble(final_notes, final_kg)
         final_prompt_len = get_prompt_len(final_prompt)
    
    # We only need the final prompt text
    return final_prompt

# Helper to create code-to-name maps
def _create_code_map(code_to_cui_map, cui_to_name_map):
    code_to_name = {}
    if code_to_cui_map is None:
        return code_to_name
    for code, cuis in code_to_cui_map.items():
        if cuis:
            name = cui_to_name_map.get(cuis[0])
            if name:
                code_to_name[code] = name
    return code_to_name

# Helper to get target text
def get_target_text(row, label_col, target_mode, code2title):
    if target_mode == "icd_titles":
        codes = [format_icd9(c) for c in to_list(row.get(label_col, [])) if c]
        codes = [c for c in codes if is_valid_icd9(c)]
        titles = [f"- {code2title.get(c, c).strip()}" for c in codes if len(code2title.get(c, "")) > 3]
        target = "\n".join(titles)
    else: # discharge_dx
        target_raw = " ".join(to_list(row.get("discharge_diagnoses",[])))
        target_raw = re.sub(r"\s+", " ", target_raw).strip()
        target = target_raw if len(target_raw) >= 5 else ""
    return target

# =================================================================================
# ---- Main Preprocessing Function ----
# =================================================================================

def process_dataframe(df, tok, code2title, G, maps, args, out_path):
    """
    Processes a dataframe (train or val) and saves the
    fully formatted prompt strings to a .jsonl file.
    """
    
    # Create the code-to-name mappers
    cui_to_name = {}
    if G is not None:
        for node, data in G.nodes(data=True):
            if 'name' in data and data['name']:
                cui_to_name[node] = data['name']
    
    atc_to_name = _create_code_map(maps["atc"], cui_to_name)
    loinc_to_name = _create_code_map(maps["loinc"], cui_to_name)
    proc_to_name = _create_code_map(maps["proc"], cui_to_name)

    log.info(f"Processing {len(df)} samples for {out_path}...")
    
    start_time = time.time()
    count_processed = 0
    with open(out_path, 'w') as f:
        for idx, row in df.iterrows():
            
            # --- Logging progress ---
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                log.info(f"  ... processed sample {idx + 1} / {len(df)} ({elapsed:.1f}s elapsed)")
            
            # 1. Get Target Text
            target = get_target_text(row, args.label_col, args.target_mode, code2title)
            if not target:
                continue # Skip samples with no target

            # 2. Get KG Context (Symbolic)
            src2cuis, ev_cuis = visit_evidence_cuis(row, maps["proc"], maps["loinc"], maps["atc"])
            H1_rows, H2_rows = mine_hops_simple(G, ev_cuis, k1=args.kg_k1, k2=args.kg_k2)
            h2_block = render_h2_block(H2_rows)
            h1_block = render_h1_block(H1_rows)
            kg_text = combine_kg_blocks_with_budget(
                tok, h2_block, h1_block, args.kg_soft_budget, args.kg_h2_ratio, mode=args.kg_block
            )
            
            # 3. Build Header (Dynamically)
            header_parts = []
            header_parts.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
            header_parts.append(f"DEMOGRAPHICS: gender={row.get('gender','')} age_group={row.get('age','')}")

            if args.structured_format == "names":
                med_codes = to_list(row.get("ndc", []))[:24]
                med_names = [atc_to_name.get(_strip(c), c) for c in med_codes]
                if med_names: header_parts.append(f"MEDICATIONS: {', '.join(med_names)}")
                
                proc_codes_raw = to_list(row.get("pro_code", []))[:24]
                proc_codes_clean = [format_icd9_proc_from_pro(c) for c in proc_codes_raw if c]
                proc_names = [proc_to_name.get(c, c) for c in proc_codes_clean if c]
                if proc_names: header_parts.append(f"PROCEDURES: {', '.join(proc_names)}")

                lab_codes = to_list(row.get("lab_test_loinc", []))[:48] 
                lab_names = [loinc_to_name.get(_strip(c), c) for c in lab_codes]
                if lab_names: header_parts.append(f"LAB TESTS: {', '.join(lab_names)}")
            
            else: # self.structured_format == "codes"
                med_codes = to_list(row.get("ndc", []))[:24]
                proc_codes_raw = to_list(row.get("pro_code", []))[:24]
                proc_codes = [format_icd9_proc_from_pro(c) for c in proc_codes_raw if c]
                lab_codes = to_list(row.get("lab_test_loinc", []))[:48]
                
                if med_codes:  header_parts.append(f"MEDICATIONS: {' '.join(med_codes)}")
                if proc_codes: header_parts.append(f"PROCEDURES: {' '.join(proc_codes)}")
                if lab_codes: header_parts.append(f"LAB TESTS: {' '.join(lab_codes)}")
            
            header_text = "\n".join(header_parts)
            
            # 4. Build Prompt with Budgeting
            prompt_text = build_textgen_prompt_budgeted_with_kg(
                header_text, row, tok, args.max_len, args.min_assistant_tokens, 
                args.N_max_terms, kg_text, args.notes_soft_budget,
                target
            )
            
            # 5. Format final string for Llama-3
            system_prompt = "You are a medical expert. Predict the diagnoses for the current visit based on the provided clinical notes and medical facts."
            
            final_text_parts = [
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>",
                f"<|start_header_id|>user<|end_header_id|>\n{prompt_text}<|eot_id|>",
                f"<|start_header_id|>assistant<|end_header_id|>\n{target}{tok.eos_token}" # Add EOS here
            ]
            final_text = "".join(final_text_parts)
            
            # 6. Save to .jsonl
            f.write(json.dumps({"text": final_text}) + "\n")
            count_processed += 1

    log.info(f"Successfully processed and saved {count_processed} samples to {out_path}")


# =================================================================================
# ---- Main Function (Runs the Preprocessing) ----
# =================================================================================

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--train_data", required=True, help="Path to pre-split training data (.pkl)")
    ap.add_argument("--val_data", required=True, help="Path to pre-split validation data (.pkl)")
    ap.add_argument("--val_n", type=int, default=100, help="Process only the first N validation samples.")
    
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--target_mode", choices=["icd_titles","discharge_dx"], default="icd_titles")
    ap.add_argument("--icd_index_dir", required=True)

    # llm & budgeting
    ap.add_argument("--llm_tokenizer", required=True, help="Path to the tokenizer (e.g., the base model path)")
    ap.add_argument("--max_len", type=int, default=5120)
    ap.add_argument("--N_max_terms", type=int, default=18)
    ap.add_argument("--min_assistant_tokens", type=int, default=128)

    # --- NEW: Output files ---
    ap.add_argument("--out_train_file", required=True, help="Path to save processed train .jsonl file")
    ap.add_argument("--out_val_file", required=True, help="Path to save processed val .jsonl file")

    # --- NEW: Format for structured data ---
    ap.add_argument("--structured_format", type=str, choices=["codes", "names"], default="names")
    
    # ---- KG ARGS ----
    ap.add_argument("--notes_soft_budget",  type=int, default=3008)
    ap.add_argument("--kg_soft_budget",     type=int, default=1500)
    ap.add_argument("--kg_h2_ratio",        type=float, default=0.7)
    ap.add_argument("--kg_block",           choices=["both","h1","h2"], default="both")
    ap.add_argument("--kg_k1", type=int, default=30)
    ap.add_argument("--kg_k2", type=int, default=30)
    
    ap.add_argument("--kg_pkl", required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl",     required=True)
    ap.add_argument("--atc_map_pkl",       required=True)

    args = ap.parse_args()

    # ---- 1. Load Tokenizer ----
    log.info(f"Loading tokenizer from {args.llm_tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.llm_tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- 2. Load data ----
    log.info(f"Loading pre-split train data: {args.train_data}")
    train_df = pd.read_pickle(args.train_data)
    log.info(f"Loading pre-split val data: {args.val_data}")
    val_df = pd.read_pickle(args.val_data)
    
    # --- Apply val_n ---
    if args.val_n and args.val_n > 0:
        n_val = int(args.val_n)
        log.warning(f"--- Using subset of first {n_val} val samples ---")
        val_df = val_df.iloc[:n_val].reset_index(drop=True)

    log.info(f"Processing {len(train_df)} train samples and {len(val_df)} val samples.")
    log.info(f"Config: format={args.structured_format}, kg_block={args.kg_block}, kg_budget={args.kg_soft_budget}")

    # ---- 3. Load KG & Maps (ONCE) ----
    log.info(f"Loading KG: {args.kg_pkl}")
    G_kg = pickle.load(open(args.kg_pkl, "rb"))
    maps = {
        "proc": pickle.load(open(args.icd9_proc_map_pkl, "rb")),
        "loinc": pickle.load(open(args.loinc_map_pkl,     "rb")),
        "atc": pickle.load(open(args.atc_map_pkl,       "rb"))
    }
    log.info(f"KG loaded with {G_kg.number_of_nodes()} nodes.")

    # Load ICD-9 title map
    code2title = {}
    try:
        with open(os.path.join(args.icd_index_dir, "code2title.json"), "r") as f:
            code2title = json.load(f)
        log.info(f"Loaded {len(code2title)} ICD-9 titles")
    except Exception as e:
        log.warning(f"Could not load code2title.json: {e}")

    # ---- 4. Process and Save Train Data ----
    process_dataframe(train_df, tok, code2title, G_kg, maps, args, args.out_train_file)
    
    # ---- 5. Process and Save Val Data ----
    process_dataframe(val_df, tok, code2title, G_kg, maps, args, args.out_val_file)

    log.info("Offline preprocessing complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())