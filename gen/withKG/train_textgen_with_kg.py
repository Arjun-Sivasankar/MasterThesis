#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_textgen_with_kg.py
HF Trainer for Method 1 (Symbolic Path Mining).

*** PERFORMANCE-OPTIMIZED VERSION (v4) ***
- Fixes major performance bug in `mine_hops_simple` for H2 mining.
- Uses "lazy loading" (logic in __getitem__) for parallel data processing.
- Fixes prompt budgeting bug by accounting for actual target length.
- Uses a single, static --N_max_terms for all prompts.
- Uses pre-split --train_data and --val_data.
- Supports --subset_n (train) and --val_n (validation) for debugging.
- Supports --structured_format=["codes", "names"] for ablation.
- Formats procedure codes correctly (e.g., "54.91") when format="codes".
- Includes detailed per-epoch time logging in the callback.
"""

import os, re, json, time, argparse, logging, pickle, sys, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Set, Tuple
from collections import Counter

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
from transformers import AutoTokenizer, PreTrainedTokenizer

import networkx as nx

# ---- Import all helpers from your common file ----
from common_textgen import (
    log, is_main_process, local_rank,
    pad_collate, load_llm_with_lora,
    to_list, format_icd9, is_valid_icd9, build_eval_labels,
    serialize_structured_readable, serialize_notes,
    chat_token_len, token_len, get_icd9_parent
)

# =================================================================================
# ---- HELPER FUNCTIONS (for Symbolic KG Mining) ----
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
        
        # --- *** BUG FIX IS HERE *** ---
        for v, rela_uv_canon, score_uv, vname in first:
            if v not in G: continue
            second = [] # <-- FIX: Reset 'second' for each 1-hop neighbor
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

# =================================================================================
# ---- PROMPT BUDGETING (with KG) ----
# =================================================================================

def build_textgen_prompt_budgeted_with_kg(
    header_text: str, row: pd.Series, tok, max_len: int, min_assist_tokens: int, N_max_terms: int,
    kg_text: str, notes_soft_budget: int,
    target_text: str # <-- MODIFICATION: Pass in target text
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

    # --- MODIFICATION: Calculate max_prompt_len based on ACTUAL target_text ---
    asst_msg_for_len = {"role": "assistant", "content": target_text}
    # Add 2 tokens for safety (e.g., EOS)
    assistant_tok_len = chat_token_len(tok, [asst_msg_for_len], add_generation_prompt=False) + 2
    assistant_tok_len = max(min_assist_tokens, assistant_tok_len)
    
    max_prompt_len = max_len - assistant_tok_len
    # --- END MODIFICATION ---

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

    stats = {
        "prompt_tokens": final_prompt_len,
        "notes_kept_chars": len(final_notes),
        "notes_trimmed": len(full_notes) - len(final_notes),
        "kg_kept_chars": len(final_kg),
        "kg_trimmed": len(kg_text) - len(final_kg),
    }
    return final_prompt, stats

# =================================================================================
# ---- SFTTextGenDataset (OPTIMIZED "LAZY" VERSION) ----
# =================================================================================

class SFTTextGenDatasetWithKG(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer,
                 label_col: str,
                 target_mode: str,
                 icd_index_dir: str,
                 max_len: int,
                 N_max_terms: int,
                 min_assistant_tokens: int,
                 # ---- KG ARGS ----
                 G: nx.DiGraph,
                 icd9_proc_map: dict,
                 loinc_map: dict,
                 atc_map: dict,
                 notes_soft_budget: int,
                 kg_soft_budget: int,
                 kg_h2_ratio: float,
                 kg_block: str,
                 kg_k1: int,
                 kg_k2: int,
                 structured_format: str
                 ):
        
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label_col = label_col
        self.target_mode = target_mode
        self.max_len = max_len
        self.N_max_terms = N_max_terms
        self.min_assistant_tokens = max(1, int(min_assistant_tokens))

        self.G = G
        self.icd9_proc_map = icd9_proc_map
        self.loinc_map = loinc_map
        self.atc_map = atc_map
        self.notes_soft_budget = notes_soft_budget
        self.kg_soft_budget = kg_soft_budget
        self.kg_h2_ratio = kg_h2_ratio
        self.kg_block = kg_block
        self.kg_k1 = kg_k1
        self.kg_k2 = kg_k2
        self.structured_format = structured_format
        
        self.cui_to_name = {}
        if G is not None:
            for node, data in G.nodes(data=True):
                if 'name' in data and data['name']:
                    self.cui_to_name[node] = data['name']

        self.atc_to_name = self._create_code_map(atc_map, self.cui_to_name)
        self.loinc_to_name = self._create_code_map(loinc_map, self.cui_to_name)
        self.proc_to_name = self._create_code_map(icd9_proc_map, self.cui_to_name)
        
        if is_main_process():
            log.info(f"Created CUI-to-Name map with {len(self.cui_to_name)} entries.")
            log.info(f"Created ATC map ({len(self.atc_to_name)}), LOINC map ({len(self.loinc_to_name)}), PROC map ({len(self.proc_to_name)})")

        self.code2title = {}
        try:
            with open(os.path.join(icd_index_dir, "code2title.json"), "r") as f:
                self.code2title = json.load(f)
            if is_main_process():
                log.info(f"Loaded {len(self.code2title)} ICD-9 titles")
        except Exception as e:
            if is_main_process():
                log.warning(f"Could not load code2title.json: {e}")

        if is_main_process():
            log.info(f"SFT_KG dataset initialized with {len(self.df)} samples (lazy loading enabled).")

    def _create_code_map(self, code_to_cui_map, cui_to_name_map):
        code_to_name = {}
        if code_to_cui_map is None:
            return code_to_name
        for code, cuis in code_to_cui_map.items():
            if cuis:
                name = cui_to_name_map.get(cuis[0])
                if name:
                    code_to_name[code] = name
        return code_to_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            if self.target_mode == "icd_titles":
                codes = [format_icd9(c) for c in to_list(row.get(self.label_col, [])) if c]
                codes = [c for c in codes if is_valid_icd9(c)]
                titles = [f"- {self.code2title.get(c, c).strip()}" for c in codes if len(self.code2title.get(c, "")) > 3]
                target = "\n".join(titles)
            else:
                target_raw = " ".join(to_list(row.get("discharge_diagnoses",[])))
                target_raw = re.sub(r"\s+", " ", target_raw).strip()
                target = target_raw if len(target_raw) >= 5 else ""
            
            if not target:
                return None 
                
            src2cuis, ev_cuis = visit_evidence_cuis(row, self.icd9_proc_map, self.loinc_map, self.atc_map)
            H1_rows, H2_rows = mine_hops_simple(self.G, ev_cuis, k1=self.kg_k1, k2=self.kg_k2)
            h2_block = render_h2_block(H2_rows)
            h1_block = render_h1_block(H1_rows)
            kg_text = combine_kg_blocks_with_budget(
                self.tok, h2_block, h1_block, self.kg_soft_budget, self.kg_h2_ratio, mode=self.kg_block
            )
            
            header_parts = []
            header_parts.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
            header_parts.append(f"DEMOGRAPHICS: gender={row.get('gender','')} age_group={row.get('age','')}")

            if self.structured_format == "names":
                med_codes = to_list(row.get("ndc", []))[:24]
                med_names = [self.atc_to_name.get(_strip(c), c) for c in med_codes]
                if med_names: header_parts.append(f"MEDICATIONS: {', '.join(med_names)}")
                
                proc_codes_raw = to_list(row.get("pro_code", []))[:24]
                proc_codes_clean = [format_icd9_proc_from_pro(c) for c in proc_codes_raw if c]
                proc_names = [self.proc_to_name.get(c, c) for c in proc_codes_clean if c]
                if proc_names: header_parts.append(f"PROCEDURES: {', '.join(proc_names)}")

                lab_codes = to_list(row.get("lab_test_loinc", []))[:48] 
                lab_names = [self.loinc_to_name.get(_strip(c), c) for c in lab_codes]
                if lab_names: header_parts.append(f"LAB TESTS: {', '.join(lab_names)}")
            
            else: 
                med_codes = to_list(row.get("ndc", []))[:24]
                proc_codes_raw = to_list(row.get("pro_code", []))[:24]
                proc_codes = [format_icd9_proc_from_pro(c) for c in proc_codes_raw if c]
                lab_codes = to_list(row.get("lab_test_loinc", []))[:48]
                
                if med_codes:  header_parts.append(f"MEDICATIONS: {' '.join(med_codes)}")
                if proc_codes: header_parts.append(f"PROCEDURES: {' '.join(proc_codes)}")
                if lab_codes: header_parts.append(f"LAB TESTS: {' '.join(lab_codes)}")

            header_text = "\n".join(header_parts)

            prompt, stat = build_textgen_prompt_budgeted_with_kg(
                header_text, row, self.tok, self.max_len, self.min_assistant_tokens, 
                self.N_max_terms, kg_text, self.notes_soft_budget,
                target
            )
            
            user_msg = {"role": "user", "content": prompt}
            asst_msg = {"role": "assistant", "content": target}

            prompt_text = self.tok.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
            full_text   = self.tok.apply_chat_template([user_msg, asst_msg], tokenize=False, add_generation_prompt=False) + self.tok.eos_token
            
            tokenized_full = self.tok(full_text, truncation=True, max_length=self.max_len, add_special_tokens=False)
            input_ids = tokenized_full.input_ids
            
            tokenized_prompt = self.tok(prompt_text, truncation=False, add_special_tokens=False)
            prompt_len = len(tokenized_prompt.input_ids)
            
            if prompt_len >= len(input_ids):
                 return None

            labels = list(input_ids)
            labels[:prompt_len] = [-100] * prompt_len

            assistant_len = sum(1 for x in labels if x != -100)
            if assistant_len < 1:
                return None

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        except Exception as e:
            log.error(f"Error processing sample {idx} (HADM_ID: {row.get('hadm_id', 'N/A')}): {e}", exc_info=True)
            return None

# =================================================================================
# ---- Collate Function (to handle 'None' from lazy dataset) ----
# =================================================================================

def safe_pad_collate(features, tok):
    valid_features = [f for f in features if f is not None]
    if not valid_features:
        return {
            "input_ids": torch.empty(0, 0, dtype=torch.long),
            "attention_mask": torch.empty(0, 0, dtype=torch.long),
            "labels": torch.empty(0, 0, dtype=torch.long),
        }
    return pad_collate(valid_features, tok) # pad_collate is from common_textgen

# =================================================================================
# ---- Trainer Callbacks (MODIFIED for Time Logging) ----
# =================================================================================

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.last_log_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Start timer at the beginning of an epoch."""
        if is_main_process():
            self.last_log_time = time.time() # Set/reset timer at epoch start

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process(): return
        logs = logs or {}
        
        # Per-step training logs
        if 'loss' in logs and 'eval_loss' not in logs:
            epoch=logs.get('epoch',0); loss=logs.get('loss',0)
            self.train_losses.append(float(loss))
            parts = [f"[TRAIN] Epoch {epoch:.2f}", f"Loss {loss:.4f}"]
            if 'learning_rate' in logs: parts.append(f"LR {logs['learning_rate']:.2e}")
            if 'grad_norm' in logs: parts.append(f"Grad {logs['grad_norm']:.3f}")
            log.info(" | ".join(parts))
        
        # Per-eval logs (end of epoch)
        if 'eval_loss' in logs:
            current_epoch = int(logs.get('epoch', 0))
            if self.train_losses:
                avg_train_loss = sum(self.train_losses)/len(self.train_losses)
                self.train_losses = [] # Clear list for next epoch
            else: 
                avg_train_loss = float('nan')
                if state.global_step > args.logging_steps:
                     log.warning("Avg Train Loss is 'nan'. Is logging_steps <= steps_per_epoch?")

            eval_loss = float(logs['eval_loss'])
            try: ppl = float(np.exp(min(eval_loss, 20)))
            except Exception: ppl = float('nan')
            
            lines = [
                "\n" + "="*56, f"[EPOCH {current_epoch} SUMMARY]",
                f"- Avg Train Loss: {avg_train_loss:.4f}",
                f"- Val Loss:       {eval_loss:.4f}",
                f"- Val Perplexity: {ppl:.2f}",
            ]
            
            # Calculate and add epoch times
            current_time = time.time()
            eval_runtime = logs.get('eval_runtime', 0)
            
            if self.last_log_time is not None:
                epoch_total_time = current_time - self.last_log_time
                epoch_train_time = epoch_total_time - eval_runtime
                lines.append(f"- Epoch Train Time: {epoch_train_time:.1f}s")
                lines.append(f"- Eval Time:        {eval_runtime:.1f}s")
                lines.append(f"- Epoch Total Time: {epoch_total_time:.1f}s")
            else:
                lines.append(f"- Eval Time:        {eval_runtime:.1f}s")
            
            self.last_log_time = current_time # Reset timer for next epoch
            lines.append("="*56 + "\n")
            log.info("\n".join(lines))

class SafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is not None and (labels != -100).sum().item() == 0:
            loss = torch.zeros((), device=labels.device, dtype=torch.float32, requires_grad=True)
            return (loss, None) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

def finalize_distributed():
    import datetime, gc
    try:
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                try: torch.cuda.synchronize()
                except Exception: pass
            try: dist.barrier(timeout=datetime.timedelta(seconds=30))
            except Exception: pass
            try: dist.destroy_process_group()
            except Exception: pass
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        if is_main_process(): log.warning(f"Error during distributed cleanup: {e}")

def extract_codes(df, label_col):
    out = []
    for _, r in df.iterrows():
        lst = to_list(r.get(label_col, []))
        lst = [format_icd9(c) for c in lst if c]
        lst = [c for c in lst if is_valid_icd9(c)]
        out.append(lst)
    return out

# =================================================================================
# ---- Main Function (All changes incorporated) ----
# =================================================================================

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--train_data", default=None, help="Path to pre-split training data (.pkl)")
    ap.add_argument("--val_data", default=None, help="Path to pre-split validation data (.pkl)")
    ap.add_argument("--data_pickle", default=None, help="Fallback: Path to *single* data file for splitting.")
    ap.add_argument("--subset_n", type=int, default=0, help="For debugging: use only the first N train samples.")
    ap.add_argument("--val_n", type=int, default=0, help="For debugging: use only the first N validation samples.")
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--target_mode", choices=["icd_titles","discharge_dx"], default="icd_titles")
    ap.add_argument("--icd_index_dir", required=True)

    # llm & train
    ap.add_argument("--llm", required=True)
    ap.add_argument("--max_len", type=int, default=5120)
    ap.add_argument("--N_max_terms", type=int, default=18, help="Static N_max_terms for all prompts.")
    ap.add_argument("--min_assistant_tokens", type=int, default=128)

    # training args
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--early_stop", type=int, default=1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    
    # io
    ap.add_argument("--out_dir", default="runs_codegen/checkpoints")
    ap.add_argument("--save_adapter", action="store_true")
    ap.add_argument("--adapter_dir", default="runs_codegen/adapter")

    # DDP
    ap.add_argument("--local_rank", type=int, default=-1)

    # --- NEW: Format for structured data ---
    ap.add_argument("--structured_format", type=str, choices=["codes", "names"], default="names",
                    help="Format for structured data in prompt: raw codes or natural language names.")
    
    # ---- KG ARGS (Simplified) ----
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
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- 1. Load data (Using new logic) ----
    train_df, val_df = None, None
    if args.train_data and args.val_data:
        if is_main_process():
            log.info(f"Loading pre-split train data: {args.train_data}")
            log.info(f"Loading pre-split val data: {args.val_data}")
        try:
            train_df = pd.read_pickle(args.train_data)
            val_df = pd.read_pickle(args.val_data)
        except Exception as e:
            log.error(f"Failed to load pre-split data files: {e}"); sys.exit(1)
    elif args.data_pickle:
        if is_main_process(): log.info(f"Loading single data file for splitting: {args.data_pickle}")
        try: df = pd.read_pickle(args.data_pickle)
        except Exception:
            with open(args.data_pickle, "rb") as f: df = pickle.load(f)
        if is_main_process(): log.info("Performing subject-based train/val split...")
        subs = df[args.subject_col].dropna().unique()
        tr_subs, te_subs = train_test_split(subs, test_size=0.10, random_state=args.seed)
        tr_subs, va_subs = train_test_split(tr_subs, test_size=0.10/0.90, random_state=args.seed)
        train_df = df[df[args.subject_col].isin(tr_subs)].copy()
        val_df   = df[df[args.subject_col].isin(va_subs)].copy()
    else:
        log.error("No data provided. Must specify either --train_data AND --val_data, OR --data_pickle.")
        if is_main_process(): sys.exit(1)
    
    # --- Apply subset_n and val_n for debugging ---
    if args.subset_n and args.subset_n > 0:
        n_train = int(args.subset_n)
        if is_main_process():
            log.warning(f"--- DEBUG: Using subset of first {n_train} train samples ---")
        train_df = train_df.iloc[:n_train].reset_index(drop=True)
    
    if args.val_n and args.val_n > 0:
        n_val = int(args.val_n)
        if is_main_process():
            log.warning(f"--- DEBUG: Using subset of first {n_val} val samples ---")
        val_df = val_df.iloc[:n_val].reset_index(drop=True)
    elif args.subset_n and args.subset_n > 0 and (not args.val_n or args.val_n <= 0):
        # If train is subset but val is not, also subset val to match
        n_val = min(int(args.subset_n), len(val_df))
        if is_main_process():
            log.warning(f"--- DEBUG: Using subset of first {n_val} val samples (to match train subset) ---")
        val_df = val_df.iloc[:n_val].reset_index(drop=True)

    if len(val_df) == 0 and len(train_df) > 0:
        val_df = train_df.head(1)
    
    train_gold = extract_codes(train_df, args.label_col)
    labels_full = build_eval_labels(train_gold)
    
    if is_main_process():
        log.info(f"Split sizes: train={len(train_df)} val={len(val_df)}")
        log.info(f"Label space (FULL): {len(labels_full)} codes")
        log.info(f"Target max_len: {args.max_len} tokens")
        log.info(f"Using static N_max_terms: {args.N_max_terms}")

    # ---- 2. Load KG & Maps (ONCE) ----
    if is_main_process(): log.info(f"Loading KG: {args.kg_pkl}")
    try:
        G_kg = pickle.load(open(args.kg_pkl, "rb"))
        icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
        loinc_map     = pickle.load(open(args.loinc_map_pkl,     "rb"))
        atc_map       = pickle.load(open(args.atc_map_pkl,       "rb"))
        if is_main_process(): log.info(f"KG loaded with {G_kg.number_of_nodes()} nodes.")
    except Exception as e:
        log.error(f"FATAL: Failed to load KG or map files: {e}"); sys.exit(1)

    # ---- 3. Load model & tokenizer (from common_textgen) ----
    model, tok = load_llm_with_lora(args.llm)

    # ---- 4. Create Datasets (using NEW KG-aware class) ----
    dataset_args = {
        "tokenizer": tok,
        "label_col": args.label_col,
        "target_mode": args.target_mode,
        "icd_index_dir": args.icd_index_dir,
        "max_len": args.max_len,
        "N_max_terms": args.N_max_terms,
        "min_assistant_tokens": args.min_assistant_tokens,
        "G": G_kg,
        "icd9_proc_map": icd9_proc_map,
        "loinc_map": loinc_map,
        "atc_map": atc_map,
        "notes_soft_budget": args.notes_soft_budget,
        "kg_soft_budget": args.kg_soft_budget,
        "kg_h2_ratio": args.kg_h2_ratio,
        "kg_block": args.kg_block,
        "kg_k1": args.kg_k1,
        "kg_k2": args.kg_k2,
        "structured_format": args.structured_format,
    }
    
    train_ds = SFTTextGenDatasetWithKG(train_df, **dataset_args)
    val_ds   = SFTTextGenDatasetWithKG(val_df,   **dataset_args)
    
    if is_main_process() and len(train_ds) > 0: 
        log.info(f"Datasets created. Train items: {len(train_ds)}")
        try:
            log.info("\n" + "="*80)
            log.info("DEBUG: FIRST TRAINING SAMPLE PROMPT")
            log.info(f"(Structured Format: {args.structured_format})")
            log.info("="*80)
            sample_data = None
            for i in range(len(train_ds)):
                sample_data = train_ds[i]
                if sample_data is not None: break
            if sample_data:
                input_ids = sample_data['input_ids']
                full_text = tok.decode(input_ids, skip_special_tokens=False)
                assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
                marker_index = full_text.find(assistant_marker)
                if marker_index != -1:
                    prompt_part = full_text[:marker_index + len(assistant_marker)]
                    response_part = full_text[marker_index + len(assistant_marker):]
                    log.info("--- [USER PROMPT (Decoded)] ---")
                    log.info(prompt_part)
                    log.info("\n--- [ASSISTANT RESPONSE (Decoded)] ---")
                    log.info(response_part)
                else:
                    log.info("--- [FULL TEXT (Decoded, no split)] ---")
                    log.info(full_text)
            else:
                log.warning("Could not find a valid sample to print.")
            log.info("="*80 + "\n")
        except Exception as e:
            log.warning(f"Could not print debug sample: {e}")
    elif is_main_process():
        log.error("Training dataset is empty after processing. Check data and filters.")
        sys.exit(1)

    # ---- 5. Training args ----
    TA = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=bool(args.early_stop),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_timeout=28800,
        local_rank=args.local_rank,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        disable_tqdm=True,
        save_safetensors=True,
    )

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    callbacks.append(LoggingCallback()) # <-- Use the new LoggingCallback

    trainer = SafeTrainer(
        model=model,
        args=TA,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda feats: safe_pad_collate(feats, tok), # <-- Use new collate fn
        callbacks=callbacks,
    )

    # ---- 6. Train ----
    if is_main_process(): log.info("Starting training...")
    t0 = time.time()
    try:
        trainer.train()
    finally:
        if args.save_adapter and is_main_process():
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    log.info("CUDA synchronized")
                os.makedirs(args.adapter_dir, exist_ok=True)
                log.info(f"Saving adapter to {args.adapter_dir}...")
                trainer.model.save_pretrained(args.adapter_dir)
                tok.save_pretrained(args.adapter_dir)
                log.info(f"Adapter saved to {args.adapter_dir}")
            except Exception as e:
                log.warning(f"Adapter save failed: {e}")
        if is_main_process():
            log.info(f"Training completed in {(time.time() - t0) / 60:.2f} minutes")
        finalize_distributed()
        if is_main_process():
            log.info("All done. Exiting cleanly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())