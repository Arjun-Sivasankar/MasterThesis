#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_textgen_with_kg.py
Inference script aligned with train_textgen_with_kg.py

Key fixes:
1. Uses same prompt structure as training (no system prompt)
2. Proper metrics printing at the end
3. Aligned token counting and chat template application
"""

import os, re, json, time, argparse, pickle, glob, sys, math
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
import torch.distributed as dist
import networkx as nx

# ====== your shared utils ======
from common_textgen import (
    log, is_main_process, world_size, local_rank,
    serialize_notes, chat_token_len, token_len,
    ICDMapper, to_list, format_icd9, is_valid_icd9,
    restrict_to, multihot, eval_pack,
    get_icd9_parent
)

# ------------------------------- robust output parsing -------------------------------
_OUTPUT_RE = re.compile(r"\[OUTPUT\]", flags=re.IGNORECASE)

def _coerce_text(x) -> str:
    if x is None: return ""
    if isinstance(x, (list, tuple)): return "\n".join(_coerce_text(y) for y in x if y is not None)
    if isinstance(x, bytes):
        try: return x.decode("utf-8", errors="ignore")
        except Exception: return str(x)
    return str(x)

def extract_after_output(generation, n_max: int = 12) -> List[str]:
    s = _coerce_text(generation)
    s = s.replace("</s>", "").replace("<s>", "").strip()
    
    # Find assistant response after the header
    prompt_marker = "<|start_header_id|>assistant<|end_header_id|>"
    marker_index = s.rfind(prompt_marker)
    if marker_index != -1:
        s = s[marker_index + len(prompt_marker):]
    
    # Find [OUTPUT] marker
    m = _OUTPUT_RE.search(s)
    block = s[m.end():].strip() if m else s.strip()
    
    lines_out = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line: continue
        line = re.sub(r"^(?:[-*]\s*|\d+\.\s*|\(\d+\)\s*)", "", line)
        if line: lines_out.append(line)
        if len(lines_out) >= n_max: break
    return lines_out

# ------------------------------- ICD-9 helpers -------------------------------
def get_icd9_parent_safe(code: str) -> str:
    try: return get_icd9_parent(code)
    except Exception:
        c = (code or "").upper()
        if not c: return c
        if c[0].isdigit(): return c.split('.')[0][:3]
        if c[0] == 'V':    return c.split('.')[0][:3]
        if c[0] == 'E':    return c.split('.')[0][:4]
        return c

def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
    g = [[get_icd9_parent_safe(c) for c in lst] for lst in gold_lists]
    p = [[get_icd9_parent_safe(c) for c in lst] for lst in pred_lists]
    labels = sorted({x for lst in g for x in lst})
    Yg = multihot(g, labels)
    Yp = multihot(p, labels)
    metrics_dict.update({
        "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
        "recall_macro_parent":    float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent":        float(f1_score(Yg, Yp, average="macro", zero_division=0)),
    })
    return labels, Yg, Yp

# ------------------------------- Minimal dist utils -------------------------------
def maybe_init_dist():
    if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl")
    return dist.is_initialized()

def shard_indices(N:int, rank:int, W:int): 
    return list(range(rank, N, W))

def barrier():
    if dist.is_available() and dist.is_initialized():
        try: dist.barrier()
        except Exception: pass

def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        try: dist.destroy_process_group()
        except Exception: pass

# ------------------------------- Token budgeting (aligned with training) -------------------------------
def trim_to_token_budget(tok, text: str, max_tokens: int) -> str:
    """Trims a string to a maximum token count - ALIGNED WITH TRAINING."""
    if max_tokens <= 0 or not text: return ""
    if token_len(tok, text) <= max_tokens: return text
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi)//2
        cand = text[:mid]
        if token_len(tok, cand) <= max_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best

# ------------------------------- Evidence & KG plumbing -------------------------------
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

# ------------------------------- KG paths (H2/H1) -------------------------------
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

# ------------------------------- Prompt builders (ALIGNED WITH TRAINING) -------------------------------
def build_tail(N_max_terms:int) -> str:
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

def build_prompt_for_row(header_text: str,
                         row: pd.Series,
                         tok,
                         kg_text: str,
                         notes_soft_budget: int,
                         N_max_terms: int) -> str:
    """Build prompt EXACTLY as in training (no system prompt)."""
    full_notes = serialize_notes(row)
    notes = trim_to_token_budget(tok, full_notes, notes_soft_budget) if notes_soft_budget > 0 else full_notes
    tail = build_tail(N_max_terms)
    
    # Assemble parts
    parts = [header_text]
    if notes: parts.append(notes)
    if kg_text: parts.append(kg_text)
    parts.append(tail)
    
    user_content = "\n".join(parts)
    
    # Apply chat template (SAME AS TRAINING - no system prompt)
    user_msg = {"role": "user", "content": user_content}
    prompt = tok.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    
    return prompt

# ------------------------------- Generation -------------------------------
def build_generate_kwargs(decoding: str,
                          max_new: int,
                          eos_id: int,
                          pad_id: int,
                          num_beams: int = 2,
                          temperature: float = 1.0,
                          top_p: float = 0.95,
                          top_k: int = 50,
                          no_repeat_ngram: int = 0):
    if decoding == "greedy":
        return dict(
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None),
        )
    if decoding == "beam":
        return dict(
            max_new_tokens=max_new,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None),
        )
    return dict(
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None),
    )

@torch.no_grad()
def generate_texts(model, tok, prompts: List[str], max_len: int, gen_kwargs: dict, batch_size: int = 8, device=None) -> List[str]:
    device = device or (model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outs = []
    bs = max(1, int(batch_size))
    tok.padding_side = "left"
    
    for i in tqdm(range(0, len(prompts), bs), disable=not is_main_process(), desc="Generating"):
        batch = prompts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len if max_len and max_len>0 else None, add_special_tokens=False)
        for k in enc:
            enc[k] = enc[k].to(device)
        
        gen = model.generate(**enc, **{k:v for k,v in gen_kwargs.items() if v is not None})
        dec = tok.batch_decode(gen, skip_special_tokens=False)
        outs.extend(dec)
    return outs

# ------------------------------- Pretty printer -------------------------------
def _pretty_print_metrics(title: str, metrics: dict):
    """Print metrics in a clean, readable format."""
    log.info("")
    log.info("="*60)
    log.info(f"  {title}")
    log.info("="*60)
    
    # Group metrics
    micro = {k: v for k, v in metrics.items() if 'micro' in k}
    macro = {k: v for k, v in metrics.items() if 'macro' in k and 'parent' not in k}
    parent = {k: v for k, v in metrics.items() if 'parent' in k}
    
    if micro:
        log.info("\nMicro-averaged metrics:")
        for k, v in sorted(micro.items()):
            log.info(f"  {k:35s}: {v:.4f}")
    
    if macro:
        log.info("\nMacro-averaged metrics:")
        for k, v in sorted(macro.items()):
            log.info(f"  {k:35s}: {v:.4f}")
    
    if parent:
        log.info("\nParent-code metrics:")
        for k, v in sorted(parent.items()):
            log.info(f"  {k:35s}: {v:.4f}")
    
    log.info("="*60)
    log.info("")

# ------------------------------- MAIN -------------------------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--subset_n", type=int, default=0)
    ap.add_argument("--print_samples", type=int, default=5)

    # prompts/generation
    ap.add_argument("--N_max_terms", type=int, default=12, help="Static cap for output lines. Must match training.")

    # token budgets
    ap.add_argument("--total_input_budget", type=int, default=5120)
    ap.add_argument("--assistant_reserve",  type=int, default=256)
    ap.add_argument("--notes_soft_budget",  type=int, default=3008)
    ap.add_argument("--kg_soft_budget",     type=int, default=1500)
    ap.add_argument("--kg_h2_ratio",        type=float, default=0.7)
    ap.add_argument("--kg_block",           choices=["both","h1","h2"], default="both")
    ap.add_argument("--structured_format",  choices=["codes", "names"], default="names")

    # decoding
    ap.add_argument("--gen_max_new", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--decoding", choices=["greedy","beam","sample"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_repeat_ngram", type=int, default=0)

    # model/adapter
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--use_bf16", action="store_true")

    # mapper
    ap.add_argument("--icd_index_dir", required=True)
    ap.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--faiss_rows", type=int, default=50)
    ap.add_argument("--tau_cos", type=float, default=0.40)
    ap.add_argument("--tau_final", type=float, default=0.60)
    ap.add_argument("--w_cos", type=float, default=0.6)
    ap.add_argument("--w_fuz", type=float, default=0.4)

    # eval label space
    ap.add_argument("--labels_space", choices=["full","head"], default="full")

    # KG inputs
    ap.add_argument("--kg_pkl", required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl",     required=True)
    ap.add_argument("--atc_map_pkl",       required=True)
    ap.add_argument("--kg_k1", type=int, default=30)
    ap.add_argument("--kg_k2", type=int, default=30)

    # distributed
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--tmp_dir", default="runs_textgen/test_shards")
    ap.add_argument("--out_metrics", default="runs_textgen/test_metrics.json")
    ap.add_argument("--stats_csv", default="")

    args = ap.parse_args()

    # ---------------- data ----------------
    if is_main_process():
        log.info(f"Loading test data: {args.data_pickle}")
    
    try:
        df = pd.read_pickle(args.data_pickle)
    except Exception:
        with open(args.data_pickle, "rb") as f: 
            df = pickle.load(f)

    if args.test_only:
        test_df = df.copy()
    else:
        from sklearn.model_selection import train_test_split
        subs = df[args.subject_col].dropna().unique()
        _, te_subs = train_test_split(subs, test_size=0.10, random_state=args.seed)
        test_df = df[df[args.subject_col].isin(te_subs)].copy()

    if args.subset_n and args.subset_n > 0:
        test_df = test_df.iloc[:args.subset_n].reset_index(drop=True)

    def extract_codes(df, label_col):
        out = []
        for _, r in df.iterrows():
            lst = to_list(r.get(label_col, []))
            lst = [format_icd9(c) for c in lst if c]
            lst = [c for c in lst if is_valid_icd9(c)]
            out.append(lst)
        return out
    
    gold_codes = extract_codes(test_df, args.label_col)
    labels_full = sorted({c for lst in gold_codes for c in lst})
    labels_eval = labels_full

    if is_main_process():
        log.info(f"Test size: {len(test_df)}")
        log.info(f"Eval label space: {len(labels_eval)} codes")
        log.info(f"Static N_max_terms: {args.N_max_terms}")
        log.info(f"Structured format: {args.structured_format}")
        log.info(f"KG block mode: {args.kg_block}")

    # ---------------- model (adapter-only, LEFT padding) ----------------
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) \
           else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None: 
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if is_main_process():
        log.info(f"Loading base model: {args.base_model}")
        log.info(f"Loading adapter: {args.adapter_dir}")

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.config.use_cache = True
    dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    # ---------------- KG + maps ----------------
    if is_main_process():
        log.info("Loading KG and mapping files...")
    
    KG = pickle.load(open(args.kg_pkl, "rb"))
    icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
    loinc_map = pickle.load(open(args.loinc_map_pkl, "rb"))
    atc_map = pickle.load(open(args.atc_map_pkl, "rb"))

    # Create CUI-to-Name and Code-to-Name maps
    cui_to_name = {}
    if KG is not None:
        for node, data in KG.nodes(data=True):
            if 'name' in data and data['name']:
                cui_to_name[node] = data['name']
    
    atc_to_name = _create_code_map(atc_map, cui_to_name)
    loinc_to_name = _create_code_map(loinc_map, cui_to_name)
    proc_to_name = _create_code_map(icd9_proc_map, cui_to_name)

    # ---------------- build prompts ----------------
    if is_main_process():
        log.info("Building prompts...")
    
    kg_prompts = []
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), disable=not is_main_process(), desc="Building prompts"):
        src2cuis, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)

        # Build header
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
        else:
            med_codes = to_list(row.get("ndc", []))[:24]
            proc_codes_raw = to_list(row.get("pro_code", []))[:24]
            proc_codes = [format_icd9_proc_from_pro(c) for c in proc_codes_raw if c]
            lab_codes = to_list(row.get("lab_test_loinc", []))[:48]
            
            if med_codes: header_parts.append(f"MEDICATIONS: {' '.join(med_codes)}")
            if proc_codes: header_parts.append(f"PROCEDURES: {' '.join(proc_codes)}")
            if lab_codes: header_parts.append(f"LAB TESTS: {' '.join(lab_codes)}")

        header_text = "\n".join(header_parts)

        # Mine & render KG paths
        H1_rows, H2_rows = mine_hops_simple(KG, ev_cuis, k1=args.kg_k1, k2=args.kg_k2)
        h2_block = render_h2_block(H2_rows)
        h1_block = render_h1_block(H1_rows)
        
        if args.kg_soft_budget <= 0:
            kg_text_combined = ""
        else:
            kg_text_combined = combine_kg_blocks_with_budget(
                tok, h2_block, h1_block, args.kg_soft_budget, args.kg_h2_ratio, mode=args.kg_block
            )

        # Build prompt (ALIGNED WITH TRAINING)
        kg_p = build_prompt_for_row(
            header_text, row, tok, kg_text_combined, 
            args.notes_soft_budget, args.N_max_terms
        )
        
        kg_prompts.append(kg_p)

    if is_main_process() and len(kg_prompts) > 0:
        log.info("\n" + "="*80)
        log.info("FIRST SAMPLE - PROMPT")
        log.info("="*80)
        log.info(kg_prompts[0])
        log.info("="*80 + "\n")

    # ---------------- decoding kwargs ----------------
    gen_kwargs = build_generate_kwargs(
        decoding=args.decoding, max_new=args.gen_max_new,
        eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
        num_beams=args.num_beams, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, no_repeat_ngram=args.no_repeat_ngram
    )

    # ---------------- distributed sharding ----------------
    if args.distributed:
        maybe_init_dist()
        rank = int(os.environ.get("RANK", "0"))
        W = world_size()
        idxs = shard_indices(len(kg_prompts), rank, W)
    else:
        rank, W = 0, 1
        idxs = list(range(len(kg_prompts)))

    shard_kg = [kg_prompts[i] for i in idxs]
    shard_gold = [gold_codes[i] for i in idxs]

    # ---------------- generation ----------------
    if is_main_process():
        log.info("Starting generation...")
    
    t0 = time.time()
    kg_out_texts = generate_texts(model, tok, shard_kg, max_len=args.total_input_budget, 
                                   gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)

    kg_terms = [extract_after_output(t, args.N_max_terms) for t in kg_out_texts]

    if is_main_process():
        per = (time.time() - t0) / max(1, len(idxs))
        log.info(f"Generation done ({per:.2f}s/sample on rank {rank}).")

    # ---------------- mapping terms -> ICD-9 ----------------
    if is_main_process():
        log.info("Mapping terms to ICD-9 codes...")
    
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    mapped_kg = mapper.map_terms(kg_terms)

    # ---------------- persist shard ----------------
    os.makedirs(args.tmp_dir, exist_ok=True)
    shard_path = os.path.join(args.tmp_dir, f"shard_{rank:03d}_of_{W:03d}.pkl")
    with open(shard_path, "wb") as f:
        pickle.dump({
            "idxs": idxs,
            "kg_texts": kg_out_texts,
            "kg_terms": kg_terms,
            "mapped_kg": mapped_kg,
            "gold": shard_gold,
        }, f)
    
    if is_main_process():
        log.info(f"[Rank {rank}] wrote shard to {shard_path}")
    barrier()

    # ---------------- merge & evaluate (rank 0) ----------------
    if rank == 0:
        log.info("Merging shards and evaluating...")
        
        shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
        all_idx, all_kgt, all_kgc, all_gold, all_kggen = [], [], [], [], []
        
        for sp in shards:
            with open(sp, "rb") as f:
                D = pickle.load(f)
            all_idx.extend(D["idxs"])
            all_kgt.extend(D["kg_terms"])
            all_kgc.extend(D["mapped_kg"])
            all_gold.extend(D["gold"])
            all_kggen.extend(D["kg_texts"])
        
        order = np.argsort(np.array(all_idx))
        kg_terms_all = [all_kgt[i] for i in order]
        pred_kg_all = [all_kgc[i] for i in order]
        gold_all = [all_gold[i] for i in order]
        kggen_all = [all_kggen[i] for i in order]

        # Print first sample generation
        if len(kggen_all) > 0:
            log.info("\n" + "="*80)
            log.info("FIRST SAMPLE - GENERATION")
            log.info("="*80)
            log.info(_coerce_text(kggen_all[0]))
            log.info("="*80 + "\n")

        # Evaluate
        gold_eval = restrict_to(gold_all, labels_eval)
        kg_eval = restrict_to(pred_kg_all, labels_eval)
        
        Yt = multihot(gold_eval, labels_eval)
        Yk = multihot(kg_eval, labels_eval)
        
        m_kg = eval_pack(Yt, Yk)
        pm_kg = {}
        add_parent_metrics_full(pm_kg, gold_eval, kg_eval)
        m_kg.update(pm_kg)

        # Print sample predictions
        n_show = min(args.print_samples, len(kg_terms_all))
        log.info("\n" + "="*80)
        log.info(f"SAMPLE PREDICTIONS (first {n_show})")
        log.info("="*80)
        
        for i in range(n_show):
            Gs = set(gold_all[i])
            K = set(pred_kg_all[i])
            tp = len(Gs & K)
            fp = len(K - Gs)
            fn = len(Gs - K)
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0
            rr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fr = (2 * pr * rr) / (pr + rr) if (pr + rr) > 0 else 0
            
            log.info(f"\n[Sample {i+1}] hadm={test_df.iloc[i].get('hadm_id','')}")
            log.info(f"  GOLD codes: {', '.join(sorted(Gs)) if Gs else '(none)'}")
            log.info(f"  Generated terms:")
            for t in kg_terms_all[i]:
                log.info(f"    - {t}")
            log.info(f"  Mapped ICD-9: {', '.join(sorted(K)) if K else '(none)'}")
            log.info(f"  P/R/F1 = {pr:.3f}/{rr:.3f}/{fr:.3f}")

        log.info("="*80 + "\n")

        # Save metrics
        payload = {
            "label_space": "FULL" if args.labels_space == "full" else "HEAD",
            "num_samples": len(kg_terms_all),
            "metrics": m_kg,
        }
        
        os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
        with open(args.out_metrics, "w") as f:
            json.dump(payload, f, indent=2)
        
        log.info(f"Metrics saved to {args.out_metrics}")
        
        # Print metrics
        _pretty_print_metrics("FINAL METRICS", m_kg)

    cleanup_dist()
    return 0

if __name__ == "__main__":
    sys.exit(main())