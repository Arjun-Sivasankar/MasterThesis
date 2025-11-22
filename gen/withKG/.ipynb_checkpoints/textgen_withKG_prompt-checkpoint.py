#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextGen with KG hints in the prompt (adapter-only), evaluated RAW vs KG.

Pipeline:
1) Build two prompts per row:
   - RAW : [VISIT]+structured+NOTES + [TASK/FORMAT/OUTPUT]
   - KG  : same + [KG HINTS] (visit evidence → CUIs → 0/1/2-hop → candidate ICD-9),
           all under a token budget so it fits within total_input_budget - assistant_reserve.
2) Generate free-text diagnoses for RAW and KG with identical decoding.
3) Extract lines AFTER [OUTPUT] and treat them as free-text terms.
4) Map terms → ICD-9 via your ICDMapper.
5) Evaluate RAW vs KG on the SAME label space (micro/macro/samples; parent micro/macro/samples).
6) Print paired samples.

Notes:
- Your dataset stores ATC in the 'ndc' column; we use ATC map directly.
- We *show ICD-9 codes in the KG context* to bias the LM, but we still instruct
  the model to output plain-text diagnoses — mapping happens afterwards.
"""

import os, re, json, time, argparse, pickle, glob, sys
from typing import List, Dict, Set, Tuple
from collections import defaultdict

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
    serialize_structured_readable, serialize_notes,
    ICDMapper, to_list, format_icd9, is_valid_icd9,
    restrict_to, multihot, eval_pack, add_parent_macro_f1,
    get_icd9_parent
)

# ------------------------------- robust output parsing -------------------------------
_OUTPUT_RE = re.compile(r"\[OUTPUT\]", flags=re.IGNORECASE)

def _coerce_text(x) -> str:
    """Turn generation outputs into a single string (handles None/list/bytes)."""
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(_coerce_text(y) for y in x if y is not None)
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return str(x)

def extract_after_output(generation, n_max: int = 12) -> List[str]:
    """
    Take a model generation, find content AFTER the first [OUTPUT] tag,
    and return up to n_max non-empty lines (bullets/numbering stripped).
    If no [OUTPUT] is present, parse the whole text (best-effort).
    """
    s = _coerce_text(generation)
    s = s.replace("</s>", "").replace("<s>", "").strip()

    m = _OUTPUT_RE.search(s)
    block = s[m.end():] if m else s

    lines_out = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        # stop if a new tag starts (avoid model echoing)
        if line.startswith("[") and not re.match(r"^\[\s*OUTPUT\s*\]$", line, flags=re.I):
            break
        # strip common list markers / numbering
        line = re.sub(r"^(?:[-*]\s*|\d+\.\s*|\(\d+\)\s*)", "", line)
        if line:
            lines_out.append(line)
        if len(lines_out) >= n_max:
            break
    return lines_out

def _safe_extract_batch(texts, nmax, label):
    out = []
    misses = 0
    for t in texts:
        s = _coerce_text(t)
        if not _OUTPUT_RE.search(s):
            misses += 1
        out.append(extract_after_output(s, nmax))
    if misses:
        log.info(f"[{label}] generations missing [OUTPUT] tag: {misses}/{len(texts)} (parsed best-effort)")
    return out

# ------------------------------- ICD-9 helpers (safe parent) -------------------------------
def get_icd9_parent_safe(code: str) -> str:
    try:
        return get_icd9_parent(code)
    except Exception:
        c = (code or "").upper()
        if not c: return c
        if c[0].isdigit(): return c.split('.')[0][:3]
        if c[0] == 'V':    return c.split('.')[0][:3]
        if c[0] == 'E':    return c.split('.')[0][:4]
        return c

# ------------------------------- Parent metrics (full) -------------------------------
def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
    g = [[get_icd9_parent_safe(c) for c in lst] for lst in gold_lists]
    p = [[get_icd9_parent_safe(c) for c in lst] for lst in pred_lists]
    labels = sorted({x for lst in g for x in lst})
    Yg = multihot(g, labels); Yp = multihot(p, labels)
    metrics_dict.update({
        "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
        "recall_macro_parent":    float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent":        float(f1_score(Yg, Yp, average="macro", zero_division=0)),
        "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
        "recall_micro_parent":    float(recall_score(Yg, Yp, average="micro", zero_division=0)),
        "f1_micro_parent":        float(f1_score(Yg, Yp, average="micro", zero_division=0)),
        "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
        "recall_samples_parent":    float(recall_score(Yg, Yp, average="samples", zero_division=0)),
        "f1_samples_parent":        float(f1_score(Yg, Yp, average="samples", zero_division=0)),
    })
    return labels, Yg, Yp

# ------------------------------- Minimal dist utils -------------------------------
def maybe_init_dist():
    if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl", timeout=torch.distributed.elastic.utils.get_default_timeout())
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

# ------------------------------- Token budgeting -------------------------------
def count_tokens(tok, text: str) -> int:
    return int(tok(text, add_special_tokens=False, return_length=True)["length"][0])

def trim_to_token_budget(tok, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text: return ""
    if count_tokens(tok, text) <= max_tokens: return text
    lo, hi = 0, len(text); best = ""
    while lo <= hi:
        mid = (lo + hi)//2
        cand = text[:mid]
        if count_tokens(tok, cand) <= max_tokens:
            best = cand; lo = mid + 1
        else:
            hi = mid - 1
    return best

# ------------------------------- Evidence helpers -------------------------------
def _strip(x: str) -> str:
    return str(x).strip().upper()

def format_icd9_proc(code: str) -> str:
    """
    Normalize ICD-9-CM procedure codes.

    Examples:
      "PRO_5491"  -> "54.91"
      "pro-549"   -> "54.9"
      "54.91"     -> "54.91"
      "PRO_54"    -> "54"
    """
    s = str(code).strip().upper()

    # remove common "PRO_" prefix variants
    s = re.sub(r'^PRO[\s_:\-]*', '', s)

    # keep only digits and dots
    s = re.sub(r'[^0-9\.]', '', s)

    # if already dotted, lightly validate and return
    if '.' in s:
        # ensure at least two digits before the dot; leave rest as-is
        m = re.match(r'^(\d{2,})\.(\d+)$', s)
        if m:
            left, right = m.groups()
            # clamp right side to at most 2 digits for ICD-9-Proc convention
            return f"{left[:2]}.{right[:2]}"
        # if it's something like "5.4" -> pad/normalize to "54" or "54.x"
        s = re.sub(r'\.', '', s)  # fall through to re-dot logic

    # re-dot after first two digits
    digits = re.sub(r'\D', '', s)
    if len(digits) <= 2:
        return digits  # e.g., "54"
    # up to two digits after the dot (ICD-9-Proc usually 2.xx)
    return f"{digits[:2]}.{digits[2:4]}"

def visit_evidence_cuis(row: pd.Series,
                        icd9_proc_map: Dict[str, List[str]],
                        loinc_map: Dict[str, List[str]],
                        atc_map: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Returns:
      src2cuis: { "ATC:XXX": [...], "LNC:YYY":[...], "PROC:12.34":[...] }
      ev_union: set(CUIs)
    Notes:
      - ATC comes from 'ndc' column in your dataset (already ATC codes).
      - LOINC from 'lab_test'.
      - ICD-9 PROC from 'pro_code'.
    """
    src2cuis: Dict[str, List[str]] = {}
    ev_union: Set[str] = set()

    # ATC via 'ndc'
    for c in to_list(row.get("ndc", [])):
        key = _strip(c)
        if not key:
            continue
        cuis = atc_map.get(key, [])
        if cuis:
            src2cuis[f"ATC:{key}"] = cuis
            ev_union.update(cuis)

    # LOINC
    for c in to_list(row.get("lab_test", [])):
        key = _strip(c)
        if not key:
            continue
        cuis = loinc_map.get(key, [])
        if cuis:
            src2cuis[f"LNC:{key}"] = cuis
            ev_union.update(cuis)

    # ICD-9 PROC
    for c in to_list(row.get("pro_code", [])):
        cc = format_icd9_proc(c)
        if not cc:
            continue
        cuis = icd9_proc_map.get(cc, [])
        if cuis:
            src2cuis[f"PROC:{cc}"] = cuis
            ev_union.update(cuis)

    return src2cuis, ev_union

# ------------------------------- KG plumbing (neighbors + allowed codes) -------------------------------
def render_neighbors_block(G: nx.DiGraph,
                           ev_cuis: Set[str],
                           hop: int,
                           rel_whitelist: Set[str] = None,
                           rela_whitelist: Set[str] = None,
                           max_neighbors_show: int = 24) -> Tuple[Set[str], str]:
    """Return (expanded_cuis, neighbor_text_block) with 'u [name] --rel/rela--> v [name]' lines."""
    if hop <= 0 or not ev_cuis:
        return set(ev_cuis), ""
    seen=set(ev_cuis); frontier=set(ev_cuis); edges=[]
    for _ in range(hop):
        nxt=set()
        for u in frontier:
            if u not in G: continue
            for v in G.successors(u):
                d = G[u][v]
                rel  = (d.get("rel")  or "").strip()
                rela = (d.get("rela") or "").strip()
                if rel_whitelist  and rel  not in rel_whitelist:   continue
                if rela_whitelist and rela not in rela_whitelist:   continue
                if v in seen: continue
                if len(edges) < max_neighbors_show:
                    nmu = G.nodes[u].get("name","Unknown") if u in G else "Unknown"
                    nmv = G.nodes[v].get("name","Unknown") if v in G else "Unknown"
                    label = (rela if rela else rel)
                    edges.append(f"- {u} [{nmu}] --{label}--> {v} [{nmv}]")
                nxt.add(v)
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if not frontier: break
    if edges:
        block = f"Neighbors within {hop} hop(s):\n" + "\n".join(edges)
        return seen, block
    return seen, ""

def allowed_icd9_from_cuis(dx_map: Dict[str, List[str]], bag_cuis: Set[str]) -> List[str]:
    """Return ICD-9 diagnoses whose CUI list intersects bag_cuis (sorted)."""
    if not bag_cuis: return []
    allowed=[]
    for code, cuis in dx_map.items():
        if bag_cuis.intersection(cuis):
            c = format_icd9(code)
            if is_valid_icd9(c): allowed.append(c)
    return sorted(set(allowed))

def build_kg_hints_text(allowed_codes: List[str],
                        src2cuis: Dict[str, List[str]],
                        tok,
                        kg_soft_budget: int,
                        neighbors_block: str = None) -> str:
    """
    Compose [KG HINTS] with:
      • Evidence lines: each source → its CUIs (budgeted)
      • Optional neighbors block (already formatted)
      • Candidate ICD-9 list (context only)
    All clamped to kg_soft_budget tokens.
    """
    if kg_soft_budget <= 0:
        return ""

    sections = ["[KG HINTS]"]

    # Evidence CUIs (compact, budget-aware)
    if src2cuis:
        sections.append("Evidence CUIs linked from visit data:")
        def _sort_key(k: str):
            if k.startswith("ATC:"):  return (0, k)
            if k.startswith("PROC:"): return (1, k)
            if k.startswith("LNC:"):  return (2, k)
            return (3, k)
        for src in sorted(src2cuis.keys(), key=_sort_key):
            cuis = list(dict.fromkeys([str(x) for x in src2cuis[src]]))  # de-dup keep order
            head = "\n".join(sections) + "\n- " + src + " -> "
            kept = []
            for cu in cuis:
                trial = head + " ".join(kept + [cu])
                if count_tokens(tok, trial) <= kg_soft_budget:
                    kept.append(cu)
                else:
                    break
            if kept:
                sections.append(f"- {src} -> " + " ".join(kept))
            else:
                trial_line = head.strip()
                if count_tokens(tok, trial_line) <= kg_soft_budget:
                    sections.append(f"- {src} ->")
                else:
                    break

    hint_so_far = "\n".join(sections)

    # Try to include neighbors block
    if neighbors_block:
        trial = hint_so_far + "\n" + neighbors_block
        if count_tokens(tok, trial) <= kg_soft_budget:
            hint_so_far = trial
        else:
            nb_trim = trim_to_token_budget(tok, neighbors_block, max(0, kg_soft_budget - count_tokens(tok, hint_so_far) - 4))
            trial2 = hint_so_far + ("\n" + nb_trim if nb_trim else "")
            if count_tokens(tok, trial2) <= kg_soft_budget:
                hint_so_far = trial2

    # Candidate ICD-9 list (context only)
    cand_head = hint_so_far + ("\nCANDIDATE ICD9 (context only):\n")
    kept_codes = []
    for c in allowed_codes:
        trial = cand_head + " ".join(kept_codes + [c])
        if count_tokens(tok, trial) <= kg_soft_budget:
            kept_codes.append(c)
        else:
            break

    if kept_codes:
        return cand_head + " ".join(kept_codes)

    # If we couldn't fit any candidates, return what we have under budget
    return trim_to_token_budget(tok, hint_so_far, kg_soft_budget)

# ------------------------------- Prompt builders -------------------------------
def build_tail(N_max_terms:int) -> str:
    lines = [
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]"
    ]
    return "\n".join(lines)

def build_prompts_for_row(row: pd.Series,
                          tok,
                          kg_text: str,
                          notes_soft_budget: int,
                          N_max_terms:int) -> Tuple[str, str, dict]:
    header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
    full_notes  = serialize_notes(row)
    notes  = trim_to_token_budget(tok, full_notes, notes_soft_budget) if notes_soft_budget > 0 else ""
    tail   = build_tail(N_max_terms)

    raw_prompt = "\n".join([x for x in [header, notes, tail] if x])
    kg_prompt  = "\n".join([x for x in [header, notes, kg_text, tail] if x])

    dbg = {
        "header_tokens": count_tokens(tok, header),
        "notes_tokens":  count_tokens(tok, notes),
        "kg_tokens":     count_tokens(tok, kg_text) if kg_text else 0,
        "tail_tokens":   count_tokens(tok, tail),
        "total_raw":     count_tokens(tok, raw_prompt),
        "total_kg":      count_tokens(tok, kg_prompt),
    }
    return raw_prompt, kg_prompt, dbg

# ------------------------------- Generation (decode-to-text) -------------------------------
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
            no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram>0 else None,
        )
    if decoding == "beam":
        return dict(
            max_new_tokens=max_new,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram>0 else None,
        )
    # sample
    return dict(
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram>0 else None,
    )

@torch.no_grad()
def generate_texts(model, tok, prompts: List[str], max_len: int, gen_kwargs: dict, batch_size: int = 8, device=None) -> List[str]:
    device = device or (model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outs = []
    bs = max(1, int(batch_size))
    tok.padding_side = "left"  # ensure left padding for decoder-only
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False)
        for k in enc:
            enc[k] = enc[k].to(device)
        gen = model.generate(**enc, **{k:v for k,v in gen_kwargs.items() if v is not None})
        dec = tok.batch_decode(gen, skip_special_tokens=False)
        outs.extend(dec)
    return outs

# ------------------------------- Pretty printer -------------------------------
def _pretty_print_block(title: str, d: dict):
    log.info(f"--- {title} ---")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            log.info(f"{k:>28s}: {v:.6f}")
        else:
            log.info(f"{k:>28s}: {v}")

# ------------------------------- MAIN -------------------------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_only", action="store_true", help="Use entire file as test set (skip internal split)")
    ap.add_argument("--subset_n", type=int, default=0, help="Run only on first N rows (0=all)")
    ap.add_argument("--print_samples", type=int, default=5)

    # prompts/generation
    ap.add_argument("--N_max_terms", type=int, default=12)

    # token budgets
    ap.add_argument("--total_input_budget", type=int, default=3072)
    ap.add_argument("--assistant_reserve",  type=int, default=128)
    ap.add_argument("--notes_soft_budget",  type=int, default=2718)
    ap.add_argument("--kg_soft_budget",     type=int, default=226)
    ap.add_argument("--max_neighbors_show", type=int, default=24)

    # decoding
    ap.add_argument("--gen_max_new", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--decoding", choices=["greedy","beam","sample"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_repeat_ngram", type=int, default=0)

    # model/adapter (adapter-only)
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
    ap.add_argument("--labels_head_k", type=int, default=0)

    # optional lists
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")

    # KG inputs
    ap.add_argument("--kg_pkl", required=True)
    ap.add_argument("--icd9_dx_map_pkl",   required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl",     required=True)
    ap.add_argument("--atc_map_pkl",       required=True)
    ap.add_argument("--hop", type=int, default=1, choices=[0,1,2])
    ap.add_argument("--rel_whitelist",  default="")
    ap.add_argument("--rela_whitelist", default="")

    # distributed
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--tmp_dir", default="runs_textgen/test_shards")
    ap.add_argument("--out_metrics", default="runs_textgen/test_metrics_raw_vs_kg.json")

    args = ap.parse_args()

    # ---------------- data ----------------
    try:
        df = pd.read_pickle(args.data_pickle)
    except Exception:
        with open(args.data_pickle, "rb") as f: df = pickle.load(f)

    if args.test_only:
        test_df = df.copy()
    else:
        from sklearn.model_selection import train_test_split
        subs = df[args.subject_col].dropna().unique()
        _, te_subs = train_test_split(subs, test_size=0.10, random_state=args.seed)
        test_df  = df[df[args.subject_col].isin(te_subs)].copy()

    if args.subset_n and args.subset_n > 0:
        test_df = test_df.iloc[:args.subset_n].reset_index(drop=True)

    # gold
    def extract_codes(df, label_col):
        out=[]
        for _, r in df.iterrows():
            lst = to_list(r.get(label_col, []))
            lst = [format_icd9(c) for c in lst if c]
            lst = [c for c in lst if is_valid_icd9(c)]
            out.append(lst)
        return out
    gold_codes = extract_codes(test_df, args.label_col)

    # label space
    labels_full = sorted({c for lst in gold_codes for c in lst})
    labels_eval = labels_full
    head_name = None
    if args.labels_space == "head" and args.labels_head_k > 0:
        from collections import Counter
        cnt = Counter([c for lst in gold_codes for c in lst])
        labels_eval = [c for c,_ in cnt.most_common(args.labels_head_k)]
        head_name = f"HEAD_{args.labels_head_k}"

    if is_main_process():
        log.info(f"Test size: {len(test_df)}")
        log.info(f"Eval label space: {len(labels_eval)} codes ({'FULL' if head_name is None else head_name})")

    # ---------------- model (adapter-only, LEFT padding) ----------------
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) \
           else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # important for decoder-only

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.config.use_cache = True
    dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    # ---------------- KG + maps ----------------
    KG  = pickle.load(open(args.kg_pkl, "rb"))
    icd9_dx_map   = pickle.load(open(args.icd9_dx_map_pkl,   "rb"))
    icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
    loinc_map     = pickle.load(open(args.loinc_map_pkl,     "rb"))
    atc_map       = pickle.load(open(args.atc_map_pkl,       "rb"))

    rel_w  = {s.strip() for s in args.rel_whitelist.split(",")  if s.strip()} or None
    rela_w = {s.strip() for s in args.rela_whitelist.split(",") if s.strip()} or None

    # ---------------- build prompts (RAW & KG), token-budgeted ----------------
    print("Building RAW & KG prompts...")
    raw_prompts=[]; kg_prompts=[]; dbg_rows=[]
    max_prompt = max(1, args.total_input_budget - args.assistant_reserve)

    for i, row in test_df.iterrows():
        # evidence → CUIs (per-source) and union set
        src2cuis, ev_cuis = visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
        # neighbors & expansion
        expanded_cuis, neighbors_block = render_neighbors_block(
            KG, ev_cuis, hop=args.hop,
            rel_whitelist=rel_w, rela_whitelist=rela_w,
            max_neighbors_show=args.max_neighbors_show
        )
        bag_cuis = expanded_cuis
        allowed  = allowed_icd9_from_cuis(icd9_dx_map, bag_cuis)

        kg_text  = build_kg_hints_text(
            allowed, src2cuis, tok,
            kg_soft_budget=args.kg_soft_budget,
            neighbors_block=neighbors_block
        )

        raw_p, kg_p, d = build_prompts_for_row(row, tok, kg_text, args.notes_soft_budget, args.N_max_terms)

        # final clamp
        if d["total_raw"] > max_prompt:
            over = d["total_raw"] - max_prompt
            new_notes = max(0, d["notes_tokens"] - over - 8)
            notes_trim = trim_to_token_budget(tok, serialize_notes(row), new_notes)
            header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
            tail   = build_tail(args.N_max_terms)
            raw_p = "\n".join([x for x in [header, notes_trim, tail] if x])
            d["total_raw"] = count_tokens(tok, raw_p)
        if d["total_kg"] > max_prompt:
            # try shrinking KG block first
            shrink = max(0, args.kg_soft_budget - (d["total_kg"] - max_prompt))
            kg_text2 = trim_to_token_budget(tok, kg_text, shrink)
            raw_p2, kg_p2, d2 = build_prompts_for_row(row, tok, kg_text2, args.notes_soft_budget, args.N_max_terms)
            if count_tokens(tok, kg_p2) <= max_prompt:
                kg_p, d = kg_p2, d2
            else:
                kg_p = raw_p
                d["kg_tokens"] = 0
                d["total_kg"]  = d["total_raw"]

        raw_prompts.append(raw_p)
        kg_prompts.append(kg_p)
        dbg_rows.append({
            "idx": i,
            "hadm_id": row.get("hadm_id",""),
            "ev_cuis": len(ev_cuis),
            "exp_cuis": len(bag_cuis),
            "allowed_icd9": len(allowed),
            **d
        })

    print("Done building prompts....")

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
        idxs = shard_indices(len(raw_prompts), rank, W)
    else:
        rank, W = 0, 1
        idxs = list(range(len(raw_prompts)))

    shard_raw = [raw_prompts[i] for i in idxs]
    shard_kg  = [kg_prompts[i]  for i in idxs]
    shard_gold= [gold_codes[i]  for i in idxs]

    # ---------------- generation -> decoded text -> parsed terms ----------------
    t0 = time.time()
    raw_out_texts = generate_texts(model, tok, shard_raw, max_len=args.total_input_budget, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)
    kg_out_texts  = generate_texts(model, tok, shard_kg,  max_len=args.total_input_budget, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)

    raw_terms = _safe_extract_batch(raw_out_texts, args.N_max_terms, "RAW")
    kg_terms  = _safe_extract_batch(kg_out_texts,  args.N_max_terms, "KG")

    if is_main_process():
        per = (time.time()-t0)/max(1,len(idxs))
        log.info(f"Generation done ({per:.2f}s/sample on rank {rank}).")

    # ---------------- mapping terms -> ICD-9 ----------------
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    mapped_raw = mapper.map_terms(raw_terms)
    mapped_kg  = mapper.map_terms(kg_terms)

    # ---------------- persist shard ----------------
    os.makedirs(args.tmp_dir, exist_ok=True)
    shard_path = os.path.join(args.tmp_dir, f"shard_{rank:03d}_of_{W:03d}.pkl")
    with open(shard_path, "wb") as f:
        pickle.dump({
            "idxs": idxs,
            "raw_texts": raw_out_texts,
            "kg_texts":  kg_out_texts,
            "raw_terms": raw_terms,
            "kg_terms":  kg_terms,
            "mapped_raw": mapped_raw,
            "mapped_kg":  mapped_kg,
            "gold": shard_gold,
            "dbg": dbg_rows,
        }, f)
    log.info(f"[Rank {rank}] wrote shard to {shard_path}")
    barrier()

    # ---------------- merge & evaluate (rank 0) ----------------
    if rank == 0:
        shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
        all_idx, all_rawt, all_kgt, all_rawc, all_kgc, all_gold = [], [], [], [], [], []
        for sp in shards:
            with open(sp, "rb") as f:
                D = pickle.load(f)
            all_idx.extend(D["idxs"])
            all_rawt.extend(D["raw_terms"]); all_kgt.extend(D["kg_terms"])
            all_rawc.extend(D["mapped_raw"]); all_kgc.extend(D["mapped_kg"])
            all_gold.extend(D["gold"])

        order = np.argsort(np.array(all_idx))
        raw_terms_all = [all_rawt[i] for i in order]
        kg_terms_all  = [all_kgt[i]  for i in order]
        pred_raw_all  = [all_rawc[i] for i in order]
        pred_kg_all   = [all_kgc[i]  for i in order]
        gold_all      = [all_gold[i] for i in order]

        # restrict to evaluation space
        gold_eval = restrict_to(gold_all, labels_eval)
        raw_eval  = restrict_to(pred_raw_all, labels_eval)
        kg_eval   = restrict_to(pred_kg_all,  labels_eval)

        # metrics
        Yt = multihot(gold_eval, labels_eval)
        Yr = multihot(raw_eval,  labels_eval)
        Yk = multihot(kg_eval,   labels_eval)

        m_raw = eval_pack(Yt, Yr)
        pm_raw = {}
        _ = add_parent_metrics_full(pm_raw, gold_eval, raw_eval)
        add_parent_macro_f1(pm_raw, gold_eval, raw_eval)
        m_raw.update(pm_raw)

        m_kg = eval_pack(Yt, Yk)
        pm_kg = {}
        _ = add_parent_metrics_full(pm_kg, gold_eval, kg_eval)
        add_parent_macro_f1(pm_kg, gold_eval, kg_eval)
        m_kg.update(pm_kg)

        # sample-level set PRF
        def sample_set_prf(gold, pred):
            vals=[]
            for g, p in zip(gold, pred):
                G, P = set(g), set(p)
                tp = len(G & P); fp = len(P - G); fn = len(G - P)
                prec = tp / (tp+fp) if tp+fp>0 else 0.0
                rec  = tp / (tp+fn) if tp+fn>0 else 0.0
                f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
                vals.append((prec, rec, f1))
            arr = np.array(vals) if vals else np.zeros((0,3))
            return float(arr[:,0].mean() if arr.size else 0.0), float(arr[:,1].mean() if arr.size else 0.0), float(arr[:,2].mean() if arr.size else 0.0)

        ps, rs, fs = sample_set_prf(gold_eval, raw_eval)
        m_raw["precision_samples_set"] = ps
        m_raw["recall_samples_set"] = rs
        m_raw["f1_samples_set"] = fs

        ps, rs, fs = sample_set_prf(gold_eval, kg_eval)
        m_kg["precision_samples_set"] = ps
        m_kg["recall_samples_set"] = rs
        m_kg["f1_samples_set"] = fs

        # per-label CSV for FULL
        out_dir = os.path.dirname(os.path.abspath(args.out_metrics))
        os.makedirs(out_dir, exist_ok=True)
        def per_label_table(y_true, y_pred, labels, out_csv_path=None):
            p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            df = pd.DataFrame({"code": labels,"precision": p,"recall": r,"f1": f1,"support": support})
            if out_csv_path:
                df.to_csv(out_csv_path, index=False)
            return df
        per_label_table(Yt, Yr, labels_eval, os.path.join(out_dir, "per_label_FULL_RAW.csv"))
        per_label_table(Yt, Yk, labels_eval, os.path.join(out_dir, "per_label_FULL_KG.csv"))

        # optional TOP/BOTTOM/PARENTS CSVs
        def _read_first_col_codes(path) -> list:
            if not path: return []
            try:
                df = pd.read_csv(path)
                if df.shape[1] == 0: return []
                col = df.columns[0]
                vals = [format_icd9(x) for x in df[col].tolist()]
                return sorted(set([v for v in vals if is_valid_icd9(v)]))
            except Exception:
                return []

        def _read_first_col_parents(path) -> list:
            if not path: return []
            try:
                df = pd.read_csv(path)
                if df.shape[1] == 0: return []
                col = df.columns[0]
                raw = [format_icd9(x) for x in df[col].tolist()]
                return sorted(set([get_icd9_parent_safe(x) for x in raw if x]))
            except Exception:
                return []

        results_ext = {}

        top_codes = _read_first_col_codes(args.top_codes_csv)
        bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
        top_parents = _read_first_col_parents(args.top_parent_csv)

        if top_codes:
            g = restrict_to(gold_all, top_codes)
            r = restrict_to(pred_raw_all, top_codes)
            k = restrict_to(pred_kg_all,  top_codes)
            Yg = multihot(g, top_codes)
            Yr2= multihot(r, top_codes)
            Yk2= multihot(k, top_codes)
            results_ext["TOP_50_CODES_RAW"] = eval_pack(Yg, Yr2)
            results_ext["TOP_50_CODES_KG"]  = eval_pack(Yg, Yk2)
            per_label_table(Yg, Yr2, top_codes, os.path.join(out_dir, "per_label_TOP_50_CODES_RAW.csv"))
            per_label_table(Yg, Yk2, top_codes, os.path.join(out_dir, "per_label_TOP_50_CODES_KG.csv"))

        if bottom_codes:
            g = restrict_to(gold_all, bottom_codes)
            r = restrict_to(pred_raw_all, bottom_codes)
            k = restrict_to(pred_kg_all,  bottom_codes)
            Yg = multihot(g, bottom_codes)
            Yr2= multihot(r, bottom_codes)
            Yk2= multihot(k, bottom_codes)
            results_ext["BOTTOM_50_CODES_RAW"] = eval_pack(Yg, Yr2)
            results_ext["BOTTOM_50_CODES_KG"]  = eval_pack(Yg, Yk2)
            per_label_table(Yg, Yr2, bottom_codes, os.path.join(out_dir, "per_label_BOTTOM_50_CODES_RAW.csv"))
            per_label_table(Yg, Yk2, bottom_codes, os.path.join(out_dir, "per_label_BOTTOM_50_CODES_KG.csv"))

        if top_parents:
            g_par = [[get_icd9_parent_safe(c) for c in lst] for lst in gold_all]
            r_par = [[get_icd9_parent_safe(c) for c in lst] for lst in pred_raw_all]
            k_par = [[get_icd9_parent_safe(c) for c in lst] for lst in pred_kg_all]
            g_par = restrict_to(g_par, top_parents)
            r_par = restrict_to(r_par, top_parents)
            k_par = restrict_to(k_par, top_parents)
            YgP = multihot(g_par, top_parents)
            YrP = multihot(r_par, top_parents)
            YkP = multihot(k_par, top_parents)
            results_ext["TOP_50_PARENTS_RAW"] = eval_pack(YgP, YrP)
            results_ext["TOP_50_PARENTS_KG"]  = eval_pack(YgP, YkP)
            per_label_table(YgP, YrP, top_parents, os.path.join(out_dir, "per_label_TOP_50_PARENTS_RAW.csv"))
            per_label_table(YgP, YkP, top_parents, os.path.join(out_dir, "per_label_TOP_50_PARENTS_KG.csv"))

        # print side-by-side samples
        n_show = min(args.print_samples, len(raw_terms_all))
        log.info("=== Sample predictions (RAW vs KG) ===")
        for i in range(n_show):
            G = set(gold_all[i]); R = set(pred_raw_all[i]); K = set(pred_kg_all[i])
            # sample PRF (RAW)
            tp = len(G & R); fp = len(R - G); fn = len(G - R)
            pr = tp/(tp+fp) if tp+fp>0 else 0.0
            rr = tp/(tp+fn) if tp+fn>0 else 0.0
            fr = (2*pr*rr)/(pr+rr) if pr+rr>0 else 0.0
            # sample PRF (KG)
            tp2 = len(G & K); fp2 = len(K - G); fn2 = len(G - K)
            pr2 = tp2/(tp2+fp2) if tp2+fp2>0 else 0.0
            rr2 = tp2/(tp2+fn2) if tp2+fn2>0 else 0.0
            fr2 = (2*pr2*rr2)/(pr2+rr2) if pr2+rr2>0 else 0.0

            log.info(f"[Sample {i+1}] hadm={test_df.iloc[i].get('hadm_id','')}")
            log.info(f"  GOLD codes: {', '.join(sorted(G)) if G else '(none)'}")
            log.info(  "  RAW free-text terms:")
            for t in raw_terms_all[i][:args.N_max_terms]:
                log.info(f"    - {t}")
            log.info(f"  RAW mapped ICD-9: {', '.join(sorted(R)) if R else '(none)'}  | P/R/F1 = {pr:.3f}/{rr:.3f}/{fr:.3f}")
            log.info(  "  KG  free-text terms:")
            for t in kg_terms_all[i][:args.N_max_terms]:
                log.info(f"    - {t}")
            log.info(f"  KG  mapped ICD-9: {', '.join(sorted(K)) if K else '(none)'}  | P/R/F1 = {pr2:.3f}/{rr2:.3f}/{fr2:.3f}")

        # save metrics JSON
        payload = {
            "label_space": ("FULL" if head_name is None else head_name),
            "num_samples": len(raw_terms_all),
            "metrics_raw": m_raw,
            "metrics_kg":  m_kg,
            **results_ext
        }
        with open(args.out_metrics, "w") as f:
            json.dump(payload, f, indent=2)
        log.info(f"Metrics saved to {args.out_metrics}")

        _pretty_print_block("OVERALL RAW (code-level)", m_raw)
        _pretty_print_block("OVERALL KG  (code-level)", m_kg)
        for k,v in results_ext.items():
            _pretty_print_block(k, v)

    cleanup_dist()
    return 0

if __name__ == "__main__":
    sys.exit(main())
