#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- RAW prompt:   [VISIT] + structured + NOTES + [TASK/CODES]
- KG  prompt:   RAW + [KG HINTS] block that shows:
    * Evidence CUIs by source (ATC/PROC/LOINC)
    * Neighbor edges up to K hops: "CUIu [NameU] --REL/RELA--> CUIv [NameV]"
    * Candidate ICD-9 codes suggested by KG (context only)
- Budgets (defaults from your token study): NOTES≈2307, KG≈637, assistant reserve=128,
  total input budget=3072 → prompt budget≈2944.

The model is expected to output ICD-9 codes (space-separated). We parse/validate those.
"""

import os, re, json, time, argparse, pickle
from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd
import torch
import networkx as nx

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.utils import logging as hf_logging
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------- Env & logging ----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()


# ---------------- ICD9 helpers ----------------
def format_icd9(code: str) -> str:
    code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
    if not code: return ""
    if code[0].isdigit():  # 3.xx
        return (code[:3]+"."+code[3:]) if (len(code)>3 and "." not in code) else code
    if code[0] == "V":
        return (code[:3]+"."+code[3:]) if (len(code)>3 and "." not in code) else code
    if code[0] == "E":
        return (code[:4]+"."+code[4:]) if (len(code)>4 and "." not in code) else code
    return code

def is_valid_icd9(code: str) -> bool:
    if not code: return False
    if code[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", code))
    if code.startswith('V'): return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", code))
    if code.startswith('E'): return bool(re.match(r"^E\d{3}(\.\d{1})?$", code))
    return False

def get_icd9_parent(code: str) -> str:
    if not code: return code
    if code[0].isdigit(): return code.split('.')[0][:3]
    if code[0]=='V':     return code.split('.')[0][:3]
    if code[0]=='E':     return code.split('.')[0][:4]
    return code


# ---------------- text utils ------------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x) -> str:
    if x is None: return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def to_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        try: it = list(x)
        except Exception: it = [x]
    else:
        s = str(x).strip()
        if not s or s.lower() in ("nan","none","null"): return []
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                it = list(ast.literal_eval(s))
            except Exception:
                it = re.split(r"[,\s]+", s)
        else:
            it = re.split(r"[,\s]+", s)
    out=[]
    for v in it:
        if v is None: continue
        if isinstance(v, float) and np.isnan(v): continue
        sv = str(v).strip()
        if sv: out.append(sv)
    return out


# ---------------- model loading ---------------
def load_model_and_tokenizer(base_model: str, adapter_dir: str):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # IMPORTANT for decoder-only generation

    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = True

    model = PeftModel.from_pretrained(base, adapter_dir)
    return model, tok


# ---------------- token budgeting -------------
def count_tokens(tok, text: str) -> int:
    return int(tok(text, add_special_tokens=False, return_length=True)["length"][0])

def trim_to_token_budget(tok, text: str, max_tokens: int) -> str:
    """Binary-search by characters until token count <= max_tokens."""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(tok, text) <= max_tokens:
        return text
    lo, hi = 0, len(text); best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid]
        if count_tokens(tok, cand) <= max_tokens:
            best = cand; lo = mid + 1
        else:
            hi = mid - 1
    return best


# ---------------- KG plumbing -----------------
def build_visit_evidence_cuis(row: pd.Series,
                              icd9_proc_map: Dict[str, List[str]],
                              loinc_map: Dict[str, List[str]],
                              atc_map: Dict[str, List[str]]) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Use visit columns to collect evidence CUIs:
      - 'pro_code' -> ICD9 PROC map
      - 'lab_test' -> LOINC map
      - 'ndc'      -> ATC map (dataset contains ATC here)
    Returns (CUIs set, raw_evidence_dict for display).
    """
    ev_cuis = set()
    evid_raw = {"PROC":[],"LOINC":[],"ATC":[]}

    for c in to_list(row.get("pro_code", [])):
        evid_raw["PROC"].append(c)
        cc = format_icd9(c)
        ev_cuis.update(icd9_proc_map.get(cc, []))

    for c in to_list(row.get("lab_test", [])):
        evid_raw["LOINC"].append(c)
        ev_cuis.update(loinc_map.get(str(c).strip().upper(), []))

    # ATC via 'ndc'
    for c in to_list(row.get("ndc", [])):
        evid_raw["ATC"].append(c)
        ev_cuis.update(atc_map.get(str(c).strip().upper(), []))

    return ev_cuis, evid_raw


def expand_k_hops_edges(G: nx.DiGraph,
                        sources: Set[str],
                        k: int,
                        rel_whitelist: Set[str] = None,
                        rela_whitelist: Set[str] = None,
                        max_edges: int = 24) -> Tuple[Set[str], List[Tuple[str,str,str]]]:
    """
    Return (neighbor_set, edge_tuples) where edge_tuples are (u, reltxt, v).
    reltxt prefers 'rela' when present, else 'rel'.
    """
    if k <= 0 or not sources:
        return set(), []

    seen = set(sources)
    frontier = set(sources)
    all_new = set()
    edges: List[Tuple[str,str,str]] = []

    for _ in range(k):
        nxt=set()
        for u in frontier:
            if u not in G: continue
            for v in G.successors(u):
                d = G[u][v]
                rel  = (d.get('rel')  or '').strip()
                rela = (d.get('rela') or '').strip()
                if rel_whitelist  and rel  not in rel_whitelist:  continue
                if rela_whitelist and rela not in rela_whitelist: continue
                if v in seen: continue
                if len(edges) < max_edges:
                    edges.append((u, (rela if rela else rel), v))
                nxt.add(v)
        nxt -= seen
        seen |= nxt
        all_new |= nxt
        if not nxt:
            break
        frontier = nxt

    return all_new, edges


def allowed_icd9_from_cuis(dx_map: Dict[str, List[str]], bag_cuis: Set[str]) -> List[str]:
    """Return ICD-9 diagnoses whose CUI list intersects bag_cuis (sorted)."""
    allowed=[]
    if not bag_cuis:
        return allowed
    for code, cuis in dx_map.items():
        if bag_cuis.intersection(cuis):
            c = format_icd9(code)
            if is_valid_icd9(c):
                allowed.append(c)
    return sorted(set(allowed))


# --------------- prompt building --------------
def serialize_structured(row: pd.Series) -> str:
    parts=[]
    parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
                 f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
    atc = to_list(row.get("ndc", []))     # ATC codes live here in your data
    proc= to_list(row.get("pro_code", []))
    labs= to_list(row.get("lab_test", []))
    if atc:  parts.append("[ATC] "  + " ".join(atc[:32]))
    if proc: parts.append("[PROC] " + " ".join(proc[:32]))
    if labs: parts.append("[LAB] "  + " ".join(labs[:64]))
    return "\n".join(parts)

def serialize_notes(row: pd.Series) -> str:
    chunks=[]
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t: chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)

def build_tail_codes() -> str:
    return "\n".join([
        "[TASK] You are a medical coding expert. Based on the patient information above, generate the appropriate ICD-9-CM diagnosis codes. Follow these guidelines:",
        "1. List only the ICD-9 codes separated by spaces",
        "2. Use proper ICD-9 format with decimal points (e.g., 250.00 not 25000)",
        "3. Include only codes directly supported by the clinical information",
        "4. Do not include any explanations or text besides the codes themselves",
        "[CODES]"
    ])

def build_kg_hints_text_with_neighbors(
    tok,
    KG: nx.DiGraph,
    ev_cuis: Set[str],
    evid_raw: Dict[str, List[str]],
    hop: int,
    rel_whitelist: Set[str],
    rela_whitelist: Set[str],
    max_neighbors_show: int,
    allowed_codes: List[str],
    kg_soft_budget: int
) -> str:
    """Compose the rich [KG HINTS] block within the token budget."""
    if kg_soft_budget <= 0:
        return ""

    lines = ["[KG HINTS]"]

    # Evidence CUIs: compact but explicit
    if evid_raw.get("ATC"):
        lines.append("Evidence CUIs linked from visit data:")
        # For each source show a few CUIs (if available) from maps
        for src in ["ATC","PROC","LOINC"]:
            vals = evid_raw.get(src, [])
            if not vals: continue
            # we don't have per-code CUI list here (already baked into ev_cuis), so just show the codes
            codes_preview = " ".join([str(v) for v in vals[:10]])
            lines.append(f"- {src}: {codes_preview}")
    else:
        lines.append("Evidence CUIs linked from visit data: (none)")

    # Neighbors: up to K hops, show up to max_neighbors_show edges
    neighbor_set, edges = expand_k_hops_edges(
        KG, ev_cuis, k=hop,
        rel_whitelist=rel_whitelist, rela_whitelist=rela_whitelist,
        max_edges=max_neighbors_show
    )
    if edges:
        lines.append(f"Neighbors within {hop} hop(s):")
        for (u, reltxt, v) in edges[:max_neighbors_show]:
            nmu = KG.nodes[u].get("name","Unknown") if u in KG else "Unknown"
            nmv = KG.nodes[v].get("name","Unknown") if v in KG else "Unknown"
            lines.append(f"- {u} [{nmu}] --{reltxt}--> {v} [{nmv}]")
    else:
        lines.append(f"Neighbors within {hop} hop(s): (none)")

    # Candidate ICD-9 (context only)
    if allowed_codes:
        lines.append("Candidate ICD-9-CM diagnosis codes suggested by KG (optional):")
        head = "\n".join(lines) + "\n  "
        kept=[]
        for c in allowed_codes:
            trial = head + " ".join(kept + [c])
            if count_tokens(tok, trial) <= kg_soft_budget:
                kept.append(c)
            else:
                break
        if kept:
            lines.append("  " + " ".join(kept))
    else:
        lines.append("Candidate ICD-9-CM diagnosis codes suggested by KG: (none)")

    text = "\n".join(lines)
    # final strict clamp (just in case)
    return trim_to_token_budget(tok, text, kg_soft_budget)


def build_prompts_for_row(row: pd.Series,
                          tok,
                          kg_text: str,
                          notes_soft_budget: int) -> Tuple[str, str, dict]:
    """
    Returns (raw_prompt, kg_prompt, debug_info).
    """
    header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured(row)
    tail   = build_tail_codes()
    notes  = serialize_notes(row)
    notes  = trim_to_token_budget(tok, notes, notes_soft_budget) if notes_soft_budget > 0 else ""

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


# --------------- generation & eval -------------
@torch.no_grad()
def generate_codes(model, tok, prompts: List[str],
                   allowed_vocab: Set[str],
                   max_len: int,
                   gen_max_new: int,
                   batch_size: int) -> List[List[str]]:
    model.eval()
    device = next(model.parameters()).device
    preds = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.amp.autocast('cuda', enabled=(device.type=='cuda' and torch.cuda.get_device_capability(0)[0] >= 8)):
            out = model.generate(
                **enc,
                max_new_tokens=gen_max_new,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=2,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True
            )
        seq = out.sequences
        gen_only = seq[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)

        for t in texts:
            tokens = re.split(r"[^A-Za-z0-9\.]+", t)
            cand = [format_icd9(z) for z in tokens if z]
            keep=[]
            seen=set()
            for c in cand:
                if not is_valid_icd9(c): continue
                if allowed_vocab and (c not in allowed_vocab): continue
                if c in seen: continue
                seen.add(c); keep.append(c)
            preds.append(keep)

        if device.type == "cuda" and (i//batch_size) % 10 == 0:
            torch.cuda.empty_cache()

    return preds


def codes_to_multihot(code_lists: List[List[str]], label_vocab: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(label_vocab)}
    Y = np.zeros((len(code_lists), len(label_vocab)), dtype=np.int32)
    for i, lst in enumerate(code_lists):
        for c in lst:
            j = idx.get(c)
            if j is not None: Y[i, j] = 1
    return Y


def eval_sets(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_precision":   float(precision_score(y_true, y_pred, average="micro",   zero_division=0)),
        "micro_recall":      float(recall_score(y_true, y_pred, average="micro",      zero_division=0)),
        "micro_f1":          float(f1_score(y_true, y_pred, average="micro",          zero_division=0)),
        "macro_precision":   float(precision_score(y_true, y_pred, average="macro",   zero_division=0)),
        "macro_recall":      float(recall_score(y_true, y_pred, average="macro",      zero_division=0)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro",          zero_division=0)),
        "samples_precision": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "samples_recall":    float(recall_score(y_true, y_pred, average="samples",    zero_division=0)),
        "samples_f1":        float(f1_score(y_true, y_pred, average="samples",        zero_division=0)),
    }


def hierarchical_parent_recall(y_true: np.ndarray, y_pred: np.ndarray, label_vocab: List[str]) -> float:
    code_to_parent = {code: get_icd9_parent(code) for code in label_vocab}
    n = y_true.shape[0]
    hits = 0
    denom = 0
    for i in range(n):
        pred_par = {code_to_parent[label_vocab[j]] for j in np.where(y_pred[i])[0]}
        true_par = {code_to_parent[label_vocab[j]] for j in np.where(y_true[i])[0]}
        hits += len(pred_par & true_par)
        denom += len(true_par)
    return float(hits/denom) if denom>0 else 0.0


# ---------------- main ------------------------
def main():
    ap = argparse.ArgumentParser("KG-in-prompt codegen (adapter-only, rich hints)")
    # data
    ap.add_argument("--test_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--subset_n", type=int, default=0, help=">0 to run on first N rows for quick tests")
    ap.add_argument("--show_n", type=int, default=5)

    # model (adapter-only)
    ap.add_argument("--base_model", required=True, help="Base LM dir/name")
    ap.add_argument("--adapter_dir", required=True, help="LoRA adapter directory")

    # KG & maps
    ap.add_argument("--kg_pkl", required=True)
    ap.add_argument("--icd9_dx_map_pkl",   required=True)
    ap.add_argument("--icd9_proc_map_pkl", required=True)
    ap.add_argument("--loinc_map_pkl",     required=True)
    ap.add_argument("--atc_map_pkl",       required=True)

    # KG expansion
    ap.add_argument("--hop", type=int, default=1, choices=[0,1,2])
    ap.add_argument("--rel_whitelist",  default="", help="comma-separated REL filter (empty=all)")
    ap.add_argument("--rela_whitelist", default="", help="comma-separated RELA filter (empty=all)")
    ap.add_argument("--max_neighbors_show", type=int, default=24)

    # token budgets (new p95 split)
    ap.add_argument("--total_input_budget", type=int, default=3072)
    ap.add_argument("--assistant_reserve",  type=int, default=128)
    ap.add_argument("--notes_soft_budget",  type=int, default=2307)
    ap.add_argument("--kg_soft_budget",     type=int, default=637)

    # generation
    ap.add_argument("--gen_max_new", type=int, default=96)
    ap.add_argument("--batch_size",  type=int, default=16)

    args = ap.parse_args()

    # data
    df = pd.read_pickle(args.test_pickle)
    if args.subset_n and args.subset_n > 0:
        df = df.iloc[:args.subset_n].reset_index(drop=True)

    # gold & label vocab
    gold=[]
    for codes in df[args.label_col].tolist():
        lst = [format_icd9(c) for c in to_list(codes)]
        lst = [c for c in lst if is_valid_icd9(c)]
        gold.append(sorted(set(lst)))
    label_vocab = sorted({c for lst in gold for c in lst})
    label_set   = set(label_vocab)

    # model
    model, tok = load_model_and_tokenizer(args.base_model, args.adapter_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # KG & maps
    KG: nx.DiGraph = pickle.load(open(args.kg_pkl, "rb"))
    icd9_dx_map   = pickle.load(open(args.icd9_dx_map_pkl,   "rb"))
    icd9_proc_map = pickle.load(open(args.icd9_proc_map_pkl, "rb"))
    loinc_map     = pickle.load(open(args.loinc_map_pkl,     "rb"))
    atc_map       = pickle.load(open(args.atc_map_pkl,       "rb"))

    rel_w  = {s.strip() for s in args.rel_whitelist.split(",")  if s.strip()} or None
    rela_w = {s.strip() for s in args.rela_whitelist.split(",") if s.strip()} or None

    # prompts
    raw_prompts=[]; kg_prompts=[]; dbg_rows=[]
    max_prompt = max(1, args.total_input_budget - args.assistant_reserve)

    for i, row in df.iterrows():
        ev_cuis, evid_raw = build_visit_evidence_cuis(row, icd9_proc_map, loinc_map, atc_map)
        neighbors, _ = expand_k_hops_edges(KG, ev_cuis, k=args.hop, rel_whitelist=rel_w, rela_whitelist=rela_w, max_edges=args.max_neighbors_show)
        bag_cuis = set(ev_cuis) | set(neighbors)
        allowed  = allowed_icd9_from_cuis(icd9_dx_map, bag_cuis)

        kg_text  = build_kg_hints_text_with_neighbors(
            tok, KG, ev_cuis, evid_raw, args.hop, rel_w, rela_w,
            args.max_neighbors_show, allowed, args.kg_soft_budget
        )

        raw_p, kg_p, d = build_prompts_for_row(row, tok, kg_text, args.notes_soft_budget)

        # ---- KG-first final clamp ----
        if d["total_kg"] > max_prompt:
            over = d["total_kg"] - max_prompt
            # try to reduce NOTES first
            new_notes_budget = max(128, d["notes_tokens"] - over - 8)  # keep a small minimum
            notes_trimmed = trim_to_token_budget(tok, serialize_notes(row), new_notes_budget)
            header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured(row)
            tail   = build_tail_codes()
            kg_p_alt = "\n".join([x for x in [header, notes_trimmed, kg_text, tail] if x])
            if count_tokens(tok, kg_p_alt) <= max_prompt:
                # accept trimmed-notes version
                kg_p = kg_p_alt
                d["notes_tokens"] = count_tokens(tok, notes_trimmed)
                d["total_kg"] = count_tokens(tok, kg_p)
            else:
                # then try shrinking KG block
                shrink = max(0, args.kg_soft_budget - (count_tokens(tok, kg_p_alt) - max_prompt))
                kg_text2 = trim_to_token_budget(tok, kg_text, shrink)
                kg_p_alt2 = "\n".join([x for x in [header, notes_trimmed, kg_text2, tail] if x])
                if count_tokens(tok, kg_p_alt2) <= max_prompt:
                    kg_p = kg_p_alt2
                    d["kg_tokens"] = count_tokens(tok, kg_text2)
                    d["notes_tokens"] = count_tokens(tok, notes_trimmed)
                    d["total_kg"] = count_tokens(tok, kg_p)
                else:
                    # last resort: drop KG entirely
                    kg_p = "\n".join([x for x in [header, notes_trimmed, tail] if x])
                    d["kg_tokens"] = 0
                    d["total_kg"]  = count_tokens(tok, kg_p)

        if d["total_raw"] > max_prompt:
            over = d["total_raw"] - max_prompt
            new_notes_budget = max(128, d["notes_tokens"] - over - 8)
            notes_trimmed = trim_to_token_budget(tok, serialize_notes(row), new_notes_budget)
            header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured(row)
            tail   = build_tail_codes()
            raw_p  = "\n".join([x for x in [header, notes_trimmed, tail] if x])
            d["notes_tokens"] = count_tokens(tok, notes_trimmed)
            d["total_raw"] = count_tokens(tok, raw_p)

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

    # generate
    preds_raw = generate_codes(model, tok, raw_prompts, label_set,
                               max_len=args.total_input_budget,
                               gen_max_new=args.gen_max_new,
                               batch_size=args.batch_size)
    preds_kg  = generate_codes(model, tok, kg_prompts, label_set,
                               max_len=args.total_input_budget,
                               gen_max_new=args.gen_max_new,
                               batch_size=args.batch_size)

    # eval
    Y_true = codes_to_multihot(gold, label_vocab)
    Y_raw  = codes_to_multihot(preds_raw, label_vocab)
    Y_kg   = codes_to_multihot(preds_kg,  label_vocab)

    def _eval(m_true, m_pred):
        m = {
            "micro_precision":   float(precision_score(m_true, m_pred, average="micro",   zero_division=0)),
            "micro_recall":      float(recall_score(m_true, m_pred, average="micro",      zero_division=0)),
            "micro_f1":          float(f1_score(m_true, m_pred, average="micro",          zero_division=0)),
            "macro_precision":   float(precision_score(m_true, m_pred, average="macro",   zero_division=0)),
            "macro_recall":      float(recall_score(m_true, m_pred, average="macro",      zero_division=0)),
            "macro_f1":          float(f1_score(m_true, m_pred, average="macro",          zero_division=0)),
            "samples_precision": float(precision_score(m_true, m_pred, average="samples", zero_division=0)),
            "samples_recall":    float(recall_score(m_true, m_pred, average="samples",    zero_division=0)),
            "samples_f1":        float(f1_score(m_true, m_pred, average="samples",        zero_division=0)),
        }
        # hierarchical parent recall
        m["hierarchical_parent_recall"] = hierarchical_parent_recall(m_true, m_pred, label_vocab)
        return m

    m_raw = _eval(Y_true, Y_raw)
    m_kg  = _eval(Y_true, Y_kg)

    # print
    def _print_metrics(tag, m):
        print(f"\n=== {tag} ===")
        for k in ["micro_precision","micro_recall","micro_f1",
                  "macro_precision","macro_recall","macro_f1",
                  "samples_precision","samples_recall","samples_f1",
                  "hierarchical_parent_recall"]:
            print(f"{k}: {m[k]:.4f}")

    _print_metrics("RAW (no KG in prompt)", m_raw)
    _print_metrics("KG in prompt",          m_kg)

    # samples
    n_show = min(args.show_n, len(df))
    if n_show > 0:
        print(f"\n=== Sample predictions (n={n_show}) ===")
        for i in range(n_show):
            row = df.iloc[i]
            dbg = dbg_rows[i]
            print("-"*120)
            print(f"Example {i}: hadm_id={row.get('hadm_id','')}")
            print("GOLD:", " ".join(gold[i]) if gold[i] else "(none)")
            print("RAW :", " ".join(preds_raw[i]) if preds_raw[i] else "(none)")
            print("KG  :", " ".join(preds_kg[i])  if preds_kg[i]  else "(none)")
            print(f"[TOKENS RAW] total={dbg['total_raw']} header={dbg['header_tokens']} notes={dbg['notes_tokens']} tail={dbg['tail_tokens']} (reserve={args.assistant_reserve})")
            print(f"[TOKENS KG ] total={dbg['total_kg']} header={dbg['header_tokens']} notes={dbg['notes_tokens']} kg={dbg['kg_tokens']} tail={dbg['tail_tokens']} (reserve={args.assistant_reserve}) | ev={dbg['ev_cuis']} exp={dbg['exp_cuis']} allowed={dbg['allowed_icd9']}")

    # small artifact for debug
    try:
        outdir = os.path.join(os.path.dirname(__file__) or ".", "outputs")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "dbg_rows_codegen.json"), "w") as f:
            json.dump(dbg_rows[:200], f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
