#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextGen with KG-RAG natural-language hints (adapter-only), evaluated RAW vs KG.

Pipeline:
1) Build two prompts per visit:
   - RAW: [VISIT]+structured+NOTES + [TASK/FORMAT/OUTPUT]
   - KG : same + [KG HINTS] where hints are in plain English:
          * expand structured codes (ATC/LOINC/PROC) via code→name
          * retrieve top ICD-9 candidates from a FAISS index built over KG-enriched ICD-9 profiles
          * render those candidate names (and codes) in the hint block under a token budget
2) Generate free-text diagnoses for RAW and KG with identical decoding.
3) Extract lines after [OUTPUT] and treat them as diagnosis terms.
4) Map terms → ICD-9 via ICDMapper (SapBERT+FAISS).
5) Evaluate RAW vs KG with micro/macro/samples and parent metrics.
"""

import os, re, json, time, argparse, pickle, glob, sys
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel

from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
import torch.distributed as dist

# ====== project utils you already have ======
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
    if x is None: return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(_coerce_text(y) for y in x if y is not None)
    if isinstance(x, bytes):
        try: return x.decode("utf-8", errors="ignore")
        except Exception: return str(x)
    return str(x)

def extract_after_output(generation, n_max: int = 12) -> List[str]:
    s = _coerce_text(generation).replace("</s>", "").replace("<s>", "").strip()
    m = _OUTPUT_RE.search(s)
    block = s[m.end():] if m else s

    lines_out = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line: continue
        if line.startswith("[") and not re.match(r"^\[\s*OUTPUT\s*\]$", line, flags=re.I):
            break
        line = re.sub(r"^(?:[-*]\s*|\d+\.\s*|\(\d+\)\s*)", "", line)  # strip bullets/numbers
        if line:
            lines_out.append(line)
        if len(lines_out) >= n_max:
            break
    return lines_out

def _safe_extract_batch(texts, nmax, label):
    out, misses = [], 0
    for t in texts:
        s = _coerce_text(t)
        if not _OUTPUT_RE.search(s): misses += 1
        out.append(extract_after_output(s, nmax))
    if misses:
        log.info(f"[{label}] generations missing [OUTPUT] tag: {misses}/{len(texts)} (parsed best-effort)")
    return out

# ------------------------------- Parent helpers -------------------------------
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
    return labels

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
    if not text: return 0
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

# ------------------------------- Simple KG retriever over FAISS index -------------------------------
class SimpleKGRetriever:
    """
    Loads FAISS index (kg_index_dir) built over KG-enriched ICD-9 profiles.
    Uses a sentence embedding model to encode queries.
    Title lookup is provided externally (code -> title).
    """
    def __init__(self, kg_index_dir: str, encoder_model: str, title_lookup: Dict[str, str] = None):
        import faiss
        self.faiss = faiss
        self.index_path = os.path.join(kg_index_dir, "faiss.index")
        self.codes_path = os.path.join(kg_index_dir, "codes.pkl")
        assert os.path.exists(self.index_path), f"Missing {self.index_path}"
        assert os.path.exists(self.codes_path), f"Missing {self.codes_path}"
        self.index = self.faiss.read_index(self.index_path)
        with open(self.codes_path, "rb") as f:
            self.codes = pickle.load(f)  # list[str]

        self.titles = title_lookup or {}

        # encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(encoder_model)
            self.encoder = None
            self.tokenizer = None
        except Exception:
            self.st = None
            self.encoder = AutoModel.from_pretrained(encoder_model)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model, use_fast=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is not None:
            self.encoder.to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        if self.st is not None:
            embs = self.st.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return embs.astype("float32")
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        toks = {k: v.to(self.device) for k,v in toks.items()}
        out = self.encoder(**toks)
        last = out.last_hidden_state
        attn = toks["attention_mask"].unsqueeze(-1)
        vec = (last * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        vec = torch.nn.functional.normalize(vec, dim=1)
        return vec.detach().cpu().numpy().astype("float32")

    def search(self, query: str, topk: int = 50) -> List[Tuple[str, float]]:
        q = self.encode([query])
        D, I = self.index.search(q, min(topk, len(self.codes)))
        res = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0: continue
            code = self.codes[idx]
            res.append((code, float(score)))
        return res

    def code_title(self, code: str) -> str:
        return self.titles.get(code.strip().upper(), "")

# ------------------------------- KG hints text -------------------------------
def build_kg_hints_text_from_candidates(
    tok,
    kg_hint_budget: int,
    expanded_structured_names: Dict[str, List[str]],
    candidates: List[Tuple[str, str]],  # (code, title)
    top_n: int
) -> str:
    if kg_hint_budget <= 0:
        return ""
    lines = ["[KG HINTS]"]
    # Compact evidence lines in natural language
    if expanded_structured_names.get("ATC"):
        lines.append("Medication class evidence: " + "; ".join(expanded_structured_names["ATC"][:10]))
    if expanded_structured_names.get("PROC"):
        lines.append("Procedure evidence: " + "; ".join(expanded_structured_names["PROC"][:10]))
    if expanded_structured_names.get("LOINC"):
        lines.append("Lab evidence: " + "; ".join(expanded_structured_names["LOINC"][:10]))

    lines.append("Likely ICD-9 discharge diagnoses (context only):")
    text = "\n".join(lines)

    kept = []
    for code, title in candidates[:top_n]:
        item = f"- {title} ({code})" if title else f"- {code}"
        trial = text + "\n" + "\n".join(kept + [item])
        if count_tokens(tok, trial) <= kg_hint_budget:
            kept.append(item)
        else:
            break

    if kept:
        return text + "\n" + "\n".join(kept)
    # if nothing fits, try trimming header lines
    return trim_to_token_budget(tok, text, kg_hint_budget)
    

def build_title_lookup(code2name_path: str, kg_nodes_csv: str) -> Dict[str, str]:
    """
    Build a dict {ICD9_code -> preferred title} from:
      1) code2name.pkl
      2) kg_nodes.csv entries with sab == 'ICD9CM'
    """
    title = {}

    # 1) code2name.pkl
    try:
        with open(code2name_path, "rb") as f:
            c2n = pickle.load(f)
        # keep only ICD-9-ish keys
        for k, v in c2n.items():
            if not isinstance(k, str): 
                continue
            kk = k.strip().upper()
            # very light filter: ICD-9 codes are alnum with optional dot and may start with E/V/digits
            if re.match(r"^[0-9EV][0-9A-Z]*\.?[0-9A-Z]*$", kk):
                if isinstance(v, str) and v.strip():
                    title[kk] = v.strip()
    except Exception:
        pass

    # 2) kg_nodes.csv (SAB=ICD9CM)
    try:
        df_nodes = pd.read_csv(kg_nodes_csv)
        # expected columns: ['cui', 'name', 'sab', 'code', 'semantic_type']
        df_icd9 = df_nodes[df_nodes['sab'].astype(str).str.upper() == 'ICD9CM'][['code', 'name']].dropna()
        for _, r in df_icd9.iterrows():
            code = str(r['code']).strip().upper()
            name = str(r['name']).strip()
            if code and name and code not in title:
                title[code] = name
    except Exception:
        pass

    return title


# ------------------------------- Prompt pieces -------------------------------
def build_tail(N_max_terms:int) -> str:
    return "\n".join([
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- Do not include ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]"
    ])

def compute_terms_caps(gold_lists: List[List[str]]) -> Tuple[int, int, int]:
    sizes = sorted(len(x) for x in gold_lists)
    if not sizes: return 12, 10, 18
    def pct(p): 
        k = max(0, min(len(sizes)-1, int(round((p/100.0)*(len(sizes)-1)))))
        return sizes[k]
    p50, p90 = pct(50), pct(90)
    base = int(max(6, min(24, round(p50 + 2))))
    return base, p50, p90

def row_cap_from_candidates(base_N: int, p90: int, allowed_count: int) -> int:
    bonus = 0
    if   allowed_count >= 30: bonus = 6
    elif allowed_count >= 20: bonus = 4
    elif allowed_count >= 10: bonus = 2
    return int(max(6, min(p90, base_N + bonus)))

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
    ap.add_argument("--N_max_terms", type=int, default=12)  # fixed; set 0 to auto
    ap.add_argument("--max_len", type=int, default=4096)    # tokenizer input budget
    ap.add_argument("--kg_hint_budget", type=int, default=600)

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

    # ICDMapper (for mapping generated terms → ICD-9)
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

    # optional per-label CSV lists
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")

    # KG-RAG inputs
    ap.add_argument("--kg_nodes_csv", required=True)      # used only for sanity or future; not required at runtime here
    ap.add_argument("--kg_edges_csv", required=True)      # same
    ap.add_argument("--code2name_pkl", required=True)     # code→name for ATC/LOINC/PROC (and maybe some ICD9)
    ap.add_argument("--kg_index_dir", required=True)      # faiss.index + codes.pkl + meta.json
    ap.add_argument("--kg_retr_topk", type=int, default=200)
    ap.add_argument("--kg_hint_top",  type=int, default=80)
    ap.add_argument("--kg_neighbor_hops", type=int, default=1)  # reserved; not used directly in this simplified retriever
    ap.add_argument("--whitelist_strict", action="store_true")  # reserve for mapper gating (not used here)

    # runtime
    ap.add_argument("--tmp_dir", default="runs_textgen/test_shards")
    ap.add_argument("--out_metrics", default="runs_textgen/test_metrics_kg_rag.json")

    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---------- data ----------
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

    # gold codes
    def extract_codes(df, label_col):
        out=[]
        for _, r in df.iterrows():
            lst = to_list(r.get(label_col, []))
            lst = [format_icd9(c) for c in lst if c]
            lst = [c for c in lst if is_valid_icd9(c)]
            out.append(lst)
        return out
    gold_codes = extract_codes(test_df, args.label_col)

    # eval label space
    labels_full = sorted({c for lst in gold_codes for c in lst})
    labels_eval = labels_full
    head_name = None
    if args.labels_space == "head" and args.labels_head_k > 0:
        cnt = Counter([c for lst in gold_codes for c in lst])
        labels_eval = [c for c,_ in cnt.most_common(args.labels_head_k)]
        head_name = f"HEAD_{args.labels_head_k}"

    if is_main_process():
        log.info(f"Test size: {len(test_df)}")
        log.info(f"Eval label space: {len(labels_eval)} codes ({'FULL' if head_name is None else head_name})")

    # ---------- dynamic term caps ----------
    base_N, p50_codes, p90_codes = compute_terms_caps(gold_codes)
    if args.N_max_terms and args.N_max_terms > 0:
        base_N = args.N_max_terms

    # ---------- model (adapter-only) ----------
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) \
           else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.config.use_cache = True
    dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    # ---------- KG retriever (for hints) ----------
    title_lookup = build_title_lookup(args.code2name_pkl, args.kg_nodes_csv)
    retr = SimpleKGRetriever(args.kg_index_dir, encoder_model="gen/withKG/RAG/biobert-mnli-snli-scinli-scitail-mednli-stsb", title_lookup=title_lookup)

    # ---------- code2name (expand structured codes) ----------
    with open(args.code2name_pkl, "rb") as f:
        code2name = pickle.load(f)  # expects dict[str -> str]

    def expand_structured_names(row: pd.Series) -> Dict[str, List[str]]:
        out = {"ATC": [], "PROC": [], "LOINC": []}
        # ATC via 'ndc' column (per your dataset)
        for c in to_list(row.get("ndc", [])):
            c = 'ATC:' + str(c)
            n = code2name.get(str(c).strip().upper(), "")
            if n: out["ATC"].append(n)
        # PROC via 'pro_code' like 'PRO_5491' or '54.91'
        for c in to_list(row.get("pro_code", [])):
            key = str(c).strip().upper()
            if key.startswith("PRO_"):
                key = key[4:]
                if len(key) >= 3:
                    key = key[:2] + "." + key[2:]
            key='PROC:' + key
            n = code2name.get(key, "")
            if n: out["PROC"].append(n)
        # LOINC via 'lab_test'
        for c in to_list(row.get("lab_test_loinc", [])):
            c = 'LNC:' + str(c)
            n = code2name.get(str(c).strip().upper(), "")
            if n: out["LOINC"].append(n)
        return out

    # ---------- build prompts (RAW & KG) ----------
    raw_prompts, kg_prompts, dbg_rows = [], [], []
    tail_cache = {}  # cache tail by N_this to avoid recomputing tokens

    for i, row in test_df.iterrows():
        # structured header + readable block
        header = f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}\n" + serialize_structured_readable(row)
        notes_full = serialize_notes(row)

        # retrieve KG candidates using query composed of structured names + a short notes snippet
        names = expand_structured_names(row)
        query_bits = []
        if names["ATC"]:  query_bits.append("Medications: " + "; ".join(names["ATC"][:10]))
        if names["PROC"]: query_bits.append("Procedures: "  + "; ".join(names["PROC"][:10]))
        if names["LOINC"]:query_bits.append("Labs: "        + "; ".join(names["LOINC"][:10]))
        # keep a small notes snippet to guide retrieval (not full notes to avoid noise)
        if notes_full: query_bits.append(notes_full[:800])
        query = "\n".join(query_bits) if query_bits else notes_full[:800]

        cand = retr.search(query, topk=max(1, args.kg_retr_topk))
        # prepare candidate (code, title)
        cand_named = [(code, retr.code_title(code)) for (code, _) in cand]

        # N cap depends weakly on how many candidates we got (more → allow a few more lines up to p90)
        N_this = row_cap_from_candidates(base_N, p90_codes, allowed_count=len(cand_named))

        # tail with correct N
        if N_this not in tail_cache:
            tail_cache[N_this] = build_tail(N_this)
        tail = tail_cache[N_this]

        # Build KG HINTS block (token-limited)
        kg_hints = build_kg_hints_text_from_candidates(
            tok=tok,
            kg_hint_budget=args.kg_hint_budget,
            expanded_structured_names=names,
            candidates=cand_named,
            top_n=args.kg_hint_top
        )

        # Compose RAW and KG prompts and clamp to max_len
        def clamp_prompt(*parts):
            txt = "\n".join([p for p in parts if p])
            # If too long, trim notes first (not hints)
            total = count_tokens(tok, txt)
            if total <= args.max_len:
                return txt
            # Try trimming notes portion aggressively
            # parts = [header, notes, kg_hints?, tail]
            head, notes, extra, tailp = parts[0], parts[1], (parts[2] if len(parts) == 4 else ""), parts[-1]
            # keep header+tail and optional hints under budget; give notes remaining space
            fixed = "\n".join([x for x in [head, extra, tailp] if x])
            fixed_tokens = count_tokens(tok, fixed)
            budget_for_notes = max(0, args.max_len - fixed_tokens - 4)
            notes_trim = trim_to_token_budget(tok, notes, budget_for_notes)
            return "\n".join([x for x in [head, notes_trim, extra, tailp] if x])

        raw_prompt = clamp_prompt(header, notes_full, "", tail)
        kg_prompt  = clamp_prompt(header, notes_full, kg_hints, tail)

        raw_prompts.append(raw_prompt)
        kg_prompts.append(kg_prompt)
        dbg_rows.append({
            "idx": i,
            "hadm_id": row.get("hadm_id",""),
            "cand_count": len(cand_named),
            "N_cap": N_this,
            "raw_tokens": count_tokens(tok, raw_prompt),
            "kg_tokens":  count_tokens(tok, kg_prompt),
        })

    # ---------- decoding kwargs ----------
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
            return dict(max_new_tokens=max_new, do_sample=False, eos_token_id=eos_id, pad_token_id=pad_id,
                        no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))
        if decoding == "beam":
            return dict(max_new_tokens=max_new, num_beams=num_beams, do_sample=False, eos_token_id=eos_id,
                        pad_token_id=pad_id, no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))
        return dict(max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                    eos_token_id=eos_id, pad_token_id=pad_id,
                    no_repeat_ngram_size=(no_repeat_ngram if no_repeat_ngram>0 else None))

    gen_kwargs = build_generate_kwargs(
        decoding=args.decoding, max_new=args.gen_max_new,
        eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
        num_beams=args.num_beams, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, no_repeat_ngram=args.no_repeat_ngram
    )

    # ---------- (optional) distributed sharding ----------
    rank, W = 0, 1
    if False:  # keep simple; enable if you torchrun
        maybe_init_dist()
        rank = int(os.environ.get("RANK", "0")); W = world_size()
    idxs = list(range(len(raw_prompts)))

    # ---------- generation ----------
    @torch.no_grad()
    def generate_texts(model, tok, prompts: List[str], max_len: int, gen_kwargs: dict, batch_size: int = 8, device=None) -> List[str]:
        device = device or (model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        outs = []
        bs = max(1, int(batch_size))
        tok.padding_side = "left"
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

    t0 = time.time()
    raw_out_texts = generate_texts(model, tok, raw_prompts, max_len=args.max_len, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)
    kg_out_texts  = generate_texts(model, tok, kg_prompts,  max_len=args.max_len, gen_kwargs=gen_kwargs, batch_size=args.gen_batch_size, device=dev)

    raw_terms = _safe_extract_batch(raw_out_texts,  max(d["N_cap"] for d in dbg_rows), "RAW")
    kg_terms  = _safe_extract_batch(kg_out_texts,   max(d["N_cap"] for d in dbg_rows), "KG")

    if is_main_process():
        per = (time.time()-t0)/max(1,len(idxs))
        log.info(f"Generation done ({per:.2f}s/sample).")

    # ---------- map terms → ICD-9 ----------
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    mapped_raw = mapper.map_terms(raw_terms)
    mapped_kg  = mapper.map_terms(kg_terms)

    # ---------- metrics ----------
    gold_eval = restrict_to(gold_codes, labels_eval)
    raw_eval  = restrict_to(mapped_raw, labels_eval)
    kg_eval   = restrict_to(mapped_kg,  labels_eval)

    Yt = multihot(gold_eval, labels_eval)
    Yr = multihot(raw_eval,  labels_eval)
    Yk = multihot(kg_eval,   labels_eval)

    m_raw = eval_pack(Yt, Yr)
    m_kg  = eval_pack(Yt, Yk)

    pm_raw = {}; pm_kg = {}
    add_parent_metrics_full(pm_raw, gold_eval, raw_eval); add_parent_macro_f1(pm_raw, gold_eval, raw_eval)
    add_parent_metrics_full(pm_kg,  gold_eval, kg_eval);  add_parent_macro_f1(pm_kg,  gold_eval, kg_eval)
    m_raw.update(pm_raw); m_kg.update(pm_kg)

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

    ps, rs, fs = sample_set_prf(gold_eval, raw_eval); m_raw["precision_samples_set"]=ps; m_raw["recall_samples_set"]=rs; m_raw["f1_samples_set"]=fs
    ps, rs, fs = sample_set_prf(gold_eval, kg_eval);  m_kg["precision_samples_set"]=ps;  m_kg["recall_samples_set"]=rs;  m_kg["f1_samples_set"]=fs

    # per-label CSV: FULL
    out_dir = os.path.dirname(os.path.abspath(args.out_metrics))
    os.makedirs(out_dir, exist_ok=True)

    def per_label_table(y_true, y_pred, labels, out_csv_path=None):
        p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        df = pd.DataFrame({"code": labels,"precision": p,"recall": r,"f1": f1,"support": support})
        if out_csv_path: df.to_csv(out_csv_path, index=False)
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
    top_codes    = _read_first_col_codes(args.top_codes_csv)
    bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
    top_parents  = _read_first_col_parents(args.top_parent_csv)

    if top_codes:
        g = restrict_to(gold_codes, top_codes)
        r = restrict_to(mapped_raw,  top_codes)
        k = restrict_to(mapped_kg,   top_codes)
        Yg = multihot(g, top_codes); Yr2= multihot(r, top_codes); Yk2= multihot(k, top_codes)
        results_ext["TOP_50_CODES_RAW"] = eval_pack(Yg, Yr2)
        results_ext["TOP_50_CODES_KG"]  = eval_pack(Yg, Yk2)
        per_label_table(Yg, Yr2, top_codes, os.path.join(out_dir, "per_label_TOP_50_CODES_RAW.csv"))
        per_label_table(Yg, Yk2, top_codes, os.path.join(out_dir, "per_label_TOP_50_CODES_KG.csv"))

    if bottom_codes:
        g = restrict_to(gold_codes, bottom_codes)
        r = restrict_to(mapped_raw,  bottom_codes)
        k = restrict_to(mapped_kg,   bottom_codes)
        Yg = multihot(g, bottom_codes); Yr2= multihot(r, bottom_codes); Yk2= multihot(k, bottom_codes)
        results_ext["BOTTOM_50_CODES_RAW"] = eval_pack(Yg, Yr2)
        results_ext["BOTTOM_50_CODES_KG"]  = eval_pack(Yg, Yk2)
        per_label_table(Yg, Yr2, bottom_codes, os.path.join(out_dir, "per_label_BOTTOM_50_CODES_RAW.csv"))
        per_label_table(Yg, Yk2, bottom_codes, os.path.join(out_dir, "per_label_BOTTOM_50_CODES_KG.csv"))

    if top_parents:
        g_par = [[get_icd9_parent_safe(c) for c in lst] for lst in gold_codes]
        r_par = [[get_icd9_parent_safe(c) for c in lst] for lst in mapped_raw]
        k_par = [[get_icd9_parent_safe(c) for c in lst] for lst in mapped_kg]
        g_par = restrict_to(g_par, top_parents); r_par = restrict_to(r_par, top_parents); k_par = restrict_to(k_par, top_parents)
        YgP = multihot(g_par, top_parents); YrP = multihot(r_par, top_parents); YkP = multihot(k_par, top_parents)
        results_ext["TOP_50_PARENTS_RAW"] = eval_pack(YgP, YrP)
        results_ext["TOP_50_PARENTS_KG"]  = eval_pack(YgP, YkP)
        per_label_table(YgP, YrP, top_parents, os.path.join(out_dir, "per_label_TOP_50_PARENTS_RAW.csv"))
        per_label_table(YgP, YkP, top_parents, os.path.join(out_dir, "per_label_TOP_50_PARENTS_KG.csv"))

    # Save metrics
    payload = {
        "label_space": ("FULL" if head_name is None else head_name),
        "num_samples": len(raw_prompts),
        "metrics_raw": m_raw,
        "metrics_kg":  m_kg,
        **results_ext
    }
    with open(args.out_metrics, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Metrics saved to {args.out_metrics}")

    # Print a few samples
    n_show = min(args.print_samples, len(raw_terms))
    log.info("=== Sample predictions (RAW vs KG) ===")
    for i in range(n_show):
        G = set(gold_codes[i]); R = set(mapped_raw[i]); K = set(mapped_kg[i])
        # RAW
        tp = len(G & R); fp = len(R - G); fn = len(G - R)
        pr = tp/(tp+fp) if tp+fp>0 else 0.0
        rr = tp/(tp+fn) if tp+fn>0 else 0.0
        fr = (2*pr*rr)/(pr+rr) if pr+rr>0 else 0.0
        # KG
        tp2 = len(G & K); fp2 = len(K - G); fn2 = len(G - K)
        pr2 = tp2/(tp2+fp2) if tp2+fp2>0 else 0.0
        rr2 = tp2/(tp2+fn2) if tp2+fn2>0 else 0.0
        fr2 = (2*pr2*rr2)/(pr2+rr2) if pr2+rr2>0 else 0.0

        log.info(f"[Sample {i+1}] hadm={test_df.iloc[i].get('hadm_id','')}")
        log.info(f"  GOLD codes: {', '.join(sorted(G)) if G else '(none)'}")
        log.info(  "  RAW free-text terms:")
        for t in raw_terms[i][:base_N]:
            log.info(f"    - {t}")
        log.info(f"  RAW mapped ICD-9: {', '.join(sorted(R)) if R else '(none)'}  | P/R/F1 = {pr:.3f}/{rr:.3f}/{fr:.3f}")
        log.info(  "  KG  free-text terms:")
        for t in kg_terms[i][:base_N]:
            log.info(f"    - {t}")
        log.info(f"  KG  mapped ICD-9: {', '.join(sorted(K)) if K else '(none)'}  | P/R/F1 = {pr2:.3f}/{rr2:.3f}/{fr2:.3f}")

    cleanup_dist()
    return 0

if __name__ == "__main__":
    sys.exit(main())
