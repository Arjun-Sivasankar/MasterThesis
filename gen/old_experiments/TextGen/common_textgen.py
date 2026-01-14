# common_textgen.py
import os, re, json, time, logging, math, pickle
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model

from sklearn.metrics import f1_score, precision_score, recall_score

import faiss
from rapidfuzz.fuzz import token_set_ratio
from transformers import AutoTokenizer as HFTok, AutoModel as HFModel

# ---------- logging & env ----------
hf_logging.set_verbosity_error()
log = logging.getLogger("textgen_common")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- rank helpers ----------
def _env_int(k, default=0):
    try: return int(os.environ.get(k, default))
    except: return default

def is_main_process(): return _env_int("RANK", 0) == 0
def world_size(): return _env_int("WORLD_SIZE", 1)
def local_rank(): return _env_int("LOCAL_RANK", 0)

# ------------------ utilities ------------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x):
    if x is None: return ""
    try:
        s = " ".join(map(str, x.tolist())) if isinstance(x, (np.ndarray, pd.Series)) else str(x)
    except Exception:
        s = str(x)
    s = s.replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def to_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        it = x.tolist() if hasattr(x, "tolist") else x
        out=[]
        for v in it:
            if v is None: continue
            if isinstance(v, float) and np.isnan(v): continue
            sv = str(v).strip()
            if sv and sv.lower() not in ("nan","none"): out.append(sv)
        return out
    s = str(x).strip()
    if not s or s.lower() in ("nan","none"): return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)): return [str(t).strip() for t in v if str(t).strip()]
        except Exception: pass
    return [t for t in re.split(r"[,\s]+", s) if t]

def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def format_icd9(code: str) -> str:
    code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
    if not code: return ""
    if code[0].isdigit():
        if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        return code
    if code[0] == "V":
        if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        return code
    if code[0] == "E":
        if len(code)>4 and "." not in code: return code[:4]+"."+code[4:]
        return code
    return code

def is_valid_icd9(code: str) -> bool:
    if not code: return False
    c = code.upper()
    if c[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", c))
    if c[0]=="V":      return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", c))
    if c[0]=="E":      return bool(re.match(r"^E\d{3}(\.\d{1})?$", c))
    return False

def get_icd9_parent(code: str) -> str:
    if not code or len(code) < 3: 
        return code
    code = str(code).upper()
    if code[0].isdigit():
        return code.split('.')[0][:3]
    if code.startswith('V'):
        base = code.split('.')[0]
        return base[:3]
    if code.startswith('E'):
        base = code.split('.')[0]
        return base[:4] if len(base) >= 4 else base
    return code

# Backward-compat shim (some older code may import this)
def to_icd9_parent(c: str) -> str:
    return get_icd9_parent(c)

def serialize_structured_readable(row: pd.Series) -> str:
    ndc  = " ".join(to_list(row.get("ndc", []))[:24])
    proc = " ".join(to_list(row.get("pro_code", []))[:24])
    labs = " ".join(to_list(row.get("lab_test", []))[:48])
    parts=[]
    parts.append(f"DEMOGRAPHICS: gender={row.get('gender','')} age_group={row.get('age','')}")
    if ndc:  parts.append(f"MEDICATIONS: {ndc}")
    if proc: parts.append(f"PROCEDURES: {proc}")
    if labs: parts.append(f"LAB TESTS: {labs}")
    return "\n".join(parts)

def serialize_notes(row: pd.Series) -> str:
    parts=[]
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t: parts.append(f"{col}: {t}")
    return "\n".join(parts)

def token_len(tok, text: str) -> int:
    return int(tok(text, add_special_tokens=False, return_length=True)["length"][0])

def chat_token_len(tok, msgs: List[Dict], add_generation_prompt: bool) -> int:
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    return token_len(tok, txt)

def build_textgen_prompt_budgeted(row: pd.Series, tok, max_len: int,
                                  min_assist_tokens: int, N_max_terms: int
                                  ) -> Tuple[str, Dict[str, int]]:
    head = []
    head.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
    head.append(serialize_structured_readable(row))
    notes_full = serialize_notes(row)

    tail = [
        "[TASK] List the final clinical diagnoses for this admission.",
        "[FORMAT]",
        "- One diagnosis per line",
        "- Avoid abbreviations if possible",
        "- No ICD codes or explanations",
        f"- Maximum: {N_max_terms} lines",
        "[OUTPUT]",
    ]
    def assemble(notes_text: str) -> str:
        parts = [*head]
        if notes_text: parts.append(notes_text)
        parts.extend(tail)
        return "\n".join([p for p in parts if p])

    prompt = assemble(notes_full)
    prompt_len_tokens = chat_token_len(tok, [{"role":"user","content":prompt}], add_generation_prompt=True)
    if prompt_len_tokens <= max_len - min_assist_tokens:
        return prompt, {"prompt_tokens": prompt_len_tokens, "notes_kept_chars": len(notes_full), "notes_trimmed": 0}

    # binary search to fit
    left, right = 0, len(notes_full)
    best_prompt, best_len, best_mid = None, None, 0
    while left <= right:
        mid = (left + right) // 2
        trial_notes = notes_full[:mid]
        trial_prompt = assemble(trial_notes)
        L = chat_token_len(tok, [{"role":"user","content":trial_prompt}], add_generation_prompt=True)
        if L <= max_len - min_assist_tokens:
            best_prompt, best_len, best_mid = trial_prompt, L, mid
            left = mid + 1
        else:
            right = mid - 1
    if best_prompt is None:
        trial_prompt = assemble("")
        L = chat_token_len(tok, [{"role":"user","content":trial_prompt}], add_generation_prompt=True)
        best_prompt, best_len, best_mid = trial_prompt, L, 0
    kept_chars = best_mid
    return best_prompt, {"prompt_tokens": best_len, "notes_kept_chars": kept_chars, "notes_trimmed": len(notes_full)-kept_chars}

# ------------------ datasets ------------------
class SFTTextGenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer,
                 label_col: str,
                 target_mode: str,
                 icd_index_dir: str,
                 max_len: int,
                 N_max_terms: int,
                 min_assistant_tokens: int):
        self.tok = tokenizer
        self.label_col = label_col
        self.target_mode = target_mode
        self.max_len = max_len
        self.N_max_terms = N_max_terms
        self.min_assistant_tokens = max(1, int(min_assistant_tokens))

        self.code2title = {}
        if target_mode == "icd_titles":
            try:
                with open(os.path.join(icd_index_dir, "code2title.json"), "r") as f:
                    self.code2title = json.load(f)
                if is_main_process():
                    log.info(f"Loaded {len(self.code2title)} ICD-9 titles")
            except Exception as e:
                if is_main_process():
                    log.warning(f"Could not load code2title.json: {e}")

        inputs, targets, kept_idx = [], [], []
        dropped_empty_targets = 0
        dropped_truncated = 0
        trimmed_notes = 0
        prompt_tok_lens, assistant_tok_lens = [], []

        for idx, row in df.reset_index(drop=True).iterrows():
            if target_mode == "icd_titles":
                codes = [format_icd9(c) for c in to_list(row.get(label_col, [])) if c]
                codes = [c for c in codes if is_valid_icd9(c)]
                titles = []
                for c in codes:
                    t = self.code2title.get(c, "").strip()
                    if len(t) > 3:
                        titles.append(f"- {t}")
                target = "\n".join(titles)
                has_supervision = len(titles) > 0
            else:
                target_raw = clean_text(row.get("Discharge Diagnosis",""))
                target = target_raw if len(target_raw) >= 5 else ""
                has_supervision = len(target) > 0

            if not has_supervision:
                dropped_empty_targets += 1
                continue

            prompt, stat = build_textgen_prompt_budgeted(
                row, self.tok, self.max_len, self.min_assistant_tokens, self.N_max_terms
            )
            if stat.get("notes_trimmed", 0) > 0:
                trimmed_notes += 1

            user_msg = {"role": "user", "content": prompt}
            asst_msg = {"role": "assistant", "content": target}

            prompt_text = self.tok.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
            full_text   = self.tok.apply_chat_template([user_msg, asst_msg], tokenize=False, add_generation_prompt=False)

            prompt_ids = self.tok(prompt_text, return_tensors="pt", truncation=True, max_length=self.max_len).input_ids[0]
            full       = self.tok(full_text,   return_tensors="pt", truncation=True, max_length=self.max_len)
            input_ids  = full.input_ids[0]
            labels     = input_ids.clone()

            prompt_len = min(len(prompt_ids), len(input_ids))
            labels[:prompt_len] = -100  # ignore prompt

            assistant_len = int((labels != -100).sum().item())
            if assistant_len < 1:
                dropped_truncated += 1
                continue

            inputs.append(input_ids)
            targets.append(labels)
            kept_idx.append(idx)
            prompt_tok_lens.append(int(prompt_len))
            assistant_tok_lens.append(int(assistant_len))

        self.inputs = inputs
        self.targets = targets
        self.kept_idx = kept_idx

        if is_main_process():
            log.info(
                f"SFT dataset: kept={len(self.inputs)} "
                f"dropped_empty_targets={dropped_empty_targets} "
                f"dropped_truncated_targets={dropped_truncated} "
                f"trimmed_notes={trimmed_notes} (mode={target_mode})"
            )
            if prompt_tok_lens and assistant_tok_lens:
                p = np.array(prompt_tok_lens); a = np.array(assistant_tok_lens)
                log.info(
                    "Token stats (kept): "
                    f"prompt_mean={p.mean():.1f}, p95={np.percentile(p,95):.0f} | "
                    f"assistant_mean={a.mean():.1f}, p95={np.percentile(a,95):.0f}"
                )

    def __len__(self): return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        labels = self.targets[idx]
        attn = torch.ones_like(input_ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels, "idx": self.kept_idx[idx]}

def pad_collate(features, tok):
    pad_id = tok.pad_token_id
    max_len = max(len(f["input_ids"]) for f in features)
    B = len(features)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        L = len(f["input_ids"])
        input_ids[i,:L] = f["input_ids"]
        attention_mask[i,:L] = f["attention_mask"]
        labels[i,:L] = f["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ------------------ LLM & LoRA ------------------
def load_llm_with_lora(model_name):
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False  # for grad checkpointing

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        try:
            model.get_input_embeddings().weight.requires_grad_(True)
        except Exception:
            pass

    if is_main_process():
        model.print_trainable_parameters()
    if hasattr(model, "generation_config"):
        gc = model.generation_config
        for a in ("temperature","top_p","top_k"):
            if hasattr(gc, a): setattr(gc, a, None)
        if hasattr(gc, "do_sample"): gc.do_sample = False
    return model, tok

# ------------------ generation ------------------
def build_generate_kwargs(
    decoding:str, max_new:int,
    eos_id:int, pad_id:int,
    num_beams:int=1, temperature:float=1.0, top_p:float=1.0, top_k:int=0,
    no_repeat_ngram:int=0
):
    decoding = decoding.lower()
    if decoding == "beam":
        return dict(
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=max(2, num_beams),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=no_repeat_ngram,
            return_dict_in_generate=True,
        )
    if decoding == "sample":
        return dict(
            max_new_tokens=max_new,
            do_sample=True,
            temperature=max(0.1, float(temperature)),
            top_p=min(1.0, max(0.0, float(top_p))),
            top_k=max(0, int(top_k)),
            num_beams=1,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=no_repeat_ngram,
            return_dict_in_generate=True,
        )
    # greedy
    return dict(
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        no_repeat_ngram_size=no_repeat_ngram,
        return_dict_in_generate=True,
    )

@torch.no_grad()
def generate_terms(model, tokenizer, prompts: List[str],
                   max_len: int, gen_kwargs: dict, batch_size: int):
    model.eval()
    device = next(model.parameters()).device
    out_all = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role":"user","content":p}],
                tokenize=False, add_generation_prompt=True
            ) for p in batch_prompts
        ]
        enc = tokenizer(
            prompt_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_len
        ).to(device)

        with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
            gen = model.generate(**enc, **gen_kwargs)

        seq = gen.sequences  # [B, L_in + L_new]
        in_lens = enc["attention_mask"].sum(dim=1).tolist()
        for b in range(seq.size(0)):
            new_tokens = seq[b, in_lens[b]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # normalize to list of lines
            terms=[]
            for line in text.split("\n"):
                t = re.sub(r"^[\-\*\u2022]+\s*", "", line).strip()
                if t: terms.append(t)
            out_all.append(terms)
    return out_all

# ------------------ HF SapBERT mean encoder ------------------
class HFMeanEncoder:
    def __init__(self, model_name: str):
        self.tokenizer = HFTok.from_pretrained(model_name)
        self.model = HFModel.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        s = torch.sum(last_hidden * mask, dim=1)
        d = torch.clamp(mask.sum(dim=1), min=1e-9)
        return s / d

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size=32) -> np.ndarray:
        if not texts: return np.zeros((0,768), dtype=np.float32)
        chunks=[]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            emb = self._mean_pool(out.last_hidden_state, enc.attention_mask)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            chunks.append(emb.cpu().numpy())
        return np.vstack(chunks)

# ------------------ Mapper ------------------
class ICDMapper:
    def __init__(self, index_dir, encoder_model_cli=None,
                 tau_cos=0.40, tau_final=0.60,
                 w_cos=0.6, w_fuz=0.4,
                 faiss_rows=20):
        self.dir = index_dir
        self.faiss_rows = faiss_rows
        self.tau_cos = tau_cos
        self.tau_final = tau_final
        self.w_cos = w_cos
        self.w_fuz = w_fuz
        self.last_stats=[]

        idx_path = os.path.join(self.dir, "icd.faiss")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        self.index = faiss.read_index(idx_path)

        rows_path = os.path.join(self.dir, "rows.json")
        if not os.path.exists(rows_path):
            raise FileNotFoundError(f"rows.json not found: {rows_path}")
        with open(rows_path, "r") as f:
            self.rows = json.load(f)
        if not isinstance(self.rows, list):
            raise ValueError("rows.json must be a list of {text, code}")
        if len(self.rows) != self.index.ntotal:
            raise ValueError(f"rows.json length ({len(self.rows)}) must match FAISS ntotal ({self.index.ntotal})")

        meta = {}
        meta_path = os.path.join(self.dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path,"r") as f:
                meta = json.load(f)
        self.metric = meta.get("metric","ip").lower()  # "ip" or "l2"
        enc_name = meta.get("encoder_model", encoder_model_cli) or "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.encoder = HFMeanEncoder(enc_name)

        if is_main_process():
            log.info(f"FAISS index loaded: {self.index.ntotal} rows (metric={self.metric})")
            log.info(f"Encoder model: {enc_name}")

    def map_terms(self, term_lists: List[List[str]]):
        self.last_stats=[]
        all_codes=[]
        for terms in term_lists:
            mapped=set()
            if not terms:
                all_codes.append([])
                self.last_stats.append((0,0))
                continue

            embs = self.encoder.encode(terms)
            n_mapped=0
            for t_idx, term in enumerate(terms):
                if not term or len(term) < 3: continue
                norm_t = norm_text(term)
                if len(norm_t) < 3: continue

                D, I = self.index.search(embs[t_idx:t_idx+1], self.faiss_rows)  # (1,k)
                D, I = D[0], I[0]

                cand=[]
                for j, row_idx in enumerate(I):
                    if row_idx < 0: continue
                    entry = self.rows[row_idx]
                    cand_text = entry.get("text","")
                    cand_code = entry.get("code","")
                    if not cand_text or not cand_code: continue

                    if self.metric == "ip":
                        cos = float(D[j])  # inner product on normalized vectors ≈ cosine
                    else:
                        cos = 1.0 - float(D[j]) / 2.0  # L2 on normalized vectors -> cosine-like

                    if cos < self.tau_cos: continue

                    fuzzy = token_set_ratio(norm_t, norm_text(cand_text)) / 100.0
                    score = self.w_cos * cos + self.w_fuz * fuzzy
                    if score >= self.tau_final:
                        cand.append((cand_code, score))

                if cand:
                    cand.sort(key=lambda x: x[1], reverse=True)
                    mapped.add(cand[0][0]); n_mapped += 1

            self.last_stats.append((len(terms), n_mapped))
            all_codes.append(sorted(mapped))
        return all_codes

# ------------------ eval helpers ------------------
def build_eval_labels(train_gold_lists, head_k=0):
    counter = Counter([c for codes in train_gold_lists for c in codes])
    if head_k and head_k > 0:
        return [c for c,_ in counter.most_common(head_k)]
    return sorted(counter.keys())

def restrict_to(codes_lists, allowed):
    S=set(allowed)
    return [[c for c in codes if c in S] for codes in codes_lists]

def multihot(codes_lists, labels):
    idx = {c:i for i,c in enumerate(labels)}
    Y = np.zeros((len(codes_lists), len(labels)), dtype=np.int32)
    for i, lst in enumerate(codes_lists):
        for c in lst:
            j = idx.get(c)
            if j is not None: Y[i,j]=1
    return Y

def eval_pack(y_true, y_pred):
    return {
        "precision_micro": float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        "recall_micro":    float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        "f1_micro":        float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro":        float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "precision_samples": float(precision_score(y_true, y_pred, average='samples', zero_division=0)),
        "recall_samples":    float(recall_score(y_true, y_pred, average='samples', zero_division=0)),
        "f1_samples":        float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
    }

def add_parent_macro_f1(metrics, gold_lists, pred_lists):
    g = [[to_icd9_parent(c) for c in lst] for lst in gold_lists]
    p = [[to_icd9_parent(c) for c in lst] for lst in pred_lists]                                                                
    labels = sorted({x for lst in g for x in lst})
    Yg = multihot(g, labels); Yp = multihot(p, labels)
    metrics["f1_macro_parent"] = float(f1_score(Yg, Yp, average="macro", zero_division=0))
