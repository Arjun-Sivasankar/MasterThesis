import os, re, json, time, argparse, logging, pickle, atexit, sys, math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import faiss
from rapidfuzz.fuzz import token_set_ratio
from transformers import AutoTokenizer as HFTok, AutoModel as HFModel

# ---------------- Env & logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train_textgen_icd")
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

# Safer NCCL defaults (avoid silent hangs)
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
os.environ.setdefault("NCCL_TIMEOUT", "3600")  # seconds
# Optional toggles if your fabric is finicky; uncomment if needed:
# os.environ.setdefault("NCCL_IB_DISABLE", "1")
# os.environ.setdefault("NCCL_P2P_DISABLE", "1")
# Reduce CUDA fragmentation on long runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ------------------ DDP helpers ------------------
def _env_rank():
    for k in ("LOCAL_RANK", "RANK"):
        v = os.environ.get(k)
        if v is not None:
            try: return int(v)
            except: pass
    return 0

def is_main_process(): return _env_rank() == 0

def dist_is_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def barrier():
    if dist_is_initialized():
        try: torch.distributed.barrier()
        except Exception: pass

def _cleanup_dist():
    # Let HF handle PG lifecycle; nothing to destroy manually.
    pass
atexit.register(_cleanup_dist)

if not is_main_process():
    logging.getLogger().setLevel(logging.WARNING)

# ------------------ utilities ------------------
TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]

def clean_text(x):
    if isinstance(x, (np.ndarray, pd.Series)):
        try:
            x = " ".join(map(str, x.tolist()))
        except Exception:
            x = str(x)
    if x is None:
        return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def to_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple, set)):
        out=[]
        for v in x:
            if v is None: continue
            if isinstance(v, float) and np.isnan(v): continue
            sv = str(v).strip()
            if sv: out.append(sv)
        return out
    if isinstance(x, (np.ndarray, pd.Series)):
        arr = x.tolist()
        if isinstance(arr, list):
            out=[]
            for v in arr:
                if v is None: continue
                if isinstance(v, float) and np.isnan(v): continue
                sv = str(v).strip()
                if sv: out.append(sv)
            return out
        return [str(arr)] if arr is not None and str(arr).strip() else []
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                out=[]
                for z in v:
                    if z is None: continue
                    if isinstance(z, float) and np.isnan(z): continue
                    sz = str(z).strip()
                    if sz: out.append(sz)
                return out
        except Exception:
            pass
    return [t for t in re.split(r"[,\s]+", s) if t]

def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def format_icd9(code: str) -> str:
    code = re.sub(r"\s+","", str(code)).upper()
    if code.endswith("."): code = code[:-1]
    if not code: return ""
    if code[0].isdigit():
        if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        return code
    if code[0] in ("V","E"):
        if code[0]=="V" and len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        if code[0]=="E" and len(code)>4 and "." not in code: return code[:4]+"."+code[4:]
        return code
    return code

def is_valid_icd9(code: str) -> bool:
    if not code: return False
    c = code.upper()
    if c[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", c))
    if c[0]=="V":      return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", c))
    if c[0]=="E":      return bool(re.match(r"^E\d{3}(\.\d{1})?$", c))
    return False

def to_icd9_parent(c: str) -> str:
    if not c: return c
    return c.split(".")[0][:3].upper()

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
            if isinstance(t, str) and t:
                parts.append(f"{col}: {t}")
    return "\n".join(parts)

# ---------- Token helpers ----------
def token_len(tok, text: str) -> int:
    return int(tok(text, add_special_tokens=False, return_length=True)["length"][0])

def chat_token_len(tok, msgs: List[Dict], add_generation_prompt: bool) -> int:
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    return token_len(tok, txt)

def build_textgen_prompt_budgeted(row: pd.Series, tok, max_len: int,
                                  min_assist_tokens: int, N_max_terms: int
                                  ) -> Tuple[str, Dict[str, int]]:
    """
    Build a prompt that leaves at least `min_assist_tokens` for assistant tokens.
    Trim notes via binary search in chars if needed.
    """
    head = []
    head.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
    head.append(serialize_structured_readable(row))
    notes_full = serialize_notes(row)

    tail = []
    tail.append("[TASK] List the final clinical diagnoses for this admission.")
    tail.append("[FORMAT]")
    tail.append("- One diagnosis per line")
    tail.append("- Avoid abbreviations if possible")
    tail.append("- No ICD codes or explanations")
    tail.append(f"- Maximum: {N_max_terms} lines")
    tail.append("[OUTPUT]")

    def assemble(notes_text: str) -> str:
        parts = [*head]
        if notes_text:
            parts.append(notes_text)
        parts.extend(tail)
        return "\n".join([p for p in parts if p])

    prompt = assemble(notes_full)
    prompt_len_tokens = chat_token_len(tok, [{"role":"user","content":prompt}], add_generation_prompt=True)

    if prompt_len_tokens <= max_len - min_assist_tokens:
        return prompt, {
            "prompt_tokens": prompt_len_tokens,
            "notes_kept_chars": len(notes_full),
            "notes_trimmed": 0
        }

    # Binary search to fit budget
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
    return best_prompt, {
        "prompt_tokens": best_len,
        "notes_kept_chars": kept_chars,
        "notes_trimmed": len(notes_full) - kept_chars
    }

def build_textgen_prompt(row: pd.Series, N_max_terms: int) -> str:
    s=[]
    s.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
    s.append(serialize_structured_readable(row))
    notes = serialize_notes(row)
    if notes: s.append(notes)
    s.append("[TASK] List the final clinical diagnoses for this admission.")
    s.append("[FORMAT]")
    s.append("- One diagnosis per line")
    s.append("- Avoid abbreviations if possible")
    s.append("- No ICD codes or explanations")
    s.append(f"- Maximum: {N_max_terms} lines")
    s.append("[OUTPUT]")
    return "\n".join([x for x in s if x])

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
        prompt_tok_lens, assistant_tok_lens = [], []
        dropped_empty_targets = 0
        dropped_truncated = 0
        trimmed_notes = 0

        for idx, row in df.reset_index(drop=True).iterrows():
            # Target text (assistant)
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

            # Build budgeted prompt
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
            attn       = full.attention_mask[0]

            labels = input_ids.clone()
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels[:prompt_len] = -100

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
                f"trimmed_notes={trimmed_notes} "
                f"(mode={target_mode})"
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
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False  # needed with gradient checkpointing

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

# ------------------ helpers ------------------
def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

# ------------------ generation ------------------
@torch.no_grad()
def generate_terms(model, tokenizer, prompts: List[str],
                   max_len: int, max_new: int, batch_size: int):
    model = unwrap_model(model)
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

        use_cuda = (device.type == "cuda")
        with torch.amp.autocast('cuda', enabled=use_cuda):
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,  # simpler + faster; avoids long collectives
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True
            )

        seq = gen.sequences  # [B, L_in + L_new]
        in_lens = enc["attention_mask"].sum(dim=1).tolist()
        for b in range(seq.size(0)):
            new_tokens = seq[b, in_lens[b]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            terms=[]
            for line in text.split("\n"):
                t = re.sub(r"^[\-\*\u2022]+\s*", "", line).strip()
                if t: terms.append(t)
            out_all.append(terms)

        # periodic cleanup during long test loops
        if use_cuda and ((i // batch_size) % 10 == 0):
            torch.cuda.empty_cache()

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
        enc_name = meta.get("encoder_model", encoder_model_cli)
        if not enc_name:
            enc_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
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
                        cos = float(D[j])  # IP on normalized vectors ≈ cosine
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

def per_sample_prf(gold_codes, pred_codes):
    """Return (precision, recall, f1, tp, |gold|, |pred|) for a single sample."""
    gs, ps = set(gold_codes), set(pred_codes)
    tp = len(gs & ps); g = len(gs); p = len(ps)
    precision = tp / p if p else 0.0
    recall    = tp / g if g else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
    return precision, recall, f1, tp, g, p

# ------------------ logging callback ------------------
class BetterLoggingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.step_losses_by_epoch = defaultdict(list)
        self.eval_loss_by_epoch = {}
        self.last_lr = None

    def on_train_begin(self, args, state, control, **kwargs):
        if is_main_process():
            log.info(f"=== Training starts: {args.num_train_epochs} epochs, LR={args.learning_rate}, "
                     f"BS(train)={args.per_device_train_batch_size} × GA={args.gradient_accumulation_steps} ===")

    def on_epoch_begin(self, args, state, control, **kwargs):
        if is_main_process():
            self.epoch_start_time = time.time()
            e = int(state.epoch) + 1 if state.epoch is not None else 1
            log.info(f"--- Epoch {e} ---")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process() or not logs:
            return
        if "loss" in logs:
            e = int(max(1, math.ceil(logs.get("epoch", state.epoch or 1))))
            self.step_losses_by_epoch[e].append(float(logs["loss"]))
        if "learning_rate" in logs:
            self.last_lr = float(logs["learning_rate"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not is_main_process() or not metrics:
            return
        e = int(state.epoch) if state.epoch is not None else len(self.eval_loss_by_epoch) + 1
        ev = float(metrics.get("eval_loss", float("nan")))
        self.eval_loss_by_epoch[e] = ev
        ppl = math.exp(ev) if ev == ev else float("nan")
        log.info(f"[Eval] epoch={e} eval_loss={ev:.4f} ppl={ppl:.2f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if not is_main_process():
            return
        e = int(state.epoch) if state.epoch is not None else 1
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        tr_list = self.step_losses_by_epoch.get(e, [])
        tr_loss = float(np.mean(tr_list)) if len(tr_list) > 0 else float("nan")
        ev_loss = self.eval_loss_by_epoch.get(e, None)
        lr_str = f"{self.last_lr:.6f}" if self.last_lr is not None else "n/a"
        if ev_loss is None:
            log.info(f"[EpochEnd] {e} | time={elapsed:.2f}s | train_loss≈{tr_loss:.4f} | lr={lr_str}")
        else:
            ppl = math.exp(ev_loss) if ev_loss == ev_loss else float("nan")
            log.info(f"[EpochEnd] {e} | time={elapsed:.2f}s | train_loss≈{tr_loss:.4f} "
                     f"| eval_loss={ev_loss:.4f} | ppl={ppl:.2f} | lr={lr_str}")

    def on_train_end(self, args, state, control, **kwargs):
        if not is_main_process():
            return
        log.info("=== Training finished ===")
        for e in sorted(set(self.step_losses_by_epoch) | set(self.eval_loss_by_epoch)):
            tr_list = self.step_losses_by_epoch.get(e, [])
            tr_loss = float(np.mean(tr_list)) if len(tr_list) > 0 else float("nan")
            ev_loss = self.eval_loss_by_epoch.get(e, float("nan"))
            ppl = math.exp(ev_loss) if ev_loss == ev_loss else float("nan")
            log.info(f"[Summary] epoch={e} train_loss≈{tr_loss:.4f} eval_loss={ev_loss:.4f} ppl={ppl:.2f}")

# ------------------ Safe Trainer ------------------
class SafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)
        if labels is not None:
            valid = (labels != -100).sum()
            if valid.item() == 0:
                loss = torch.zeros((), device=labels.device, dtype=torch.float32, requires_grad=True)
                return (loss, None) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

# ------------------ test logging helpers ------------------
def log_test_results(results, labels_full, labels_head=None, eval_head_k=0):
    log.info("=== Test Results Summary ===")
    m_full = results["FULL"]
    log.info(f"FULL label space ({len(labels_full)} codes):")
    log.info(f"  F1 micro: {m_full['f1_micro']:.4f}")
    log.info(f"  Precision micro: {m_full['precision_micro']:.4f}")
    log.info(f"  Recall micro: {m_full['recall_micro']:.4f}")
    log.info(f"  F1 macro: {m_full['f1_macro']:.4f}")
    log.info(f"  F1 samples: {m_full['f1_samples']:.4f}")
    if "f1_macro_parent" in m_full:
        log.info(f"  F1 macro (parent): {m_full['f1_macro_parent']:.4f}")
    head_key = f"HEAD_{eval_head_k}"
    if head_key in results and labels_head:
        m_head = results[head_key]
        log.info(f"HEAD-{eval_head_k} label space ({len(labels_head)} codes):")
        log.info(f"  F1 micro: {m_head['f1_micro']:.4f}")
        log.info(f"  Precision micro: {m_head['precision_micro']:.4f}")
        log.info(f"  Recall micro: {m_head['recall_micro']:.4f}")
        log.info(f"  F1 macro: {m_head['f1_macro']:.4f}")
        log.info(f"  F1 samples: {m_head['f1_samples']:.4f}")
        if "f1_macro_parent" in m_head:
            log.info(f"  F1 macro (parent): {m_head['f1_macro_parent']:.4f}")
    if "mean_terms_per_visit" in m_full:
        log.info("Generation statistics:")
        log.info(f"  Mean terms per visit: {m_full['mean_terms_per_visit']:.2f}")
        log.info(f"  Mean mapped terms: {m_full['mean_mapped_terms_per_visit']:.2f}")
        log.info(f"  Unmappable term rate: {m_full['unmappable_term_rate']:.2f}")
    if "rougeL_vs_DischargeDiagnosis_mean" in m_full:
        log.info(f"Text quality:")
        log.info(f"  ROUGE-L vs Discharge Diagnosis: {m_full['rougeL_vs_DischargeDiagnosis_mean']:.4f}")

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--icd9_csv", default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.csv")

    # targets for SFT
    ap.add_argument("--target_mode", choices=["icd_titles","discharge_dx"], default="icd_titles")
    ap.add_argument("--icd_index_dir", default="./gen/TextGen/icd_index_v9")

    # llm & train
    ap.add_argument("--llm", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=128)
    ap.add_argument("--N_max_terms", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--early_stop", type=int, default=1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_assistant_tokens", type=int, default=128)

    # DDP (torchrun sets LOCAL_RANK)
    ap.add_argument("--local_rank", type=int, default=-1)

    # mapping (SapBERT+FAISS+fuzzy)
    ap.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--faiss_rows", type=int, default=50)
    ap.add_argument("--tau_cos", type=float, default=0.40)
    ap.add_argument("--tau_final", type=float, default=0.60)
    ap.add_argument("--w_cos", type=float, default=0.6)
    ap.add_argument("--w_fuz", type=float, default=0.4)

    # evaluation label space
    ap.add_argument("--eval_head_k", type=int, default=0)  # 0=full space

    args = ap.parse_args()

    # GPU niceties
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if torch.cuda.get_device_capability(0)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            try: torch.set_float32_matmul_precision("high")
            except Exception: pass
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # -------- load data --------
    if is_main_process():
        log.info(f"Loading data: {args.data_pickle}")
    try:
        df = pd.read_pickle(args.data_pickle)
    except Exception:
        with open(args.data_pickle, "rb") as f: df = pickle.load(f)

    subs = df[args.subject_col].dropna().unique()
    tr_subs, te_subs = train_test_split(subs, test_size=0.10, random_state=args.seed)
    tr_subs, va_subs = train_test_split(tr_subs, test_size=0.10/0.90, random_state=args.seed)
    train_df = df[df[args.subject_col].isin(tr_subs)].copy()
    val_df   = df[df[args.subject_col].isin(va_subs)].copy()
    test_df  = df[df[args.subject_col].isin(te_subs)].copy()
    if is_main_process():
        log.info(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # gold code lists
    def extract_codes(df_):
        out=[]
        for _, r in df_.iterrows():
            raw = r.get(args.label_col, [])
            lst = to_list(raw)
            lst = [format_icd9(c) for c in lst if c]
            lst = [c for c in lst if is_valid_icd9(c)]
            out.append(lst)
        return out
    train_gold = extract_codes(train_df)
    val_gold   = extract_codes(val_df)
    test_gold  = extract_codes(test_df)
    train_df["gold_codes"] = train_gold
    val_df["gold_codes"]   = val_gold
    test_df["gold_codes"]  = test_gold

    labels_full = build_eval_labels(train_gold)
    labels_head = build_eval_labels(train_gold, head_k=args.eval_head_k) if args.eval_head_k>0 else []

    if is_main_process():
        log.info(f"Label space (FULL): {len(labels_full)} codes")
        if labels_head: log.info(f"Head@{args.eval_head_k}: {len(labels_head)}")

    # -------- mapper --------
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )

    # -------- model & tokenizer --------
    model, tok = load_llm_with_lora(args.llm)

    # -------- datasets (token-budgeted) --------
    train_ds = SFTTextGenDataset(train_df, tok, args.label_col,
                                 target_mode=args.target_mode,
                                 icd_index_dir=args.icd_index_dir,
                                 max_len=args.max_len,
                                 N_max_terms=args.N_max_terms,
                                 min_assistant_tokens=args.min_assistant_tokens)
    val_ds   = SFTTextGenDataset(val_df, tok, args.label_col,
                                 target_mode=args.target_mode,
                                 icd_index_dir=args.icd_index_dir,
                                 max_len=args.max_len,
                                 N_max_terms=args.N_max_terms,
                                 min_assistant_tokens=args.min_assistant_tokens)
    if is_main_process():
        log.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")

    # -------- training args --------
    TA = TrainingArguments(
        output_dir="runs_textgen/checkpoints",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=1,
        load_best_model_at_end=bool(args.early_stop),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",

        # DDP & perf
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_timeout=28800,              # seconds (your request)
        local_rank=args.local_rank,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        # Memory
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        disable_tqdm=True,
    )

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    callbacks.append(BetterLoggingCallback())

    trainer = SafeTrainer(
        model=model,
        args=TA,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda feats: pad_collate(feats, tok),
        callbacks=callbacks
    )

    # -------- train --------
    t0 = time.time()
    if is_main_process():
        log.info("Starting training...")
    trainer.train()
    if is_main_process():
        log.info(f"Training completed in {(time.time()-t0)/60.0:.1f} min")

    # -------- test (rank-0 only; no manual barriers/destroy) --------
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    rank        = int(os.environ.get("RANK", "0"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    run_test = (world_size == 1) or (rank == 0)

    if run_test:
        gen_model = unwrap_model(trainer.model)
        device_id = 0 if world_size == 1 else local_rank
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        gen_model.to(device).eval()

        if hasattr(gen_model, "generation_config"):
            gc = gen_model.generation_config
            for a in ("temperature","top_p","top_k"):
                if hasattr(gc, a): setattr(gc, a, None)
            if hasattr(gc, "do_sample"): gc.do_sample = False

        log.info("=== Starting test evaluation ===")
        test_start = time.time()

        # Build prompts
        log.info("Generating diagnosis terms...")
        gen_start = time.time()
        def _safe_build_prompt(r):
            s=[]
            s.append(f"[VISIT] subject_id={r.get('subject_id_x','?')} hadm_id={r.get('hadm_id','?')}")
            s.append(serialize_structured_readable(r))
            notes = serialize_notes(r)
            if notes: s.append(notes)
            s.append("[TASK] List the final clinical diagnoses for this admission.")
            s.append("[FORMAT]")
            s.append("- One diagnosis per line")
            s.append("- Avoid abbreviations if possible")
            s.append("- No ICD codes or explanations")
            s.append(f"- Maximum: {args.N_max_terms} lines")
            s.append("[OUTPUT]")
            return "\n".join([x for x in s if x])

        test_prompts = [_safe_build_prompt(r) for _, r in test_df.iterrows()]

        terms_test = []
        bs = max(1, args.per_device_eval_batch_size)
        for i in range(0, len(test_prompts), bs):
            batch = test_prompts[i:i+bs]
            batch_results = generate_terms(
                gen_model, tok, batch,
                max_len=args.max_len, max_new=args.gen_max_new,
                batch_size=bs
            )
            terms_test.extend(batch_results)

        gen_time = time.time() - gen_start
        log.info(f"Generation completed in {gen_time:.2f}s ({gen_time/len(test_prompts):.2f}s per sample)")

        if terms_test:
            log.info("Sample generations:")
            for i in range(min(3, len(terms_test))):
                terms_to_show = terms_test[i][:3]
                log.info(f"  Sample {i+1}: {', '.join(terms_to_show)}" + ("..." if len(terms_test[i]) > 3 else ""))

        # Mapping
        log.info("Mapping terms to ICD codes...")
        map_start = time.time()
        mapped_test = mapper.map_terms(terms_test)
        map_time = time.time() - map_start
        log.info(f"Mapping completed in {map_time:.2f}s ({map_time/len(terms_test):.2f}s per sample)")

        if mapped_test:
            log.info("Sample ICD mappings:")
            for i in range(min(3, len(mapped_test))):
                codes_to_show = mapped_test[i][:5]
                log.info(f"  Sample {i+1}: {', '.join(codes_to_show)}" + ("..." if len(mapped_test[i]) > 5 else ""))

        # FULL space metrics
        log.info("Evaluating on FULL label space...")
        gold_full = restrict_to(test_df["gold_codes"].tolist(), labels_full)
        pred_full = restrict_to(mapped_test, labels_full)
        Yt = multihot(gold_full, labels_full); Yp = multihot(pred_full, labels_full)
        m_full = eval_pack(Yt, Yp)
        add_parent_macro_f1(m_full, gold_full, pred_full)

        results = {"FULL": m_full}

        # ---- Show 5 detailed samples (generated free text, mapped ICD, gold, and per-sample metrics) ----
        n_show = min(5, len(test_df))
        if n_show > 0:
            log.info("=== Sample diagnostics (FULL label space) ===")
            for i in range(n_show):
                g = gold_full[i]
                p = pred_full[i]
                terms = terms_test[i] if i < len(terms_test) else []

                prec, rec, f1, tp, gN, pN = per_sample_prf(g, p)
                log.info(f"[Sample {i+1}] P={prec:.2f} R={rec:.2f} F1={f1:.2f} | TP={tp} Pred={pN} Gold={gN}")

                if terms:
                    # Free-text diagnoses exactly as generated (top N lines)
                    shown_terms = terms[:args.N_max_terms]
                    log.info("  Generated terms:")
                    for t in shown_terms:
                        log.info(f"    - {t}")
                else:
                    log.info("  Generated terms: (none)")

                log.info("  Mapped ICD: " + (", ".join(p) if p else "(none)"))
                log.info("  GOLD ICD:   " + (", ".join(g) if g else "(none)"))


        # HEAD-K (optional)
        if labels_head:
            log.info(f"Evaluating on HEAD-{args.eval_head_k} label space...")
            gold_head = restrict_to(test_df["gold_codes"].tolist(), labels_head)
            pred_head = restrict_to(mapped_test, labels_head)
            Yt_h = multihot(gold_head, labels_head); Yp_h = multihot(pred_head, labels_head)
            m_head = eval_pack(Yt_h, Yp_h)
            add_parent_macro_f1(m_head, gold_head, pred_head)
            results[f"HEAD_{args.eval_head_k}"] = m_head

        # Diagnostics
        if mapper.last_stats:
            n_terms = np.array([n for (n,m) in mapper.last_stats], dtype=np.float32)
            n_map   = np.array([m for (n,m) in mapper.last_stats], dtype=np.float32)
            results["FULL"]["mean_terms_per_visit"] = float(n_terms.mean())
            results["FULL"]["mean_mapped_terms_per_visit"] = float(n_map.mean())
            results["FULL"]["unmappable_term_rate"] = float(np.mean(np.where(n_terms>0, 1.0 - (n_map/np.maximum(n_terms,1)), 0.0)))

        # Optional ROUGE-L vs Discharge Diagnosis text
        if "Discharge Diagnosis" in test_df.columns:
            try:
                log.info("Computing ROUGE-L against Discharge Diagnosis...")
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                gold_texts = [clean_text(x) for x in test_df["Discharge Diagnosis"].tolist()]
                gen_texts  = ["\n".join(t) for t in terms_test]
                R = [scorer.score(g, p)["rougeL"].fmeasure for g,p in zip(gold_texts, gen_texts)]
                results["FULL"]["rougeL_vs_DischargeDiagnosis_mean"] = float(np.mean(R))
            except Exception as e:
                log.warning(f"ROUGE not computed: {e}")

        # Log summarized results & save
        log_test_results(results, labels_full, labels_head, args.eval_head_k)
        os.makedirs("runs_textgen", exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join("runs_textgen", f"test_metrics_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {results_path}")

        test_time = time.time() - test_start
        log.info(f"Test evaluation completed in {test_time/60:.1f} min ({test_time:.2f}s)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
