# -*- coding: utf-8 -*-
"""
Simple finetuning script with a single flag to control TRAIN SIZE.
(change --train_size per run):
"""

import os, re, json, random, logging, pickle, datetime, time, atexit, math, argparse
from typing import List, Any, Dict
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, EarlyStoppingCallback, Trainer
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- Env & logging ----------------
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


print("CUDA:", torch.cuda.is_available(),
      "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Enable TF32 on Ampere+ (A100/H100, etc.)
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass


def _cleanup_dist():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try: torch.distributed.barrier()
            except Exception: pass
            torch.distributed.destroy_process_group()
    except Exception:
        pass
atexit.register(_cleanup_dist)

# ---------------- Args ----------------

def get_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", default=None, help="If provided, will subject-split into train/val/test.")
    ap.add_argument("--train_pickle", default=None)
    ap.add_argument("--val_pickle", default=None)
    ap.add_argument("--test_pickle", default=None)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")

    # model/prompt
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--tgt_reserve_tok", type=int, default=128)
    ap.add_argument("--gen_max_new", type=int, default=96)

    # training
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--early_stop", type=int, default=1)
    ap.add_argument("--patience", type=int, default=2)

    # size & seed
    ap.add_argument("--train_size", type=int, default=-1, help="Number of training rows to use (subject-safe subset). -1=all")
    ap.add_argument("--seed", type=int, default=42)

    # run dirs
    ap.add_argument("--run_root", default="runs_gen/diffsize")
    ap.add_argument("--run_name", default=None)

    # misc
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--merge_after", type=int, default=0)

    return ap.parse_args()

# ---------------- Prompting helpers ----------------

TEXT_COLS_SAFE = [
    "Chief Complaint","History of Present Illness","Past Medical History",
    "Family History","Physical Exam","Pertinent Results",
    "Brief Hospital Course","Medications on Admission"
]


def clean_text(x: Any) -> str:
    if pd.isna(x): return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s)
    return re.sub(r"\s+"," ", s).strip()


def to_list(x) -> List[str]:
    if isinstance(x, list): return [str(v) for v in x]
    if pd.isna(x): return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast; v = ast.literal_eval(s)
            if isinstance(v, list): return [str(z) for z in v]
        except Exception: pass
    return [t for t in re.split(r"[,\s]+", s) if t]


def serialize_structured(row: pd.Series) -> str:
    parts = []
    parts.append(f"[DEM] gender={row.get('gender','')} age_group={row.get('age','')} "
                 f"stay_days={row.get('stay_days','')} death={row.get('death','')}")
    ndc  = to_list(row.get("ndc", []))
    proc = to_list(row.get("pro_code", []))
    labs = to_list(row.get("lab_test", []))
    if ndc:  parts.append("[NDC] "  + " ".join(ndc[:32]))
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


def build_input_text(row: pd.Series, use_structured=True, use_notes=True,
                     subject_col="subject_id_x") -> str:
    s = [f"[VISIT] subject_id={row.get(subject_col,'?')} hadm_id={row.get('hadm_id','?')}"]
    if use_structured: s.append(serialize_structured(row))
    if use_notes:
        t = serialize_notes(row)
        if t: s.append(t)
    s.append("[TASK] Predict ICD diagnosis codes (space-separated). Output ONLY the codes, separated by single spaces.")
    s.append("[CODES]")
    return "\n".join([x for x in s if x])

# ---------------- Splits & labels ----------------

def subject_splits(df: pd.DataFrame, subject_col: str,
                   test_size=0.10, val_size=0.10, seed=42):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    logging.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te


def lock_label_space(frames: List[pd.DataFrame], label_col: str) -> MultiLabelBinarizer:
    all_codes = set()
    for fr in frames:
        for codes in fr[label_col]:
            all_codes.update(str(c) for c in codes)
    all_codes = sorted(all_codes)
    mlb = MultiLabelBinarizer(classes=all_codes); mlb.fit([all_codes])
    logging.info(f"Total unique ICD codes: {len(all_codes)}")
    return mlb


def y_multi_hot(mlb: MultiLabelBinarizer, lists):
    return mlb.transform([[str(c) for c in row] for row in lists])

# ---------------- Subject-safe subsetting ----------------

def nested_subject_sample(train_df, target_n, subject_col="subject_id_x", seed=13):
    if target_n is None or target_n < 0 or target_n >= len(train_df):
        return train_df.copy()
    rng = np.random.default_rng(seed)
    subjects = train_df[subject_col].drop_duplicates().tolist()
    rng.shuffle(subjects)
    chosen, count = [], 0
    for s in subjects:
        rows = train_df[train_df[subject_col] == s]
        if count + len(rows) <= target_n or len(chosen) == 0:
            chosen.append(s)
            count += len(rows)
        if count >= target_n: break
    sub = train_df[train_df[subject_col].isin(chosen)].copy()
    logging.info(f"[subset] requested={target_n} got={len(sub)} unique_subjects={sub[subject_col].nunique()}")
    return sub

# ---------------- Dataset (pre-tokenized) ----------------

class GenCodesDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tok, max_len: int, tgt_reserve: int, label_col: str):
        self.tok = tok
        self.max_len = max_len
        self.tgt_reserve = max(8, int(tgt_reserve))
        self.label_col = label_col
        prompts = rows["input_text"].astype(str).tolist()
        targets = [" ".join(sorted({str(c) for c in codes})) for codes in rows[label_col].tolist()]
        self.prompt_ids = [tok.encode(p + "\n", add_special_tokens=True) for p in prompts]
        eos = (tok.eos_token or "")
        self.ans_ids    = [tok.encode(t + eos, add_special_tokens=False) for t in targets]

    def __len__(self): return len(self.prompt_ids)

    def __getitem__(self, i):
        prompt_ids = self.prompt_ids[i]
        ans_ids    = self.ans_ids[i]
        max_prompt_len = max(1, self.max_len - self.tgt_reserve)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]
        remaining = max(1, self.max_len - len(prompt_ids))
        if len(ans_ids) > remaining:
            ans_ids = ans_ids[:remaining]
        input_ids = prompt_ids + ans_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + ans_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pad_collate(features, tok):
    input_ids = [f["input_ids"] for f in features]
    attn     = [f["attention_mask"] for f in features]
    labels   = [f["labels"] for f in features]
    pad_out  = tok.pad({"input_ids": input_ids, "attention_mask": attn}, return_tensors="pt")
    max_len = pad_out["input_ids"].size(1)
    lab_pad = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, lab in enumerate(labels):
        lab_pad[i, :lab.size(0)] = lab
    return {"input_ids": pad_out["input_ids"], "attention_mask": pad_out["attention_mask"], "labels": lab_pad}

# ---------------- Model ----------------

def load_lm_and_tokenizer(model_name):
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora_cfg)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tok

# ---------------- Generation & metrics ----------------

def normalize_code(c: str) -> str:
    c = c.strip().upper()
    c = re.sub(r"\s+", "", c)
    return c[:-1] if c.endswith(".") else c


@torch.no_grad()
def generate_codes(model, tok, prompts: List[str], labels_vocab: List[str],
                   max_new=96, batch_size=4, max_len=3072):
    model.eval()
    allowed = set(labels_vocab)
    preds = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tok(batch_prompts, return_tensors="pt",
                     padding=True, truncation=True, max_length=max_len).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=2,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        seq = out.sequences
        gen_only = seq[:, inputs["input_ids"].shape[1]:]  # only new tokens
        texts = tok.batch_decode(gen_only, skip_special_tokens=True)
        for t in texts:
            tokens = re.split(r"[^A-Za-z0-9\.]+", t)
            cand = [normalize_code(z) for z in tokens if z]
            seen, keep = set(), []
            for c in cand:
                if c in allowed and c not in seen:
                    seen.add(c); keep.append(c)
            preds.append(keep)
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
        "micro_f1":   f1_score(y_true, y_pred, average="micro",   zero_division=0),
        "macro_f1":   f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "micro_precision":   precision_score(y_true, y_pred, average="micro",   zero_division=0),
        "macro_precision":   precision_score(y_true, y_pred, average="macro",   zero_division=0),
        "samples_precision": precision_score(y_true, y_pred, average="samples", zero_division=0),
        "micro_recall":      recall_score(y_true, y_pred, average="micro",   zero_division=0),
        "macro_recall":      recall_score(y_true, y_pred, average="macro",   zero_division=0),
        "samples_recall":    recall_score(y_true, y_pred, average="samples", zero_division=0),
    }

# ---------------- Run helpers ----------------

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(base, tag):
    path = os.path.join(base, tag)
    os.makedirs(path, exist_ok=False)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


def save_json(path: str, obj: dict):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ---------------- Main ----------------

def main():
    args = get_args()
    set_seed(args.seed)

    # Load data
    if args.train_pickle and args.val_pickle and args.test_pickle:
        train_df = pickle.load(open(args.train_pickle, "rb"))
        val_df   = pickle.load(open(args.val_pickle, "rb"))
        test_df  = pickle.load(open(args.test_pickle, "rb"))
    elif args.data_pickle:
        full_df = pickle.load(open(args.data_pickle, "rb"))
        train_df, val_df, test_df = subject_splits(full_df, subject_col=args.subject_col, test_size=0.10, val_size=0.10, seed=args.seed)
    else:
        raise ValueError("Provide either --data_pickle OR all of --train_pickle/--val_pickle/--test_pickle")

    for df_ in (train_df, val_df, test_df):
        assert args.label_col in df_.columns and args.subject_col in df_.columns

    # Subject-safe subsetting of TRAIN
    train_df = nested_subject_sample(train_df, args.train_size, subject_col=args.subject_col, seed=args.seed)

    # Build prompts
    for df_, name in ((train_df, 'train'), (val_df, 'val'), (test_df, 'test')):
        df_["input_text"] = df_.apply(lambda r: build_input_text(r, args.use_structured==1, args.use_notes==1, args.subject_col), axis=1)
        logging.info(f"[{name}] rows with input_text: {df_['input_text'].notna().sum()}")

    # Label space union (consistent filter)
    mlb = lock_label_space([train_df, val_df, test_df], args.label_col)
    labels_vocab = mlb.classes_.tolist()
    y_val  = y_multi_hot(mlb, val_df[args.label_col].tolist())
    y_test = y_multi_hot(mlb, test_df[args.label_col].tolist())

    # Model & tokenizer
    model, tok = load_lm_and_tokenizer(args.llama_model)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")

    # Datasets
    train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col)
    val_ds   = GenCodesDataset(val_df,   tok, args.max_len, args.tgt_reserve_tok, args.label_col)

    # Run dir
    tag = args.run_name or f"{now_tag()}_N{args.train_size}"
    RUN_DIR = make_run_dir(args.run_root, tag)
    logging.info(f"Run dir: {RUN_DIR}")

    save_json(os.path.join(RUN_DIR, "config.json"), {
        "model": args.llama_model, "max_len": args.max_len, "lr": args.learning_rate,
        "epochs": args.epochs, "gen_max_new": args.gen_max_new, "tgt_reserve_tok": args.tgt_reserve_tok,
        "seed": args.seed, "train_rows": len(train_df)
    })
    save_json(os.path.join(RUN_DIR, "label_space.json"), {"labels": labels_vocab})

    # Training args (simple)
    train_args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        logging_strategy="epoch",
        eval_strategy="epoch",
        prediction_loss_only=True,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=bool(args.early_stop),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim="adamw_torch", dataloader_num_workers=2,
        run_name=os.path.basename(RUN_DIR),
        disable_tqdm=True,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.early_stop else []

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=lambda feats: pad_collate(feats, tok),
    )

    # Train
    t0 = time.perf_counter()
    logging.info("Starting training…")
    trainer.train()
    train_secs = time.perf_counter() - t0
    logging.info(f"[TIME] train: {train_secs:.2f}s")

    # Save adapter
    tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
    trainer.model.save_pretrained(os.path.join(RUN_DIR, "adapter_best"))

    if args.merge_after:
        try:
            merged_dir = os.path.join(RUN_DIR, "model_merged")
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir)
            tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
            logging.info(f"Merged model saved to: {merged_dir}")
        except Exception as e:
            logging.warning(f"Could not merge adapters into base: {e}")

    # Final TEST generation
    test_prompts = test_df["input_text"].astype(str).tolist()
    t_gen = time.perf_counter()
    pred_code_lists = generate_codes(
        model, tok, test_prompts, labels_vocab,
        max_new=args.gen_max_new, batch_size=args.per_device_eval_batch_size, max_len=args.max_len
    )
    test_gen_secs = time.perf_counter() - t_gen

    Y_pred = codes_to_multihot(pred_code_lists, labels_vocab)
    metrics = eval_sets(y_test, Y_pred)
    metrics["train_seconds"] = train_secs
    metrics["test_generate_seconds"] = test_gen_secs

    with open(os.path.join(RUN_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Generative TEST metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
