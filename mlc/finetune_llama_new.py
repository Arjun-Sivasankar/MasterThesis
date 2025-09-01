# -*- coding: utf-8 -*-

import os, re, json, random, logging, pickle, datetime
import yaml
import wandb
from typing import List, Tuple, Any
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, EarlyStoppingCallback, Trainer, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


def load_secrets(path="secrets.yaml"):
    """Read secrets.yaml and set env vars for WANDB_API_KEY and HUGGINGFACE_HUB_TOKEN.
       Returns dict with wandb config & flags."""
    cfg = {"auth": {}, "wandb": {}}
    if not os.path.isfile(path):
        logging.warning(f"secrets file not found: {path}. W&B/HF auth may fail if private models or logging enabled.")
        return {"wandb_ok": False, "hf_ok": False, "wandb_cfg": {}}

    if yaml is None:
        logging.warning("PyYAML missing; cannot parse secrets.yaml.")
        return {"wandb_ok": False, "hf_ok": False, "wandb_cfg": {}}

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    auth = (data.get("auth") or {})
    wb   = (data.get("wandb") or {})

    wandb_token = auth.get("wandb_token")
    hf_token    = auth.get("hf_token")

    if wandb_token:
        os.environ.setdefault("WANDB_API_KEY", wandb_token)
    if hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    # let wandb know the project
    if "project" in wb:
        os.environ.setdefault("WANDB_PROJECT", wb["project"])

    return {
        "wandb_ok": bool(wandb_token),
        "hf_ok": bool(hf_token),
        "wandb_cfg": wb,
    }

# ===== 0) Repro, logging, TF32, quiet progress =====
os.environ["HF_DISABLE_PROGRESS_BAR"] = "1"  # kill tqdm bars

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
print("CUDA:", torch.cuda.is_available(), "| Device:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

# ===== 1) Config =====
SMOKE_TEST  = False      # fast pipeline check
EARLY_STOP  = True       # <<< toggle early stopping on/off
PATIENCE    = 2
WANDB_ONLINE = True

RUN_NAME    = None       # set custom name, else timestamp
DATA_PICKLE = "mergeddf.pkl"

USE_STRUCTURED, USE_NOTES = True, True
SUBJECT_COL, LABEL_COL = "subject_id_x", "icd_code"

# You can trim/expand this list as needed; avoid discharge-time leakage columns
TEXT_COLS_SAFE = [
    "Chief Complaint", "History of Present Illness", "Past Medical History",
    "Family History", "Physical Exam", "Pertinent Results",
    "Brief Hospital Course", "Medications on Admission"
]
STRUCT_COLS = {"gender":"gender","age":"age","stay_days":"stay_days","death":"death",
               "ndc":"ndc","pro_code":"pro_code","lab_test":"lab_test"}
CAP_NDC, CAP_PROC, CAP_LABS = 32, 32, 64

LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LEN = 3072
LR = 2e-4
EPOCHS = 20

print(f"\nModel Configs: \nMAX_LEN:{MAX_LEN}, LR:{LR}, EPOCHS:{EPOCHS}")
print(f"\nOther Configs: Smoke Test:{SMOKE_TEST}, Early Stop:{EARLY_STOP}, W&B online:{WANDB_ONLINE}")

# smoke-test knobs
DEBUG_MAX_LEN = 768; DEBUG_TRAIN_SUBJ = 120; DEBUG_VAL_SUBJ = 30; DEBUG_TEST_SUBJ = 30
MAX_STEPS = 200; EVAL_STEPS = 50; LOG_STEPS = 10

# ===== 2) Small I/O helpers =====
def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def make_run_dir(base="runs", run_name=None):
    tag = run_name or f"{now_tag()}_llama1b_len{MAX_LEN}_lr{LR}"
    path = os.path.join(base, tag)
    os.makedirs(path, exist_ok=False)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path

def save_df(df: pd.DataFrame, path: str): df.to_pickle(path)
def save_npz(path: str, **arrays): np.savez_compressed(path, **arrays)
def save_json(path: str, obj: dict):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ===== 3) Input serialization =====
def clean_text(x: Any) -> str:
    if pd.isna(x): return ""
    s = str(x).replace("\x00"," ").replace("\r"," ")
    s = re.sub(r"_+"," ", s); s = re.sub(r"\s+"," ", s).strip()
    return s

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
    ndc  = to_list(row.get("ndc", []))[:CAP_NDC]
    proc = to_list(row.get("pro_code", []))[:CAP_PROC]
    labs = to_list(row.get("lab_test", []))[:CAP_LABS]
    if ndc:  parts.append("[NDC] "  + " ".join(ndc))
    if proc: parts.append("[PROC] " + " ".join(proc))
    if labs: parts.append("[LAB] "  + " ".join(labs))
    return "\n".join(parts)

def serialize_notes(row: pd.Series) -> str:
    chunks=[]
    for col in TEXT_COLS_SAFE:
        if col in row:
            t = clean_text(row[col])
            if t: chunks.append(f"[{col.upper()}] {t}")
    return "\n".join(chunks)

def build_input_text(row: pd.Series) -> str:
    s = [f"[VISIT] subject_id={row.get(SUBJECT_COL,'?')} hadm_id={row.get('hadm_id','?')}"]
    if USE_STRUCTURED: s.append(serialize_structured(row))
    if USE_NOTES:
        t = serialize_notes(row)
        if t: s.append(t)
    s.append("[TASK] Predict ICD diagnosis codes (space-separated).")
    return "\n".join([x for x in s if x])

# ===== 4) Splits & label space =====
def subject_splits(df: pd.DataFrame, subject_col=SUBJECT_COL,
                   test_size=0.10, val_size=0.10, seed=42):
    subs = df[subject_col].dropna().unique()
    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()
    logging.info(f"Split sizes (visits): train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

def limit_subjects(df: pd.DataFrame, subject_col: str, n: int) -> pd.DataFrame:
    subs = df[subject_col].dropna().unique().tolist()
    random.shuffle(subs); keep = set(subs[:min(n,len(subs))])
    return df[df[subject_col].isin(keep)].copy()

def lock_label_space(full_df: pd.DataFrame) -> MultiLabelBinarizer:
    all_codes = sorted({str(code) for codes in full_df[LABEL_COL] for code in codes})
    mlb = MultiLabelBinarizer(classes=all_codes); mlb.fit([all_codes])
    logging.info(f"Total unique ICD codes: {len(all_codes)}"); return mlb

def to_multi_hot(mlb: MultiLabelBinarizer, code_lists):
    return mlb.transform([[str(c) for c in row] for row in code_lists])

def compute_pos_weight(train_Y: np.ndarray) -> torch.Tensor:
    N = train_Y.shape[0]; pos = np.clip(train_Y.sum(axis=0), 1.0, None)
    neg = N - pos; w = np.clip(neg/pos, 1.0, 50.0)
    return torch.tensor(w, dtype=torch.float32)

# ===== 5) Dataset =====
class EHRNotesDataset(Dataset):
    def __init__(self, texts, labels, tok, max_length):
        self.texts = texts; self.labels = labels.astype(np.float32)
        self.tok = tok; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

# ===== 6) Model loader =====
def load_model_and_tokenizer(num_labels: int):
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    config = AutoConfig.from_pretrained(
        LLAMA_MODEL, num_labels=num_labels,
        problem_type="multi_label_classification",
        pad_token_id=tok.pad_token_id
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        LLAMA_MODEL, config=config, torch_dtype=dtype, device_map="auto"
    )
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
    return model, tok, "adamw_torch"

# ===== 7) Trainer & metrics =====
class PosWeightTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs); self.pos_weight = pos_weight
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits = outputs.logits
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))(logits, labels)
        return (loss, outputs) if return_outputs else loss

class EpochLogger(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_loss_by_epoch = {}
        self.eval_by_epoch = {}
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "epoch" not in logs: return
        ep = int(round(float(logs["epoch"])))
        if "loss" in logs and "eval_loss" not in logs:
            self.train_loss_by_epoch.setdefault(ep, float(logs["loss"]))
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        ep = int(np.ceil(state.epoch or 0))
        ev = {k: float(v) for k, v in (metrics or {}).items()}
        self.eval_by_epoch[ep] = ev
        tr = self.train_loss_by_epoch.get(ep, float("nan"))
        logging.info(
            f"[Epoch {ep}] "
            f"train_loss={tr:.4f} | "
            f"val_loss={ev.get('eval_loss', float('nan')):.4f} | "
            f"micro_f1={ev.get('eval_micro_f1', float('nan')):.4f} | "
            f"macro_f1={ev.get('eval_macro_f1', float('nan')):.4f} | "
            f"samples_f1={ev.get('eval_samples_f1', float('nan')):.4f}"
        )

def make_compute_metrics(th=0.5):
    def _metrics(eval_pred):
        if isinstance(eval_pred, tuple): 
            logits, labels = eval_pred
        else: 
            logits, labels = eval_pred.predictions, eval_pred.label_ids
            
        probs = 1/(1+np.exp(-logits))
        y_pred = (probs >= th).astype(int)
        y_true = labels
        
        return {
            # F1
            "micro_f1":   f1_score(y_true, y_pred, average="micro",   zero_division=0),
            "macro_f1":   f1_score(y_true, y_pred, average="macro",   zero_division=0),
            "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),

            # Precision
            "micro_precision":   precision_score(y_true, y_pred, average="micro",   zero_division=0),
            "macro_precision":   precision_score(y_true, y_pred, average="macro",   zero_division=0),
            "samples_precision": precision_score(y_true, y_pred, average="samples", zero_division=0),

            # Recall
            "micro_recall":   recall_score(y_true, y_pred, average="micro",   zero_division=0),
            "macro_recall":   recall_score(y_true, y_pred, average="macro",   zero_division=0),
            "samples_recall": recall_score(y_true, y_pred, average="samples", zero_division=0),
        }
    return _metrics

def precision_recall_at_k(y_true, y_score, k:int):
    N,L = y_true.shape; k=min(k,L)
    topk = np.argpartition(-y_score, kth=min(k-1,L-1), axis=1)[:, :k]
    pred = np.zeros_like(y_true, dtype=np.int32); pred[np.arange(N)[:,None], topk]=1
    tp = (pred & (y_true==1)).sum()
    return float(tp/(k*N+1e-12)), float(tp/(y_true.sum()+1e-12))

def tune_threshold(y_true, y_score, grid=np.arange(0.1,0.91,0.05)):
    best_t, best_micro = 0.5, -1.0
    for t in grid:
        micro = f1_score(y_true, (y_score>=t).astype(int), average="micro", zero_division=0)
        if micro>best_micro: best_micro, best_t = micro, float(t)
    return best_t

def macro_auroc(y_true, y_score):
    vals=[]
    for j in range(y_true.shape[1]):
        if y_true[:,j].max()!=y_true[:,j].min():
            try: vals.append(roc_auc_score(y_true[:,j], y_score[:,j]))
            except Exception: pass
    return float(np.mean(vals)) if vals else float("nan")

# ===== 8) Load secrets, data & build splits =====
secrets_info = load_secrets("secrets.yaml")
if not WANDB_ONLINE:
    os.environ["WANDB_MODE"] = "offline"
wandb_enabled = secrets_info["wandb_ok"]

final_df = pickle.load(open(DATA_PICKLE,"rb"))
assert LABEL_COL in final_df.columns and SUBJECT_COL in final_df.columns

df = final_df.copy()
df["input_text"] = df.apply(build_input_text, axis=1)
df = df[df["input_text"].str.len() > 0]

train_df, val_df, test_df = subject_splits(df, subject_col=SUBJECT_COL, test_size=0.10, val_size=0.10, seed=42)

if SMOKE_TEST:
    MAX_LEN = DEBUG_MAX_LEN
    train_df = limit_subjects(train_df, SUBJECT_COL, DEBUG_TRAIN_SUBJ)
    val_df   = limit_subjects(val_df,   SUBJECT_COL, DEBUG_VAL_SUBJ)
    test_df  = limit_subjects(test_df,  SUBJECT_COL, DEBUG_TEST_SUBJ)
    logging.info(f"[SMOKE] visits -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

mlb = lock_label_space(df)
y_train = to_multi_hot(mlb, train_df[LABEL_COL].tolist())
y_val   = to_multi_hot(mlb, val_df[LABEL_COL].tolist())
y_test  = to_multi_hot(mlb, test_df[LABEL_COL].tolist())

pos_weight = compute_pos_weight(y_train)
num_labels = len(mlb.classes_)
print(f"num_labels={num_labels}")

# ===== 9) Create unique run dir + persist splits/labels/config =====
RUN_DIR = make_run_dir(base="runs", run_name=RUN_NAME)
print("Run dir:", RUN_DIR)

#save_df(train_df, os.path.join(RUN_DIR, "train_df.pkl"))
#save_df(val_df,   os.path.join(RUN_DIR, "val_df.pkl"))
#save_df(test_df,  os.path.join(RUN_DIR, "test_df.pkl"))
save_npz(os.path.join(RUN_DIR, "labels.npz"), y_train=y_train, y_val=y_val, y_test=y_test)
save_json(os.path.join(RUN_DIR, "config.json"), {
    "model": LLAMA_MODEL, "max_len": MAX_LEN, "lr": LR, "epochs": EPOCHS,
    "use_structured": USE_STRUCTURED, "use_notes": USE_NOTES,
    "smoke_test": SMOKE_TEST, "early_stop": EARLY_STOP, "patience": PATIENCE
})


if wandb_enabled:
    run_name = os.path.basename(RUN_DIR)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", secrets_info["wandb_cfg"].get("project", "finetune")),
        name=run_name,
        tags=secrets_info["wandb_cfg"].get("tags", []),
        config={
            "model": LLAMA_MODEL, "max_len": MAX_LEN, "lr": LR, "epochs": EPOCHS,
            "smoke_test": SMOKE_TEST, "early_stop": EARLY_STOP, "patience": PATIENCE,
            "num_labels": num_labels,
        },
        reinit=True,
    )

# ===== 10) Train =====
model, tok, optim_hint = load_model_and_tokenizer(num_labels=num_labels)

train_ds = EHRNotesDataset(train_df["input_text"].tolist(), y_train, tok, MAX_LEN)
val_ds   = EHRNotesDataset(val_df["input_text"].tolist(),   y_val,   tok, MAX_LEN)
test_ds  = EHRNotesDataset(test_df["input_text"].tolist(),  y_test,  tok, MAX_LEN)

REPORT_TO = ["wandb"] if wandb_enabled else "none"

if SMOKE_TEST:
    args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=100, max_steps=MAX_STEPS,
        learning_rate=LR,
        per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
        per_device_eval_batch_size=2 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.0, weight_decay=0.0,
        logging_strategy="steps", logging_steps=LOG_STEPS,
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="no", load_best_model_at_end=False,
        report_to=REPORT_TO,
        gradient_checkpointing=True,
        remove_unused_columns=False, label_names=["labels"],
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim=optim_hint, dataloader_num_workers=2,
        disable_tqdm=True,
        run_name=os.path.basename(RUN_DIR),
    )
    callbacks = [EpochLogger()]
else:
    args = TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
        per_device_eval_batch_size=2 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.03, weight_decay=0.0,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=EARLY_STOP,                 # if early stop, keep best in memory
        metric_for_best_model="eval_micro_f1",             # used by early stopping & load_best
        greater_is_better=True,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False, label_names=["labels"],
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim=optim_hint, dataloader_num_workers=2,
        disable_tqdm=True,
        run_name=os.path.basename(RUN_DIR),
    )
    callbacks = [EpochLogger()] + ([EarlyStoppingCallback(early_stopping_patience=PATIENCE)] if EARLY_STOP else [])

trainer = PosWeightTrainer(
    pos_weight=pos_weight, model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok,
    compute_metrics=make_compute_metrics(th=0.5), callbacks=callbacks
)

logging.info("Starting training…")
trainer.train()
logging.info("Training complete.")

# Save artifacts: tokenizer, current (best or last) adapter, label space
tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
trainer.model.save_pretrained(os.path.join(RUN_DIR, "adapter_best"))
save_json(os.path.join(RUN_DIR, "label_space.json"), {"labels": mlb.classes_.tolist()})

# === Save merged full model (adapters + trained classifier head) ===
try:
    merged_dir = os.path.join(RUN_DIR, "model_merged")
    merged = trainer.model.merge_and_unload()    # fold LoRA into base
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(os.path.join(RUN_DIR, "tokenizer"))
    logging.info(f"Merged model saved to: {merged_dir}")
except Exception as e:
    logging.warning(f"Could not merge adapters into base: {e}")

# ===== 11) Tune threshold on val + save =====
val_out = trainer.predict(val_ds)
val_probs = 1/(1+np.exp(-val_out.predictions))
best_t = tune_threshold(y_val, val_probs)
print(f"Best threshold (best_t) = {best_t}")
save_json(os.path.join(RUN_DIR, "val_threshold.json"), {"best_threshold": float(best_t)})

# ===== 12) Test metrics (in-memory model) =====
test_out = trainer.predict(test_ds)
test_probs = 1/(1+np.exp(-test_out.predictions))
y_pred = (test_probs >= best_t).astype(int)

metrics = {
    # F1
    "micro_f1":   f1_score(y_test, y_pred, average="micro",   zero_division=0),
    "macro_f1":   f1_score(y_test, y_pred, average="macro",   zero_division=0),
    "samples_f1": f1_score(y_test, y_pred, average="samples", zero_division=0),
    
    # Precision
    "micro_precision":   precision_score(y_test, y_pred, average="micro",   zero_division=0),
    "macro_precision":   precision_score(y_test, y_pred, average="macro",   zero_division=0),
    "samples_precision": precision_score(y_test, y_pred, average="samples", zero_division=0),

    # Recall
    "micro_recall":   recall_score(y_test, y_pred, average="micro",   zero_division=0),
    "macro_recall":   recall_score(y_test, y_pred, average="macro",   zero_division=0),
    "samples_recall": recall_score(y_test, y_pred, average="samples", zero_division=0),
}
try: metrics["micro_auroc"] = roc_auc_score(y_test, test_probs, average="micro")
except Exception: metrics["micro_auroc"] = float("nan")
def macro_auroc_local(y_true, y_score):
    vals=[]
    for j in range(y_true.shape[1]):
        if y_true[:,j].max()!=y_true[:,j].min():
            try: vals.append(roc_auc_score(y_true[:,j], y_score[:,j]))
            except Exception: pass
    return float(np.mean(vals)) if vals else float("nan")
metrics["macro_auroc"] = macro_auroc_local(y_test, test_probs)
for k in (5, 8, 10):
    p, r = precision_recall_at_k(y_test, test_probs, k=k)
    metrics[f"precision@{k}"] = p; metrics[f"recall@{k}"] = r

save_json(os.path.join(RUN_DIR, "test_metrics.json"), metrics)
print("\n=== Test metrics (this run) ===")
print(json.dumps(metrics, indent=2))

# Log to W&B summary and close
if wandb_enabled:
    wandb.summary.update({"best_threshold": float(best_t)})
    wandb.summary.update(metrics)
    wandb.finish()

# ===== 13) Reload from saved run and re-evaluate on test =====
def load_from_run(run_dir: str):
    tok = AutoTokenizer.from_pretrained(os.path.join(run_dir, "tokenizer"), use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    merged_dir = os.path.join(run_dir, "model_merged")
    if os.path.isdir(merged_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            merged_dir, torch_dtype=dtype, device_map="auto"
        )
        return model, tok

    # fallback (not recommended): base + adapter (classifier head will be fresh)
    label_space = json.load(open(os.path.join(run_dir, "label_space.json")))
    num_labels = len(label_space["labels"])
    config = AutoConfig.from_pretrained(
        LLAMA_MODEL, num_labels=num_labels, problem_type="multi_label_classification",
        pad_token_id=tok.pad_token_id
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        LLAMA_MODEL, config=config, torch_dtype=dtype, device_map="auto"
    )
    base.config.pad_token_id = tok.pad_token_id; base.config.use_cache = False
    model = PeftModel.from_pretrained(base, os.path.join(run_dir, "adapter_best"))
    model.eval()
    return model, tok

print("\n[Reload test] loading from:", RUN_DIR)
best_model, best_tok = load_from_run(RUN_DIR)
test_ds_reload = EHRNotesDataset(test_df["input_text"].tolist(), y_test, best_tok, MAX_LEN)

eval_trainer = Trainer(
    model=best_model,
    args=TrainingArguments(
        output_dir=os.path.join(RUN_DIR, "eval_tmp"),
        per_device_eval_batch_size=2 if torch.cuda.is_available() else 1,
        dataloader_num_workers=2,
        remove_unused_columns=False, label_names=["labels"],
        report_to=REPORT_TO,
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        disable_tqdm=True,
        run_name=os.path.basename(RUN_DIR) + "_reload_eval",
    ),
    eval_dataset=test_ds_reload,
    tokenizer=best_tok,
)

best_t_loaded = json.load(open(os.path.join(RUN_DIR, "val_threshold.json")))["best_threshold"]
test_logits_reload = eval_trainer.predict(test_ds_reload).predictions
test_probs_reload  = 1/(1+np.exp(-test_logits_reload))
y_pred_reload = (test_probs_reload >= best_t_loaded).astype(int)

metrics_reload = {
    "micro_f1":   f1_score(y_test, y_pred_reload, average="micro",   zero_division=0),
    "macro_f1":   f1_score(y_test, y_pred_reload, average="macro",   zero_division=0),
    "samples_f1": f1_score(y_test, y_pred_reload, average="samples", zero_division=0),
}
try: metrics_reload["micro_auroc"] = roc_auc_score(y_test, test_probs_reload, average="micro")
except Exception: metrics_reload["micro_auroc"] = float("nan")
metrics_reload["macro_auroc"] = macro_auroc_local(y_test, test_probs_reload)
for k in (5, 8, 10):
    p, r = precision_recall_at_k(y_test, test_probs_reload, k=k)
    metrics_reload[f"precision@{k}"] = p; metrics_reload[f"recall@{k}"] = r

print("\n=== Test metrics (reloaded run) ===")
print(json.dumps(metrics_reload, indent=2))
save_json(os.path.join(RUN_DIR, "test_metrics_reload.json"), metrics_reload)
if wandb_enabled:
    wandb.summary.update({f"reload_{k}": v for k, v in metrics_reload.items()})
