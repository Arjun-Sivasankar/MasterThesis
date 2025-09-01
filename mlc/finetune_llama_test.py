# -*- coding: utf-8 -*-

from peft import PeftModel
import os

import os, re, json, random, logging
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
    BitsAndBytesConfig,  
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

CKPT_DIR = "llama32_1b_icd_multimodal/checkpoint-5856"  

def load_from_checkpoint(ckpt_dir: str, num_labels: int):
    # tokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # dtype
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    # base model (same as training)
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

    # attach LoRA adapter weights from checkpoint
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()
    return model, tok

if os.path.isdir(CKPT_DIR):
    print(f"\n[checkpoint] loading: {CKPT_DIR}")
    ckpt_model, ckpt_tok = load_from_checkpoint(CKPT_DIR, num_labels=len(mlb.classes_))

    # build a lightweight eval Trainer
    eval_args = TrainingArguments(
        output_dir="ckpt_eval_tmp",
        per_device_eval_batch_size=2 if torch.cuda.is_available() else 1,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
    )

    # 1) tune threshold on VAL with the checkpoint model
    eval_trainer = PosWeightTrainer(
        pos_weight=compute_pos_weight(y_train),  # any tensor on same device is fine for BCE shape
        model=ckpt_model, args=eval_args, eval_dataset=val_ds,
        tokenizer=ckpt_tok, compute_metrics=make_compute_metrics(th=0.5)
    )
    ckpt_val = eval_trainer.predict(val_ds)
    ckpt_val_probs = 1.0 / (1.0 + np.exp(-ckpt_val.predictions))
    ckpt_t = tune_threshold(y_val, ckpt_val_probs)
    print(f"[checkpoint] best threshold on val: {ckpt_t:.2f}")

    # 2) evaluate on TEST with tuned threshold
    eval_trainer.compute_metrics = make_compute_metrics(th=ckpt_t)
    ckpt_test_eval = eval_trainer.evaluate(eval_dataset=test_ds)
    print("\n=== Quick test metrics (checkpoint, Trainer.evaluate) ===")
    print(json.dumps(ckpt_test_eval, indent=2))

    # rich metrics
    ckpt_test = eval_trainer.predict(test_ds)
    ckpt_test_probs = 1.0 / (1.0 + np.exp(-ckpt_test.predictions))
    ckpt_pred = (ckpt_test_probs >= ckpt_t).astype(int)

    ckpt_metrics = {
        "micro_f1": f1_score(y_test, ckpt_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test, ckpt_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_test, ckpt_pred, average="samples", zero_division=0),
    }
    try:
        ckpt_metrics["micro_auroc"] = roc_auc_score(y_test, ckpt_test_probs, average="micro")
    except Exception:
        ckpt_metrics["micro_auroc"] = float("nan")
    ckpt_metrics["macro_auroc"] = macro_auroc(y_test, ckpt_test_probs)
    for k in (5, 8, 10):
        p, r = precision_recall_at_k(y_test, ckpt_test_probs, k=k)
        ckpt_metrics[f"precision@{k}"] = p
        ckpt_metrics[f"recall@{k}"] = r

    print("\n=== Rich test metrics (checkpoint) ===")
    print(json.dumps(ckpt_metrics, indent=2))
else:
    print(f"[info] checkpoint dir not found: {CKPT_DIR}")
