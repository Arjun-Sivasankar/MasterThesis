# train_textgen.py
import os, json, time, argparse, logging, pickle, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.distributed as dist
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from common_textgen import (
    log, is_main_process, SFTTextGenDataset, pad_collate,
    load_llm_with_lora, build_eval_labels, to_list, format_icd9, is_valid_icd9
)

# ---- Env / NCCL hygiene ----
os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Prefer torch-prefixed controls (the NCCL_* ones are deprecated in PyTorch logs)
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# Pick the right fabric on your cluster; exclude lo/docker
os.environ.setdefault("NCCL_SOCKET_IFNAME", "ib,eth,^lo,docker")

def extract_codes(df, label_col):
    out=[]
    for _, r in df.iterrows():
        lst = to_list(r.get(label_col, []))
        lst = [format_icd9(c) for c in lst if c]
        lst = [c for c in lst if is_valid_icd9(c)]
        out.append(lst)
    return out

class SafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is not None and (labels != -100).sum().item() == 0:
            loss = torch.zeros((), device=labels.device, dtype=torch.float32, requires_grad=True)
            return (loss, None) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

def finalize_distributed():
    """
    Best-effort barrier + destroy to avoid NCCL/Gloo teardown hangs.
    Only runs if torch.distributed is actually initialized.
    """
    try:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
            except Exception:
                # barrier can throw if a rank already exited; safe to ignore
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")
    ap.add_argument("--target_mode", choices=["icd_titles","discharge_dx"], default="icd_titles")
    ap.add_argument("--icd_index_dir", required=True)

    # llm & train
    ap.add_argument("--llm", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--N_max_terms", type=int, default=12)
    ap.add_argument("--min_assistant_tokens", type=int, default=128)

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

    # io
    ap.add_argument("--out_dir", default="runs_textgen/checkpoints")
    ap.add_argument("--save_adapter", action="store_true", help="Save PEFT adapter (recommended)")
    ap.add_argument("--adapter_dir", default="runs_textgen/adapter")

    # DDP (torchrun sets LOCAL_RANK)
    ap.add_argument("--local_rank", type=int, default=-1)

    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- data ----
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

    # gold label space for info (not used by Trainer)
    train_gold = extract_codes(train_df, args.label_col)
    val_gold   = extract_codes(val_df, args.label_col)
    test_gold  = extract_codes(test_df, args.label_col)
    labels_full = build_eval_labels(train_gold)
    if is_main_process():
        log.info(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        log.info(f"Label space (FULL): {len(labels_full)} codes")

    # ---- model & tokenizer ----
    model, tok = load_llm_with_lora(args.llm)

    # ---- datasets ----
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

    # ---- training args ----
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
        logging_steps=50,
        eval_strategy="epoch",   # <- canonical arg name
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

        # DDP stability
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_timeout=28800,  # 8h
        local_rank=args.local_rank,

        # DataLoader stability (prevents teardown hang)
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,

        # Misc
        disable_tqdm=True,
        save_safetensors=True,
    )

    callbacks=[]
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = SafeTrainer(
        model=model,
        args=TA,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda feats: pad_collate(feats, tok),
        callbacks=callbacks
    )

    if is_main_process():
        log.info("Starting training...")
    t0 = time.time()

    # ---- TRAIN ----
    try:
        trainer.train()
        # If you asked to load best at end, HF will do it on rank-0 only.
    finally:
        # Make sure any CUDA kernels are done before we proceed to save/exit
        if torch.cuda.is_available():
            try: torch.cuda.synchronize()
            except Exception: pass

    if is_main_process():
        mins = (time.time()-t0)/60.0
        log.info(f"Training completed in {mins:.1f} min")

    # ---- Save adapter only (compact) ----
    if args.save_adapter and is_main_process():
        os.makedirs(args.adapter_dir, exist_ok=True)
        trainer.model.save_pretrained(args.adapter_dir)
        tok.save_pretrained(args.adapter_dir)
        log.info(f"Adapter saved to {args.adapter_dir}")

    # ---- Clean DDP nicely to avoid post-train hangs ----
    finalize_distributed()

    if is_main_process():
        log.info("All done. Exiting cleanly.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
