# pipelines/train_codegen.py
import os, time, json, argparse, datetime, inspect, torch
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
import pandas as pd
from .util_codegen_core import (
    log, set_seed, is_main_process, rank0_print,
    subject_splits, nested_subject_sample, build_input_text,
    GenCodesDataset, pad_collate, load_lm_and_tokenizer, save_json
)

def get_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_pickle", default=None)
    ap.add_argument("--train_pickle", default=None)
    ap.add_argument("--val_pickle", default=None)
    ap.add_argument("--subject_col", default="subject_id_x")
    ap.add_argument("--label_col", default="icd_code")

    # model/prompt
    ap.add_argument("--llama_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--use_structured", type=int, default=1)
    ap.add_argument("--use_notes", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--tgt_reserve_tok", type=int, default=128)

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
    ap.add_argument("--train_size", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)

    # run dirs
    ap.add_argument("--run_root", default="runs_codegen")
    ap.add_argument("--run_name", default=None)

    # misc
    ap.add_argument("--compile", type=int, default=0)
    return ap.parse_args()

def make_training_args(args, run_dir):
    TA = TrainingArguments
    sig = inspect.signature(TA.__init__).parameters
    kwargs = dict(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
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
        optim="adamw_torch",
        dataloader_num_workers=2,
        run_name=os.path.basename(run_dir),
        disable_tqdm=True,
    )
    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"
    if "ddp_backend" in sig: kwargs["ddp_backend"] = "nccl"
    if "ddp_find_unused_parameters" in sig: kwargs["ddp_find_unused_parameters"] = False
    if "ddp_timeout" in sig: kwargs["ddp_timeout"] = 28800
    return TA(**kwargs)

def main():
    args = get_args()
    set_seed(args.seed)

    # Load data
    if args.train_pickle and args.val_pickle:
        train_df = pd.read_pickle(args.train_pickle)
        val_df   = pd.read_pickle(args.val_pickle)
    elif args.data_pickle:
        full_df = pd.read_pickle(args.data_pickle)
        train_df, val_df, _ = subject_splits(full_df, subject_col=args.subject_col, test_size=0.10, val_size=0.10, seed=args.seed)
    else:
        raise ValueError("Provide --data_pickle OR both --train_pickle/--val_pickle")

    train_df = nested_subject_sample(train_df, args.train_size, subject_col=args.subject_col, seed=args.seed)

    for df_, name in ((train_df, 'train'), (val_df, 'val')):
        df_["input_text"] = df_.apply(lambda r: build_input_text(r, args.use_structured==1, args.use_notes==1, args.subject_col), axis=1)

    model, tok = load_lm_and_tokenizer(args.llama_model)
    if args.compile:
        try: model = torch.compile(model)
        except Exception: pass

    train_ds = GenCodesDataset(train_df, tok, args.max_len, args.tgt_reserve_tok, args.label_col)
    val_ds   = GenCodesDataset(val_df,   tok, args.max_len, args.tgt_reserve_tok, args.label_col)

    tag = args.run_name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.run_root, tag)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    if is_main_process():
        rank0_print(f"Run dir: {run_dir}")

    train_args = make_training_args(args, run_dir)
    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda feats: pad_collate(feats, tok),
        callbacks=callbacks
    )

    t0 = time.perf_counter()
    trainer.train()
    train_secs = time.perf_counter() - t0

    if is_main_process():
        # Save adapter and tokenizer
        tok.save_pretrained(os.path.join(run_dir, "tokenizer"))
        trainer.model.save_pretrained(os.path.join(run_dir, "adapter_best"))
        save_json(os.path.join(run_dir, "label_space.json"), {
            "labels": sorted({format_icd9_properly(str(c)) for row in train_df[args.label_col] for c in row if is_valid_icd9(format_icd9_properly(str(c)))})
        })
        save_json(os.path.join(run_dir, "train_summary.json"), {"train_seconds": train_secs})

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
