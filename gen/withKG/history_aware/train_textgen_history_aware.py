#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_history_aware.py
Unified training script for History Aware generation.
"""

import os, json, time, argparse, logging, sys
import numpy as np
from pathlib import Path
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model

# ============================================================================
# ENVIRONMENT
# ============================================================================

os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log = logging.getLogger(__name__)

def is_main_process():
    return int(os.environ.get('RANK', 0)) == 0

class RankFilter(logging.Filter):
    def filter(self, record):
        return is_main_process()

log.addFilter(RankFilter())

# ============================================================================
# DATASET
# ============================================================================

class HistoryAwareDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_len=5120):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Original Logic: Read CSV/TSV
        df = pd.read_csv(tsv_path, sep='\t')
        self.examples = df.to_dict(orient='records')
        
        if is_main_process():
            log.info(f"Loaded {len(self.examples)} examples from {tsv_path}")
            log.info(f"  Max Length: {max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example['prompt']
        target = example['target']
        
        # Original Logic: Construct conversation
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target}
        ]
        
        full_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        
        encodings = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding=False, return_tensors=None
        )
        
        input_ids = encodings['input_ids']
        labels = input_ids.copy()
        
        # Original Logic: Mask everything before the assistant's response
        try:
            assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
            # Find the sequence of tokens corresponding to "assistant" (header of the response)
            found = False
            for i in range(len(input_ids) - len(assistant_tokens)):
                if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    # Mask everything up to the assistant tokens + 2 (usually structural tokens)
                    mask_end_idx = i + len(assistant_tokens) + 2
                    labels[:mask_end_idx] = [-100] * mask_end_idx
                    found = True
                    break
            
            if not found:
                # Fallback if specific token sequence not found
                mask_len = int(len(labels) * 0.8)
                labels[:mask_len] = [-100] * mask_len
        except:
            # Fallback on error
            mask_len = int(len(labels) * 0.8)
            labels[:mask_len] = [-100] * mask_len
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': encodings['attention_mask']
        }

# ============================================================================
# COLLATOR
# ============================================================================

def pad_collate(features, tokenizer):
    max_len = max(len(f['input_ids']) for f in features)
    
    batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    
    for f in features:
        pad_len = max_len - len(f['input_ids'])
        batch['input_ids'].append(f['input_ids'] + [tokenizer.pad_token_id] * pad_len)
        batch['attention_mask'].append(f['attention_mask'] + [0] * pad_len)
        batch['labels'].append(f['labels'] + [-100] * pad_len)
    
    return {
        'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
        'labels': torch.tensor(batch['labels'], dtype=torch.long)
    }

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_llm_with_lora(model_name: str, lora_r: int = 16, lora_alpha: int = 32,
                        lora_dropout: float = 0.1):
    """Load model with LoRA - memory optimized for long sequences."""
    
    if is_main_process():
        log.info(f"Loading model: {model_name}")
        log.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if is_main_process():
        log.info(f"  World size: {world_size}")
        log.info(f"  Mode: {'Multi-GPU (DDP)' if world_size > 1 else 'Single GPU'}")
    
    # Try Flash Attention 2 first, fallback to eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,  # DDP handles placement
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        if is_main_process():
            log.info("   Using Flash Attention 2")
    except Exception as e:
        if is_main_process():
            log.warning(f"  Flash Attention 2 not available: {e}")
            log.info("  Using eager attention (will use more memory)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )
    
    if is_main_process():
        log.info(f"  Model dtype: {model.dtype}")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    if is_main_process():
        log.info("   Gradient checkpointing enabled")
    
    # LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    if is_main_process():
        log.info("   LoRA adapters added")
        model.print_trainable_parameters()
    
    return model, tokenizer

# ============================================================================
# CALLBACKS
# ============================================================================

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process():
            return
        
        logs = logs or {}
        
        if 'loss' in logs and 'eval_loss' not in logs:
            epoch = logs.get('epoch', 0)
            loss = logs.get('loss', 0)
            self.train_losses.append(float(loss))
            
            parts = [f"[TRAIN] Epoch {epoch:.2f}", f"Loss {loss:.4f}"]
            if 'learning_rate' in logs:
                parts.append(f"LR {logs['learning_rate']:.2e}")
            if 'grad_norm' in logs:
                parts.append(f"Grad {logs['grad_norm']:.3f}")
            log.info(" | ".join(parts))
        
        if 'eval_loss' in logs:
            if self.train_losses:
                avg_train = sum(self.train_losses) / len(self.train_losses)
                self.train_losses = []
            else:
                avg_train = float('nan')
            
            eval_loss = float(logs['eval_loss'])
            try:
                ppl = float(np.exp(min(eval_loss, 20)))
            except:
                ppl = float('nan')
            
            log.info("\n" + "="*56)
            log.info(f"[EPOCH {int(logs.get('epoch', 0))} SUMMARY]")
            log.info(f"- Avg Train Loss: {avg_train:.4f}")
            log.info(f"- Val Loss:       {eval_loss:.4f}")
            log.info(f"- Val Perplexity: {ppl:.2f}")
            if 'eval_runtime' in logs:
                log.info(f"- Eval Time:     {logs['eval_runtime']:.1f}s")
            log.info("="*56 + "\n")

# ============================================================================
# SAFE TRAINER
# ============================================================================

class SafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is not None and (labels != -100).sum().item() == 0:
            loss = torch.zeros((), device=labels.device, dtype=torch.float32, requires_grad=True)
            return (loss, None) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

# ============================================================================
# CLEANUP
# ============================================================================

def finalize_distributed():
    import datetime, gc
    try:
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    pass
            try:
                dist.barrier(timeout=datetime.timedelta(seconds=30))
            except:
                pass
            try:
                dist.destroy_process_group()
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        if is_main_process():
            log.warning(f"Error during distributed cleanup: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Dataset specific args
    parser.add_argument("--train_tsv", required=True)
    parser.add_argument("--val_tsv", required=True)
    parser.add_argument("--llm", default="meta-llama/Llama-3.1-8B-Instruct")
    
    parser.add_argument("--max_len", type=int, default=5120)
    
    # LoRA args
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    
    # IO args
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_adapter", action="store_true")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    if is_main_process():
        log.info("=" * 80)
        log.info("TRAINING CONFIGURATION")
        log.info("=" * 80)
        log.info(f"Experiment: {args.experiment_name}")
        log.info(f"Train: {args.train_tsv}")
        log.info(f"Val:   {args.val_tsv}")
        log.info(f"Tokens: total={args.max_len}")
        log.info(f"Training: {args.epochs} epochs, LR={args.learning_rate}")
        log.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        log.info("=" * 80)
    
    model, tokenizer = load_llm_with_lora(
        args.llm, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    
    if is_main_process():
        log.info("\nLoading datasets...")
    
    train_dataset = HistoryAwareDataset(args.train_tsv, tokenizer, args.max_len)
    val_dataset = HistoryAwareDataset(args.val_tsv, tokenizer, args.max_len)
    
    if is_main_process():
        log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    training_args = TrainingArguments(
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=bool(args.early_stop),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        bf16=True, # Enforced bfloat16 as per reference
        
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_timeout=28800,
        local_rank=args.local_rank,
        
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        
        disable_tqdm=True,
        save_safetensors=True,
    )
    
    callbacks = [LoggingCallback()]
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda f: pad_collate(f, tokenizer),
        callbacks=callbacks,
    )
    
    if is_main_process():
        log.info("\n Starting training...")
    
    t0 = time.time()
    
    try:
        trainer.train()
        
    finally:
        if args.save_adapter and args.adapter_dir and is_main_process():
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                os.makedirs(args.adapter_dir, exist_ok=True)
                log.info(f"\n Saving adapter to {args.adapter_dir}...")
                trainer.model.save_pretrained(args.adapter_dir)
                tokenizer.save_pretrained(args.adapter_dir)
                
                info = {
                    'model': args.llm,
                    'epochs': args.epochs,
                    'max_len': args.max_len,
                    'training_time_min': (time.time() - t0) / 60
                }
                
                with open(Path(args.adapter_dir) / 'training_info.json', 'w') as f:
                    json.dump(info, f, indent=2)
                
                log.info(" Adapter saved")
            except Exception as e:
                log.warning(f"Adapter save failed: {e}")
        
        if is_main_process():
            log.info(f"\n Training completed in {(time.time() - t0) / 60:.2f} min")
        
        finalize_distributed()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())