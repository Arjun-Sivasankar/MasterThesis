#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_textgen_baseline.py
Training script for baseline diagnosis generation (no KG retrieval).
"""

import os, json, time, argparse, logging, sys
import numpy as np
from pathlib import Path

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

class SFTTextGenDataset(Dataset):
    """Dataset for supervised fine-tuning (baseline, no KG facts)."""
    
    def __init__(self, jsonl_path: str, tokenizer, 
                 max_len: int = 5120,
                 max_prompt_tokens: int = 4572,
                 max_target_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.examples = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        
        if is_main_process():
            log.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")
            log.info(f"  Max prompt tokens: {max_prompt_tokens}")
            log.info(f"  Max target tokens: {max_target_tokens}")
            log.info(f"  Max sequence length: {max_len}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example['prompt']
        target = example['target']
        
        # Truncate prompt if needed
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > self.max_prompt_tokens:
            prompt_tokens = prompt_tokens[:self.max_prompt_tokens]
            prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
        # Truncate target if needed
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        if len(target_tokens) > self.max_target_tokens:
            target_tokens = target_tokens[:self.max_target_tokens]
            target = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
        
        # Build conversation
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
        
        # Mask prompt tokens (only train on assistant response)
        try:
            assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
            for i in range(len(input_ids) - len(assistant_tokens)):
                if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    labels[:i+len(assistant_tokens)+2] = [-100] * (i+len(assistant_tokens)+2)
                    break
            else:
                # Fallback: mask first 80%
                mask_len = int(len(labels) * 0.8)
                labels[:mask_len] = [-100] * mask_len
        except:
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
    """Pad batch to max length."""
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
    """Load model with LoRA - memory optimized."""
    
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
            log.info("  Using Flash Attention 2")
    except Exception as e:
        if is_main_process():
            log.warning(f"  Flash Attention 2 not available: {e}")
            log.info("  Using eager attention")
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
        log.info("  Gradient checkpointing enabled")
    
    # LoRA configuration
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
        log.info("  LoRA adapters added")
        model.print_trainable_parameters()
    
    return model, tokenizer

# ============================================================================
# CALLBACKS
# ============================================================================

class LoggingCallback(TrainerCallback):
    """Custom logging callback."""
    
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
    """Trainer with safe loss computation."""
    
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
    """Clean up distributed training."""
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
    parser = argparse.ArgumentParser(
        description="Train baseline diagnosis generation model (no KG)"
    )
    
    # Data
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL file")
    parser.add_argument("--val_jsonl", required=True, help="Validation JSONL file")
    parser.add_argument("--llm", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base LLM model")
    
    # Token budgets
    parser.add_argument("--max_len", type=int, default=5120,
                       help="Maximum sequence length")
    parser.add_argument("--max_prompt_tokens", type=int, default=4572,
                       help="Maximum tokens for prompt (clinical notes)")
    parser.add_argument("--max_target_tokens", type=int, default=512,
                       help="Maximum tokens for target (diagnoses)")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--early_stop", type=int, default=1,
                       help="Enable early stopping (0=disable, 1=enable)")
    parser.add_argument("--patience", type=int, default=3)
    
    # Output
    parser.add_argument("--out_dir", required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--save_adapter", action="store_true",
                       help="Save LoRA adapter")
    parser.add_argument("--adapter_dir", default="",
                       help="Directory to save adapter")
    parser.add_argument("--experiment_name", default="baseline",
                       help="Experiment name")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    if is_main_process():
        log.info("=" * 80)
        log.info("BASELINE TRAINING CONFIGURATION")
        log.info("=" * 80)
        log.info(f"Experiment: {args.experiment_name}")
        log.info(f"Train: {args.train_jsonl}")
        log.info(f"Val:   {args.val_jsonl}")
        log.info(f"\nToken Budget:")
        log.info(f"  Max total:  {args.max_len}")
        log.info(f"  Prompt:     {args.max_prompt_tokens} (clinical notes)")
        log.info(f"  Target:     {args.max_target_tokens} (diagnoses)")
        log.info(f"  Overhead:   {args.max_len - args.max_prompt_tokens - args.max_target_tokens} (chat template)")
        log.info(f"\nTraining:")
        log.info(f"  Epochs:     {args.epochs}")
        log.info(f"  Batch size: {args.per_device_train_batch_size}")
        log.info(f"  Grad accum: {args.grad_accum}")
        log.info(f"  LR:         {args.learning_rate}")
        log.info(f"\nLoRA:")
        log.info(f"  r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        log.info("=" * 80)
    
    # Load model
    model, tokenizer = load_llm_with_lora(
        args.llm, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    
    if is_main_process():
        log.info("\nLoading datasets...")
    
    # Load datasets
    train_dataset = SFTTextGenDataset(
        args.train_jsonl, tokenizer, args.max_len,
        args.max_prompt_tokens, args.max_target_tokens
    )
    val_dataset = SFTTextGenDataset(
        args.val_jsonl, tokenizer, args.max_len,
        args.max_prompt_tokens, args.max_target_tokens
    )
    
    if is_main_process():
        log.info(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
    
    # Training arguments
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
        bf16=True,
        
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
    
    # Callbacks
    callbacks = [LoggingCallback()]
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    
    # Trainer
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda f: pad_collate(f, tokenizer),
        callbacks=callbacks,
    )
    
    if is_main_process():
        log.info("\nStarting training...")
    
    t0 = time.time()
    
    try:
        trainer.train()
        
    finally:
        # Save adapter
        if args.save_adapter and args.adapter_dir and is_main_process():
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                os.makedirs(args.adapter_dir, exist_ok=True)
                log.info(f"\nSaving adapter to {args.adapter_dir}...")
                trainer.model.save_pretrained(args.adapter_dir)
                tokenizer.save_pretrained(args.adapter_dir)
                
                info = {
                    'model': args.llm,
                    'mode': 'baseline',
                    'epochs': args.epochs,
                    'max_len': args.max_len,
                    'max_prompt_tokens': args.max_prompt_tokens,
                    'max_target_tokens': args.max_target_tokens,
                    'training_time_min': (time.time() - t0) / 60
                }
                
                with open(Path(args.adapter_dir) / 'training_info.json', 'w') as f:
                    json.dump(info, f, indent=2)
                
                log.info("Adapter saved")
            except Exception as e:
                log.warning(f"Adapter save failed: {e}")
        
        if is_main_process():
            log.info(f"\nTraining completed in {(time.time() - t0) / 60:.2f} min")
        
        finalize_distributed()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())