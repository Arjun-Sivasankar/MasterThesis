# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# train_textgen_ragKG.py - WORKING VERSION
# Unified training script for RAG-enhanced diagnosis generation.
# """

# import os, json, time, argparse, logging, sys
# import numpy as np
# from pathlib import Path

# import torch
# import torch.distributed as dist
# from torch.utils.data import Dataset
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM,
#     TrainingArguments, Trainer, EarlyStoppingCallback
# )
# from transformers.trainer_callback import TrainerCallback
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# # ============================================================================
# # ENVIRONMENT
# # ============================================================================

# os.environ.setdefault("HF_DISABLE_PROGRESS_BAR", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# # ============================================================================
# # LOGGING
# # ============================================================================

# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# log = logging.getLogger(__name__)

# def is_main_process():
#     return int(os.environ.get('RANK', 0)) == 0

# class RankFilter(logging.Filter):
#     def filter(self, record):
#         return is_main_process()

# log.addFilter(RankFilter())

# # ============================================================================
# # DATASET
# # ============================================================================

# class RAGTextGenDataset(Dataset):
#     def __init__(self, jsonl_path: str, tokenizer, 
#                  max_len: int = 5120,
#                  max_prompt_tokens: int = 3072,
#                  max_kg_tokens: int = 1500,
#                  max_target_tokens: int = 512):
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.max_prompt_tokens = max_prompt_tokens
#         self.max_kg_tokens = max_kg_tokens
#         self.max_target_tokens = max_target_tokens
#         self.examples = []
        
#         with open(jsonl_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if line.strip():
#                     self.examples.append(json.loads(line))
        
#         if is_main_process():
#             log.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")
#             log.info(f"  Tokens: prompt={max_prompt_tokens}, kg={max_kg_tokens}, target={max_target_tokens}")
    
#     def __len__(self):
#         return len(self.examples)
    
#     def _split_prompt_and_kg(self, full_prompt: str):
#         kg_marker = "[KNOWLEDGE GRAPH FACTS]"
#         task_marker = "[TASK]"
        
#         if kg_marker not in full_prompt:
#             return full_prompt, ""
        
#         parts = full_prompt.split(kg_marker, 1)
#         before_kg = parts[0]
        
#         if len(parts) > 1 and task_marker in parts[1]:
#             kg_and_after = parts[1].split(task_marker, 1)
#             kg_section = kg_marker + kg_and_after[0]
#             after_kg = task_marker + kg_and_after[1] if len(kg_and_after) > 1 else ""
#             return before_kg + after_kg, kg_section
        
#         return before_kg, kg_marker + parts[1] if len(parts) > 1 else ""
    
#     def _truncate_kg_section(self, kg_section: str, max_tokens: int):
#         if not kg_section or max_tokens == 0:
#             return ""
        
#         lines = kg_section.split('\n')
#         header_lines = []
#         fact_lines = []
        
#         in_facts = False
#         for line in lines:
#             if '[KNOWLEDGE GRAPH FACTS]' in line:
#                 header_lines.append(line)
#                 in_facts = True
#             elif in_facts and line.strip() and line.strip()[0].isdigit():
#                 fact_lines.append(line)
#             elif in_facts:
#                 header_lines.append(line)
        
#         header_text = '\n'.join(header_lines)
#         header_tokens = self.tokenizer.encode(header_text, add_special_tokens=False)
#         available = max_tokens - len(header_tokens)
        
#         if available <= 0:
#             return header_text
        
#         kept_facts = []
#         current = 0
#         for fact in fact_lines:
#             tokens = self.tokenizer.encode(fact + '\n', add_special_tokens=False)
#             if current + len(tokens) <= available:
#                 kept_facts.append(fact)
#                 current += len(tokens)
#             else:
#                 break
        
#         return header_text + '\n' + '\n'.join(kept_facts) if kept_facts else header_text
    
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         full_prompt = example['prompt']
#         target = example['target']
        
#         prompt_part, kg_part = self._split_prompt_and_kg(full_prompt)
        
#         prompt_tokens = self.tokenizer.encode(prompt_part, add_special_tokens=False)
#         if len(prompt_tokens) > self.max_prompt_tokens:
#             prompt_tokens = prompt_tokens[:self.max_prompt_tokens]
#             prompt_part = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
#         if kg_part and self.max_kg_tokens > 0:
#             kg_part = self._truncate_kg_section(kg_part, self.max_kg_tokens)
#         elif self.max_kg_tokens == 0:
#             kg_part = ""
        
#         if kg_part:
#             if "[TASK]" in prompt_part:
#                 parts = prompt_part.split("[TASK]", 1)
#                 reconstructed = parts[0] + "\n" + kg_part + "\n[TASK]" + parts[1]
#             else:
#                 reconstructed = prompt_part + "\n" + kg_part
#         else:
#             reconstructed = prompt_part
        
#         target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
#         if len(target_tokens) > self.max_target_tokens:
#             target_tokens = target_tokens[:self.max_target_tokens]
#             target = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
        
#         conversation = [
#             {"role": "user", "content": reconstructed},
#             {"role": "assistant", "content": target}
#         ]
        
#         full_text = self.tokenizer.apply_chat_template(
#             conversation, tokenize=False, add_generation_prompt=False
#         )
        
#         encodings = self.tokenizer(
#             full_text, truncation=True, max_length=self.max_len,
#             padding=False, return_tensors=None
#         )
        
#         input_ids = encodings['input_ids']
#         labels = input_ids.copy()
        
#         try:
#             assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
#             for i in range(len(input_ids) - len(assistant_tokens)):
#                 if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
#                     labels[:i+len(assistant_tokens)+2] = [-100] * (i+len(assistant_tokens)+2)
#                     break
#             else:
#                 mask_len = int(len(labels) * 0.8)
#                 labels[:mask_len] = [-100] * mask_len
#         except:
#             mask_len = int(len(labels) * 0.8)
#             labels[:mask_len] = [-100] * mask_len
        
#         return {
#             'full_prompt': full_prompt,
#             'prompt': prompt_part,
#             'kg': kg_part,
#             'target': target,
#             'input_ids': input_ids,
#             'labels': labels,
#             'attention_mask': encodings['attention_mask']
#         }

# # ============================================================================
# # COLLATOR
# # ============================================================================

# def pad_collate(features, tokenizer):
#     max_len = max(len(f['input_ids']) for f in features)
    
#     batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    
#     for f in features:
#         pad_len = max_len - len(f['input_ids'])
#         batch['input_ids'].append(f['input_ids'] + [tokenizer.pad_token_id] * pad_len)
#         batch['attention_mask'].append(f['attention_mask'] + [0] * pad_len)
#         batch['labels'].append(f['labels'] + [-100] * pad_len)
    
#     return {
#         'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
#         'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
#         'labels': torch.tensor(batch['labels'], dtype=torch.long)
#     }

# # ============================================================================
# # MODEL LOADING - FIXED FOR GRADIENT CHECKPOINTING
# # ============================================================================

# # def load_llm_with_lora(model_name: str, lora_r: int = 16, lora_alpha: int = 32,
# #                        lora_dropout: float = 0.1):
# #     """Load model with LoRA - gradient checkpointing compatible."""
    
# #     if is_main_process():
# #         log.info(f"Loading model: {model_name}")
# #         log.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #         tokenizer.pad_token_id = tokenizer.eos_token_id
    
# #     world_size = int(os.environ.get('WORLD_SIZE', 1))
    
# #     if is_main_process():
# #         log.info(f"  World size: {world_size}")
# #         log.info(f"  Mode: {'Multi-GPU (DDP)' if world_size > 1 else 'Single GPU'}")
    
# #     # Load model
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16,
# #         device_map="auto" if world_size == 1 else None,
# #         trust_remote_code=True
# #     )
    
# #     if is_main_process():
# #         log.info(f"  Model dtype: {model.dtype}")
    
# #     # ✓ CRITICAL: Disable cache BEFORE prepare_model_for_kbit_training
# #     model.config.use_cache = False
    
# #     # ✓ CRITICAL: Enable gradient checkpointing BEFORE prepare_model_for_kbit_training
# #     if hasattr(model, 'gradient_checkpointing_enable'):
# #         model.gradient_checkpointing_enable()
    
# #     # ✓ CRITICAL: Pass use_gradient_checkpointing=True
# #     model = prepare_model_for_kbit_training(
# #         model, 
# #         use_gradient_checkpointing=True
# #     )
    
# #     # LoRA
# #     lora_config = LoraConfig(
# #         r=lora_r,
# #         lora_alpha=lora_alpha,
# #         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
# #         lora_dropout=lora_dropout,
# #         bias="none",
# #         task_type="CAUSAL_LM"
# #     )
    
# #     model = get_peft_model(model, lora_config)
    
# #     # ✓ CRITICAL: Explicitly enable gradients for LoRA parameters
# #     for name, param in model.named_parameters():
# #         if 'lora' in name.lower():
# #             param.requires_grad = True
    
# #     if is_main_process():
# #         log.info("✓ LoRA adapters added")
# #         log.info("✓ Gradient checkpointing enabled")
# #         model.print_trainable_parameters()
    
# #     return model, tokenizer

# def load_llm_with_lora(model_name: str, lora_r: int = 16, lora_alpha: int = 32,
#                        lora_dropout: float = 0.1):
#     """Load model with LoRA - memory optimized for long sequences."""
    
#     if is_main_process():
#         log.info(f"Loading model: {model_name}")
#         log.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     world_size = int(os.environ.get('WORLD_SIZE', 1))
    
#     if is_main_process():
#         log.info(f"  World size: {world_size}")
#         log.info(f"  Mode: {'Multi-GPU (DDP)' if world_size > 1 else 'Single GPU'}")
    
#     # ✓ CRITICAL: Load with torch_dtype to save memory
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,  # Always use bfloat16
#         device_map=None,  # Let DDP handle placement
#         trust_remote_code=True,
#         attn_implementation="flash_attention_2",  # ✓ NEW: Use Flash Attention 2
#     )
    
#     if is_main_process():
#         log.info(f"  Model dtype: {model.dtype}")
    
#     # ✓ CRITICAL: Enable gradient checkpointing
#     model.gradient_checkpointing_enable()
#     model.config.use_cache = False  # Required for gradient checkpointing
    
#     if is_main_process():
#         log.info("✓ Gradient checkpointing enabled")
    
#     # LoRA
#     lora_config = LoraConfig(
#         r=lora_r,
#         lora_alpha=lora_alpha,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         lora_dropout=lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
    
#     model = get_peft_model(model, lora_config)
    
#     if is_main_process():
#         log.info("✓ LoRA adapters added")
#         model.print_trainable_parameters()
    
#     return model, tokenizer

# # ============================================================================
# # CALLBACKS
# # ============================================================================

# class LoggingCallback(TrainerCallback):
#     def __init__(self):
#         self.train_losses = []
    
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if not is_main_process():
#             return
        
#         logs = logs or {}
        
#         if 'loss' in logs and 'eval_loss' not in logs:
#             epoch = logs.get('epoch', 0)
#             loss = logs.get('loss', 0)
#             self.train_losses.append(float(loss))
            
#             parts = [f"[TRAIN] Epoch {epoch:.2f}", f"Loss {loss:.4f}"]
#             if 'learning_rate' in logs:
#                 parts.append(f"LR {logs['learning_rate']:.2e}")
#             log.info(" | ".join(parts))
        
#         if 'eval_loss' in logs:
#             if self.train_losses:
#                 avg_train = sum(self.train_losses) / len(self.train_losses)
#                 self.train_losses = []
#             else:
#                 avg_train = float('nan')
            
#             eval_loss = float(logs['eval_loss'])
#             try:
#                 ppl = float(np.exp(min(eval_loss, 20)))
#             except:
#                 ppl = float('nan')
            
#             log.info("\n" + "="*60)
#             log.info(f"[EPOCH {int(logs.get('epoch', 0))} SUMMARY]")
#             log.info(f"  Avg Train Loss: {avg_train:.4f}")
#             log.info(f"  Val Loss:       {eval_loss:.4f}")
#             log.info(f"  Val Perplexity: {ppl:.2f}")
#             log.info("="*60 + "\n")

# # ============================================================================
# # SAFE TRAINER
# # ============================================================================

# class SafeTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.get("labels")
#         if labels is not None and (labels != -100).sum().item() == 0:
#             loss = torch.zeros((), device=labels.device, dtype=torch.float32, requires_grad=True)
#             return (loss, None) if return_outputs else loss
#         return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("--train_jsonl", required=True)
#     parser.add_argument("--val_jsonl", required=True)
#     parser.add_argument("--llm", default="meta-llama/Llama-3.1-8B-Instruct")
    
#     parser.add_argument("--max_len", type=int, default=5120)
#     parser.add_argument("--max_prompt_tokens", type=int, default=3072)
#     parser.add_argument("--max_kg_tokens", type=int, default=1500)
#     parser.add_argument("--max_target_tokens", type=int, default=512)
    
#     parser.add_argument("--lora_r", type=int, default=16)
#     parser.add_argument("--lora_alpha", type=int, default=32)
#     parser.add_argument("--lora_dropout", type=float, default=0.1)
    
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--per_device_train_batch_size", type=int, default=1)
#     parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
#     parser.add_argument("--grad_accum", type=int, default=16)
#     parser.add_argument("--learning_rate", type=float, default=2e-4)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     parser.add_argument("--warmup_ratio", type=float, default=0.03)
#     parser.add_argument("--early_stop", type=int, default=1)
#     parser.add_argument("--patience", type=int, default=3)
#     parser.add_argument("--seed", type=int, default=42)
    
#     parser.add_argument("--out_dir", required=True)
#     parser.add_argument("--save_adapter", action="store_true")
#     parser.add_argument("--adapter_dir", default="")
#     parser.add_argument("--experiment_name", default="")
#     parser.add_argument("--local_rank", type=int, default=-1)
    
#     args = parser.parse_args()
    
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
    
#     if is_main_process():
#         log.info("=" * 80)
#         log.info("TRAINING CONFIGURATION")
#         log.info("=" * 80)
#         log.info(f"Experiment: {args.experiment_name}")
#         log.info(f"Train: {args.train_jsonl}")
#         log.info(f"Val:   {args.val_jsonl}")
#         log.info(f"\nTokens: total={args.max_len}, prompt={args.max_prompt_tokens}, "
#                  f"kg={args.max_kg_tokens}, target={args.max_target_tokens}")
#         log.info(f"\nTraining: {args.epochs} epochs, LR={args.learning_rate}")
#         log.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
#         log.info("=" * 80)
    
#     model, tokenizer = load_llm_with_lora(
#         args.llm, args.lora_r, args.lora_alpha, args.lora_dropout
#     )
    
#     if is_main_process():
#         log.info("\nLoading datasets...")
    
#     train_dataset = RAGTextGenDataset(
#         args.train_jsonl, tokenizer, args.max_len,
#         args.max_prompt_tokens, args.max_kg_tokens, args.max_target_tokens
#     )
#     val_dataset = RAGTextGenDataset(
#         args.val_jsonl, tokenizer, args.max_len,
#         args.max_prompt_tokens, args.max_kg_tokens, args.max_target_tokens
#     )
    
#     if is_main_process():
#         log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
#     training_args = TrainingArguments(
#         output_dir=args.out_dir,
#         num_train_epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         per_device_train_batch_size=args.per_device_train_batch_size,
#         per_device_eval_batch_size=args.per_device_eval_batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         warmup_ratio=args.warmup_ratio,
#         weight_decay=args.weight_decay,
        
#         logging_strategy="steps",
#         logging_steps=50,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         save_total_limit=1,
#         load_best_model_at_end=bool(args.early_stop),
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#         report_to="none",
        
#         gradient_checkpointing=True,
#         gradient_checkpointing_kwargs={"use_reentrant": False},  # ✓ CRITICAL
#         remove_unused_columns=False,
#         optim="adamw_torch",
#         bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
#         fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),
        
#         ddp_find_unused_parameters=False,
#         local_rank=args.local_rank,
        
#         dataloader_num_workers=0,  # ✓ Set to 0 to avoid multiprocessing issues
#         dataloader_pin_memory=True,
        
#         disable_tqdm=True,
#         save_safetensors=True,
#     )
    
#     callbacks = [LoggingCallback()]
#     if args.early_stop:
#         callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    
#     trainer = SafeTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         data_collator=lambda f: pad_collate(f, tokenizer),
#         callbacks=callbacks,
#     )
    
#     if is_main_process():
#         log.info("\n🚀 Starting training...")
    
#     t0 = time.time()
    
#     try:
#         trainer.train()
        
#         if args.save_adapter and args.adapter_dir and is_main_process():
#             os.makedirs(args.adapter_dir, exist_ok=True)
#             log.info(f"\n💾 Saving adapter to {args.adapter_dir}...")
#             trainer.model.save_pretrained(args.adapter_dir)
#             tokenizer.save_pretrained(args.adapter_dir)
            
#             info = {
#                 'model': args.llm,
#                 'epochs': args.epochs,
#                 'max_len': args.max_len,
#                 'max_prompt_tokens': args.max_prompt_tokens,
#                 'max_kg_tokens': args.max_kg_tokens,
#                 'max_target_tokens': args.max_target_tokens,
#                 'training_time_min': (time.time() - t0) / 60
#             }
            
#             with open(Path(args.adapter_dir) / 'training_info.json', 'w') as f:
#                 json.dump(info, f, indent=2)
            
#             log.info("✅ Adapter saved")
        
#         if is_main_process():
#             log.info(f"\n✅ Training completed in {(time.time() - t0) / 60:.2f} min")
    
#     except Exception as e:
#         if is_main_process():
#             log.error(f"❌ Training failed: {e}")
#         raise
    
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_textgen_ragKG.py
Unified training script for RAG-enhanced diagnosis generation.
Memory-optimized for long sequences (5120 tokens).
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

class RAGTextGenDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, 
                 max_len: int = 5120,
                 max_prompt_tokens: int = 3072,
                 max_kg_tokens: int = 1500,
                 max_target_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_prompt_tokens = max_prompt_tokens
        self.max_kg_tokens = max_kg_tokens
        self.max_target_tokens = max_target_tokens
        self.examples = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        
        if is_main_process():
            log.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")
            log.info(f"  Tokens: prompt={max_prompt_tokens}, kg={max_kg_tokens}, target={max_target_tokens}")
    
    def __len__(self):
        return len(self.examples)
    
    def _split_prompt_and_kg(self, full_prompt: str):
        kg_marker = "[KNOWLEDGE GRAPH FACTS]"
        task_marker = "[TASK]"
        
        if kg_marker not in full_prompt:
            return full_prompt, ""
        
        parts = full_prompt.split(kg_marker, 1)
        before_kg = parts[0]
        
        if len(parts) > 1 and task_marker in parts[1]:
            kg_and_after = parts[1].split(task_marker, 1)
            kg_section = kg_marker + kg_and_after[0]
            after_kg = task_marker + kg_and_after[1] if len(kg_and_after) > 1 else ""
            return before_kg + after_kg, kg_section
        
        return before_kg, kg_marker + parts[1] if len(parts) > 1 else ""
    
    def _truncate_kg_section(self, kg_section: str, max_tokens: int):
        if not kg_section or max_tokens == 0:
            return ""
        
        lines = kg_section.split('\n')
        header_lines = []
        fact_lines = []
        
        in_facts = False
        for line in lines:
            if '[KNOWLEDGE GRAPH FACTS]' in line:
                header_lines.append(line)
                in_facts = True
            elif in_facts and line.strip() and line.strip()[0].isdigit():
                fact_lines.append(line)
            elif in_facts:
                header_lines.append(line)
        
        header_text = '\n'.join(header_lines)
        header_tokens = self.tokenizer.encode(header_text, add_special_tokens=False)
        available = max_tokens - len(header_tokens)
        
        if available <= 0:
            return header_text
        
        kept_facts = []
        current = 0
        for fact in fact_lines:
            tokens = self.tokenizer.encode(fact + '\n', add_special_tokens=False)
            if current + len(tokens) <= available:
                kept_facts.append(fact)
                current += len(tokens)
            else:
                break
        
        return header_text + '\n' + '\n'.join(kept_facts) if kept_facts else header_text
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        full_prompt = example['prompt']
        target = example['target']
        
        prompt_part, kg_part = self._split_prompt_and_kg(full_prompt)
        
        prompt_tokens = self.tokenizer.encode(prompt_part, add_special_tokens=False)
        if len(prompt_tokens) > self.max_prompt_tokens:
            prompt_tokens = prompt_tokens[:self.max_prompt_tokens]
            prompt_part = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
        if kg_part and self.max_kg_tokens > 0:
            kg_part = self._truncate_kg_section(kg_part, self.max_kg_tokens)
        elif self.max_kg_tokens == 0:
            kg_part = ""
        
        if kg_part:
            if "[TASK]" in prompt_part:
                parts = prompt_part.split("[TASK]", 1)
                reconstructed = parts[0] + "\n" + kg_part + "\n[TASK]" + parts[1]
            else:
                reconstructed = prompt_part + "\n" + kg_part
        else:
            reconstructed = prompt_part
        
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        if len(target_tokens) > self.max_target_tokens:
            target_tokens = target_tokens[:self.max_target_tokens]
            target = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
        
        conversation = [
            {"role": "user", "content": reconstructed},
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
        
        try:
            assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
            for i in range(len(input_ids) - len(assistant_tokens)):
                if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    labels[:i+len(assistant_tokens)+2] = [-100] * (i+len(assistant_tokens)+2)
                    break
            else:
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
# MODEL LOADING - Memory Optimized for 5120 tokens
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
            log.info("  ✓ Using Flash Attention 2")
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
        log.info("  ✓ Gradient checkpointing enabled")
    
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
        log.info("  ✓ LoRA adapters added")
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
    
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--llm", default="meta-llama/Llama-3.1-8B-Instruct")
    
    parser.add_argument("--max_len", type=int, default=5120)
    parser.add_argument("--max_prompt_tokens", type=int, default=3072)
    parser.add_argument("--max_kg_tokens", type=int, default=1500)
    parser.add_argument("--max_target_tokens", type=int, default=512)
    
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_adapter", action="store_true")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if is_main_process():
        log.info("=" * 80)
        log.info("TRAINING CONFIGURATION")
        log.info("=" * 80)
        log.info(f"Experiment: {args.experiment_name}")
        log.info(f"Train: {args.train_jsonl}")
        log.info(f"Val:   {args.val_jsonl}")
        log.info(f"\nTokens: total={args.max_len}, prompt={args.max_prompt_tokens}, "
                 f"kg={args.max_kg_tokens}, target={args.max_target_tokens}")
        log.info(f"\nTraining: {args.epochs} epochs, LR={args.learning_rate}")
        log.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        log.info("=" * 80)
    
    model, tokenizer = load_llm_with_lora(
        args.llm, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    
    if is_main_process():
        log.info("\nLoading datasets...")
    
    train_dataset = RAGTextGenDataset(
        args.train_jsonl, tokenizer, args.max_len,
        args.max_prompt_tokens, args.max_kg_tokens, args.max_target_tokens
    )
    val_dataset = RAGTextGenDataset(
        args.val_jsonl, tokenizer, args.max_len,
        args.max_prompt_tokens, args.max_kg_tokens, args.max_target_tokens
    )
    
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
        bf16=True,  # Always use bfloat16
        
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
        log.info("\n🚀 Starting training...")
    
    t0 = time.time()
    
    try:
        trainer.train()
        
    finally:
        if args.save_adapter and args.adapter_dir and is_main_process():
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                os.makedirs(args.adapter_dir, exist_ok=True)
                log.info(f"\n💾 Saving adapter to {args.adapter_dir}...")
                trainer.model.save_pretrained(args.adapter_dir)
                tokenizer.save_pretrained(args.adapter_dir)
                
                info = {
                    'model': args.llm,
                    'epochs': args.epochs,
                    'max_len': args.max_len,
                    'max_prompt_tokens': args.max_prompt_tokens,
                    'max_kg_tokens': args.max_kg_tokens,
                    'max_target_tokens': args.max_target_tokens,
                    'training_time_min': (time.time() - t0) / 60
                }
                
                with open(Path(args.adapter_dir) / 'training_info.json', 'w') as f:
                    json.dump(info, f, indent=2)
                
                log.info("✅ Adapter saved")
            except Exception as e:
                log.warning(f"Adapter save failed: {e}")
        
        if is_main_process():
            log.info(f"\n✅ Training completed in {(time.time() - t0) / 60:.2f} min")
        
        finalize_distributed()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())