#!/bin/bash
#SBATCH --job-name=rag_textgen_train
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/train_textgen_rag_%j.out

### ============================================================================
### SLURM SETTINGS
### ============================================================================
# --gres=gpu:3          → Use 3 GPUs (change to 1 for single GPU)
# --cpus-per-task=16    → 16 CPUs total (distributes across GPUs)
# --mem=128G            → Total memory (adjust based on GPU count)
# --time=24:00:00       → Max runtime (adjust as needed)
### ============================================================================

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

### 3) Virtualenv
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"

### 4) Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_DISABLE_PROGRESS_BAR=1
export TRANSFORMERS_VERBOSITY=error

# DDP stability
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

# GPU info
GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using ${GPUS} GPU(s) for training"

SCRIPT=gen/withKG/train_textgen_ragKG.py
echo "[INFO] Training Script: ${SCRIPT}"

### ============================================================================
### 5) CONFIGURATION - EDIT THIS SECTION
### ============================================================================

# Choose dataset mode: "baseline", "rag_unweighted", or "rag_weighted"
# MODE="baseline"
MODE="rag_unweighted"
# MODE="rag_weighted"

echo "[INFO] Selected MODE: ${MODE}"

# Base model
# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct
BASE_LLM=models/Meditron3-8B
echo "[INFO] Using base LLM: ${BASE_LLM}"
if [ "$BASE_LLM" = "models/Llama-3.1-8B-Instruct" ]; then
    echo "[INFO] Using Llama-3.1-8B model"
    LLM="llama3.1-8B"
elif [ "$BASE_LLM" = "meta-llama/Llama-3.2-1B-Instruct" ]; then
    echo "[INFO] Using Llama-3.2-1B model"
    LLM="llama3.2-1B"
elif [ "$BASE_LLM" = "models/Meditron3-8B" ]; then
    echo "[INFO] Using Meditron3-8B model"
    LLM="Meditron3-8B"
else
    echo "[WARNING] Base LLM is not configured"
fi

# Training parameters
EPOCHS=10

### ============================================================================
### TOKEN BUDGETS - CONFIGURED PER MODE
### ============================================================================

if [ "$MODE" = "baseline" ]; then
    # BASELINE: No KG facts
    TRAIN_JSONL=dataset/preprocessed_rag_62k/train_rag_unweighted.jsonl  
    VAL_JSONL=dataset/preprocessed_rag_62k/val_rag_unweighted.jsonl  
    OUT_DIR=runs_textgen_rag/${MODE}/${LLM}/baseline_checkpoints_$EPOCHS
    ADAPTER_DIR=runs_textgen_rag/${MODE}/${LLM}/baseline_adapter_$EPOCHS
    EXP_NAME="baseline"
    
    # Token budgets - SAME total as RAG for fair comparison
    MAX_LEN=5120
    MAX_PROMPT_TOKENS=4572      # 3072 + 1500 (absorbs KG space)
    MAX_KG_TOKENS=0             # No KG facts (Removes [KNOWLEDGE GRAPH FACTS] section)
    MAX_TARGET_TOKENS=512       # Diagnosis output
    
elif [ "$MODE" = "rag_unweighted" ]; then
    # RAG UNWEIGHTED: With unweighted KG facts
    TRAIN_JSONL=dataset/preprocessed_rag_62k/train_rag_unweighted.jsonl
    VAL_JSONL=dataset/preprocessed_rag_62k/val_rag_unweighted.jsonl
    OUT_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_unweighted_checkpoints_$EPOCHS
    ADAPTER_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_unweighted_adapter_$EPOCHS
    EXP_NAME="rag_unweighted"
    
    # Token budgets
    MAX_LEN=5120
    MAX_PROMPT_TOKENS=3072      # Clinical notes + instructions
    MAX_KG_TOKENS=1500          # Knowledge graph facts (based on data median)
    MAX_TARGET_TOKENS=512       # Diagnosis output
    
elif [ "$MODE" = "rag_weighted" ]; then
    # RAG WEIGHTED: With weighted KG facts (alpha=0.3)
    TRAIN_JSONL=dataset/preprocessed_rag_62k/train_rag_weighted_alpha0.3.jsonl
    VAL_JSONL=dataset/preprocessed_rag_62k/val_rag_weighted_alpha0.3.jsonl
    OUT_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_weighted_checkpoints_$EPOCHS
    ADAPTER_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_weighted_adapter_$EPOCHS
    EXP_NAME="rag_weighted_alpha0.3"
    
    # Token budgets (same as unweighted)
    MAX_LEN=5120
    MAX_PROMPT_TOKENS=3072      # Clinical notes + instructions
    MAX_KG_TOKENS=1500          # Knowledge graph facts (weighted)
    MAX_TARGET_TOKENS=512       # Diagnosis output
    
else
    echo "[ERROR] Unknown MODE: $MODE"
    echo "Valid options: baseline, rag_unweighted, rag_weighted"
    exit 1
fi

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

### ============================================================================
### 6) DISPLAY CONFIGURATION
### ============================================================================

echo ""
echo "=========================================================================="
echo "TRAINING CONFIGURATION"
echo "=========================================================================="
echo "Mode:           ${MODE}"
echo "Train data:     ${TRAIN_JSONL}"
echo "Val data:       ${VAL_JSONL}"
echo "Base LLM:       ${BASE_LLM}"
echo ""
echo "TOKEN BUDGETS:"
echo "  Max total length:    ${MAX_LEN} tokens"
echo "  Max prompt tokens:   ${MAX_PROMPT_TOKENS} tokens (clinical notes)"
echo "  Max KG tokens:       ${MAX_KG_TOKENS} tokens (knowledge graph facts)"
echo "  Max target tokens:   ${MAX_TARGET_TOKENS} tokens (diagnosis output)"
echo "  Sum (content):       $((MAX_PROMPT_TOKENS + MAX_KG_TOKENS + MAX_TARGET_TOKENS)) tokens"
echo "  Overhead:            $((MAX_LEN - MAX_PROMPT_TOKENS - MAX_KG_TOKENS - MAX_TARGET_TOKENS)) tokens (chat template)"
echo ""
echo "TRAINING:"
echo "  Epochs:              ${EPOCHS}"
echo "  Batch size:          1 per device"
echo "  Gradient accum:      16 steps"
echo "  Learning rate:       2e-4"
echo "  Warmup ratio:        0.03"
echo "  Weight decay:        0.01"
echo "  Early stopping:      Yes (patience=2)"
echo ""
echo "LORA:"
echo "  Rank (r):            ${LORA_R}"
echo "  Alpha:               ${LORA_ALPHA}"
echo "  Dropout:             ${LORA_DROPOUT}"
echo ""
echo "OUTPUT:"
echo "  Checkpoints:         ${OUT_DIR}"
echo "  Adapter:             ${ADAPTER_DIR}"
echo "  Experiment name:     ${EXP_NAME}"
echo ""
echo "HARDWARE:"
echo "  GPUs:                ${GPUS}"
echo "  CPUs per task:       ${SLURM_CPUS_PER_TASK}"
echo "  Memory:              ${SLURM_MEM_PER_NODE}M"
echo "  Nodes:               ${SLURM_NNODES}"
echo "=========================================================================="
echo ""

### ============================================================================
### 7) RUN TRAINING
### ============================================================================

start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME}"
echo ""

# Check if datasets exist
if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "[ERROR] Training data not found: ${TRAIN_JSONL}"
    exit 1
fi

if [ ! -f "${VAL_JSONL}" ]; then
    echo "[ERROR] Validation data not found: ${VAL_JSONL}"
    exit 1
fi

echo "[INFO] Launching training with torchrun..."
echo "[INFO] Command: torchrun --standalone --nproc_per_node=${GPUS}"
echo ""

# Run with torchrun for DDP (works for single GPU too)
srun torchrun \
  --standalone \
  --nproc_per_node=${GPUS} \
  ${SCRIPT} \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl "${VAL_JSONL}" \
  --llm "${BASE_LLM}" \
  --max_len ${MAX_LEN} \
  --max_prompt_tokens ${MAX_PROMPT_TOKENS} \
  --max_kg_tokens ${MAX_KG_TOKENS} \
  --max_target_tokens ${MAX_TARGET_TOKENS} \
  --epochs ${EPOCHS} \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --grad_accum 16 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --out_dir "${OUT_DIR}" \
  --save_adapter \
  --adapter_dir "${ADAPTER_DIR}" \
  --experiment_name "${EXP_NAME}" \
  --early_stop 1 \
  --patience 2 \
  --seed 42

status=$?
end=$(date +%s)
elapsed=$((end - start))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo ""
echo "=========================================================================="
echo "[TIME] Training completed"
echo "[TIME] Elapsed: ${elapsed} seconds (${minutes}m ${seconds}s)"
echo "[INFO] Exit code: ${status}"
echo "[INFO] Job finished: $(date)"
echo "=========================================================================="

if [ $status -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully! ✓"
    echo "[INFO] Adapter saved to: ${ADAPTER_DIR}"
else
    echo "[ERROR] Training failed with exit code: ${status}"
fi

exit $status