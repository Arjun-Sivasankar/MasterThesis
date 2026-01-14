#!/bin/bash
#SBATCH --job-name=baseline_train
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen/train_baseline_%j.out

set -e

### ============================================================================
### BASELINE TRAINING CONFIGURATION
### ============================================================================
# This script trains a diagnosis generation model WITHOUT knowledge graph facts
# Uses preprocessed data from preprocess_baseline.py
### ============================================================================

echo "========================================"
echo "BASELINE TRAINING"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

### 1) Setup modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project directory
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "[ERROR] Project dir not found"; exit 1; }

### 3) Virtual environment
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

### 5) Configuration
GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using ${GPUS} GPU(s) for training"

# Paths
SCRIPT=gen/pipeline/train_textgen_baseline.py
DATA_DIR=dataset/baseline  
TRAIN_JSONL=${DATA_DIR}/train_baseline.jsonl
VAL_JSONL=${DATA_DIR}/val_baseline.jsonl

# Model selection
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=meta-llama/Llama-3.1-8B-Instruct
# BASE_LLM=models/Meditron3-8B

echo "[INFO] Using base LLM: ${BASE_LLM}"

# Model name for output directories
if [ "$BASE_LLM" = "models/Llama-3.1-8B-Instruct" ] || [ "$BASE_LLM" = "meta-llama/Llama-3.1-8B-Instruct" ]; then
    LLM="llama3.1-8B"
elif [ "$BASE_LLM" = "meta-llama/Llama-3.2-1B-Instruct" ]; then
    LLM="llama3.2-1B"
elif [ "$BASE_LLM" = "models/Meditron3-8B" ]; then
    LLM="Meditron3-8B"
else
    LLM="unknown"
fi

# Training parameters
EPOCHS=10
BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE=2e-4
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
EARLY_STOP=1  
PATIENCE=3

# Token budgets (baseline has no KG facts)
MAX_LEN=5120
MAX_PROMPT_TOKENS=4572  
MAX_TARGET_TOKENS=512
# Overhead: 5120 - 4572 - 512 = 36 tokens for chat template

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Output directories
OUT_DIR=runs_textgen/baseline/${LLM}/checkpoints_${EPOCHS}
ADAPTER_DIR=runs_textgen/baseline/${LLM}/adapter_${EPOCHS}

EXP_NAME="baseline"

### ============================================================================
### 6) VALIDATION
### ============================================================================

echo ""
echo "=========================================================================="
echo "CONFIGURATION VALIDATION"
echo "=========================================================================="

# Check if data exists
if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "[ERROR] Training data not found: ${TRAIN_JSONL}"
    echo "[INFO] Please run preprocess_baseline.py first"
    exit 1
fi

if [ ! -f "${VAL_JSONL}" ]; then
    echo "[ERROR] Validation data not found: ${VAL_JSONL}"
    echo "[INFO] Please run preprocess_baseline.py first"
    exit 1
fi

echo "[INFO] Training data found: ${TRAIN_JSONL}"
echo "[INFO] Validation data found: ${VAL_JSONL}"

# Check script exists
if [ ! -f "${SCRIPT}" ]; then
    echo "[ERROR] Training script not found: ${SCRIPT}"
    exit 1
fi

echo "[INFO] Training script found: ${SCRIPT}"

### ============================================================================
### 7) DISPLAY CONFIGURATION
### ============================================================================

echo ""
echo "=========================================================================="
echo "TRAINING CONFIGURATION"
echo "=========================================================================="
echo "Mode:           BASELINE (no KG facts)"
echo "Train data:     ${TRAIN_JSONL}"
echo "Val data:       ${VAL_JSONL}"
echo "Base LLM:       ${BASE_LLM}"
echo ""
echo "TOKEN BUDGETS:"
echo "  Max total length:    ${MAX_LEN} tokens"
echo "  Max prompt tokens:   ${MAX_PROMPT_TOKENS} tokens (clinical notes)"
echo "  Max target tokens:   ${MAX_TARGET_TOKENS} tokens (diagnosis output)"
echo "  Overhead:            $((MAX_LEN - MAX_PROMPT_TOKENS - MAX_TARGET_TOKENS)) tokens (chat template)"
echo ""
echo "TRAINING:"
echo "  Epochs:              ${EPOCHS}"
echo "  Batch size:          ${BATCH_SIZE} per device"
echo "  Gradient accum:      ${GRAD_ACCUM} steps"
echo "  Effective batch:     $((BATCH_SIZE * GRAD_ACCUM * GPUS))"
echo "  Learning rate:       ${LEARNING_RATE}"
echo "  Warmup ratio:        ${WARMUP_RATIO}"
echo "  Weight decay:        ${WEIGHT_DECAY}"
echo "  Early stopping:      $([ $EARLY_STOP -eq 1 ] && echo 'Yes (patience='$PATIENCE')' || echo 'No')"
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
echo "  Node:                ${SLURMD_NODENAME}"
echo "=========================================================================="
echo ""

### ============================================================================
### 8) CREATE OUTPUT DIRECTORIES
### ============================================================================

mkdir -p "$OUT_DIR"
mkdir -p "$ADAPTER_DIR"
mkdir -p logs/Textgen

echo "[INFO] Output directories created"

### ============================================================================
### 9) RUN TRAINING
### ============================================================================

start=$(date +%s)

echo ""
echo "=========================================================================="
echo "LAUNCHING TRAINING"
echo "=========================================================================="
echo "[INFO] Command: srun torchrun --standalone --nproc_per_node=${GPUS}"
echo ""

srun torchrun \
  --standalone \
  --nproc_per_node=${GPUS} \
  ${SCRIPT} \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl "${VAL_JSONL}" \
  --llm "${BASE_LLM}" \
  --max_len ${MAX_LEN} \
  --max_prompt_tokens ${MAX_PROMPT_TOKENS} \
  --max_target_tokens ${MAX_TARGET_TOKENS} \
  --epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --grad_accum ${GRAD_ACCUM} \
  --learning_rate ${LEARNING_RATE} \
  --warmup_ratio ${WARMUP_RATIO} \
  --weight_decay ${WEIGHT_DECAY} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --out_dir "${OUT_DIR}" \
  --save_adapter \
  --adapter_dir "${ADAPTER_DIR}" \
  --experiment_name "${EXP_NAME}" \
  --early_stop ${EARLY_STOP} \
  --patience ${PATIENCE} \

status=$?
end=$(date +%s)
elapsed=$((end - start))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

### ============================================================================
### 10) SUMMARY
### ============================================================================

echo ""
echo "=========================================================================="
echo "TRAINING SUMMARY"
echo "=========================================================================="
echo "[TIME] Elapsed: ${elapsed} seconds (${minutes}m ${seconds}s)"
echo "[INFO] Exit code: ${status}"
echo "[INFO] Job finished: $(date)"
echo ""

if [ $status -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully"
    echo ""
    echo "Output locations:"
    echo "  Checkpoints: ${OUT_DIR}"
    echo "  Adapter:     ${ADAPTER_DIR}"
    echo ""
    echo "To use the trained model:"
    echo "  python gen/pipeline/inference_baseline.py \\"
    echo "    --adapter_dir ${ADAPTER_DIR} \\"
    echo "    --test_data dataset/baseline/test_baseline.jsonl"
else
    echo "[ERROR] Training failed with exit code: ${status}"
    echo ""
    echo "Check logs for details"
fi

echo "=========================================================================="

exit $status