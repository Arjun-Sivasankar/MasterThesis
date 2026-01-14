#!/bin/bash
#SBATCH --job-name=rag_textgen_train
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/train_textgen_rag_%j.out

set -e

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

# Choose dataset mode: "rag_unweighted", or "rag_weighted"
MODE="rag_unweighted"
# MODE="rag_weighted"

# Path config for H1/H2
H1=true
H2=false
COMBINED=false

BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct
# BASE_LLM=models/Meditron3-8B
echo "[INFO] Using base LLM: ${BASE_LLM}"

if [ "$BASE_LLM" = "models/Llama-3.1-8B-Instruct" ]; then
    LLM="llama3.1-8B"
elif [ "$BASE_LLM" = "meta-llama/Llama-3.2-1B-Instruct" ]; then
    LLM="llama3.2-1B"
elif [ "$BASE_LLM" = "models/Meditron3-8B" ]; then
    LLM="Meditron3-8B"
else
    LLM="unknown"
fi

EPOCHS=10

# Set data directory and output directory structure
if [ "$MODE" = "rag_unweighted" ]; then
    # Path config selection for RAG modes
    if [ "$H1" = true ] && [ "$H2" = false ] && [ "$COMBINED" = false ]; then
        path_config="h1"
    elif [ "$H1" = false ] && [ "$H2" = true ] && [ "$COMBINED" = false ]; then
        path_config="h2"
    elif [ "$H1" = false ] && [ "$H2" = false ] && [ "$COMBINED" = true ]; then
        path_config="combined"
    else
        echo "[ERROR] Invalid path configuration for rag_unweighted mode."
        exit 1
    fi
    DATA_PARENT_DIR=dataset/preprocessed_rag_full/map_desc/${path_config}/unweighted
    OUT_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_unweighted_checkpoints_$EPOCHS
    ADAPTER_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_unweighted_adapter_$EPOCHS
    EXP_NAME="rag_unweighted"
elif [ "$MODE" = "rag_weighted" ]; then
    # Path config selection for RAG modes
    if [ "$H1" = true ] && [ "$H2" = false ] && [ "$COMBINED" = false ]; then
        path_config="h1"
    elif [ "$H1" = false ] && [ "$H2" = true ] && [ "$COMBINED" = false ]; then
        path_config="h2"
    elif [ "$H1" = false ] && [ "$H2" = false ] && [ "$COMBINED" = true ]; then
        path_config="combined"
    else
        echo "[ERROR] Invalid path configuration for rag_weighted mode."
        exit 1
    fi
    DATA_PARENT_DIR=dataset/preprocessed_rag_full/map_desc/${path_config}/weighted_alpha0.3
    OUT_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_weighted_checkpoints_$EPOCHS
    ADAPTER_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_weighted_adapter_$EPOCHS
    EXP_NAME="rag_weighted_alpha0.3"
else
    echo "[ERROR] Unknown MODE: ${MODE}"
    exit 1
fi

echo "[INFO] Selected MODE: ${MODE}"
echo "[INFO] Data parent dir: ${DATA_PARENT_DIR}"

# Set train/val jsonl paths and token budgets
if [ "$MODE" = "rag_unweighted" ]; then
    TRAIN_JSONL=${DATA_PARENT_DIR}/train_rag_${path_config}_unweighted.jsonl
    VAL_JSONL=${DATA_PARENT_DIR}/val_rag_${path_config}_unweighted.jsonl
    MAX_LEN=5120
    MAX_PROMPT_TOKENS=3072
    MAX_KG_TOKENS=1500
    MAX_TARGET_TOKENS=512
elif [ "$MODE" = "rag_weighted" ]; then
    TRAIN_JSONL=${DATA_PARENT_DIR}/train_rag_${path_config}_weighted_alpha0.3.jsonl
    VAL_JSONL=${DATA_PARENT_DIR}/val_rag_${path_config}_weighted_alpha0.3.jsonl
    MAX_LEN=5120
    MAX_PROMPT_TOKENS=3072
    MAX_KG_TOKENS=1500
    MAX_TARGET_TOKENS=512
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
  --early_stop 0 \
  --patience 2 \

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
