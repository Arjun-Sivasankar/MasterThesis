#!/bin/bash
#SBATCH --job-name=Train_NextDiag
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/TextgenNextDiag/train_textgen_%j.out
#SBATCH --mail-type=start,end
#SBATCH --mail-user=arjun.sivasankar@mailbox.tu-dresden.de

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

### 3) Virtualenv
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"

### 4) HPC / Torch runtime
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

# Prefer Torch-prefixed envs (NCCL_* variants now warn)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"

### ========================================
### USER CONFIG - ADJUST THESE
### ========================================

# Prediction window: "6M" or "12M"
PREDICTION_WINDOW="6M"

# Model size: "1B", '7B' or "8B"
MODEL_SIZE="7B"

# Include index ICD codes in prompt?
INCLUDE_INDEX_ICD="--include_index_icd"

# Train on emergent diagnoses only? (new codes not in index visit)
# NEW_ONLY_FLAG="--new_only"
NEW_ONLY_FLAG=""

# Enable adaptive max_terms based on current visit complexity?
# ADAPTIVE_FLAG="--adaptive_max_terms"
ADAPTIVE_FLAG=""

### ========================================
### DERIVED PATHS AND SETTINGS
### ========================================

# Data paths based on prediction window - NOW USING PRE-SPLIT DATA
if [[ "${PREDICTION_WINDOW}" == "6M" ]]; then
  TRAIN_PKL="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/nextdiag_data/6M/train_df.pkl"
  VAL_PKL="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/nextdiag_data/6M/val_df.pkl"
  LABEL_COL="NEXT_DIAG_6M"
elif [[ "${PREDICTION_WINDOW}" == "12M" ]]; then
  TRAIN_PKL="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/nextdiag_data/12M/train_df.pkl"
  VAL_PKL="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/nextdiag_data/12M/val_df.pkl"
  LABEL_COL="NEXT_DIAG_12M"
else
  echo "[ERROR] Invalid PREDICTION_WINDOW: ${PREDICTION_WINDOW}. Use '6M' or '12M'."
  exit 1
fi

# Check if files exist
if [[ ! -f "${TRAIN_PKL}" ]]; then
  echo "[ERROR] Training data not found: ${TRAIN_PKL}"
  exit 1
fi
if [[ ! -f "${VAL_PKL}" ]]; then
  echo "[ERROR] Validation data not found: ${VAL_PKL}"
  exit 1
fi

# Base LLM path
if [[ "${MODEL_SIZE}" == "1B" ]]; then
  BASE_LLM="meta-llama/Llama-3.2-1B-Instruct"
elif [[ "${MODEL_SIZE}" == "8B" ]]; then
  BASE_LLM="models/Llama-3.1-8B-Instruct"
elif [[ "${MODEL_SIZE}" == "7B" ]]; then
  BASE_LLM="models/Meditron3-8B"
else
  echo "[ERROR] Invalid MODEL_SIZE: ${MODEL_SIZE}. Use '1B', '7B' or '8B'."
  exit 1
fi

# Adapter output directory
MODE_SUFFIX=""
if [[ "${INCLUDE_INDEX_ICD}" == "--include_index_icd" ]]; then
  MODE_SUFFIX="${MODE_SUFFIX}_withIndex"
fi
if [[ "${NEW_ONLY_FLAG}" == "--new_only" ]]; then
  MODE_SUFFIX="${MODE_SUFFIX}_emergentOnly"
fi
if [[ "${ADAPTIVE_FLAG}" == "--adaptive_max_terms" ]]; then
  MODE_SUFFIX="${MODE_SUFFIX}_adaptive"
else
  MODE_SUFFIX="${MODE_SUFFIX}_static"
fi

### ========================================
### TRAINING HYPERPARAMETERS
### ========================================

EPOCHS=10
BATCH_SIZE=1
GRAD_ACCUM=16
LR=2e-4
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
EARLY_STOP=1
PATIENCE=2

# Prompt budget parameters - Window-specific
# MAX_LEN=3072
MAX_LEN=4096

# Set N_max_terms and min_assistant_tokens based on prediction window
if [[ "${PREDICTION_WINDOW}" == "6M" ]]; then
  N_MAX_TERMS=22           # 6M: p90=25, use 22 to cover most cases
  MIN_ASSIST_TOKENS=140    # Allow more tokens for longer outputs
elif [[ "${PREDICTION_WINDOW}" == "12M" ]]; then
  N_MAX_TERMS=20           # 12M: p90=24, use 20 to cover most cases
  MIN_ASSIST_TOKENS=130    # Slightly fewer tokens needed
else
  N_MAX_TERMS=20
  MIN_ASSIST_TOKENS=128
fi

ADAPTER_DIR="runs_textgen_nextdiag/adapter_v1_${MODEL_SIZE}_nextdiag_${PREDICTION_WINDOW}${MODE_SUFFIX}_${EPOCHS}"
CHECKPOINT_DIR="runs_textgen_nextdiag/checkpoints_${MODEL_SIZE}_${PREDICTION_WINDOW}${MODE_SUFFIX}_${EPOCHS}"

# ICD index directory for title mapping
ICD_INDEX_DIR="./gen/TextGen/icd_index_v9"

# KG nodes mapping file (for code descriptions)
KG_NODES_PATH="./KG/kg_output4/kg_nodes.csv"
# if --kg_path is provided, then adapter dir and checkpoint dir should reflect that
if [[ -n "${KG_NODES_PATH}" ]]; then
  ADAPTER_DIR="${ADAPTER_DIR}_withKGnodes"
  CHECKPOINT_DIR="${CHECKPOINT_DIR}_withKGnodes"
fi

ADAPTER_DIR="${ADAPTER_DIR}_${MAX_LEN}"
CHECKPOINT_DIR="${CHECKPOINT_DIR}_${MAX_LEN}"

### ========================================
### LOGGING
### ========================================

echo "[INFO] ========================================"
echo "[INFO] NEXT DIAGNOSIS PREDICTION TRAINING"
echo "[INFO] ========================================"
echo "[INFO] Prediction Window: ${PREDICTION_WINDOW}"
echo "[INFO] Model Size: ${MODEL_SIZE}"
echo "[INFO] Base LLM: ${BASE_LLM}"
echo "[INFO] Training for Epochs: ${EPOCHS}"
echo "[INFO] Train data: ${TRAIN_PKL}"
echo "[INFO] Val data: ${VAL_PKL}"
echo "[INFO] Label Column: ${LABEL_COL}"
echo "[INFO] Include Index ICD: ${INCLUDE_INDEX_ICD:-NO}"
echo "[INFO] Emergent Only: ${NEW_ONLY_FLAG:-NO}"
echo "[INFO] Adaptive Max Terms: ${ADAPTIVE_FLAG:-NO (static)}"
echo "[INFO] N_max_terms: ${N_MAX_TERMS}"
echo "[INFO] Min assistant tokens: ${MIN_ASSIST_TOKENS}"
echo "[INFO] Adapter Dir: ${ADAPTER_DIR}"
echo "[INFO] Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "[INFO] Job started: $(date)"
echo "[INFO] Early Stop: ${EARLY_STOP} (Patience: ${PATIENCE})"
echo "[INFO] Learning Rate: ${LR}, Weight Decay: ${WEIGHT_DECAY}, Warmup Ratio: ${WARMUP_RATIO}"
echo "[INFO] Batch Size: ${BATCH_SIZE}, Grad Accum: ${GRAD_ACCUM}"
echo "[INFO] Max Input Length: ${MAX_LEN} tokens"
echo "[INFO] KG Nodes Path: ${KG_NODES_PATH}"
echo "[INFO] ========================================"

### ========================================
### RUN TRAINING
### ========================================

python gen/pipeline/train_textgen_nextdiag.py \
  --train_pickle "${TRAIN_PKL}" \
  --val_pickle "${VAL_PKL}" \
  --label_col "${LABEL_COL}" \
  --target_mode "icd_titles" \
  --icd_index_dir "${ICD_INDEX_DIR}" \
  --seed 42 \
  --future_window "${PREDICTION_WINDOW}" \
  ${INCLUDE_INDEX_ICD} \
  ${NEW_ONLY_FLAG} \
  ${ADAPTIVE_FLAG} \
  --N_max_terms ${N_MAX_TERMS} \
  --max_len ${MAX_LEN} \
  --min_assistant_tokens ${MIN_ASSIST_TOKENS} \
  --llm "${BASE_LLM}" \
  --epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --grad_accum ${GRAD_ACCUM} \
  --learning_rate ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --early_stop ${EARLY_STOP} \
  --patience ${PATIENCE} \
  --out_dir "${CHECKPOINT_DIR}" \
  --save_adapter \
  --adapter_dir "${ADAPTER_DIR}" \
  --kg_path "${KG_NODES_PATH}"

status=$?

echo "[INFO] ========================================"
echo "[INFO] Training finished: $(date)"
echo "[INFO] Exit status: ${status}"
echo "[INFO] ========================================"

exit ${status}


# ### 1) Modules
# module purge
# module load release/24.04 GCCcore/11.3.0
# module load Python/3.10.4

# ### 2) Project dir
# cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

# ### 3) Virtualenv
# source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
# echo "[INFO] Virtual env: $VIRTUAL_ENV"

# ### 4) HPC / Torch runtime
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# export TOKENIZERS_PARALLELISM=false

# # Prefer Torch-prefixed envs (NCCL_* variants now warn)
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# # If you have IB + Ethernet, list them; avoid loopback & docker
# export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
# export NCCL_DEBUG=INFO

# # Respect SLURM's GPU allocation (do NOT hardcode CUDA_VISIBLE_DEVICES)
# GPUS=${SLURM_GPUS_ON_NODE:-3}
# echo "[INFO] Using $GPUS GPUs for training"

# SCRIPT=gen/pipeline/train_textgen_nextdiag.py

# ### 5) Configuration: Set prediction window here
# PREDICTION_WINDOW="6M"  # Options: "6M" or "12M"
# echo "[INFO] Prediction Window: ${PREDICTION_WINDOW}"

# # Set data path and label column based on window
# if [[ "${PREDICTION_WINDOW}" == "6M" ]]; then
#   DATA_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/mimic_diag_6m.pkl
#   LABEL_COL="NEXT_DIAG_6M"
# elif [[ "${PREDICTION_WINDOW}" == "12M" ]]; then
#   DATA_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/mimic_diag_12m.pkl
#   LABEL_COL="NEXT_DIAG_12M"
# else
#   echo "[ERROR] Invalid PREDICTION_WINDOW: ${PREDICTION_WINDOW}"
#   exit 1
# fi

# echo "[INFO] Data: ${DATA_PKL}"
# echo "[INFO] Label Column: ${LABEL_COL}"

# ICD_INDEX=./gen/TextGen/icd_index_v9

# EPOCHS=1  
# echo "[INFO] Training for Epochs: ${EPOCHS}"

# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# # BASE_LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
# echo "[INFO] Using base LLM: ${BASE_LLM}"

# # if BASE_LLM is "meta-llama/Llama-3.2-1B-Instruct", set llm='1B', otherwise '8B'
# if [[ "${BASE_LLM}" == "meta-llama/Llama-3.2-1B-Instruct" ]]; then
#   llm='1B'
# else
#   llm='8B'
# fi

# OUT_DIR=runs_textgen_nextdiag/checkpoints/llm_${llm}_nextdiag_${PREDICTION_WINDOW}
# ADAPTER_DIR=runs_textgen_nextdiag/adapter_v1_${llm}_nextdiag_${PREDICTION_WINDOW}

# echo "[INFO] Output Dir: ${OUT_DIR}"
# echo "[INFO] Adapter Dir: ${ADAPTER_DIR}"

# # Optional: Enable emergent diagnosis mode (new diagnoses only)
# NEW_ONLY_FLAG=""
# # Uncomment the next line to predict only NEW diagnoses (not in index visit)
# # NEW_ONLY_FLAG="--new_only"

# ### 6) Run: multi-GPU training (DDP) via torchrun
# start=$(date +%s)
# echo "[INFO] Job started: $(date)"
# echo "[INFO] Running Script: ${SCRIPT}"
# echo "[INFO] Launching training…"

# srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
#   --data_pickle "${DATA_PKL}" \
#   --llm "${BASE_LLM}" \
#   --label_col "${LABEL_COL}" \
#   --target_mode icd_titles \
#   --icd_index_dir "${ICD_INDEX}" \
#   --future_window "${PREDICTION_WINDOW}" \
#   --include_index_icd \
#   ${NEW_ONLY_FLAG} \
#   --epochs ${EPOCHS} \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --grad_accum 16 \
#   --learning_rate 2e-4 \
#   --weight_decay 0.0 \
#   --warmup_ratio 0.03 \
#   --N_max_terms 12 \
#   --min_assistant_tokens 128 \
#   --early_stop 1 \
#   --patience 2 \
#   --out_dir "${OUT_DIR}" \
#   --save_adapter \
#   --adapter_dir "${ADAPTER_DIR}" \
#   --seed 42

# status=$?
# end=$(date +%s)
# echo "[TIME] Elapsed: $((end-start)) s"
# echo "[INFO] Train exit code: $status"
# exit $status