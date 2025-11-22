#!/bin/bash
#SBATCH --job-name=Test_NextDiag
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/TextgenNextDiag/test_textgen_%j.out
#SBATCH --mail-type=start,end
#SBATCH --mail-user=arjun.sivasankar@mailbox.tu-dresden.de

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
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
# export NCCL_DEBUG=INFO

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

# SCRIPT=gen/pipeline/test_textgen_nextdiag.py
# ICD_INDEX=./gen/TextGen/icd_index_v9

# # Base model selection
# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# # BASE_LLM=models/Llama-3.1-8B-Instruct
# echo "[INFO] Using base LLM: ${BASE_LLM}"

# # Determine model size for paths
# if [[ "${BASE_LLM}" == "meta-llama/Llama-3.2-1B-Instruct" ]]; then
#   llm='1B'
# else
#   llm='8B'
# fi

# # Adapter and output paths
# ADAPTER_DIR=runs_textgen_nextdiag/adapter_v1_${llm}_nextdiag_${PREDICTION_WINDOW}
# TMP_DIR=runs_textgen_nextdiag/adapter_v1_${llm}_nextdiag_${PREDICTION_WINDOW}/test_shards
# OUT_METRICS=runs_textgen_nextdiag/adapter_v1_${llm}_nextdiag_${PREDICTION_WINDOW}/test_metrics.json

# echo "[INFO] Data: ${DATA_PKL}"
# echo "[INFO] Label Column: ${LABEL_COL}"
# echo "[INFO] Adapter Dir: ${ADAPTER_DIR}"
# echo "[INFO] Output Metrics: ${OUT_METRICS}"

# # Next-diagnosis specific flags
# INCLUDE_INDEX_ICD_FLAG="--include_index_icd"  # Include current diagnoses in prompt
# NEW_ONLY_FLAG=""  # Uncomment next line to evaluate only emergent diagnoses
# # NEW_ONLY_FLAG="--new_only"

# # Decoding config
# DECODING=greedy      # greedy | beam | sample
# NUM_BEAMS=2
# GEN_MAX_NEW=128
# GEN_BS=8
# NO_REPEAT_NGRAM=0
# TEMP=1.0
# TOPP=0.95
# TOPK=50

# # Label space
# LABELS_SPACE=full    # full | head
# HEAD_K=0             # only used if LABELS_SPACE=head

# PRINT_SAMPLES=5
# USE_BF16=1           # 1 to request bf16 if GPU allows it

# # Optional: CSV files for per-label metrics
# TOP_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_codes.csv
# BOT_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/bottom_50_codes.csv
# TOP_PARENTS=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_category_levels.csv

# # Mapper configs
# w_cos=0.6
# w_fuz=0.4

# ### 6) Launch test

# ##############
# # A) SINGLE-GPU test
# ##############
# if true ; then
#   echo "[INFO] Job started: $(date)"
#   echo "[INFO] Running Script: ${SCRIPT}"
#   echo "[INFO] Running SINGLE-GPU test on GPU 0"
#   export CUDA_VISIBLE_DEVICES=0
  
#   start=$(date +%s)
  
#   python ${SCRIPT} \
#     --data_pickle "${DATA_PKL}" \
#     --label_col "${LABEL_COL}" \
#     --future_window "${PREDICTION_WINDOW}" \
#     ${INCLUDE_INDEX_ICD_FLAG} \
#     ${NEW_ONLY_FLAG} \
#     --base_model "${BASE_LLM}" \
#     --adapter_dir "${ADAPTER_DIR}" \
#     --icd_index_dir "${ICD_INDEX}" \
#     --decoding "${DECODING}" \
#     --num_beams ${NUM_BEAMS} \
#     --gen_batch_size ${GEN_BS} \
#     --gen_max_new ${GEN_MAX_NEW} \
#     --no_repeat_ngram ${NO_REPEAT_NGRAM} \
#     --temperature ${TEMP} \
#     --top_p ${TOPP} \
#     --top_k ${TOPK} \
#     --labels_space ${LABELS_SPACE} \
#     --labels_head_k ${HEAD_K} \
#     --print_samples ${PRINT_SAMPLES} \
#     --tmp_dir "${TMP_DIR}" \
#     --out_metrics "${OUT_METRICS}" \
#     --top_codes_csv "${TOP_CODES}" \
#     --bottom_codes_csv "${BOT_CODES}" \
#     --top_parent_csv "${TOP_PARENTS}" \
#     --w_cos ${w_cos} \
#     --w_fuz ${w_fuz} \
#     $( [[ "${USE_BF16}" == "1" ]] && echo --use_bf16 )
  
#   status=$?
#   end=$(date +%s)
#   echo "[TIME] Elapsed: $((end-start)) s"
# fi

# ##############
# # B) MULTI-GPU sharded inference (uncomment to use)
# ##############
# : '
# GPUS=${SLURM_GPUS_ON_NODE:-3}
# echo "[INFO] Running MULTI-GPU sharded inference on $GPUS GPUs"

# start=$(date +%s)

# srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
#   --distributed \
#   --data_pickle "${DATA_PKL}" \
#   --label_col "${LABEL_COL}" \
#   --future_window "${PREDICTION_WINDOW}" \
#   ${INCLUDE_INDEX_ICD_FLAG} \
#   ${NEW_ONLY_FLAG} \
#   --base_model "${BASE_LLM}" \
#   --adapter_dir "${ADAPTER_DIR}" \
#   --icd_index_dir "${ICD_INDEX}" \
#   --decoding "${DECODING}" \
#   --num_beams ${NUM_BEAMS} \
#   --gen_batch_size ${GEN_BS} \
#   --gen_max_new ${GEN_MAX_NEW} \
#   --no_repeat_ngram ${NO_REPEAT_NGRAM} \
#   --temperature ${TEMP} \
#   --top_p ${TOPP} \
#   --top_k ${TOPK} \
#   --labels_space ${LABELS_SPACE} \
#   --labels_head_k ${HEAD_K} \
#   --print_samples ${PRINT_SAMPLES} \
#   --tmp_dir "${TMP_DIR}" \
#   --out_metrics "${OUT_METRICS}" \
#   --top_codes_csv "${TOP_CODES}" \
#   --bottom_codes_csv "${BOT_CODES}" \
#   --top_parent_csv "${TOP_PARENTS}" \
#   --w_cos ${w_cos} \
#   --w_fuz ${w_fuz} \
#   $( [[ "${USE_BF16}" == "1" ]] && echo --use_bf16 )

# status=$?
# end=$(date +%s)
# echo "[TIME] Elapsed: $((end-start)) s"
# '

# echo "[INFO] Test exit code: $status"
# exit $status

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

# Evaluate emergent diagnoses only?
# NEW_ONLY_FLAG="--new_only"
NEW_ONLY_FLAG=""

# Enable adaptive max_terms based on current visit complexity?
# ADAPTIVE_FLAG="--adaptive_max_terms"
ADAPTIVE_FLAG=""

### ========================================
### DERIVED PATHS AND SETTINGS
### ========================================

# Data path based on prediction window - NOW USING PRE-SPLIT TEST DATA
if [[ "${PREDICTION_WINDOW}" == "6M" ]]; then
  TEST_PKL="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/nextdiag_data/6M/test_df.pkl"
  LABEL_COL="NEXT_DIAG_6M"
elif [[ "${PREDICTION_WINDOW}" == "12M" ]]; then
  TEST_PKL="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/nextdiag_data/12M/test_df.pkl"
  LABEL_COL="NEXT_DIAG_12M"
else
  echo "[ERROR] Invalid PREDICTION_WINDOW: ${PREDICTION_WINDOW}. Use '6M' or '12M'."
  exit 1
fi

# Check if test file exists
if [[ ! -f "${TEST_PKL}" ]]; then
  echo "[ERROR] Test data not found: ${TEST_PKL}"
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
  echo "[ERROR] Invalid MODEL_SIZE: ${MODEL_SIZE}. Use '1B' or '8B'."
  exit 1
fi

# Adapter directory (must match training output)
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

EPOCHS="10"  # Must match training epochs

ADAPTER_DIR="runs_textgen_nextdiag/adapter_v1_${MODEL_SIZE}_nextdiag_${PREDICTION_WINDOW}${MODE_SUFFIX}_${EPOCHS}"
TMP_DIR="${ADAPTER_DIR}/test_shards"

# ICD index directory
ICD_INDEX_DIR="./gen/TextGen/icd_index_v9"

# Encoder model for semantic mapping
ENCODER_MODEL="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

KG_NODES_PATH="./KG/kg_output4/kg_nodes.csv"
# if --kg_path is provided, then adapter dir and checkpoint dir should reflect that
if [[ -n "${KG_NODES_PATH}" ]]; then
  ADAPTER_DIR="${ADAPTER_DIR}_withKGnodes"
  CHECKPOINT_DIR="${CHECKPOINT_DIR}_withKGnodes"
fi

### ========================================
### GENERATION PARAMETERS
### ========================================

# Decoding config - Window-specific
DECODING=greedy
NUM_BEAMS=2
NO_REPEAT_NGRAM=3        # Prevent repetitive outputs
TEMP=1.0
TOPP=0.95
TOPK=50

# Window-specific generation parameters
if [[ "${PREDICTION_WINDOW}" == "6M" ]]; then
  GEN_MAX_NEW=140        # 6M: median=14 codes × ~20 tokens/code + buffer
  N_MAX_TERMS=22         # Cover p90=25
elif [[ "${PREDICTION_WINDOW}" == "12M" ]]; then
  GEN_MAX_NEW=320        # 12M: median=12 codes × ~20 tokens/code + buffer
  N_MAX_TERMS=20         # Cover p90=24
else
  GEN_MAX_NEW=384
  N_MAX_TERMS=22
fi

GEN_BS=8

# Label space
LABELS_SPACE=full
HEAD_K=0

# Max input length
# MAX_LEN=3072
MAX_LEN=4096

PRINT_SAMPLES=5
USE_BF16=1

### ========================================
### SEMANTIC MAPPER CONFIG
### ========================================

FAISS_ROWS=50
TAU_COS=0.40
TAU_FINAL=0.60
W_COS=0.6
W_FUZ=0.4

### ========================================
### OPTIONAL CSV PATHS FOR STRATIFIED EVAL
### ========================================

# Top/bottom codes and parent categories (optional)
TOP_CODES_CSV=""
BOTTOM_CODES_CSV=""
TOP_PARENT_CSV=""

ADAPTER_DIR="${ADAPTER_DIR}_${MAX_LEN}"
CHECKPOINT_DIR="${CHECKPOINT_DIR}_${MAX_LEN}"
OUT_METRICS="${ADAPTER_DIR}/test_metrics.json"

### ========================================
### LOGGING
### ========================================

echo "[INFO] ========================================"
echo "[INFO] NEXT DIAGNOSIS PREDICTION TESTING"
echo "[INFO] ========================================"
echo "[INFO] Prediction Window: ${PREDICTION_WINDOW}"
echo "[INFO] Using base LLM: ${BASE_LLM}"
echo "[INFO] Test data: ${TEST_PKL}"
echo "[INFO] Label Column: ${LABEL_COL}"
echo "[INFO] Adapter Dir: ${ADAPTER_DIR}"
echo "[INFO] Output Metrics: ${OUT_METRICS}"
echo "[INFO] Include Index ICD: ${INCLUDE_INDEX_ICD:-NO}"
echo "[INFO] Emergent Only: ${NEW_ONLY_FLAG:-NO}"
echo "[INFO] Adaptive Max Terms: ${ADAPTIVE_FLAG:-NO (static)}"
echo "[INFO] N_max_terms: ${N_MAX_TERMS}"
echo "[INFO] Gen max new tokens: ${GEN_MAX_NEW}"
echo "[INFO] Job started: $(date)"
echo "[INFO] Max input length: ${MAX_LEN}"
echo "[INFO] ========================================"

### ========================================
### CHECK IF ADAPTER EXISTS
### ========================================

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "[ERROR] Adapter directory not found: ${ADAPTER_DIR}"
  echo "[ERROR] Please train the model first using train_textgen_nextdiag.sh"
  exit 1
fi

### ========================================
### RUN TESTING
### ========================================

echo "[INFO] Running Script: gen/pipeline/test_textgen_nextdiag.py"

# Check if running in distributed mode
if [[ -n "${SLURM_NTASKS}" ]] && [[ ${SLURM_NTASKS} -gt 1 ]]; then
  echo "[INFO] Running DISTRIBUTED test across ${SLURM_NTASKS} GPUs"
  
  srun python gen/pipeline/test_textgen_nextdiag.py \
    --test_pickle "${TEST_PKL}" \
    --label_col "${LABEL_COL}" \
    --seed 42 \
    --future_window "${PREDICTION_WINDOW}" \
    ${INCLUDE_INDEX_ICD} \
    ${NEW_ONLY_FLAG} \
    ${ADAPTIVE_FLAG} \
    --N_max_terms ${N_MAX_TERMS} \
    --max_len ${MAX_LEN} \
    --gen_max_new ${GEN_MAX_NEW} \
    --gen_batch_size ${GEN_BS} \
    --decoding ${DECODING} \
    --num_beams ${NUM_BEAMS} \
    --temperature ${TEMP} \
    --top_p ${TOPP} \
    --top_k ${TOPK} \
    --no_repeat_ngram ${NO_REPEAT_NGRAM} \
    --base_model "${BASE_LLM}" \
    --adapter_dir "${ADAPTER_DIR}" \
    $(if [ ${USE_BF16} -eq 1 ]; then echo "--use_bf16"; fi) \
    --icd_index_dir "${ICD_INDEX_DIR}" \
    --encoder_model "${ENCODER_MODEL}" \
    --faiss_rows ${FAISS_ROWS} \
    --tau_cos ${TAU_COS} \
    --tau_final ${TAU_FINAL} \
    --w_cos ${W_COS} \
    --w_fuz ${W_FUZ} \
    --labels_space ${LABELS_SPACE} \
    --labels_head_k ${HEAD_K} \
    --print_samples ${PRINT_SAMPLES} \
    $(if [ -n "${TOP_CODES_CSV}" ]; then echo "--top_codes_csv ${TOP_CODES_CSV}"; fi) \
    $(if [ -n "${BOTTOM_CODES_CSV}" ]; then echo "--bottom_codes_csv ${BOTTOM_CODES_CSV}"; fi) \
    $(if [ -n "${TOP_PARENT_CSV}" ]; then echo "--top_parent_csv ${TOP_PARENT_CSV}"; fi) \
    --distributed \
    --tmp_dir "${TMP_DIR}" \
    --out_metrics "${OUT_METRICS}" \
    --kg_path "${KG_NODES_PATH}"
else
  echo "[INFO] Running SINGLE-GPU test on GPU 0"
  
  CUDA_VISIBLE_DEVICES=0 python gen/pipeline/test_textgen_nextdiag.py \
    --test_pickle "${TEST_PKL}" \
    --label_col "${LABEL_COL}" \
    --seed 42 \
    --future_window "${PREDICTION_WINDOW}" \
    ${INCLUDE_INDEX_ICD} \
    ${NEW_ONLY_FLAG} \
    ${ADAPTIVE_FLAG} \
    --N_max_terms ${N_MAX_TERMS} \
    --max_len ${MAX_LEN} \
    --gen_max_new ${GEN_MAX_NEW} \
    --gen_batch_size ${GEN_BS} \
    --decoding ${DECODING} \
    --num_beams ${NUM_BEAMS} \
    --temperature ${TEMP} \
    --top_p ${TOPP} \
    --top_k ${TOPK} \
    --no_repeat_ngram ${NO_REPEAT_NGRAM} \
    --base_model "${BASE_LLM}" \
    --adapter_dir "${ADAPTER_DIR}" \
    $(if [ ${USE_BF16} -eq 1 ]; then echo "--use_bf16"; fi) \
    --icd_index_dir "${ICD_INDEX_DIR}" \
    --encoder_model "${ENCODER_MODEL}" \
    --faiss_rows ${FAISS_ROWS} \
    --tau_cos ${TAU_COS} \
    --tau_final ${TAU_FINAL} \
    --w_cos ${W_COS} \
    --w_fuz ${W_FUZ} \
    --labels_space ${LABELS_SPACE} \
    --labels_head_k ${HEAD_K} \
    --print_samples ${PRINT_SAMPLES} \
    $(if [ -n "${TOP_CODES_CSV}" ]; then echo "--top_codes_csv ${TOP_CODES_CSV}"; fi) \
    $(if [ -n "${BOTTOM_CODES_CSV}" ]; then echo "--bottom_codes_csv ${BOTTOM_CODES_CSV}"; fi) \
    $(if [ -n "${TOP_PARENT_CSV}" ]; then echo "--top_parent_csv ${TOP_PARENT_CSV}"; fi) \
    --tmp_dir "${TMP_DIR}" \
    --out_metrics "${OUT_METRICS}" \
    --kg_path "${KG_NODES_PATH}"
fi

status=$?

echo "[INFO] ========================================"
echo "[INFO] Testing finished: $(date)"
echo "[INFO] Exit status: ${status}"
echo "[INFO] Results saved to: ${OUT_METRICS}"
echo "[INFO] ========================================"

exit ${status}