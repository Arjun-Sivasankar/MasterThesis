#!/bin/bash
#SBATCH --job-name=test_codegen
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/CodeGen-DDP/test_codegen_%j.out

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "${PROJECT_DIR}" || { echo "Project dir not found: ${PROJECT_DIR}"; exit 1; }

### 3) Virtualenv
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"

### 4) Runtime envs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

echo "[INFO] Job started: $(date)"
echo "[INFO] GPUs allocated by SLURM: ${SLURM_GPUS_ON_NODE:-1}"

### ============================================================================
### CONFIGURATION
### ============================================================================

# MODE: "trained" or "base_model_only" (Uncomment whichever mode you want)
# MODE="trained"
MODE="base_model_only"

### Paths & args
SCRIPT_MOD=gen/pipeline/test_codegen.py

# Data
TEST_PKL=./dataset/final_data/test_df.pkl

# Model selection
# LLAMA_MODEL=meta-llama/Llama-3.2-1B-Instruct
# LLAMA_MODEL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
LLAMA_MODEL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Meditron3-8B

if [[ "$LLAMA_MODEL" == *"Llama-3.1-8B"* ]]; then
    LLM="llama3.1-8B"
elif [[ "$LLAMA_MODEL" == *"Llama-3.2-1B"* ]]; then
    LLM="llama3.2-1B"
elif [[ "$LLAMA_MODEL" == *"Meditron3-8B"* ]]; then
    LLM="Meditron3-8B"
else
    LLM="unknown"
fi

echo "[INFO] Using model: ${LLAMA_MODEL} (${LLM})"

# Set paths based on MODE
if [ "$MODE" = "base_model_only" ]; then
    # Base model ablation - no adapter
    RUN_ROOT=runs_gen/icd9_gen_ddp
    RUN_NAME=base_model_ablation_${LLM}
    ADAPTER_DIR=""  # No adapter for base model only
    LABELS_JSON=runs_gen/icd9_gen_ddp/20251209-094701_N54981_icd9_complete/label_space.json  # Use existing label space
    OUT_DIR=${RUN_ROOT}/${RUN_NAME}/eval_${SLURM_JOB_ID}
    
elif [ "$MODE" = "trained" ]; then
    RUN_ROOT=runs_gen/icd9_gen_ddp
    RUN_NAME=${RUN_NAME:-20251209-094701_N54981_icd9_complete}   # e.g., from training
    ADAPTER_DIR=${RUN_ROOT}/${RUN_NAME}/adapter_best
    LABELS_JSON=${RUN_ROOT}/${RUN_NAME}/label_space.json
    OUT_DIR=${RUN_ROOT}/${RUN_NAME}/eval_${SLURM_JOB_ID}
    
else
    echo "[ERROR] Unknown MODE: $MODE"
    exit 1
fi

# Generation/eval settings
MAX_LEN=3072
GEN_MAX_NEW=96
TEST_BATCH_SIZE=16
USE_STRUCTURED=1
USE_NOTES=1

# Bucket files (updated paths)
TOP_CODES_CSV=./analysis_results/top_50_codes.csv
BOTTOM_CODES_CSV=./analysis_results/bottom_50_codes.csv
TOP_PARENTS_CSV=./analysis_results/top_50_category_levels.csv

# Number of sample predictions to show
TEST_EXAMPLES=5

### ============================================================================
### DISPLAY CONFIGURATION
### ============================================================================

echo ""
echo "=========================================================================="
echo "CODEGEN TESTING CONFIGURATION"
echo "=========================================================================="
echo "Mode:                 ${MODE}"
echo "Model:                ${LLAMA_MODEL}"
echo "LLM identifier:       ${LLM}"
if [ "$MODE" = "base_model_only" ]; then
    echo "ABLATION:          Base model only (no adapter)"
else
    echo "Adapter:              ${ADAPTER_DIR}"
    echo "Run name:             ${RUN_NAME}"
fi
echo ""
echo "DATA:"
echo "  Test data:          ${TEST_PKL}"
echo "  Labels JSON:        ${LABELS_JSON}"
echo ""
echo "GENERATION:"
echo "  Max length:         ${MAX_LEN}"
echo "  Max new tokens:     ${GEN_MAX_NEW}"
echo "  Batch size:         ${TEST_BATCH_SIZE}"
echo "  Use structured:     ${USE_STRUCTURED}"
echo "  Use notes:          ${USE_NOTES}"
echo "  Test examples:      ${TEST_EXAMPLES}"
echo ""
echo "BUCKET EVALUATION:"
echo "  Top codes:          ${TOP_CODES_CSV}"
echo "  Bottom codes:       ${BOTTOM_CODES_CSV}"
echo "  Top parents:        ${TOP_PARENTS_CSV}"
echo ""
echo "OUTPUT:"
echo "  Output dir:         ${OUT_DIR}"
echo ""
echo "HARDWARE:"
echo "  GPUs:               ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs:               ${SLURM_CPUS_PER_TASK}"
echo "  Memory:             ${SLURM_MEM_PER_NODE}M"
echo "=========================================================================="
echo ""

### 6) Sanity checks
[[ -f "${TEST_PKL}" ]] || { echo "[ERR] TEST_PKL not found: ${TEST_PKL}"; exit 2; }
[[ -f "${LABELS_JSON}" ]] || { echo "[ERR] LABELS_JSON not found: ${LABELS_JSON}"; exit 2; }

if [ "$MODE" = "trained" ]; then
    [[ -d "${ADAPTER_DIR}" ]] || { echo "[ERR] ADAPTER_DIR not found: ${ADAPTER_DIR}"; exit 2; }
fi

mkdir -p "${OUT_DIR}"

### 7) Launch tester (single process; no torchrun)
start=$(date +%s)
echo "[INFO] Running: ${SCRIPT_MOD}"
if [ "$MODE" = "trained" ]; then
    echo "[INFO] CodeGen TEST -> ADAPTER=${ADAPTER_DIR}"
else
    echo "[INFO] CodeGen BASE MODEL TEST (no adapter)"
fi
echo "[INFO] Outputs -> ${OUT_DIR}"
echo ""

CMD="srun python ${SCRIPT_MOD} \
  --test_pickle \"${TEST_PKL}\" \
  --llama_model \"${LLAMA_MODEL}\" \
  --labels_json \"${LABELS_JSON}\" \
  --max_len \"${MAX_LEN}\" \
  --gen_max_new \"${GEN_MAX_NEW}\" \
  --test_batch_size \"${TEST_BATCH_SIZE}\" \
  --use_structured \"${USE_STRUCTURED}\" \
  --use_notes \"${USE_NOTES}\" \
  --seed \"${SEED}\" \
  --top_codes_csv \"${TOP_CODES_CSV}\" \
  --bottom_codes_csv \"${BOTTOM_CODES_CSV}\" \
  --top_parent_csv \"${TOP_PARENTS_CSV}\" \
  --out_dir \"${OUT_DIR}\" \
  --test_examples \"${TEST_EXAMPLES}\""

# Add adapter or base_model_only flag
if [ "$MODE" = "base_model_only" ]; then
    CMD="${CMD} --base_model_only"
else
    CMD="${CMD} --adapter_dir \"${ADAPTER_DIR}\""
fi

echo "[CMD] ${CMD}"
echo ""

eval ${CMD}

status=$?
end=$(date +%s)
elapsed=$((end-start))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo ""
echo "=========================================================================="
echo "[TIME] Testing completed"
echo "[TIME] Elapsed: ${elapsed} seconds (${minutes}m ${seconds}s)"
echo "[INFO] Test exit code: ${status}"
echo "[INFO] Job finished: $(date)"
echo "=========================================================================="

if [ $status -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Testing completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  - ${OUT_DIR}/test_metrics.json"
    echo "  - ${OUT_DIR}/test_metrics_buckets.json"
    echo "  - ${OUT_DIR}/per_label_FULL.csv"
    echo "  - ${OUT_DIR}/per_label_TOP_50_CODES.csv"
    echo "  - ${OUT_DIR}/per_label_BOTTOM_50_CODES.csv"
    echo "  - ${OUT_DIR}/per_label_TOP_50_PARENTS.csv"
    echo ""
else
    echo ""
    echo "[ERROR] Testing failed with exit code: ${status}"
    echo ""
    echo "Check the log file for details:"
    echo "   logs/CodeGen-DDP/test_codegen_${SLURM_JOB_ID}.out"
    echo ""
fi

exit $status