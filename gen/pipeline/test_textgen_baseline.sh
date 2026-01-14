#!/bin/bash
#SBATCH --job-name=test_baseline
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen/test_baseline_%j.out

set -e

### ============================================================================
### BASELINE TESTING CONFIGURATION
### ============================================================================
# This script tests baseline diagnosis generation models (no KG retrieval)
# Uses SapBERT for mapping generated text to ICD-9 codes
### ============================================================================

echo "========================================"
echo "BASELINE TESTING"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "[ERROR] Project dir not found"; exit 1; }

### 3) Virtualenv
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"

### 4) Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_DISABLE_PROGRESS_BAR=1
export TRANSFORMERS_VERBOSITY=error

SCRIPT=gen/pipeline/test_textgen_baseline.py
echo "[INFO] Testing Script: ${SCRIPT}"

### ============================================================================
### CONFIGURATION
### ============================================================================

# Test mode
BASE_MODEL_ONLY=false  

# Model selection
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct
# BASE_LLM=models/Meditron3-8B

if [ "$BASE_LLM" = "models/Llama-3.1-8B-Instruct" ] || [ "$BASE_LLM" = "meta-llama/Llama-3.1-8B-Instruct" ]; then
    LLM="llama3.1-8B"
elif [ "$BASE_LLM" = "meta-llama/Llama-3.2-1B-Instruct" ]; then
    LLM="llama3.2-1B"
elif [ "$BASE_LLM" = "models/Meditron3-8B" ]; then
    LLM="Meditron3-8B"
else
    LLM="unknown"
fi

EPOCHS=10

# Paths
TEST_JSONL=dataset/baseline/test_baseline.jsonl
ADAPTER_DIR=runs_textgen/baseline/${LLM}/adapter_${EPOCHS}

# SapBERT Mapper paths
ICD_INDEX=gen/pipeline/icd_index_v9
ENCODER_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext

# Subset testing
SUBSET_SIZE=0        # Set >0 for subset, 0 for full test set
SUBSET_SEED=42

# Bucket CSV paths
TOP_CODES_CSV=./analysis_results/top_50_codes.csv
BOTTOM_CODES_CSV=./analysis_results/bottom_50_codes.csv
TOP_PARENTS_CSV=./analysis_results/top_50_category_levels.csv

# Output paths
if [ "$BASE_MODEL_ONLY" = true ]; then
    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen/baseline/${LLM}/base_model_test_metrics_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen/baseline/${LLM}/base_model_test_shards_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen/baseline/${LLM}/base_model_test_metrics.json
        TMP_DIR=runs_textgen/baseline/${LLM}/base_model_test_shards
    fi
else
    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen/baseline/${LLM}/baseline_test_metrics_${EPOCHS}_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen/baseline/${LLM}/baseline_test_shards_${EPOCHS}_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen/baseline/${LLM}/baseline_test_metrics_${EPOCHS}.json
        TMP_DIR=runs_textgen/baseline/${LLM}/baseline_test_shards_${EPOCHS}
    fi
fi

# Generation config
DECODING=greedy
NUM_BEAMS=2
GEN_MAX_NEW=128
GEN_BS=8
N_MAX_TERMS=12
TEMP=1.0
TOPP=0.95
TOPK=50

# Mapper config
FAISS_ROWS=50
TAU_COS=0.40
TAU_FINAL=0.60
W_COS=1.0
W_FUZ=0.0

PRINT_SAMPLES=5
USE_BF16=1

# (set to true to update prompt k to match N_MAX_TERMS)
UPDATE_PROMPT_K=false

### ============================================================================
### DISPLAY CONFIGURATION
### ============================================================================

echo ""
echo "=========================================================================="
echo "TESTING CONFIGURATION"
echo "=========================================================================="
echo "Mode:           BASELINE (no KG retrieval)"
if [ "$BASE_MODEL_ONLY" = true ]; then
    echo "ABLATION:       Base model only (no adapter)"
fi
echo "Test data:      ${TEST_JSONL}"
echo "Base LLM:       ${BASE_LLM}"
echo "LLM identifier: ${LLM}"
if [ "$BASE_MODEL_ONLY" != true ]; then
    echo "Adapter:        ${ADAPTER_DIR}"
    echo "Epochs:         ${EPOCHS}"
fi
echo ""
if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
    echo "SUBSET MODE:"
    echo "  Samples:      ${SUBSET_SIZE}"
    echo "  Seed:         ${SUBSET_SEED}"
else
    echo "FULL TEST SET"
fi
echo ""
echo "GENERATION:"
echo "  Decoding:     ${DECODING}"
echo "  Num beams:    ${NUM_BEAMS}"
echo "  Max new:      ${GEN_MAX_NEW}"
echo "  Batch size:   ${GEN_BS}"
echo "  Max terms:    ${N_MAX_TERMS}"
echo "  Update k:     ${UPDATE_PROMPT_K}"
echo "  Print samples: ${PRINT_SAMPLES}"
echo ""
echo "SAPBERT MAPPER:"
echo "  Index:        ${ICD_INDEX}"
echo "  Encoder:      ${ENCODER_MODEL}"
echo "  FAISS rows:   ${FAISS_ROWS}"
echo "  Weights:      w_cos=${W_COS}, w_fuz=${W_FUZ}"
echo ""
echo "BUCKET EVALUATION:"
echo "  Top codes:    ${TOP_CODES_CSV}"
echo "  Bottom codes: ${BOTTOM_CODES_CSV}"
echo "  Top parents:  ${TOP_PARENTS_CSV}"
echo ""
echo "OUTPUT:"
echo "  Metrics:      ${OUT_METRICS}"
echo "  Temp dir:     ${TMP_DIR}"
echo ""
echo "HARDWARE:"
echo "  GPUs:         ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "  Memory:       ${SLURM_MEM_PER_NODE}M"
echo "  Use BF16:     ${USE_BF16}"
echo "=========================================================================="
echo ""

### ============================================================================
### VALIDATION
### ============================================================================

# Check if test data exists
if [ ! -f "${TEST_JSONL}" ]; then
    echo "[ERROR] Test data not found: ${TEST_JSONL}"
    echo "[INFO] Please run preprocess_baseline.py first"
    exit 1
fi
echo "[INFO] Test data found: ${TEST_JSONL}"

# Check if adapter exists (skip for base_model_only mode)
if [ "$BASE_MODEL_ONLY" != true ]; then
    if [ ! -d "${ADAPTER_DIR}" ]; then
        echo "[ERROR] Adapter directory not found: ${ADAPTER_DIR}"
        echo "[INFO] Please run train_textgen_baseline.sh first"
        exit 1
    fi
    echo "[INFO] Adapter found: ${ADAPTER_DIR}"
fi

# Check if ICD index exists
if [ ! -d "${ICD_INDEX}" ]; then
    echo "[ERROR] ICD index directory not found: ${ICD_INDEX}"
    exit 1
fi
echo "[INFO] ICD index found: ${ICD_INDEX}"

# Check if script exists
if [ ! -f "${SCRIPT}" ]; then
    echo "[ERROR] Testing script not found: ${SCRIPT}"
    exit 1
fi
echo "[INFO] Testing script found: ${SCRIPT}"

### ============================================================================
### RUN TESTING
### ============================================================================

start=$(date +%s)
echo ""
echo "=========================================================================="
echo "LAUNCHING TESTING"
echo "=========================================================================="
echo ""

# Create output directories
mkdir -p "$(dirname "${OUT_METRICS}")"
mkdir -p "${TMP_DIR}"

export CUDA_VISIBLE_DEVICES=0

CMD="python ${SCRIPT} \
  --test_jsonl \"${TEST_JSONL}\" \
  --base_model \"${BASE_LLM}\""

# Add adapter dir only if not in base_model_only mode
if [ "$BASE_MODEL_ONLY" = true ]; then
    CMD="${CMD} --base_model_only"
else
    CMD="${CMD} --adapter_dir \"${ADAPTER_DIR}\""
fi

CMD="${CMD} \
  --icd_index_dir \"${ICD_INDEX}\" \
  --encoder_model \"${ENCODER_MODEL}\" \
  --max_len 5120 \
  --gen_max_new ${GEN_MAX_NEW} \
  --gen_batch_size ${GEN_BS} \
  --N_max_terms ${N_MAX_TERMS} \
  --decoding \"${DECODING}\" \
  --num_beams ${NUM_BEAMS} \
  --temperature ${TEMP} \
  --top_p ${TOPP} \
  --top_k ${TOPK} \
  --faiss_rows ${FAISS_ROWS} \
  --tau_cos ${TAU_COS} \
  --tau_final ${TAU_FINAL} \
  --w_cos ${W_COS} \
  --w_fuz ${W_FUZ} \
  --print_samples ${PRINT_SAMPLES} \
  --tmp_dir \"${TMP_DIR}\" \
  --out_metrics \"${OUT_METRICS}\" \
  --top_codes_csv \"${TOP_CODES_CSV}\" \
  --bottom_codes_csv \"${BOTTOM_CODES_CSV}\" \
  --top_parent_csv \"${TOP_PARENTS_CSV}\""

if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
    CMD="${CMD} --subset_size ${SUBSET_SIZE} --subset_seed ${SUBSET_SEED}"
fi

if [ "${USE_BF16}" = "1" ]; then
    CMD="${CMD} --use_bf16"
fi

if [ "${UPDATE_PROMPT_K}" = true ]; then
    CMD="${CMD} --update_prompt_k"
fi

echo "[INFO] Command: ${CMD}"
echo ""

eval ${CMD}

status=$?
end=$(date +%s)
elapsed=$((end - start))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

### ============================================================================
### SUMMARY
### ============================================================================

echo ""
echo "=========================================================================="
echo "TESTING SUMMARY"
echo "=========================================================================="
echo "[TIME] Elapsed: ${elapsed} seconds (${minutes}m ${seconds}s)"
echo "[INFO] Exit code: ${status}"
echo "[INFO] Job finished: $(date)"
echo ""

if [ $status -eq 0 ]; then
    echo "[SUCCESS] Testing completed successfully"
    echo ""
    echo "Results saved to:"
    echo "  Metrics: ${OUT_METRICS}"
    echo "  Buckets: $(dirname ${OUT_METRICS})/test_metrics_buckets.json"
    echo "  Per-label tables:"
    echo "    - $(dirname ${OUT_METRICS})/per_label_FULL.csv"
    echo "    - $(dirname ${OUT_METRICS})/per_label_TOP_50_CODES.csv"
    echo "    - $(dirname ${OUT_METRICS})/per_label_BOTTOM_50_CODES.csv"
    echo "    - $(dirname ${OUT_METRICS})/per_label_TOP_50_PARENTS.csv"
    echo "  Shards:  ${TMP_DIR}"
    echo ""
else
    echo "[ERROR] Testing failed with exit code: ${status}"
    echo ""
    echo "Check the log file for details:"
    echo "  logs/Textgen/test_baseline_${SLURM_JOB_ID}.out"
    echo ""
fi

echo "=========================================================================="

exit $status