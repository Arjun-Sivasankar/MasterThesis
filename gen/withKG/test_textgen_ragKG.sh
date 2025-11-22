#!/bin/bash
#SBATCH --job-name=test_rag_textgen
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/test_textgen_rag_%j.out

### ============================================================================
### SLURM SETTINGS
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

SCRIPT=gen/withKG/test_textgen_ragKG.py
echo "[INFO] Testing Script: ${SCRIPT}"

### ============================================================================
### 5) CONFIGURATION
### ============================================================================

# Choose which experiment to test
# MODE="rag_unweighted"
MODE="baseline"
# MODE="rag_weighted"

echo "[INFO] Testing MODE: ${MODE}"

# Model selection
# BASE_LLM=models/Llama-3.1-8B-Instruct
# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
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

EPOCHS=10

# ✓ SapBERT Mapper paths (REQUIRED - same as baseline)
ICD_INDEX=./gen/TextGen/icd_index_v9
ENCODER_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext

# Subset testing
SUBSET_SIZE=0        # Quick test on 100 samples
# SUBSET_SIZE=""       # Uncomment for full test set
SUBSET_SEED=42

# Display subset info
if [ -n "$SUBSET_SIZE" ] && [ "$SUBSET_SIZE" != "0" ]; then
    echo "[INFO] ⚠️  SUBSET MODE: Testing on ${SUBSET_SIZE} samples (seed=${SUBSET_SEED})"
else
    echo "[INFO] 📊 FULL TEST: Testing on entire test set"
    SUBSET_SIZE=0
fi

# Set paths based on MODE
if [ "$MODE" = "baseline" ]; then
    TEST_JSONL=dataset/preprocessed_rag_62k/test_rag_unweighted.jsonl
    ADAPTER_DIR=runs_textgen_rag/${MODE}/${LLM}/baseline_adapter_${EPOCHS}
    
    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen_rag/${MODE}/${LLM}/baseline_test_metrics_${EPOCHS}_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen_rag/${MODE}/${LLM}/baseline_test_shards_${EPOCHS}_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen_rag/${MODE}/${LLM}/baseline_test_metrics_${EPOCHS}.json
        TMP_DIR=runs_textgen_rag/${MODE}/${LLM}/baseline_test_shards_${EPOCHS}
    fi
    
elif [ "$MODE" = "rag_unweighted" ]; then
    TEST_JSONL=dataset/preprocessed_rag_62k/test_rag_unweighted.jsonl
    ADAPTER_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_unweighted_adapter_${EPOCHS}

    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen_rag/${MODE}/${LLM}/rag_unweighted_test_metrics_${EPOCHS}_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_unweighted_test_shards_${EPOCHS}_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen_rag/${MODE}/${LLM}/rag_unweighted_test_metrics_${EPOCHS}.json
        TMP_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_unweighted_test_shards_${EPOCHS}
    fi
    
elif [ "$MODE" = "rag_weighted" ]; then
    TEST_JSONL=dataset/preprocessed_rag_62k/test_rag_weighted_alpha0.3.jsonl
    ADAPTER_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_weighted_adapter_${EPOCHS}

    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen_rag/${MODE}/${LLM}/rag_weighted_test_metrics_${EPOCHS}_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_weighted_test_shards_${EPOCHS}_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen_rag/${MODE}/${LLM}/rag_weighted_test_metrics_${EPOCHS}.json
        TMP_DIR=runs_textgen_rag/${MODE}/${LLM}/rag_weighted_test_shards_${EPOCHS}
    fi
    
else
    echo "[ERROR] Unknown MODE: $MODE"
    exit 1
fi

# Generation config
DECODING=greedy
NUM_BEAMS=1
GEN_MAX_NEW=128
GEN_BS=8
N_MAX_TERMS=12
TEMP=1.0
TOPP=0.95
TOPK=50

# ✓ Mapper config (same as baseline)
FAISS_ROWS=50
TAU_COS=0.40
TAU_FINAL=0.60
W_COS=0.6
W_FUZ=0.4

PRINT_SAMPLES=5
USE_BF16=1

### ============================================================================
### 6) DISPLAY CONFIGURATION
### ============================================================================

echo ""
echo "=========================================================================="
echo "TESTING CONFIGURATION"
echo "=========================================================================="
echo "Mode:           ${MODE}"
echo "Test data:      ${TEST_JSONL}"
echo "Base LLM:       ${BASE_LLM}"
echo "LLM identifier: ${LLM}"
echo "Adapter:        ${ADAPTER_DIR}"
echo "Epochs:         ${EPOCHS}"
echo ""
if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
    echo "⚠️  SUBSET MODE:"
    echo "  Samples:      ${SUBSET_SIZE}"
    echo "  Seed:         ${SUBSET_SEED}"
else
    echo "📊 FULL TEST SET"
fi
echo ""
echo "GENERATION:"
echo "  Decoding:     ${DECODING}"
echo "  Num beams:    ${NUM_BEAMS}"
echo "  Max new:      ${GEN_MAX_NEW}"
echo "  Batch size:   ${GEN_BS}"
echo "  Max terms:    ${N_MAX_TERMS}"
echo "  Print samples: ${PRINT_SAMPLES}"
echo ""
echo "SAPBERT MAPPER:"
echo "  Index:        ${ICD_INDEX}"
echo "  Encoder:      ${ENCODER_MODEL}"
echo "  FAISS rows:   ${FAISS_ROWS}"
echo "  Weights:      w_cos=${W_COS}, w_fuz=${W_FUZ}"
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
### 7) RUN TESTING
### ============================================================================

start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: ${SLURMD_NODENAME}"
echo ""

# Check if test data exists
if [ ! -f "${TEST_JSONL}" ]; then
    echo "[ERROR] Test data not found: ${TEST_JSONL}"
    exit 1
fi

# Check if adapter exists
if [ ! -d "${ADAPTER_DIR}" ]; then
    echo "[ERROR] Adapter directory not found: ${ADAPTER_DIR}"
    exit 1
fi

# ✓ Check if ICD index exists
if [ ! -d "${ICD_INDEX}" ]; then
    echo "[ERROR] ICD index directory not found: ${ICD_INDEX}"
    exit 1
fi

# Create output directories
mkdir -p "$(dirname "${OUT_METRICS}")"
mkdir -p "${TMP_DIR}"

echo "[INFO] Launching testing..."
echo ""

# Single GPU testing
export CUDA_VISIBLE_DEVICES=0

# ✓ Build command with SapBERT mapper arguments
CMD="python ${SCRIPT} \
  --test_jsonl \"${TEST_JSONL}\" \
  --base_model \"${BASE_LLM}\" \
  --adapter_dir \"${ADAPTER_DIR}\" \
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
  --out_metrics \"${OUT_METRICS}\""

# Add subset flag only if SUBSET_SIZE > 0
if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
    CMD="${CMD} --subset_size ${SUBSET_SIZE} --subset_seed ${SUBSET_SEED}"
fi

# Add bf16 flag if enabled
if [[ "${USE_BF16}" == "1" ]]; then
    CMD="${CMD} --use_bf16"
fi

echo "[INFO] Command: ${CMD}"
echo ""

# Execute
eval ${CMD}

status=$?
end=$(date +%s)
elapsed=$((end - start))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo ""
echo "=========================================================================="
echo "[TIME] Testing completed"
echo "[TIME] Elapsed: ${elapsed} seconds (${minutes}m ${seconds}s)"
echo "[INFO] Exit code: ${status}"
echo "[INFO] Job finished: $(date)"
echo "=========================================================================="

if [ $status -eq 0 ]; then
    echo ""
    echo "✅ [SUCCESS] Testing completed successfully!"
    echo ""
    echo "📊 Results saved to:"
    echo "  Metrics: ${OUT_METRICS}"
    echo "  Shards:  ${TMP_DIR}"
    echo ""
else
    echo ""
    echo "❌ [ERROR] Testing failed with exit code: ${status}"
    echo ""
    echo "🔍 Check the log file for details:"
    echo "   logs/Textgen-withKG-withRAG/test_textgen_rag_${SLURM_JOB_ID}.out"
    echo ""
fi

exit $status