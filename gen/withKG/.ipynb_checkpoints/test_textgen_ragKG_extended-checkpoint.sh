#!/bin/bash
#SBATCH --job-name=test_rag_textgen
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/test_textgen_rag_new_%j.out

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

SCRIPT=gen/withKG/test_textgen_ragKG_extended.py
echo "[INFO] Testing Script: ${SCRIPT}"

### ============================================================================
### CONFIGURATION
### ============================================================================

# MODE: "baseline", "rag_unweighted", "rag_weighted", or "base_model_only"
# MODE="baseline"
# MODE="rag_unweighted"
MODE="rag_weighted"
# MODE="base_model_only"

# Path config for RAG modes (only used for rag_unweighted and rag_weighted)
H1=false
H2=true
COMBINED=false

# Model selection
# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct
BASE_LLM=models/Meditron3-8B

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

# SapBERT Mapper paths
ICD_INDEX=./gen/TextGen/icd_index_v9
ENCODER_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext

# Subset testing
SUBSET_SIZE=0        # Set >0 for subset, 0 for full test set
SUBSET_SEED=42

# ------------------------------------------------------------------------------
# TOGGLE: Strip KG facts from prompt for baseline testing?
# If set to 1, will remove [KNOWLEDGE GRAPH FACTS] and everything after from prompt.
# If set to 0, will keep the full prompt (including KG facts) for baseline testing.
# NOTE: For base_model_only mode, KG facts are ALWAYS stripped
# ------------------------------------------------------------------------------
STRIP_KG_FOR_BASELINE=0

# Bucket CSV paths (updated to use thesis_plots directory)
TOP_CODES_CSV=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/thesis_plots/top_50_codes.csv
BOTTOM_CODES_CSV=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/thesis_plots/bottom_50_codes.csv
TOP_PARENTS_CSV=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/thesis_plots/top_50_category_levels.csv

# Set path_config for RAG modes
if [ "$MODE" = "rag_unweighted" ] || [ "$MODE" = "rag_weighted" ]; then
    if [ "$H1" = true ] && [ "$H2" = false ] && [ "$COMBINED" = false ]; then
        path_config="h1"
    elif [ "$H1" = false ] && [ "$H2" = true ] && [ "$COMBINED" = false ]; then
        path_config="h2"
    elif [ "$H1" = false ] && [ "$H2" = false ] && [ "$COMBINED" = true ]; then
        path_config="combined"
    else
        echo "[ERROR] Invalid path configuration for RAG mode. Set H1, H2, or COMBINED."
        exit 1
    fi
fi

# Set paths based on MODE (matches training script output structure)
if [ "$MODE" = "base_model_only" ]; then
    # Base model ablation - no adapter, always strip KG
    TEST_JSONL=dataset/preprocessed_rag_full/map_desc/h1/unweighted/test_rag_h1_unweighted.jsonl
    ADAPTER_DIR=""  # No adapter for base model only
    
    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen_rag/base_model_only/${LLM}/base_model_test_metrics_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen_rag/base_model_only/${LLM}/base_model_test_shards_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen_rag/base_model_only/${LLM}/base_model_test_metrics.json
        TMP_DIR=runs_textgen_rag/base_model_only/${LLM}/base_model_test_shards
    fi

elif [ "$MODE" = "baseline" ]; then
    TEST_JSONL=dataset/preprocessed_rag_full/map_desc/h1/unweighted/test_rag_h1_unweighted.jsonl
    ADAPTER_DIR=runs_textgen_rag/baseline_new/${LLM}/baseline_adapter_${EPOCHS}

    # Suffix for output files based on STRIP_KG_FOR_BASELINE toggle
    if [ "$STRIP_KG_FOR_BASELINE" -eq 1 ]; then
        KG_SUFFIX="stripkg"
    else
        KG_SUFFIX="withkg"
    fi

    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen_rag/baseline_new/${LLM}/baseline_test_metrics_${EPOCHS}_${KG_SUFFIX}_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen_rag/baseline_new/${LLM}/baseline_test_shards_${EPOCHS}_${KG_SUFFIX}_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen_rag/baseline_new/${LLM}/baseline_test_metrics_${EPOCHS}_${KG_SUFFIX}.json
        TMP_DIR=runs_textgen_rag/baseline_new/${LLM}/baseline_test_shards_${EPOCHS}_${KG_SUFFIX}
    fi

elif [ "$MODE" = "rag_unweighted" ]; then
    TEST_JSONL=dataset/preprocessed_rag_full/map_desc/${path_config}/unweighted/test_rag_${path_config}_unweighted.jsonl
    ADAPTER_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_unweighted_adapter_${EPOCHS}
    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen_rag/${path_config}/${LLM}/rag_unweighted_test_metrics_${EPOCHS}_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_unweighted_test_shards_${EPOCHS}_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen_rag/${path_config}/${LLM}/rag_unweighted_test_metrics_${EPOCHS}.json
        TMP_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_unweighted_test_shards_${EPOCHS}
    fi

elif [ "$MODE" = "rag_weighted" ]; then
    TEST_JSONL=dataset/preprocessed_rag_full/map_desc/${path_config}/weighted_alpha0.3/test_rag_${path_config}_weighted_alpha0.3.jsonl
    ADAPTER_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_weighted_adapter_${EPOCHS}
    if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
        OUT_METRICS=runs_textgen_rag/${path_config}/${LLM}/rag_weighted_test_metrics_${EPOCHS}_subset${SUBSET_SIZE}.json
        TMP_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_weighted_test_shards_${EPOCHS}_subset${SUBSET_SIZE}
    else
        OUT_METRICS=runs_textgen_rag/${path_config}/${LLM}/rag_weighted_test_metrics_${EPOCHS}.json
        TMP_DIR=runs_textgen_rag/${path_config}/${LLM}/rag_weighted_test_shards_${EPOCHS}
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

# Mapper config
FAISS_ROWS=50
TAU_COS=0.40
TAU_FINAL=0.60
W_COS=0.6
W_FUZ=0.4

PRINT_SAMPLES=5
USE_BF16=1

### ============================================================================
### DISPLAY CONFIGURATION
### ============================================================================

echo ""
echo "=========================================================================="
echo "TESTING CONFIGURATION"
echo "=========================================================================="
echo "Mode:           ${MODE}"
if [ "$MODE" = "baseline" ]; then
    echo "Stripped KG:  ${STRIP_KG_FOR_BASELINE}"
elif [ "$MODE" = "base_model_only" ]; then
    echo "🔍 ABLATION:   Base model only (no adapter, KG stripped)"
fi
echo "Test data:      ${TEST_JSONL}"
echo "Base LLM:       ${BASE_LLM}"
echo "LLM identifier: ${LLM}"
if [ "$MODE" != "base_model_only" ]; then
    echo "Adapter:        ${ADAPTER_DIR}"
    echo "Epochs:         ${EPOCHS}"
fi
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
if [ "$MODE" = "baseline" ]; then
    if [ "$STRIP_KG_FOR_BASELINE" -eq 1 ]; then
        echo "  [BASELINE] KG facts will be STRIPPED from prompt for true baseline evaluation."
    else
        echo "  [BASELINE] KG facts will be INCLUDED in prompt for ablation."
    fi
elif [ "$MODE" = "base_model_only" ]; then
    echo "  [BASE_MODEL_ONLY] Using untrained base model (no adapter) for ablation."
    echo "  [BASE_MODEL_ONLY] KG facts are ALWAYS STRIPPED for fair comparison."
fi
echo "=========================================================================="
echo ""

### ============================================================================
### RUN TESTING
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

# Check if adapter exists (skip for base_model_only mode)
if [ "$MODE" != "base_model_only" ]; then
    if [ ! -d "${ADAPTER_DIR}" ]; then
        echo "[ERROR] Adapter directory not found: ${ADAPTER_DIR}"
        exit 1
    fi
fi

# Check if ICD index exists
if [ ! -d "${ICD_INDEX}" ]; then
    echo "[ERROR] ICD index directory not found: ${ICD_INDEX}"
    exit 1
fi

# Create output directories
mkdir -p "$(dirname "${OUT_METRICS}")"
mkdir -p "${TMP_DIR}"

echo "[INFO] Launching testing..."
echo ""

export CUDA_VISIBLE_DEVICES=0
export MODE=${MODE}
export STRIP_KG_FOR_BASELINE=${STRIP_KG_FOR_BASELINE}

CMD="python ${SCRIPT} \
  --test_jsonl \"${TEST_JSONL}\" \
  --base_model \"${BASE_LLM}\""

# Add adapter dir only if not in base_model_only mode
if [ "$MODE" != "base_model_only" ]; then
    CMD="${CMD} --adapter_dir \"${ADAPTER_DIR}\""
else
    CMD="${CMD} --base_model_only"
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

if [[ "${USE_BF16}" == "1" ]]; then
    CMD="${CMD} --use_bf16"
fi

echo "[INFO] Command: ${CMD}"
echo ""

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
    if [ "$MODE" != "base_model_only" ]; then
        echo "  Buckets: $(dirname ${OUT_METRICS})/test_metrics_buckets.json"
    fi
    echo "  Per-label tables:"
    echo "    - $(dirname ${OUT_METRICS})/per_label_FULL.csv"
    echo "    - $(dirname ${OUT_METRICS})/per_label_TOP_50_CODES.csv"
    echo "    - $(dirname ${OUT_METRICS})/per_label_BOTTOM_50_CODES.csv"
    echo "    - $(dirname ${OUT_METRICS})/per_label_TOP_50_PARENTS.csv"
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