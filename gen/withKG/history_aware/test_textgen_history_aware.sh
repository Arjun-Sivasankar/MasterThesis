#!/bin/bash
#SBATCH --job-name=test_historyaware_textgen
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/History_Aware/test_textgen_historyaware_%j.out

module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_DISABLE_PROGRESS_BAR=1
export TRANSFORMERS_VERBOSITY=error

SCRIPT=gen/withKG/history_aware/test_textgen_history_aware.py

MODE="baseline"  # Options: baseline, h1_unweighted, h1_weighted, h2_unweighted, h2_weighted

# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct
BASE_LLM=models/Meditron3-8B

EPOCHS=10
PROMPT_DIR=dataset/history_aware_data/prompts
ADAPTER_BASE=runs_historyaware

TEST_JSONL=dataset/history_aware_data/jsonl_output_final/test_modular.jsonl

# Subset testing
SUBSET_SIZE=0       # Set >0 for subset, 0 for full test set
SUBSET_SEED=42

case $MODE in
  baseline)
    TEST_TSV=${PROMPT_DIR}/test_prompts.tsv
    ADAPTER_DIR=${ADAPTER_BASE}/baseline/${BASE_LLM}/adapter_${EPOCHS}
    ;;
  h1_unweighted)
    TEST_TSV=${PROMPT_DIR}/test_prompts_h1_unweighted.tsv
    ADAPTER_DIR=${ADAPTER_BASE}/h1_unweighted/${BASE_LLM}/adapter_${EPOCHS}
    ;;
  h1_weighted)
    TEST_TSV=${PROMPT_DIR}/test_prompts_h1_weighted.tsv
    ADAPTER_DIR=${ADAPTER_BASE}/h1_weighted/${BASE_LLM}/adapter_${EPOCHS}
    ;;
  h2_unweighted)
    TEST_TSV=${PROMPT_DIR}/test_prompts_h2_unweighted.tsv
    ADAPTER_DIR=${ADAPTER_BASE}/h2_unweighted/${BASE_LLM}/adapter_${EPOCHS}
    ;;
  h2_weighted)
    TEST_TSV=${PROMPT_DIR}/test_prompts_h2_weighted.tsv
    ADAPTER_DIR=${ADAPTER_BASE}/h2_weighted/${BASE_LLM}/adapter_${EPOCHS}
    ;;
  *)
    echo "[ERROR] Unknown MODE: ${MODE}"
    exit 1
    ;;
esac

OUT_JSON=${ADAPTER_DIR}/test_generations.json
OUT_METRICS=${ADAPTER_DIR}/test_metrics.json


GEN_MAX_NEW=128
GEN_BS=8
N_MAX_TERMS=12
FAISS_ROWS=50
TAU_COS=0.40
TAU_FINAL=0.60
W_COS=1.0
W_FUZ=0.0
PRINT_SAMPLES=5

echo "=========================================================================="
echo "TESTING CONFIGURATION"
echo "=========================================================================="
echo "Mode:           $MODE"
echo "Test data:      $TEST_TSV"
echo "Gold codes:     $TEST_JSONL"
echo "Base LLM:       $BASE_LLM"
echo "Adapter:        $ADAPTER_DIR"
echo "Epochs:         $EPOCHS"
echo
if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
  echo " SUBSET TESTING"
  echo "  Size:         $SUBSET_SIZE"
  echo "  Seed:         $SUBSET_SEED"
  echo
else
  echo " FULL TEST SET"
  echo
fi
echo "GENERATION:"
echo "  Max len:      5120"
echo "  Max new:      $GEN_MAX_NEW"
echo "  Batch size:   $GEN_BS"
echo "  Max terms:    $N_MAX_TERMS"
echo "  Print samples: $PRINT_SAMPLES"
echo
echo "SAPBERT MAPPER:"
echo "  Index:        ./gen/pipeline/icd_index_v9"
echo "  Encoder:      cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
echo "  FAISS rows:   $FAISS_ROWS"
echo "  tau_cos:      $TAU_COS"
echo "  tau_final:    $TAU_FINAL"
echo "  w_cos:        $W_COS"
echo "  w_fuz:        $W_FUZ"
echo
echo "OUTPUT:"
echo "  Generations:  $OUT_JSON"
echo "  Metrics:      $OUT_METRICS"
echo
echo "HARDWARE:"
echo "  GPUs:         ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "  Memory:       ${SLURM_MEM_PER_NODE}M"
echo "=========================================================================="
echo ""

start_time=$(date +%s)

CMD="python ${SCRIPT} \
  --test_tsv \"${TEST_TSV}\" \
  --test_jsonl \"${TEST_JSONL}\" \
  --base_model \"${BASE_LLM}\" \
  --adapter_dir \"${ADAPTER_DIR}\" \
  --max_len 5120 \
  --gen_max_new $GEN_MAX_NEW \
  --gen_batch_size $GEN_BS \
  --icd_index_dir ./gen/pipeline/icd_index_v9 \
  --encoder_model cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --faiss_rows $FAISS_ROWS \
  --tau_cos $TAU_COS \
  --tau_final $TAU_FINAL \
  --w_cos $W_COS \
  --w_fuz $W_FUZ \
  --N_max_terms $N_MAX_TERMS \
  --print_samples $PRINT_SAMPLES \
  --out_json \"${OUT_JSON}\" \
  --out_metrics \"${OUT_METRICS}\" \
  --top_codes_csv \"/analysis_results_HA/top_50_codes.csv\" \
  --bottom_codes_csv \"/analysis_results_HA/bottom_50_codes.csv\" \
  --top_parent_csv \"/analysis_results_HA/top_50_category_levels.csv\""

if [ "$SUBSET_SIZE" -gt 0 ] 2>/dev/null; then
    CMD="${CMD} --subset_size ${SUBSET_SIZE} --subset_seed ${SUBSET_SEED}"
fi

echo "[INFO] Command: ${CMD}"
echo ""

eval ${CMD}

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "[INFO] Testing completed in $elapsed seconds."
status=$?
echo "[INFO] Testing finished with exit code: $status"
exit $status