#!/bin/bash
#SBATCH --job-name=test_parent_only
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/Parents/test_parent_%j.out

# Parse command line arguments for run directory
RUN_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-dir=*)
      RUN_DIR="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# --- 1) Modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Project dir ---
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

# --- 3) Venv ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"
python -V

# --- 4) Environment ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 5) Paths ---
TEST_SCRIPT="gen/test_parents.py"
ICD9_TEST=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl
LLM=meta-llama/Llama-3.2-1B-Instruct

# --- 6) Determine run directory ---
if [ -z "$RUN_DIR" ]; then
    echo "[INFO] No --run-dir specified, finding latest training run..."
    RUN_DIR=$(find runs_gen/icd9_parent -maxdepth 1 -type d -name "20*" | sort | tail -1)
    if [ -z "$RUN_DIR" ]; then
        echo "[ERROR] No training run found in runs_gen/icd9_parent/"
        echo "Usage: sbatch scripts/run_test_parents.sh --run-dir=runs_gen/icd9_parent/20XXXXXX-XXXXXX_..."
        exit 1
    fi
    echo "[INFO] Using latest run: $RUN_DIR"
else
    echo "[INFO] Using specified run: $RUN_DIR"
fi

# --- 7) Validate required files ---
ADAPTER_DIR="$RUN_DIR/adapter_best"
LABELS_JSON="$RUN_DIR/label_space.json"

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "[ERROR] Adapter directory not found: $ADAPTER_DIR"
    echo "Make sure training completed successfully."
    exit 1
fi

if [ ! -f "$LABELS_JSON" ]; then
    echo "[ERROR] Labels JSON not found: $LABELS_JSON"
    echo "Make sure training completed successfully."
    exit 1
fi

echo "[INFO] Found adapter: $ADAPTER_DIR"
echo "[INFO] Found labels: $LABELS_JSON"

start=$(date +%s)
echo "[INFO] Testing job started at $(date)"

#############################################
# ICD-9 PARENT Testing ONLY
#############################################
echo "[INFO] Starting ICD-9 PARENT testing..."
echo "[INFO] Test script: $TEST_SCRIPT"

python "${TEST_SCRIPT}" \
  --test_pickle "${ICD9_TEST}" \
  --adapter_dir "$ADAPTER_DIR" \
  --labels_json "$LABELS_JSON" \
  --subject_col subject_id_x \
  --label_col icd_code \
  --llama_model "${LLM}" \
  --max_len 3072 \
  --gen_max_new 96 \
  --test_batch_size 16 \
  --use_structured 1 \
  --use_notes 1 \
  --icd_scheme icd9cm \
  --icd_level parent \
  --seed 42 \
  --out_dir "$RUN_DIR/test_results" \
  --test_examples 5

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "[INFO] Testing completed successfully!"
    echo "[INFO] Results saved to: $RUN_DIR/test_results"
    echo "[INFO] View results:"
    echo "cat $RUN_DIR/test_results/test_metrics.json"
else
    echo "[ERROR] Testing failed with exit code: $TEST_EXIT_CODE"
fi

end=$(date +%s)
echo "[INFO] Testing job finished at $(date)"
echo "[TIME] Testing elapsed: $((end-start)) seconds"

exit $TEST_EXIT_CODE