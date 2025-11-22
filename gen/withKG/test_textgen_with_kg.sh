#!/bin/bash
#SBATCH --job-name=TEST_kg
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/Textgen-withKG/%x_%j.out
#SBATCH --licenses=horse

# -------- Modules & env --------
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found"; exit 1; }

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------- 5A) Common Paths --------
# --- THIS IS THE NEW SCRIPT ---
SCRIPT=gen/withKG/test_textgen_with_kg.py

DATA_PKL=$PROJECT_DIR/dataset/final_data/test_df.pkl
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
INDEX_DIR=$PROJECT_DIR/icd_index_v9

KG_DIR=$PROJECT_DIR/KG/kg_output4
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl
# DX_MAP no longer needed
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
ATC_MAP=$KG_DIR/code2cui_atc.pkl

# -------- 5B) CHOOSE MODEL TO TEST --------
#
# This MUST match the model you just trained
#-----------------------------------------------------------------------

# === Load the "KG + Codes" model ===
# ADAPTER_DIR=$PROJECT_DIR/runs_textgen/adapter_KG_Codes_trial
# MODEL_TAG="KG_Codes_1000_Model"
# STRUCTURED_FORMAT="codes" # Must match the model

# === Load the "KG + Names" model ===
ADAPTER_DIR=$PROJECT_DIR/runs_textgen/adapter_KG_Codes_trial
MODEL_TAG="KG_Names_Model"
STRUCTURED_FORMAT="names" # Must match the model

#-----------------------------------------------------------------------
echo "[INFO] Testing with Adapter: $ADAPTER_DIR"
echo "[INFO] Test run tag: $MODEL_TAG"
echo "[INFO] Structured Format: $STRUCTURED_FORMAT"

# -------- Budgets & Miner Caps --------
TOTAL_INPUT_BUDGET=5120
ASSISTANT_RESERVE=256
NOTES_SOFT_BUDGET=3008
KG_SOFT_BUDGET=1500
K1=30
K2=30
N_MAX_TERMS=12 # <-- CRITICAL: Must match the training value

echo "[INFO] Total Input Budget: $TOTAL_INPUT_BUDGET"
echo "[INFO] Static N_max_terms: $N_MAX_TERMS"

# -------- Decoding --------
DECODING=greedy
GEN_MAX_NEW=128
BATCH_SIZE=1

# -------- Common args (MODIFIED) --------
COMMON_ARGS=(
  --data_pickle "$DATA_PKL"
  --test_only
  --subset_n 10
  --print_samples 5

  --total_input_budget $TOTAL_INPUT_BUDGET
  --assistant_reserve $ASSISTANT_RESERVE
  --notes_soft_budget $NOTES_SOFT_BUDGET
  --N_max_terms $N_MAX_TERMS
  --structured_format "$STRUCTURED_FORMAT"

  --gen_max_new $GEN_MAX_NEW
  --gen_batch_size $BATCH_SIZE
  --decoding $DECODING

  --base_model "$BASE_MODEL"
  --adapter_dir "$ADAPTER_DIR"
  --use_bf16

  --icd_index_dir "$INDEX_DIR"
  --encoder_model cambridgeltl/SapBERT-from-PubMedBERT-fulltext
  --faiss_rows 50
  --tau_cos 0.40
  --tau_final 0.60
  --w_cos 0.6
  --w_fuz 0.4

  --labels_space full

  --kg_pkl "$KG_PKL"
  --icd9_proc_map_pkl "$PROC_MAP"
  --loinc_map_pkl "$LOINC_MAP"
  --atc_map_pkl "$ATC_MAP"

  --kg_k1 $K1
  --kg_k2 $K2
)

mkdir -p runs_textgen/ablations_v2 runs_textgen/test_shards logs/textgenKGprompt

# Helper to call python with tmp + outputs
run_case () {
  local TEST_TAG="$1"; shift
  local EXTRA=("$@")
  
  local FINAL_TAG="${MODEL_TAG}__${TEST_TAG}"
  
  echo ""
  echo "====================================================="
  echo "[TESTING] $FINAL_TAG"
  echo "====================================================="
  
  start=$(date +%s)
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --tmp_dir "runs_textgen/test_shards/${FINAL_TAG}" \
    --out_metrics "runs_textgen/ablations_v2/metrics_${FINAL_TAG}.json" \
    --stats_csv  "runs_textgen/ablations_v2/stats_${FINAL_TAG}.csv" \
    "${EXTRA[@]}"
  end=$(date +%s)
  
  echo "[TIME] Elapsed: $((end-start)) seconds"
}

#-----------------------------------------------------------------------
### 5C) CHOOSE ABLATIONS TO RUN
#
# These ablations are run on the *single model* you chose in 5B.
#-----------------------------------------------------------------------

# echo "[Ablation] RAW (no KG)"
# # Disable KG by giving no KG budget (kg_soft_budget=0).
# run_case "RAW_no_KG" \
#   --kg_soft_budget 0 \
#   --kg_block both

# echo "[Ablation] H1-only"
# run_case "Test_H1_only" \
#   --kg_soft_budget $KG_SOFT_BUDGET \
#   --kg_block h1

# echo "[Ablation] H2-only"
# run_case "Test_H2_only" \
#   --kg_soft_budget $KG_SOFT_BUDGET \
#   --kg_block h2

echo "[Ablation] H1+H2 (full context)"
run_case "Test_H1_H2_Both" \
  --kg_soft_budget $KG_SOFT_BUDGET \
  --kg_block both \
  --kg_h2_ratio 0.7 # Use the same ratio as training

#-----------------------------------------------------------------------

echo "Done. Check runs_textgen/ablations_v2 for metrics & stats."