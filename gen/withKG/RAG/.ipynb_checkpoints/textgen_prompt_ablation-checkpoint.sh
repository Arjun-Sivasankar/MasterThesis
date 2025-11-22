#!/bin/bash
#SBATCH --job-name=KG_ablation
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/textgenKGprompt/%x_%j.out

# -------- Modules & env --------
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------- Paths (edit if needed) --------
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR"

SCRIPT=gen/withKG/RAG/textgen_prompt_hpaths_ablations.py

DATA_PKL=$PROJECT_DIR/dataset/final_data/test_df.pkl
BASE_MODEL=$PROJECT_DIR/models/Llama-3.1-8B-Instruct
ADAPTER_DIR=$PROJECT_DIR/runs_textgen/adapter_v1
INDEX_DIR=$PROJECT_DIR/icd_index_v9

KG_DIR=$PROJECT_DIR/KG/kg_output4
# KG_PKL=$KG_DIR/medical_knowledge_graph.pkl
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl
DX_MAP=$KG_DIR/code2cui_icd9_dx.pkl
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
ATC_MAP=$KG_DIR/code2cui_atc.pkl

# -------- Budgets & miner caps --------
TOTAL_INPUT_BUDGET=0          # 0 => no global clamp
ASSISTANT_RESERVE=256
NOTES_SOFT_BUDGET=3008
# KG_SOFT_BUDGET=1500            # set 0 to disable KG text (used for RAW)
KG_SOFT_BUDGET=1000
K1=30
K2=30

echo "[INFO] KG soft budget: $KG_SOFT_BUDGET"

# -------- Decoding --------
DECODING=greedy
GEN_MAX_NEW=128
BATCH_SIZE=1

# -------- Common args --------
COMMON_ARGS=(
  --data_pickle "$DATA_PKL"
  --test_only
  --subset_n 10
  --print_samples 5

  --total_input_budget $TOTAL_INPUT_BUDGET
  --assistant_reserve $ASSISTANT_RESERVE
  --notes_soft_budget $NOTES_SOFT_BUDGET

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
  --icd9_dx_map_pkl "$DX_MAP"
  --icd9_proc_map_pkl "$PROC_MAP"
  --loinc_map_pkl "$LOINC_MAP"
  --atc_map_pkl "$ATC_MAP"

  --hop 1
  --kg_k1 $K1
  --kg_k2 $K2
)

mkdir -p runs_textgen/ablations runs_textgen/test_shards logs/textgenKGprompt

# Helper to call python with tmp + outputs
run_case () {
  local TAG="$1"; shift
  local EXTRA=("$@")
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --tmp_dir "runs_textgen/test_shards/${TAG}" \
    --out_metrics "runs_textgen/ablations/metrics_${TAG}.json" \
    --stats_csv  "runs_textgen/ablations/stats_${TAG}.csv" \
    "${EXTRA[@]}"
}

# echo "[Ablation] RAW (no KG)"
# # Disable KG by giving no KG budget (kg_soft_budget=0). kg_block value is irrelevant here.
# run_case "RAW" \
#   --kg_soft_budget 0 \
#   --kg_block both \
#   --kg_h2_ratio 1.0

echo "[Ablation] H1-only"
start=$(date +%s)
run_case "H1_only" \
  --kg_soft_budget $KG_SOFT_BUDGET \
  --kg_block h1 \
  --kg_h2_ratio 0.0
end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) seconds"

echo "[Ablation] H2-only"
start2=$(date +%s)
run_case "H2_only" \
  --kg_soft_budget $KG_SOFT_BUDGET \
  --kg_block h2 \
  --kg_h2_ratio 1.0
end2=$(date +%s)
echo "[TIME] Elapsed: $((end2-start2)) seconds"

# echo "[Ablation] H1+H2 (ratio=0.7 to H2)"
# start3=$(date +%s)
# run_case "H1&H2_ratio-0.7" \
#   --kg_soft_budget $KG_SOFT_BUDGET \
#   --kg_block both \
#   --kg_h2_ratio 0.7
# end3=$(date +%s)
# echo "[TIME] Elapsed: $((end3-start3)) seconds"

echo "Done. Check runs_textgen/ablations for metrics & stats."
