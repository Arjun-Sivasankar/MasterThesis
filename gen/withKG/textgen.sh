#!/bin/bash
#SBATCH --job-name=KGprompt_textgen
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/textgenKGprompt/%x_%j.out

# 0) Modules/venv
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"
python -V

# 1) Paths
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs/textgenKGprompt runs_textgen/test_shards

SCRIPT=gen/withKG/textgen_withKG_prompt.py

# Data (use your test split or full set with --test_only/--subset_n below)
DATA_PKL=$PROJECT_DIR/dataset/merged_icd9.pkl

# ICDMapper FAISS/index dir
INDEX_DIR=$PROJECT_DIR/icd_index_v9

# Base + LoRA (adapter-only)
BASE_MODEL=$PROJECT_DIR/models/Llama-3.1-8B-Instruct
# BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
ADAPTER_DIR=$PROJECT_DIR/runs_textgen/adapter_v1

# KG & maps (ATC fetched from 'ndc' column)
KG_DIR=$PROJECT_DIR/KG/kg_output2
KG_PKL=$KG_DIR/medical_knowledge_graph.pkl
DX_MAP=$KG_DIR/code2cui_icd9_dx.pkl
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
ATC_MAP=$KG_DIR/code2cui_atc.pkl

# Optional code lists
TOP_CODES=$PROJECT_DIR/icd9_analysis_improved/top_50_codes.csv
BOT_CODES=$PROJECT_DIR/icd9_analysis_improved/bottom_50_codes.csv
TOP_PARENTS=""    # leave blank if unused

# 2) Env
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$PROJECT_DIR/.hf_cache

GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using $GPUS GPU(s)"

# 3) Budgets (from your token-budget helper)
TOTAL_INPUT_BUDGET=3072
ASSISTANT_RESERVE=128
NOTES_SOFT_BUDGET=2307   # ~p95 split for notes
KG_SOFT_BUDGET=637       # ~p95 split for KG block

# 4) Decoding
DECODING=greedy
NUM_BEAMS=2
GEN_MAX_NEW=128
BATCH_SIZE=8

# 5) KG expansion
HOP=1
MAX_NEIGHBORS_SHOW=24
REL_WHITELIST=""
RELA_WHITELIST=""

# 6) Quick test controls
SUBSET_N=0          # set 0 to run all rows
PRINT_SAMPLES=5

OUT_JSON=$PROJECT_DIR/runs_textgen/test_metrics_raw_vs_kg.json
TMP_DIR=$PROJECT_DIR/runs_textgen/test_shards

echo "[INFO] Starting at: $(date)"
PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --data_pickle "$DATA_PKL" \
  --test_only \
  --subset_n $SUBSET_N \
  --print_samples $PRINT_SAMPLES \
  --N_max_terms 12 \
  --total_input_budget $TOTAL_INPUT_BUDGET \
  --assistant_reserve $ASSISTANT_RESERVE \
  --notes_soft_budget $NOTES_SOFT_BUDGET \
  --kg_soft_budget $KG_SOFT_BUDGET \
  --gen_max_new $GEN_MAX_NEW \
  --gen_batch_size $BATCH_SIZE \
  --decoding $DECODING \
  --num_beams $NUM_BEAMS \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --icd_index_dir "$INDEX_DIR" \
  --kg_pkl "$KG_PKL" \
  --icd9_dx_map_pkl "$DX_MAP" \
  --icd9_proc_map_pkl "$PROC_MAP" \
  --loinc_map_pkl "$LOINC_MAP" \
  --atc_map_pkl "$ATC_MAP" \
  --hop $HOP \
  --max_neighbors_show $MAX_NEIGHBORS_SHOW \
  --rel_whitelist "$REL_WHITELIST" \
  --rela_whitelist "$RELA_WHITELIST" \
  --top_codes_csv "$TOP_CODES" \
  --bottom_codes_csv "$BOT_CODES" \
  --top_parent_csv "$TOP_PARENTS" \
  --tmp_dir "$TMP_DIR" \
  --out_metrics "$OUT_JSON"

EXIT_CODE=$?
echo "[INFO] Finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
  echo "[INFO] Done."
else
  echo "[ERROR] Exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
