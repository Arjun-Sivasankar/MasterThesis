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

SCRIPT=gen/withKG/textgen_withKG_prompt2.py
echo "Running script: $SCRIPT"

DATA_PKL=$PROJECT_DIR/dataset/icd9/test_df.pkl     # or your test set if you prefer
INDEX_DIR=$PROJECT_DIR/icd_index_v9                # ICDMapper FAISS/index dir

# Base + LoRA (adapter-only)
BASE_MODEL=$PROJECT_DIR/models/Llama-3.1-8B-Instruct
ADAPTER_DIR=$PROJECT_DIR/runs_textgen/adapter_v1

# KG & maps (ATC fetched from 'ndc' col)
KG_DIR=$PROJECT_DIR/KG/kg_output2
KG_PKL=$KG_DIR/medical_knowledge_graph.pkl
DX_MAP=$KG_DIR/code2cui_icd9_dx.pkl
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
ATC_MAP=$KG_DIR/code2cui_atc.pkl

# Optional code lists
TOP_CODES=$PROJECT_DIR/icd9_analysis_improved/top_50_codes.csv
BOT_CODES=$PROJECT_DIR/icd9_analysis_improved/bottom_50_codes.csv
TOP_PARENTS=""    # leave blank for now

# 2) Env
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using $GPUS GPU(s)"

# 3) Budgets (from your p95 summaries; give KG more room)
TOTAL_INPUT_BUDGET=4096     # you can try 4096 to give notes/KG more headroom
ASSISTANT_RESERVE=256
NOTES_SOFT_BUDGET=3008
KG_SOFT_BUDGET=832

echo "[INFO] Using budgets - Total: $TOTAL_INPUT_BUDGET, Assistant reserve: $ASSISTANT_RESERVE, Notes soft: $NOTES_SOFT_BUDGET, KG soft: $KG_SOFT_BUDGET"

# 4) Decoding
DECODING=greedy
NUM_BEAMS=2
GEN_MAX_NEW=128
BATCH_SIZE=8

# 5) KG expansion
HOP=1
REL_WHITELIST=""
RELA_WHITELIST=""
MAX_NEIGHBORS_SHOW=24

# 6) Run size
TEST_ONLY=--test_only        # or blank to do subject split
SUBSET_N=0                   # set 0 to run all
PRINT_SAMPLES=5

# 7) ICD Mapper settings
w_cos=1.0
w_fuz=0.0

echo "[INFO] Using ICD Mapper weights - Cosine: $w_cos, Fuzzy: $w_fuz"

OUT_JSON=$PROJECT_DIR/runs_textgen/test_metrics_raw_vs_kg.json
TMP_DIR=$PROJECT_DIR/runs_textgen/test_shards

echo "[INFO] Starting at: $(date)"
PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --data_pickle "$DATA_PKL" \
  $TEST_ONLY \
  --subset_n $SUBSET_N \
  --print_samples $PRINT_SAMPLES \
  --N_max_terms 0 \
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
  --rel_whitelist "$REL_WHITELIST" \
  --rela_whitelist "$RELA_WHITELIST" \
  --max_neighbors_show $MAX_NEIGHBORS_SHOW \
  --top_codes_csv "$TOP_CODES" \
  --bottom_codes_csv "$BOT_CODES" \
  --top_parent_csv "$TOP_PARENTS" \
  --tmp_dir "$TMP_DIR" \
  --out_metrics "$OUT_JSON" \
  --w_cos $w_cos \
  --w_fuz $w_fuz

EXIT_CODE=$?
echo "[INFO] Finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
  echo "[INFO] Done."
else
  echo "[ERROR] Exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
