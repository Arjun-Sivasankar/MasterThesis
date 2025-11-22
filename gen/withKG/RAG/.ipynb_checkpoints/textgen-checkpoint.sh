#!/bin/bash
#SBATCH --job-name=KG-RAG_textgen
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/textgenKGprompt/%x_%j.out

set -euo pipefail

# ========= 0) Modules / venv =========
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"
python -V

# ========= 1) Paths =========
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR"

# If you used a different filename, point to it here:
SCRIPT=gen/withKG/RAG/textgen_RAG.py
echo "[INFO] Running script: $SCRIPT"

# Data & mapper (SapBERT) index
DATA_PKL=$PROJECT_DIR/dataset/final_data/test_df.pkl
MAPPER_INDEX_DIR=$PROJECT_DIR/icd_index_v9

# Base + LoRA (adapter-only)
BASE_MODEL=$PROJECT_DIR/models/Llama-3.1-8B-Instruct
ADAPTER_DIR=$PROJECT_DIR/runs_textgen/adapter_v1

# UMLS KG CSVs
KG_NODES_CSV=$PROJECT_DIR/KG/kg_output4/kg_nodes.csv
KG_EDGES_CSV=$PROJECT_DIR/KG/kg_output4/kg_edges.csv

# Prepped KG-RAG assets dir (already built)
KG_REC_DIR=$PROJECT_DIR/gen/withKG/RAG/kg_recommender
CODE2NAME_PKL=$KG_REC_DIR/code2name.pkl
ICD9_PROFILES_JSON=$KG_REC_DIR/icd9_profiles.json     # not passed to test script, just sanity
KG_FAISS_DIR=$KG_REC_DIR                               # contains faiss.index, codes.pkl, meta.json

# Optional analysis lists
TOP_CODES=$PROJECT_DIR/icd9_analysis_improved/top_50_codes.csv
BOT_CODES=$PROJECT_DIR/icd9_analysis_improved/bottom_50_codes.csv
TOP_PARENTS=""

# ========= 2) Env =========
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using $GPUS GPU(s)"

# ========= 3) Prompt & decoding budgets =========
MAX_LEN=4096          # total input tokens to the tokenizer
KG_HINT_BUDGET=600    # tokens reserved for [KG HINTS]
N_MAX_TERMS=12        # lines to parse after [OUTPUT]

DECODING=greedy       # greedy | beam | sample
NUM_BEAMS=2
GEN_MAX_NEW=128
BATCH_SIZE=8

# ========= 4) KG-RAG retrieval knobs =========
KG_RETR_TOPK=200      # whitelist size for mapper / retrieval
KG_HINT_TOP=80        # how many candidate names to display in [KG HINTS]
KG_NEIGHBOR_HOPS=1    # 0/1/2; 1 is usually enough
USE_WHITELIST_STRICT=0  # 1 => add --whitelist_strict, 0 => omit

# ========= 5) ICD Mapper weighting =========
W_COS=0.6
W_FUZ=0.4
echo "[INFO] ICD Mapper weights - Cosine: $W_COS, Fuzzy: $W_FUZ"

# ========= 6) Run size =========
TEST_ONLY=1           # 1 => --test_only, 0 => no flag (use subject split)
SUBSET_N=0            # 0 => all rows
PRINT_SAMPLES=5

OUT_JSON=$PROJECT_DIR/runs_textgen/test_metrics_kg_rag.json
TMP_DIR=$PROJECT_DIR/runs_textgen/test_shards
mkdir -p "$(dirname "$OUT_JSON")" "$TMP_DIR"

# ========= 7) Sanity checks =========
require() { for f in "$@"; do [[ -e "$f" ]] || { echo "[ERROR] Missing: $f"; exit 2; }; done; }
require "$SCRIPT" "$DATA_PKL" "$BASE_MODEL" "$ADAPTER_DIR"
require "$KG_NODES_CSV" "$KG_EDGES_CSV" "$CODE2NAME_PKL"
require "$KG_FAISS_DIR/faiss.index" "$KG_FAISS_DIR/codes.pkl" "$KG_FAISS_DIR/meta.json"
require "$MAPPER_INDEX_DIR"

[[ -z "$TOP_CODES" || -f "$TOP_CODES" ]] || TOP_CODES=""
[[ -z "$BOT_CODES" || -f "$BOT_CODES" ]] || BOT_CODES=""
[[ -z "$TOP_PARENTS" || -f "$TOP_PARENTS" ]] || TOP_PARENTS=""

# ========= 8) Optional flags =========
BF16_FLAG=--use_bf16   # test script safely enables bf16 only when supported
TEST_ONLY_FLAG=""
[[ $TEST_ONLY -eq 1 ]] && TEST_ONLY_FLAG="--test_only"

WHITELIST_FLAG=""
[[ $USE_WHITELIST_STRICT -eq 1 ]] && WHITELIST_FLAG="--whitelist_strict"

# ========= 9) Launch =========
echo "[INFO] Starting at: $(date)"
set -x
PYTHONUNBUFFERED=1 python "$SCRIPT" \
  --data_pickle "$DATA_PKL" \
  $TEST_ONLY_FLAG \
  --subset_n "$SUBSET_N" \
  --print_samples "$PRINT_SAMPLES" \
  --N_max_terms "$N_MAX_TERMS" \
  --max_len "$MAX_LEN" \
  --gen_max_new "$GEN_MAX_NEW" \
  --gen_batch_size "$BATCH_SIZE" \
  --decoding "$DECODING" \
  --num_beams "$NUM_BEAMS" \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --icd_index_dir "$MAPPER_INDEX_DIR" \
  --encoder_model "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" \
  --tau_cos 0.40 \
  --tau_final 0.60 \
  --w_cos "$W_COS" \
  --w_fuz "$W_FUZ" \
  --faiss_rows 50 \
  --kg_nodes_csv "$KG_NODES_CSV" \
  --kg_edges_csv "$KG_EDGES_CSV" \
  --code2name_pkl "$CODE2NAME_PKL" \
  --kg_index_dir "$KG_FAISS_DIR" \
  --kg_retr_topk "$KG_RETR_TOPK" \
  --kg_hint_top "$KG_HINT_TOP" \
  --kg_neighbor_hops "$KG_NEIGHBOR_HOPS" \
  --kg_hint_budget "$KG_HINT_BUDGET" \
  $WHITELIST_FLAG \
  --top_codes_csv "$TOP_CODES" \
  --bottom_codes_csv "$BOT_CODES" \
  --top_parent_csv "$TOP_PARENTS" \
  --tmp_dir "$TMP_DIR" \
  --out_metrics "$OUT_JSON" \
  $BF16_FLAG
set +x

EXIT_CODE=$?
echo "[INFO] Finished at: $(date)"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "[INFO] Done."
else
  echo "[ERROR] Exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
