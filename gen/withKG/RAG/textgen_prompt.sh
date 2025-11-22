#!/bin/bash
#SBATCH --job-name=KGprompt_textgen_Hpaths
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/textgenKGprompt/%x_%j.out

module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"
python -V

PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

SCRIPT=gen/withKG/RAG/textgen_prompt_hpaths.py
echo "Running script: $SCRIPT"

# Data
DATA_PKL=$PROJECT_DIR/dataset/final_data/test_df.pkl

# Models
BASE_MODEL=$PROJECT_DIR/models/Llama-3.1-8B-Instruct
ADAPTER_DIR=$PROJECT_DIR/runs_textgen/adapter_v1

# ICD Mapper index
INDEX_DIR=$PROJECT_DIR/icd_index_v9

# KG (PKL graph)
KG_PKL=$PROJECT_DIR/KG/kg_output4/medical_knowledge_graph.pkl

# code2cui maps
DX_MAP=$PROJECT_DIR/KG/kg_output4/code2cui_icd9_dx.pkl
PROC_MAP=$PROJECT_DIR/KG/kg_output4/code2cui_icd9_proc.pkl
LOINC_MAP=$PROJECT_DIR/KG/kg_output4/code2cui_loinc.pkl
ATC_MAP=$PROJECT_DIR/KG/kg_output4/code2cui_atc.pkl

# Prompt budgets (diagnostic: no total/KG clamp)
TOTAL_INPUT_BUDGET=0          # 0 => no total clamp
ASSISTANT_RESERVE=256
NOTES_SOFT_BUDGET=3008        # always clamp notes softly
KG_SOFT_BUDGET=1500              # 0 => no KG clamp
KG_H2_RATIO=0.9               # only used if KG_SOFT_BUDGET>0

echo "[INFO] KG soft budget: $KG_SOFT_BUDGET"
echo "[INFO] KG H2 ratio: $KG_H2_RATIO"

# Miner degree caps
K1=30
K2=30
HOP=1

# Decoding
DECODING=greedy
NUM_BEAMS=2
GEN_MAX_NEW=128
BATCH_SIZE=1

# Eval scope
TEST_ONLY=--test_only
SUBSET_N=10
PRINT_SAMPLES=5

echo "[INFO] Subset size: $SUBSET_N"

# Mapper weights
W_COS=0.6
W_FUZ=0.4

OUT_JSON=$PROJECT_DIR/runs_textgen/test_metrics_raw_vs_kg3.json
OUT_ROWS=$PROJECT_DIR/runs_textgen/kg_prompt_rows3.csv
TMP_DIR=$PROJECT_DIR/runs_textgen/test_shards

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --kg_h2_ratio $KG_H2_RATIO \
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
  --kg_k1 $K1 \
  --kg_k2 $K2 \
  --tmp_dir "$TMP_DIR" \
  --out_metrics "$OUT_JSON" \
  --stats_csv "$OUT_ROWS" \
  --w_cos $W_COS \
  --w_fuz $W_FUZ

EXIT_CODE=$?
echo "[INFO] Finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
  echo "[INFO] Done."
else
  echo "[ERROR] Exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
