#!/bin/bash
#SBATCH --job-name=preprocess_KG
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/Textgen-withKG/preprocess_KG_%j.out
#SBATCH --licenses=horse

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================"
echo "PREPROCESSING JOB STARTED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

### 1) Load modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Set project directory
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "ERROR: Project dir not found"; exit 1; }
echo "Working directory: $(pwd)"

### 3) Activate virtual environment
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "Virtual env: $VIRTUAL_ENV"

### 4) Environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

### 5) Define paths
SCRIPT=$PROJECT_DIR/gen/withKG/preprocess_data_with_KG2.py

# Input data
TRAIN_DATA_PKL=$PROJECT_DIR/dataset/final_data/train_df.pkl
VAL_DATA_PKL=$PROJECT_DIR/dataset/final_data/val_df.pkl

# ICD index
ICD_INDEX=$PROJECT_DIR/gen/TextGen/icd_index_v9

# Tokenizer (use base model for tokenizer)
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct

# KG files
KG_DIR=$PROJECT_DIR/KG/kg_output4
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
ATC_MAP=$KG_DIR/code2cui_atc.pkl

# Output directory
OUT_DIR=$PROJECT_DIR/preprocessed_data2
mkdir -p "$OUT_DIR"
mkdir -p "$PROJECT_DIR/logs/Textgen-withKG"

### 6) Configuration
VAL_N=100          # Process first 100 validation samples
MAX_LEN=5120
NOTES_SOFT_BUDGET=3008
N_MAX_TERMS=12

#-----------------------------------------------------------------------
### 7) ABLATION SETTINGS - Choose ONE option below
#-----------------------------------------------------------------------

# === OPTION 1: Method 1 with Names ===
# echo "[CONFIG] Method 1: KG + Names"
# KG_SOFT_BUDGET=1500
# KG_BLOCK="both"
# KG_H2_RATIO=0.7
# STRUCTURED_FORMAT="names"
# OUT_TRAIN="$OUT_DIR/train_KG_Names.jsonl"
# OUT_VAL="$OUT_DIR/val_KG_Names_100.jsonl"

# === OPTION 2: Method 1 with Codes ===
echo "[CONFIG] Method 1: KG + Codes"
KG_SOFT_BUDGET=1500
KG_BLOCK="both"
KG_H2_RATIO=0.7
STRUCTURED_FORMAT="codes"
OUT_TRAIN="$OUT_DIR/train_KG_Codes.jsonl"
OUT_VAL="$OUT_DIR/val_KG_Codes_100.jsonl"

# === OPTION 3: Baseline (No KG) ===
# echo "[CONFIG] Baseline: No KG"
# KG_SOFT_BUDGET=0
# KG_BLOCK="both"
# KG_H2_RATIO=0.0
# STRUCTURED_FORMAT="names"
# OUT_TRAIN="$OUT_DIR/train_Baseline.jsonl"
# OUT_VAL="$OUT_DIR/val_Baseline_100.jsonl"

# === OPTION 4: H1 only ===
# echo "[CONFIG] Method 1: H1 only"
# KG_SOFT_BUDGET=1500
# KG_BLOCK="h1"
# KG_H2_RATIO=0.0
# STRUCTURED_FORMAT="names"
# OUT_TRAIN="$OUT_DIR/train_KG_H1.jsonl"
# OUT_VAL="$OUT_DIR/val_KG_H1_100.jsonl"

# === OPTION 5: H2 only ===
# echo "[CONFIG] Method 1: H2 only"
# KG_SOFT_BUDGET=1500
# KG_BLOCK="h2"
# KG_H2_RATIO=1.0
# STRUCTURED_FORMAT="names"
# OUT_TRAIN="$OUT_DIR/train_KG_H2.jsonl"
# OUT_VAL="$OUT_DIR/val_KG_H2_100.jsonl"

#-----------------------------------------------------------------------

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "Tokenizer: $BASE_LLM"
echo "Max length: $MAX_LEN"
echo "N_max_terms: $N_MAX_TERMS"
echo "Notes budget: $NOTES_SOFT_BUDGET"
echo "KG budget: $KG_SOFT_BUDGET"
echo "KG block: $KG_BLOCK"
echo "KG H2 ratio: $KG_H2_RATIO"
echo "Structured format: $STRUCTURED_FORMAT"
echo "Val samples: $VAL_N"
echo "Output train: $OUT_TRAIN"
echo "Output val: $OUT_VAL"
echo "========================================"

# Verify input files exist
echo "Verifying input files..."
for file in "$TRAIN_DATA_PKL" "$VAL_DATA_PKL" "$KG_PKL" "$PROC_MAP" "$LOINC_MAP" "$ATC_MAP"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi
done
echo "✓ All input files exist"

### 8) Run preprocessing
echo "========================================"
echo "STARTING PREPROCESSING"
echo "========================================"

start=$(date +%s)

python "$SCRIPT" \
  --train_data "$TRAIN_DATA_PKL" \
  --val_data "$VAL_DATA_PKL" \
  --val_n $VAL_N \
  --llm_tokenizer "$BASE_LLM" \
  --target_mode icd_titles \
  --icd_index_dir "$ICD_INDEX" \
  --label_col "icd_code" \
  --N_max_terms $N_MAX_TERMS \
  --min_assistant_tokens 128 \
  --structured_format "$STRUCTURED_FORMAT" \
  --out_train_file "$OUT_TRAIN" \
  --out_val_file "$OUT_VAL" \
  --max_len $MAX_LEN \
  --notes_soft_budget $NOTES_SOFT_BUDGET \
  --kg_soft_budget $KG_SOFT_BUDGET \
  --kg_block "$KG_BLOCK" \
  --kg_h2_ratio $KG_H2_RATIO \
  --kg_k1 30 \
  --kg_k2 30 \
  --kg_pkl "$KG_PKL" \
  --icd9_proc_map_pkl "$PROC_MAP" \
  --loinc_map_pkl "$LOINC_MAP" \
  --atc_map_pkl "$ATC_MAP"

status=$?
end=$(date +%s)
elapsed=$((end - start))

echo "========================================"
echo "PREPROCESSING COMPLETE"
echo "========================================"
echo "Exit code: $status"
echo "Elapsed time: $elapsed seconds ($(($elapsed / 60)) minutes)"
echo "End time: $(date)"

if [ $status -eq 0 ]; then
    echo "✓ SUCCESS"
    echo "Output files:"
    echo "  Train: $OUT_TRAIN"
    echo "  Val:   $OUT_VAL"
    
    # Show file sizes
    if [ -f "$OUT_TRAIN" ]; then
        echo "  Train size: $(du -h "$OUT_TRAIN" | cut -f1)"
        echo "  Train lines: $(wc -l < "$OUT_TRAIN")"
    fi
    if [ -f "$OUT_VAL" ]; then
        echo "  Val size: $(du -h "$OUT_VAL" | cut -f1)"
        echo "  Val lines: $(wc -l < "$OUT_VAL")"
    fi
else
    echo "✗ FAILED"
fi

echo "========================================"

exit $status