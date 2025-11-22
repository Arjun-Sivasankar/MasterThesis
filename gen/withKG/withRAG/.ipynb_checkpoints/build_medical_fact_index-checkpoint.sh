#!/bin/bash
# filepath: /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/withKG/withRAG/build_medical_fact_index.sh
#SBATCH --job-name=build_fact_index
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/build_fact_index_dataset_%j.out
#SBATCH --licenses=horse

set -e
set -u

echo "========================================"
echo "BUILDING DATASET-AWARE FACT INDEX"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR"

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

### Paths
SCRIPT=$PROJECT_DIR/gen/withKG/withRAG/build_medical_fact_index_dataset_aware.py
echo "Running Script: $SCRIPT"

# Dataset
TRAIN_DATA=$PROJECT_DIR/dataset/final_data/train_df.pkl
VAL_DATA=$PROJECT_DIR/dataset/final_data/val_df.pkl
TEST_DATA=$PROJECT_DIR/dataset/final_data/test_df.pkl

# Code mappings
KG_DIR=$PROJECT_DIR/KG/kg_output4
ATC_MAP=$KG_DIR/code2cui_atc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl

# KG
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl

### Configuration - TARGET CONSTRAINT
# Set to "true" for STRICT mode (both src & target in dataset)
# Set to "false" for RELAXED mode (only src in dataset)
# REQUIRE_TARGET_IN_DATASET=true  # Change to false for relaxed mode
REQUIRE_TARGET_IN_DATASET=true  # Change to true for strict mode

### Configuration - INDEPENDENT H1/H2 LIMITS
# First run: Set to None to see statistics
# Subsequent runs: Use values from path_statistics.json
MAX_H1_PER_SOURCE=None  # Or use "None" for unlimited
MAX_H2_PER_SOURCE=None  # Or use "None" for unlimited

SAPBERT_MODEL="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BATCH_SIZE=64

# Output
FACT_INDEX_DIR=$PROJECT_DIR/fact_index_dataset_aware
mkdir -p "$FACT_INDEX_DIR"
mkdir -p "$PROJECT_DIR/logs/Textgen-withKG-withRAG"
if [ "$REQUIRE_TARGET_IN_DATASET" = true ]; then
    FACT_INDEX_DIR="$FACT_INDEX_DIR/strict_mode"
else
    FACT_INDEX_DIR="$FACT_INDEX_DIR/relaxed_mode"
fi

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "Target constraint: $REQUIRE_TARGET_IN_DATASET"
if [ "$REQUIRE_TARGET_IN_DATASET" = true ]; then
    echo "  Mode: STRICT (both source & target in dataset)"
else
    echo "  Mode: RELAXED (only source in dataset)"
fi
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Test data: $TEST_DATA"
echo "KG: $KG_PKL"
echo "Output: $FACT_INDEX_DIR"
echo "Max H1 paths per source: $MAX_H1_PER_SOURCE"
echo "Max H2 paths per source: $MAX_H2_PER_SOURCE"
echo "========================================"

# Build command with optional limits
CMD="python $SCRIPT \
  --train_data $TRAIN_DATA \
  --val_data $VAL_DATA \
  --test_data $TEST_DATA \
  --atc_map $ATC_MAP \
  --loinc_map $LOINC_MAP \
  --proc_map $PROC_MAP \
  --kg_pkl $KG_PKL \
  --output_dir $FACT_INDEX_DIR \
  --sapbert_model $SAPBERT_MODEL \
  --batch_size $BATCH_SIZE \
  --use_gpu_faiss"

# Add target constraint flag
if [ "$REQUIRE_TARGET_IN_DATASET" = true ]; then
    CMD="$CMD --require_target_in_dataset"
fi

# Add limits if not "None"
if [ "$MAX_H1_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h1_per_source $MAX_H1_PER_SOURCE"
fi

if [ "$MAX_H2_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h2_per_source $MAX_H2_PER_SOURCE"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

status=$?

echo "========================================"
echo "COMPLETE (Exit code: $status)"
echo "End time: $(date)"
echo ""
if [ $status -eq 0 ]; then
    echo "✓ SUCCESS!"
    echo ""
    echo "Next steps:"
    echo "1. Check path statistics: $FACT_INDEX_DIR/path_statistics.json"
    echo "2. If needed, adjust MAX_H1_PER_SOURCE and MAX_H2_PER_SOURCE"
    echo "3. Re-run with optimized limits"
fi
echo "========================================"

exit $status