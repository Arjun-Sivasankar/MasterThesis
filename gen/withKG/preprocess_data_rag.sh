#!/bin/bash
# filepath: /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/withKG/withRAG/preprocess_data_rag.sh
#SBATCH --job-name=preprocess_rag
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/diag_only/preprocess_rag_%j.out
#SBATCH --licenses=horse

set -e
set -u

echo "========================================"
echo "RAG PREPROCESSING (H1/H2 Ablation)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

# =============================================================================
# SETUP
# =============================================================================

module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis

cd "$PROJECT_DIR"
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_DIR/logs/Textgen-withKG-withRAG"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Which ablations to run
RUN_H1=true
RUN_H2=false
RUN_COMBINED=false

# Weighting configurations
RUN_UNWEIGHTED=false
RUN_WEIGHTED=true
ALPHA=0.3

# Combined mode ratios
H1_RATIOS="0.3"

# Relationship aggregation for H2/combined
REL_AGGREGATION="sum"  # or "max" or "mean"

# Diagnosis filtering
DIAGNOSIS_ONLY=false  # Set to true to retrieve only diagnosis-targeting facts

# Data settings
SUBSET_SIZE=0       # 0 = full dataset
K=50
# K=100                # Increase from 50 to 100 when DIAGNOSIS_ONLY=true
PROCESS_TEST=true
COMPARISON_MODE=false
MAP_DESC=true
echo "[INFO] MAP_DESC set to $MAP_DESC"

# =============================================================================
# PATHS
# =============================================================================

SCRIPT=$PROJECT_DIR/gen/withKG/preprocess_data_with_rag.py

TRAIN_DATA=$PROJECT_DIR/dataset/final_data/train_df.pkl
VAL_DATA=$PROJECT_DIR/dataset/final_data/val_df.pkl
TEST_DATA=$PROJECT_DIR/dataset/final_data/test_df.pkl

H1_INDEX_DIR=$PROJECT_DIR/fact_indexes/h1_index
H2_INDEX_DIR=$PROJECT_DIR/fact_indexes/h2_index

# Base output directory
if [ $SUBSET_SIZE -gt 0 ]; then
    OUTPUT_SUBDIR=preprocessed_rag_subset_${SUBSET_SIZE}
else
    OUTPUT_SUBDIR=preprocessed_rag_full
fi

if [ "$MAP_DESC" = true ]; then
    MAP_DIR=map_desc
else
    MAP_DIR=with_codes
fi

# Only add diagnosis_only subdirectory when flag is true
if [ "$DIAGNOSIS_ONLY" = true ]; then
    BASE_OUTPUT_DIR=$PROJECT_DIR/dataset/$OUTPUT_SUBDIR/$MAP_DIR/diagnosis_only
else
    BASE_OUTPUT_DIR=$PROJECT_DIR/dataset/$OUTPUT_SUBDIR/$MAP_DIR
fi

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
SAPBERT="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
ICD_INDEX_DIR=$PROJECT_DIR/gen/pipeline/icd_index_v9
KG_PATH=$PROJECT_DIR/KG/kg_output/medical_knowledge_graph2.pkl

# =============================================================================
# BUILD COMMON ARGUMENTS
# =============================================================================

COMMON_ARGS="--train_data $TRAIN_DATA --val_data $VAL_DATA"
COMMON_ARGS="$COMMON_ARGS --h1_index_dir $H1_INDEX_DIR --h2_index_dir $H2_INDEX_DIR"
COMMON_ARGS="$COMMON_ARGS --base_output_dir $BASE_OUTPUT_DIR"
COMMON_ARGS="$COMMON_ARGS --model_name $MODEL_NAME --sapbert_model $SAPBERT"
COMMON_ARGS="$COMMON_ARGS --k $K --icd_index_dir $ICD_INDEX_DIR --kg $KG_PATH"
COMMON_ARGS="$COMMON_ARGS --rel_aggregation $REL_AGGREGATION"

[ $SUBSET_SIZE -gt 0 ] && COMMON_ARGS="$COMMON_ARGS --subset_size $SUBSET_SIZE"
[ "$COMPARISON_MODE" = true ] && COMMON_ARGS="$COMMON_ARGS --comparison_mode"
[ "$PROCESS_TEST" = true ] && [ -f "$TEST_DATA" ] && COMMON_ARGS="$COMMON_ARGS --test_data $TEST_DATA"
[ "$MAP_DESC" = true ] && COMMON_ARGS="$COMMON_ARGS --map_desc"
[ "$DIAGNOSIS_ONLY" = true ] && COMMON_ARGS="$COMMON_ARGS --diagnosis_only"

# =============================================================================
# DETERMINE MODES
# =============================================================================

# Build --modes argument based on flags
MODES=""
if [ "$RUN_UNWEIGHTED" = true ] && [ "$RUN_WEIGHTED" = true ]; then
    MODES="both"
elif [ "$RUN_UNWEIGHTED" = true ]; then
    MODES="unweighted"
elif [ "$RUN_WEIGHTED" = true ]; then
    MODES="weighted"
fi

echo ""
echo "========================================"
echo "ABLATION STUDY CONFIGURATION"
echo "========================================"
echo "  H1 only: $RUN_H1"
echo "  H2 only: $RUN_H2"
echo "  H1+H2 combined: $RUN_COMBINED"
echo ""
echo "  Modes: $MODES"
if [ "$RUN_WEIGHTED" = true ]; then
    echo "  Alpha: $ALPHA"
    echo "  Relationship aggregation: $REL_AGGREGATION"
fi
echo ""
if [ "$RUN_COMBINED" = true ]; then
    echo "  H1 ratios: $H1_RATIOS"
fi
echo "  Diagnosis filtering: $DIAGNOSIS_ONLY"
echo "  Dataset: $([ $SUBSET_SIZE -gt 0 ] && echo "subset ($SUBSET_SIZE)" || echo "full")"
echo "  K facts: $K"
echo "  Base output: $BASE_OUTPUT_DIR"
echo "========================================"

# =============================================================================
# RUN ABLATIONS
# =============================================================================
#  H1 INDEX
if [ "$RUN_H1" = true ] && [ -n "$MODES" ]; then
    echo ""
    echo "========================================"
    echo "Running: H1 INDEX"
    echo "========================================"
    python "$SCRIPT" $COMMON_ARGS --index_mode h1 --modes $MODES --alpha $ALPHA
    echo " H1 complete"
fi

#  H2 INDEX
if [ "$RUN_H2" = true ] && [ -n "$MODES" ]; then
    echo ""
    echo "========================================"
    echo "Running: H2 INDEX"
    echo "========================================"
    python "$SCRIPT" $COMMON_ARGS --index_mode h2 --modes $MODES --alpha $ALPHA
    echo " H2 complete"
fi

#  COMBINED H1+H2
if [ "$RUN_COMBINED" = true ] && [ -n "$MODES" ]; then
    echo ""
    echo "========================================"
    echo "Running: COMBINED H1+H2"
    echo "========================================"
    
    for RATIO in $H1_RATIOS; do
        echo ""
        echo "  ──────────────────────────────────"
        echo "  H1 ratio: $RATIO"
        echo "  ──────────────────────────────────"
        python "$SCRIPT" $COMMON_ARGS --index_mode combined \
            --h1_ratio $RATIO --modes $MODES --alpha $ALPHA
    done
    
    echo " Combined complete"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "========================================"
echo " ALL PROCESSING COMPLETE"
echo "========================================"
echo "End time: $(date)"
echo ""

echo "Output structure:"
tree -L 3 "$BASE_OUTPUT_DIR" 2>/dev/null || find "$BASE_OUTPUT_DIR" -type d | head -20

exit 0