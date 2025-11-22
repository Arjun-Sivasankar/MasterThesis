#!/bin/bash
# filepath: /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/withKG/withRAG/preprocess_data_rag.sh
#SBATCH --job-name=preprocess_rag
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/preprocess_rag_%j.out
#SBATCH --licenses=horse

set -e
set -u

# =============================================================================
# CONFIGURATION - EDIT THESE TO CONTROL BEHAVIOR
# =============================================================================

# Subset mode: Set to 0 for full dataset, or any number for testing (e.g., 50, 100, 500)
SUBSET_SIZE=5

# Comparison mode: true = retrieve both weighted/unweighted for analysis, false = skip
COMPARISON_MODE=true

# Which ablations to run: true = run, false = skip
RUN_BASELINE=false
RUN_UNWEIGHTED=true
RUN_WEIGHTED_03=true
RUN_WEIGHTED_05=false
RUN_WEIGHTED_07=false

# Retrieval parameters
# K=20
K=50

# Include test data: true = process test set, false = skip
PROCESS_TEST=true

# =============================================================================
# PATHS (usually don't need to change)
# =============================================================================

PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
SCRIPT=$PROJECT_DIR/gen/withKG/withRAG/preprocess_data_with_rag.py

TRAIN_DATA=$PROJECT_DIR/dataset/final_data/train_df.pkl
VAL_DATA=$PROJECT_DIR/dataset/final_data/val_df.pkl
TEST_DATA=$PROJECT_DIR/dataset/final_data/test_df.pkl

MAP_DESC=true

FACT_INDEX_DIR=$PROJECT_DIR/fact_index_dataset_aware/strict_mode
# Output directory based on subset size => If subset size > 0, use subset directory; else use full directory
if [ $SUBSET_SIZE -gt 0 ]; then
    OUTPUT_DIR=$PROJECT_DIR/dataset/preprocessed_rag_subset_${SUBSET_SIZE}
else
    OUTPUT_DIR=$PROJECT_DIR/dataset/preprocessed_rag_full
fi

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
SAPBERT="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

ICD_INDEX_DIR=$PROJECT_DIR/gen/TextGen/icd_index_v9

KG_PATH=$PROJECT_DIR/KG/kg_output4/medical_knowledge_graph2.pkl


# =============================================================================
# SETUP
# =============================================================================

echo "========================================"
echo "RAG PREPROCESSING"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

cd "$PROJECT_DIR"
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/logs/Textgen-withKG-withRAG"

# =============================================================================
# SHOW CONFIGURATION
# =============================================================================

echo ""
echo "Configuration:"
echo "  Subset size: $([ $SUBSET_SIZE -eq 0 ] && echo 'FULL DATASET' || echo "$SUBSET_SIZE samples")"
echo "  Comparison mode: $COMPARISON_MODE"
echo "  K facts: $K"
echo "  Process test: $PROCESS_TEST"
echo ""
echo "Ablations to run:"
echo "  Baseline (no KG): $RUN_BASELINE"
echo "  Unweighted RAG: $RUN_UNWEIGHTED"
echo "  Weighted α=0.3: $RUN_WEIGHTED_03"
echo "  Weighted α=0.5: $RUN_WEIGHTED_05"
echo "  Weighted α=0.7: $RUN_WEIGHTED_07"
echo ""

# Build common arguments
COMMON_ARGS="--train_data $TRAIN_DATA --val_data $VAL_DATA --fact_index_dir $FACT_INDEX_DIR --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --sapbert_model $SAPBERT --k $K --icd_index_dir $ICD_INDEX_DIR"

[ $SUBSET_SIZE -gt 0 ] && COMMON_ARGS="$COMMON_ARGS --subset_size $SUBSET_SIZE"
[ "$COMPARISON_MODE" = true ] && COMMON_ARGS="$COMMON_ARGS --comparison_mode"
[ "$PROCESS_TEST" = true ] && [ -f "$TEST_DATA" ] && COMMON_ARGS="$COMMON_ARGS --test_data $TEST_DATA"
[ "$MAP_DESC" = true ] && COMMON_ARGS="$COMMON_ARGS --map_desc"

# =============================================================================
# RUN ABLATIONS
# =============================================================================

# Baseline (No KG)
if [ "$RUN_BASELINE" = true ]; then
    echo ""
    echo "========================================"
    echo "Running: BASELINE (No KG)"
    echo "========================================"
    python "$SCRIPT" $COMMON_ARGS
    echo "✓ Baseline complete"
fi

# RAG Unweighted
if [ "$RUN_UNWEIGHTED" = true ]; then
    echo ""
    echo "========================================"
    echo "Running: RAG UNWEIGHTED"
    echo "========================================"
    python "$SCRIPT" $COMMON_ARGS --use_kg
    echo "✓ Unweighted complete"
fi

# RAG Weighted α=0.3
if [ "$RUN_WEIGHTED_03" = true ]; then
    echo ""
    echo "========================================"
    echo "Running: RAG WEIGHTED (α=0.3)"
    echo "========================================"
    python "$SCRIPT" $COMMON_ARGS --use_kg --use_weighting --alpha 0.3
    echo "✓ Weighted α=0.3 complete"
fi

# RAG Weighted α=0.5
if [ "$RUN_WEIGHTED_05" = true ]; then
    echo ""
    echo "========================================"
    echo "Running: RAG WEIGHTED (α=0.5)"
    echo "========================================"
    python "$SCRIPT" $COMMON_ARGS --use_kg --use_weighting --alpha 0.5
    echo "✓ Weighted α=0.5 complete"
fi

# RAG Weighted α=0.7
if [ "$RUN_WEIGHTED_07" = true ]; then
    echo ""
    echo "========================================"
    echo "Running: RAG WEIGHTED (α=0.7)"
    echo "========================================"
    python "$SCRIPT" $COMMON_ARGS --use_kg --use_weighting --alpha 0.7
    echo "✓ Weighted α=0.7 complete"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "========================================"
echo "✓ ALL PROCESSING COMPLETE"
echo "========================================"
echo "End time: $(date)"
echo ""

echo "Files created:"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null | tail -20
echo ""

echo "Statistics files:"
ls "$OUTPUT_DIR"/*_stats.json 2>/dev/null | wc -l
echo ""

[ "$COMPARISON_MODE" = true ] && echo "Comparison files:" && ls "$OUTPUT_DIR"/*_comparison.json 2>/dev/null | wc -l

echo ""
echo "========================================"

exit 0