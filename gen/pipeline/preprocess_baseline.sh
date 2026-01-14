#!/bin/bash
#SBATCH --job-name=baseline_prep
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_preprocessing_%j.out

set -e

echo "========================================"
echo "BASELINE PREPROCESSING"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

# Setup
module purge
module load release/24.04 GCCcore/11.3.0 Python/3.10.4

PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR"
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export PYTHONUNBUFFERED=1

# Paths - UPDATE THESE TO YOUR DATA
TRAIN_DATA=$PROJECT_DIR/dataset/final_data/train_df.pkl
VAL_DATA=$PROJECT_DIR/dataset/final_data/val_df.pkl
TEST_DATA=$PROJECT_DIR/dataset/final_data/test_df.pkl

OUTPUT_DIR=$PROJECT_DIR/dataset/baseline

# Optional
ICD_INDEX_DIR=$PROJECT_DIR/gen/TextGen/icd_index_v9
KG_PATH=$PROJECT_DIR/KG/kg_output/medical_knowledge_graph2.pkl

# Run
python gen/withKG/withRAG/preprocess_baseline.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --test_data $TEST_DATA \
    --output_dir $OUTPUT_DIR \
    --icd_index_dir $ICD_INDEX_DIR \
    --kg $KG_PATH \
    --map_desc

echo "========================================"
echo "✓ COMPLETE"
echo "End time: $(date)"
echo "========================================"