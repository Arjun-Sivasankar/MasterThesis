#!/bin/bash
#SBATCH --job-name=build_fact_index
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/Textgen-withKG/build_fact_index_%j.out
#SBATCH --licenses=horse

set -e
set -u

echo "========================================"
echo "BUILDING MEDICAL FACT INDEX"
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
export CUDA_VISIBLE_DEVICES=0

### 5) Define paths
SCRIPT=$PROJECT_DIR/gen/withKG/build_medical_fact_index.py

# Input KG
KG_DIR=$PROJECT_DIR/KG/kg_output4
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl

# Output directory
FACT_INDEX_DIR=$PROJECT_DIR/gen/withKG/fact_index
mkdir -p "$FACT_INDEX_DIR"
mkdir -p "$PROJECT_DIR/logs/Textgen-withKG"

### 6) Configuration
SAPBERT_MODEL="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MAX_H1=100000      # Mine up to 100k H1 paths (or None for all)
MAX_H2=200000      # Mine up to 200k H2 paths
MAX_H2_PER_SRC=100 # Max 100 H2 paths per source node
BATCH_SIZE=128     # Encoding batch size

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "KG file: $KG_PKL"
echo "Output dir: $FACT_INDEX_DIR"
echo "SapBERT model: $SAPBERT_MODEL"
echo "Max H1 paths: $MAX_H1"
echo "Max H2 paths: $MAX_H2"
echo "Max H2 per source: $MAX_H2_PER_SRC"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

# Verify KG exists
if [ ! -f "$KG_PKL" ]; then
    echo "ERROR: KG file not found: $KG_PKL"
    exit 1
fi
echo "✓ KG file exists"

### 7) Run fact index builder
echo "========================================"
echo "STARTING FACT INDEX BUILD"
echo "========================================"

start=$(date +%s)

python "$SCRIPT" \
  --kg_pkl "$KG_PKL" \
  --output_dir "$FACT_INDEX_DIR" \
  --max_h1 $MAX_H1 \
  --max_h2 $MAX_H2 \
  --max_h2_per_source $MAX_H2_PER_SRC \
  --sapbert_model "$SAPBERT_MODEL" \
  --batch_size $BATCH_SIZE \
  --use_gpu_faiss

status=$?
end=$(date +%s)
elapsed=$((end - start))

echo "========================================"
echo "FACT INDEX BUILD COMPLETE"
echo "========================================"
echo "Exit code: $status"
echo "Elapsed time: $elapsed seconds ($(($elapsed / 60)) minutes)"
echo "End time: $(date)"

if [ $status -eq 0 ]; then
    echo "✓ SUCCESS"
    echo "Output directory: $FACT_INDEX_DIR"
    echo ""
    echo "Files created:"
    ls -lh "$FACT_INDEX_DIR"
else
    echo "✗ FAILED"
fi

echo "========================================"

exit $status