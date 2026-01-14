#!/bin/bash
#SBATCH --job-name=icd_index_build
#SBATCH --partition=capella            # CPU partition (no GPU needed)
#SBATCH --cpus-per-task=8              # CPU cores
#SBATCH --nodes=1
#SBATCH --mem=32G                      # Adjust as needed
#SBATCH --time=02:00:00                # 2 hours, adjust as needed
#SBATCH --output=logs/icd_index/icd_index_%j.out

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# --- 4) Run ICD index build ---
echo "[INFO] Starting ICD index build..."

python gen/pipeline/build_icd_index.py \
    --csv dataset/codes/icd9.csv \
    --icd_version 9 \
    --model cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --out_dir gen/pipeline/icd_index_v9

# --- 5) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status