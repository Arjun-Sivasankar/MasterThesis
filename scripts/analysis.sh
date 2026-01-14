#!/bin/bash
#SBATCH --job-name=icd9_analysis
#SBATCH --partition=capella            # CPU partition (no GPU needed)
#SBATCH --gres=gpu:1                   # No GPU
#SBATCH --cpus-per-task=8              # CPU cores
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=01:00:00                # 1 hour should be enough
#SBATCH --output=logs/analyse/analysis_%j.out

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# --- 4) Run streamlined analysis ---
echo "[INFO] Starting ICD-9 analysis..."

python analysis/icd9_analysis.py \
    --train_pickle dataset/final_data/train_df.pkl \
    --val_pickle dataset/final_data/val_df.pkl \
    --test_pickle dataset/final_data/test_df.pkl \
    --icd9_pickle dataset/codes/icd9.pkl \
    --output_dir analysis_results \

# --- 5) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status