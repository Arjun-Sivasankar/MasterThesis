#!/bin/bash
#SBATCH --job-name=KG
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:2                   # 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=03:00:00                # Max walltime
#SBATCH --output=logs/KGBuild2/%j.out    # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

SCRIPT=KG/buildKG.py
echo "Running script: $SCRIPT"
python $SCRIPT \
    --umls-dir ./UMLS/2025AA/META \
    --out-dir ./KG/kg_output \
    --cui-to-icd9-json ./KG/mappings/cui_to_icd9_EXACT.json \
    --icd9-dx-pkl ./dataset/codes/icd9.pkl \
    --icd9-proc-pkl ./dataset/codes/icd9proc.pkl \
    --dataset-pkl ./dataset/merged_icd9.pkl \
    --with-names

status=$?
echo "[INFO] Job finished with exit code $status"
exit $status