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

# python KG/buildKG.py
python KG/buildKG2.py
# python KG/buildKG2.py \
#   --umls-dir /data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META \
#   --out-dir  /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2 \
#   --cui-to-icd9-json /data/horse/ws/arsi805e-finetune/Thesis/mappings/cui_to_icd9_EXACT.json \
#   --icd9-dx-pkl   /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl \
#   --icd9-proc-pkl /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9proc.pkl \
#   --dataset-pkl   /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl \
#   --with_names \
#   --target-vocabs ICD9CM,LNC,ATC,SNOMEDCT_US

status=$?
echo "[INFO] Job finished with exit code $status"
exit $status