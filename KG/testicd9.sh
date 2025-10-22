#!/bin/bash
#SBATCH --job-name=KG_test
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=03:00:00                # Max walltime
#SBATCH --output=logs/KGBuild/icd9diagproc_%j.out    # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# python testicd9.py
python kg_stats.py \
  --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_nodes.csv \
  --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_edges.csv \
  --out-dir /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/stats \
  --chunk-size 500000 \
  --top-k 100 \
  --expand-sab-pairs

status=$?
echo "[INFO] Job finished with exit code $status"
exit $status