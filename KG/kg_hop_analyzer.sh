#!/bin/bash
#SBATCH --job-name=KGhop
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=03:00:00                # Max walltime
#SBATCH --output=logs/KGHop/%j.out    # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# 1-hop, both directions, default labels (CUI) and auto edge labels:
python kg_hop_analyzer.py \
  --seed C0152602 \
  --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_nodes.csv \
  --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_edges.csv \
  --out-dir /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/hop_reports

# python kg_hop_analyzer.py \
#   --seed C0011849 \
#   --rel-allow CHD,PAR \
#   --direction both \
#   --out-dir /…/kg_output/hop_reports

status=$?
echo "[INFO] Job finished with exit code $status"
exit $status