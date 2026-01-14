#!/bin/bash
#SBATCH --job-name=retrieve_weighted_paths
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/History_Aware/retrieve_weighted_paths_%j.out

# --- 1) Modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Project dir ---
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

# --- 3) Venv ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"
python -V

# --- 4) Environment ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPT="gen/withKG/history_aware/retrieve_paths_for_history_aware_weighted.py"
SPLIT="train"      # set to train, val, or test
MODE="h2"         # set to h1, h2

echo "[INFO] Running Script: ${SCRIPT} --split ${SPLIT} --mode ${MODE}"
start_time=$(date +%s)

python ${SCRIPT} --split ${SPLIT} --mode ${MODE}

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "[INFO] Weighted retrieval completed for '${SPLIT}' split, mode '${MODE}' in ${duration} seconds"
echo "[INFO] Successfully completed in ${duration} seconds"