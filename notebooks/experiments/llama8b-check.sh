#!/bin/bash
#SBATCH --job-name=trial
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/trial_%j.out

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using $GPUS GPU(s)"

# --- 5) Paths ---
TRAIN_SCRIPT="gen/trial.py"

start=$(date +%s)
echo "[INFO] Training job started at $(date)"

echo "[INFO] Training script: $TRAIN_SCRIPT"

python "${TRAIN_SCRIPT}" 

TRAIN_EXIT_CODE=$?

end=$(date +%s)
echo "[INFO] Job finished at $(date)"
echo "[TIME] Elapsed: $((end-start)) seconds"

exit $TRAIN_EXIT_CODE