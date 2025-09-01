#!/bin/bash
#SBATCH --job-name=llama32_unsloth
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=1:00:00                # Max walltime
#SBATCH --output=llama32_unsloth_%j.out    # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# --- 4) PyTorch HPC settings ---
#export TOKENIZERS_PARALLELISM=false
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 5) Optional: Hugging Face token (if model gated) ---
# export HUGGING_FACE_HUB_TOKEN=your_token_here

# --- 6) Run training ---
echo "[INFO] Starting finetuning..."
python finetune_llama_unsloth.py

# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status
