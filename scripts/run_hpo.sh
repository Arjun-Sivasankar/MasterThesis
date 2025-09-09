#!/bin/bash
#SBATCH --job-name=HPO
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU (changed from 2)
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=96:00:00                # Max walltime
#SBATCH --output=logs/hpo/%j.out       # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# --- 4) PyTorch HPC settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 5) Create directories if they don't exist ---
mkdir -p runs_hpo/llama_train_10k

# --- 7) Check disk usage before starting ---
echo "[INFO] Starting disk usage: $(du -sh runs_hpo/llama_train_10k)"

# --- 8) Run HPO script ---
echo "[INFO] Starting finetuning..."
start=$(date +%s)

# Use a single GPU for HPO
export CUDA_VISIBLE_DEVICES=0
GPUS=1

srun torchrun --standalone --nproc_per_node=${GPUS} gen/hpo/finetune_hpo.py \
  --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/train_df.pkl \
  --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/val_df.pkl \
  --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/test_df.pkl \
  --icd9_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl \
  --train_size 10000 \
  --n_trials 15 \
  --output_dir runs_hpo/llama_train_10k \
  --metric micro_f1 \
  --test_batch_size 16 \
  --save_full_model 0 \
  --compress_trials 1 \
  --keep_trials 3 \
  --auto_cleanup_threshold 0.85

end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) seconds"

# --- 9) Check disk usage after completion ---
echo "[INFO] Final disk usage: $(du -sh runs_hpo/llama_train_10k)"

# --- 10) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status