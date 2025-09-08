#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=03:00:00                # Max walltime
#SBATCH --output=logs/analyse/analysis_%j.out    # stdout+stderr log

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
#export TOKENIZERS_PARALLELISM=false
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 5) Optional: Hugging Face token (if model gated) ---
# export HUGGING_FACE_HUB_TOKEN=your_token_here

# --- 5) Run analysis ---
# echo "[INFO] Starting ICD-9 analysis..."
# python analysis/analyse_icd9_distribution.py \
#     --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/train_df.pkl \
#     --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/val_df.pkl \
#     --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/test_df.pkl \
#     --icd9_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl \
#     --output_dir icd9_analysis

echo "[INFO] Starting ICD-10 analysis..."
python analysis/analyse_icd10_distribution.py \
    --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/train_df.pkl \
    --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/val_df.pkl \
    --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/test_df.pkl \
    --icd10_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd10.pkl \
    --output_dir icd10_analysis

# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status

