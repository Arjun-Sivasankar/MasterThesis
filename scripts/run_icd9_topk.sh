#!/bin/bash
#SBATCH --job-name=ICD9_topk
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:3                   # 3 GPUs
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=24:00:00                # Max walltime
#SBATCH --output=logs/ICD9_topk50_%j.out    # stdout+stderr log
## SBATCH --output=logs/%j.out    # stdout+stderr log

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

# good to keep, reduces fragmentation:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 5) Optional: Hugging Face token (if model gated) ---
# export HUGGING_FACE_HUB_TOKEN=your_token_here
export CUDA_VISIBLE_DEVICES=0,1

# DDP launch (uses $SLURM_GPUS_ON_NODE if set)
GPUS=${SLURM_GPUS_ON_NODE:-2}
echo "[INFO] Using $GPUS GPUs"

SCRIPT="gen/finetune_llama_gen_ddp_icd9_topk.py"

# --- 6) Run training ---
start=$(date +%s)
echo "[INFO] Job started at $(date)"

echo "[INFO] Running script: $SCRIPT"

echo "[INFO] Starting ICD-9 finetuning with top-k codes ...."
srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
    --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/train_df.pkl \
    --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/val_df.pkl \
    --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl \
    --icd9_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl \
    --train_size 54981 \
    --eval_sample_size 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --test_batch_size 16 \
    --grad_accum 16 \
    --epochs 6 \
    --use_complete_icd9 1 \
    --top_k 50 \


end=$(date +%s)
echo "[INFO] Job finished at $(date)"
echo "[TIME] Elapsed: $((end-start)) seconds"

# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
echo "[INFO] Script that was run: $SCRIPT"
exit $status