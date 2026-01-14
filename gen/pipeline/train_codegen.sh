#!/bin/bash
#SBATCH --job-name=codegen_train
#SBATCH --partition=capella                               # GPU partition
#SBATCH --gres=gpu:3                                      # 3 GPUs
#SBATCH --cpus-per-task=8                                 # CPU cores for data loading
#SBATCH --nodes=1                                         # node
#SBATCH --mem=64G                                         # RAM
#SBATCH --time=24:00:00                                   # Max walltime
#SBATCH --output=logs/CodeGen-DDP/train_ddp_new_%j.out    # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# good to keep, reduces fragmentation:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 5) Optional: Hugging Face token (if model gated) ---
# export HUGGING_FACE_HUB_TOKEN=your_token_here
export CUDA_VISIBLE_DEVICES=0,1

# DDP launch (uses $SLURM_GPUS_ON_NODE if set)
GPUS=${SLURM_GPUS_ON_NODE:-2}
echo "[INFO] Using $GPUS GPUs"

SCRIPT="gen/train_codegen.py"
echo "[INFO] Script to be run: $SCRIPT"

EPOCHS=10
echo "[INFO] Training for $EPOCHS epoch(s)"

LLM=./models/Llama-3.1-8B-Instruct
# LLM=meta-llama/Llama-3.2-1B-Instruct
# LLM=./models/Meditron3-8B

echo "[INFO] Using LLM model: ${LLM}"

EARLY_STOP=1
echo "[INFO] Early stopping: ${EARLY_STOP}"

# --- 6) Run training ---
start=$(date +%s)
echo "[INFO] Job started at $(date)"

echo "[INFO] Starting ICD-9 finetuning ...."
srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
    --train_pickle ./dataset/final_data/train_df.pkl \
    --val_pickle ./dataset/final_data/val_df.pkl \
    --test_pickle ./dataset/final_data/test_df.pkl \
    --icd9_pickle ./dataset/codes/icd9.pkl \
    --llama_model ${LLM} \
    --eval_sample_size 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --test_batch_size 16 \
    --grad_accum 16 \
    --epochs ${EPOCHS} \
    --use_complete_icd9 1 \
    --early_stop ${EARLY_STOP} \

end=$(date +%s)
echo "[INFO] Job finished at $(date)"
echo "[TIME] Elapsed: $((end-start)) seconds"

# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
echo "[INFO] Script that was run: $SCRIPT"
exit $status