#!/bin/bash
#SBATCH --job-name=diffsize_train
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:3                   # 3 GPUs
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=24:00:00                # Max walltime
#SBATCH --output=logs/CodeGen-DDP/train_ddp_%j.out    # stdout+stderr log
## SBATCH --output=logs/CodeGen-DDP/%j.out    # stdout+stderr log

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

SCRIPT="gen/finetune_llama_gen_ddp.py"
echo "[INFO] Script to be run: $SCRIPT"

EPOCHS=10
echo "[INFO] Training for $EPOCHS epoch(s)"

LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
# LLM=meta-llama/Llama-3.2-1B-Instruct
echo "[INFO] Using LLM model: ${LLM}"

# --- 6) Run training ---
start=$(date +%s)
echo "[INFO] Job started at $(date)"

echo "[INFO] Starting ICD-9 finetuning (DDP - With diff model) ...."
srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
    --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/train_df.pkl \
    --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/val_df.pkl \
    --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl \
    --icd9_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl \
    --llama_model /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct \
    --train_size 54981 \
    --eval_sample_size 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --test_batch_size 16 \
    --grad_accum 16 \
    --epochs ${EPOCHS} \
    --use_complete_icd9 1


# echo "[INFO] Starting ICD-9 finetuning..."
# # srun torchrun --standalone --nproc_per_node=${GPUS} gen/finetune_llama_gen_difftrainsize_improved.py \
# #     --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/train_df.pkl \
# #     --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/val_df.pkl \
# #     --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/gen_data/test_df.pkl \
# #     --icd9_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl \
# #     --train_size 60000 \
# #     --eval_sample_size 100 \
# #     --epochs 10 --learning_rate 2e-4 

# srun torchrun --standalone --nproc_per_node=${GPUS} gen/finetune_llama_gen_ddp.py \
#     --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/train_df.pkl \
#     --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/val_df.pkl \
#     --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl \
#     --icd9_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl \
#     --train_size 54981 \
#     --eval_sample_size 100 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --test_batch_size 16 \
#     --grad_accum 16 \
#     --epochs 6 \
#     --use_complete_icd9 1

# echo "[INFO] Starting ICD-10 finetuning..."
# start=$(date +%s)

# srun torchrun --standalone --nproc_per_node=${GPUS} gen/finetune_llama_gen_ddp_icd10.py \
#     --train_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/train_df.pkl \
#     --val_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/val_df.pkl \
#     --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/test_df.pkl \
#     --icd10_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd10.pkl \
#     --label_col icd_code \
#     --train_size 40000 \
#     --eval_sample_size 100 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --test_batch_size 16 \
#     --grad_accum 16 \
#     --epochs 6 \
#     --use_complete_icd10 1 \
#     --run_root runs_gen/icd10

end=$(date +%s)
echo "[INFO] Job finished at $(date)"
echo "[TIME] Elapsed: $((end-start)) seconds"

# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
echo "[INFO] Script that was run: $SCRIPT"
exit $status