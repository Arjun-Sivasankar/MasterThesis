#!/bin/bash
#SBATCH --job-name=diffsize_train
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:3                   # 3 GPUs
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=24:00:00                # Max walltime
#SBATCH --output=logs/train_textgen_%j.out    # stdout+stderr log
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

SCRIPT="gen/TextGen/finetune_textgen.py"

# --- 6) Run training ---
start=$(date +%s)
echo "[INFO] Job started at $(date)"

echo "[INFO] Running script: $SCRIPT"
echo "[INFO] Starting ICD-9 finetuning (Text-Gen Diagnosis) ...."
srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
    --data_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl \
    --llm meta-llama/Llama-3.2-1B-Instruct \
    --target_mode icd_titles \
    --icd_index_dir ./gen/TextGen/icd_index_v9 \
    --encoder_model cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --epochs 4 --per_device_train_batch_size 1 --grad_accum 16 \
    --gen_max_new 128 --N_max_terms 12 \
    --faiss_rows 50 --tau_cos 0.40 --tau_final 0.60 --w_cos 0.6 --w_fuz 0.4 \
    --eval_head_k 0

end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) seconds"

# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
echo "[INFO] Script that was run: $SCRIPT"
exit $status