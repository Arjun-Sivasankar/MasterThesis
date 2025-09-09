#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:2                   # 3 GPUs
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=03:00:00                # Max walltime
#SBATCH --output=logs/test/%j.out    # stdout+stderr log

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
export CUDA_VISIBLE_DEVICES=0,1
GPUS=${SLURM_GPUS_ON_NODE:-2}

# --- 6) Run training ---
# echo "[INFO] Starting analysis..."
# python analyse_model.py --data_pickle 'mergeddf.pkl' --run_dir '/data/horse/ws/arsi805e-finetune/Thesis/runs_gen/20250820-133502_llama1b_gen_len3072_lr0.0002' 
#python single_sample_compare.py \
#  --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#  --data_pickle mergeddf.pkl \
#  --row_index 112

# python compare_filter.py \
#   --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#   --data_pickle mergeddf.pkl \
#   # --limit 2000 \
#   --batch_size 4

# Top 5 most-different rows (by removed_non_vocab_count)
# python compare_.py \
#   --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#   --data_pickle mergeddf.pkl \
#   # --limit 1000 \
#   --batch_size 4 \
#   --examples_out runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002/examples_norm_leading_zero.txt \
#   --examples_k 6

echo "[INFO] Starting base evaluation..."
#python eval_base_llm.py --data_pickle mergeddf.pkl --filter_mode both --structured --notes 
# python eval_base_llm2.py --data_pickle mergeddf.pkl --model meta-llama/Llama-3.2-1B-Instruct

srun torchrun --nproc_per_node=${GPUS} gen/test/base_model_eval/eval_base_llm.py \
    --data_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --show_prompts \

# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status
