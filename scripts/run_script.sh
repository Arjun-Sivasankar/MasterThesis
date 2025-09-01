#!/bin/bash
#SBATCH --job-name=analyse
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=01:00:00                # Max walltime
#SBATCH --output=llama_test_%j.out    # stdout+stderr log

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
# echo "[INFO] Starting dump predictions..."
#python dump_preds.py \
#  --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#  --data_pickle mergeddf.pkl \
#  --batch_size 8 \
#  --max_new 128
  
#python analyse_predictions.py \
#  --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002
  

echo "[INFO] Analysing predictions and plotting results ..."
python analyse_predictions1.py \
 --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
 --preds_jsonl predictions.jsonl \
 --head_n 50 \
 --torso_n 450 \
 --save_plots 1
  

# # Greedy (default), with post-filter
# echo "[INFO] Dump predictions with post filter..."
# # With post-filter (default)
# python dump_preds_modes.py \
#   --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#   --data_pickle mergeddf.pkl \
#   --decoding greedy

# echo "[INFO] Dump predictions without post filter..."
# # Without post-filter (and also save the raw, unfiltered tokens)
# python dump_preds_modes.py \
#   --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#   --data_pickle mergeddf.pkl \
#   --decoding greedy \
#   --no_post_filter \
#   --save_raw \
#   --save_suffix _nopf


## Beam search
#python dump_preds.py --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 --data_pickle mergeddf.pkl --decoding beam --num_beams 4
#
## Top-k sampling
#python dump_preds.py --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 --data_pickle mergeddf.pkl --decoding topk --top_k 50 --temperature 0.7
#
## Top-p sampling
#python dump_preds.py --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 --data_pickle mergeddf.pkl --decoding topp --top_p 0.9 --temperature 0.7
#
## Turn OFF post-filter to quantify OOV/duplicates and bag metrics
#python dump_preds.py --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 --data_pickle mergeddf.pkl --no_post_filter
#
## Compare several modes at once on a subset (faster):
#python dump_preds.py --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 --data_pickle mergeddf.pkl --compare_all --limit 1000

# echo "[INFO] Comparing decoding modes..."
# or specify files explicitly:
# python analyse_predictions_compare.py \
#   --run_dir runs_gen/20250820-232601_llama1b_gen_len3072_lr0.0002 \
#   --data_pickle mergeddf.pkl \
#   --pred_files preds_raw_greedy.jsonl preds_post_greedy.jsonl 
  
# --- 7) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status