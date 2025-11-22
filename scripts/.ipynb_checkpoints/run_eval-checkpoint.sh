#!/bin/bash
#SBATCH --job-name=diffsize_eval
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU (evaluation doesn't need multi-GPU)
#SBATCH --cpus-per-task=4              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=32G                      # RAM
#SBATCH --time=04:00:00                # Max walltime (evaluation is much faster)
#SBATCH --output=logs/test/eval_%j.out      # stdout+stderr log

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
export CUDA_VISIBLE_DEVICES=0

# --- 5) Run evaluation ---
start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] Running Script: ${SCRIPT}"
echo "[INFO] Running SINGLE-GPU test on GPU 0"

# Specify the run directory from your training (adjust path as needed)
RUN_DIR="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs/runs_gen/diffsize/20250909-223713_N54981_icd9_complete"  # Update this path
# SCRIPT="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/test/finetuned_eval/eval.py"  # Name of your evaluation script
SCRIPT="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/eval_icd9_codegen.py"

echo "[INFO] Run directory: $RUN_DIR"
echo "[INFO] Job started: $(date)"
echo "[INFO] Running Script: ${SCRIPT}"
echo "[INFO] Running SINGLE-GPU test on GPU 0"

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "[ERROR] Run directory not found: $RUN_DIR"
    exit 1
fi

# Check required files
if [ ! -d "$RUN_DIR/adapter_best" ]; then
    echo "[ERROR] Adapter not found at $RUN_DIR/adapter_best"
    exit 1
fi

if [ ! -d "$RUN_DIR/tokenizer" ]; then
    echo "[ERROR] Tokenizer not found at $RUN_DIR/tokenizer"
    exit 1
fi

if [ ! -f "$RUN_DIR/config.json" ]; then
    echo "[ERROR] Config not found at $RUN_DIR/config.json"
    exit 1
fi

if [ ! -f "$RUN_DIR/label_space.json" ]; then
    echo "[ERROR] Label space not found at $RUN_DIR/label_space.json"
    exit 1
fi

BASE_LLM=meta-llama/Llama-3.2-1B-Instruct

TOP_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_codes.csv
BOT_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/bottom_50_codes.csv
TOP_PARENTS=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_category_levels.csv

echo "[INFO] Starting ICD-9 evaluation..."
# srun python ${SCRIPT} \
#     --run_dir "$RUN_DIR" \
#     --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl \
#     --test_batch_size 16 \
#     --test_examples 5 \
#     --seed 42 \
#     --output_json "$RUN_DIR/test_metrics_eval.json"

srun python ${SCRIPT} \
  --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl \
  --base_model "${BASE_LLM}" \
  --adapter_dir /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs/runs_gen/diffsize/20250909-223713_N54981_icd9_complete/adapter_best \
  --label_space_json /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs/runs_gen/diffsize/20250909-223713_N54981_icd9_complete/label_space.json \
  --top_codes_csv "${TOP_CODES}" \
  --bottom_codes_csv "${BOT_CODES}" \
  --top_parent_csv "${TOP_PARENTS}" \
  --out_dir /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs/runs_gen/diffsize/20250909-223713_N54981_icd9_complete/eval_out \
  --batch_size 16 --gen_max_new 96 --print_samples 5


end=$(date +%s)
echo "[INFO] Job finished at $(date)"
echo "[TIME] Elapsed: $((end-start)) seconds"

# --- 6) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
echo "[INFO] Script that was run: $SCRIPT"
echo "[INFO] Run directory: $RUN_DIR"
exit $status