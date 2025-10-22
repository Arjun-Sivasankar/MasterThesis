#!/bin/bash
#SBATCH --job-name=train_parent_only
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/Parents/train_parent_%j.out

# --- 1) Modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Project dir ---
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

# --- 3) Venv ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"
python -V

# --- 4) Environment ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using $GPUS GPU(s)"

# --- 5) Paths ---
TRAIN_SCRIPT="gen/train_parents.py"

ICD9_TRAIN=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/train_df.pkl
ICD9_VAL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/val_df.pkl
ICD9_ALLCODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl

LLM=meta-llama/Llama-3.2-1B-Instruct
echo "[INFO] Using LLM model: ${LLM}"

EPOCHS=10
echo "[INFO] Training for $EPOCHS epoch(s)"

mkdir -p logs runs_gen

start=$(date +%s)
echo "[INFO] Training job started at $(date)"

#############################################
# ICD-9 PARENT Training ONLY
#############################################
echo "[INFO] Starting ICD-9 PARENT training..."
echo "[INFO] Training script: $TRAIN_SCRIPT"

python "${TRAIN_SCRIPT}" \
  --llama_model "${LLM}" \
  --train_pickle  "${ICD9_TRAIN}" \
  --val_pickle    "${ICD9_VAL}" \
  --codes_pickle  "${ICD9_ALLCODES}" \
  --subject_col   subject_id_x \
  --label_col     icd_code \
  --icd_scheme    icd9cm \
  --icd_level     parent \
  --use_complete_codes 1 \
  --train_size 54981 \
  --eval_sample_size 100 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --grad_accum 16 \
  --epochs ${EPOCHS} \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --early_stop 1 \
  --patience 2 \
  --run_root runs_gen/icd9_parent \
  --seed 42

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "[INFO] Training completed successfully!"
    # Find and display the training run directory
    LATEST_RUN=$(find runs_gen/icd9_parent -maxdepth 1 -type d -name "20*" | sort | tail -1)
    echo "[INFO] Model saved to: $LATEST_RUN"
    echo "[INFO] To test this model, use:"
    echo "sbatch gen/test_parents.sh --run-dir=$LATEST_RUN"
else
    echo "[ERROR] Training failed with exit code: $TRAIN_EXIT_CODE"
fi

end=$(date +%s)
echo "[INFO] Training job finished at $(date)"
echo "[TIME] Training elapsed: $((end-start)) seconds"

exit $TRAIN_EXIT_CODE