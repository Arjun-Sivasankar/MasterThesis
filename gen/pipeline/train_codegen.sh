#!/bin/bash
#SBATCH --job-name=pipe_codegen_train
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/CodeGen/train_codegen_%j.out

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

### 3) Virtualenv
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"

### 4) Runtime envs (no DDP here, but harmless to keep)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

# Single-GPU trainer -> request one GPU above; don't launch with torchrun.
echo "[INFO] GPUs allocated by SLURM: ${SLURM_GPUS_ON_NODE:-1}"

### 5) Paths & args (edit as needed)
SCRIPT=gen/pipeline/train_codegen.py

# Data: EITHER (A) one pickle to auto-split by subjects, OR (B) explicit train/val pickles
DATA_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl
# TRAIN_PKL=dataset/train_icd9.pkl
# VAL_PKL=dataset/val_icd9.pkl

# LLAMA_MODEL=meta-llama/Llama-3.2-1B-Instruct
LLAMA_MODEL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
echo "[INFO] Using LLM model: ${LLAMA_MODEL}"

# Run bookkeeping
RUN_ROOT=runs_codegen
RUN_NAME=exp_${SLURM_JOB_ID}

# Trainer hparams
EPOCHS=10
PER_DEV_BS=1
GRAD_ACCUM=16
LR=2e-4
WARMUP=0.03
MAX_LEN=3072
TGT_RESERVE_TOK=128
USE_STRUCTURED=1
USE_NOTES=1
COMPILE=0   # set to 1 if you want torch.compile()

### 6) Launch (non-distributed, single process)
start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] Running: ${SCRIPT}"
echo "[INFO] Training for Epochs: ${EPOCHS}"
echo "[INFO] CodeGen TRAIN -> RUN_ROOT=${RUN_ROOT} RUN_NAME=${RUN_NAME}"

mkdir -p logs "${RUN_ROOT}"

# (A) Auto subject split from a single pickle:
srun python "${SCRIPT}" \
  --data_pickle "${DATA_PKL}" \
  --llama_model "${LLAMA_MODEL}" \
  --use_structured "${USE_STRUCTURED}" \
  --use_notes "${USE_NOTES}" \
  --max_len "${MAX_LEN}" \
  --tgt_reserve_tok "${TGT_RESERVE_TOK}" \
  --epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEV_BS}" \
  --per_device_eval_batch_size "${PER_DEV_BS}" \
  --grad_accum "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --weight_decay 0.0 \
  --warmup_ratio "${WARMUP}" \
  --early_stop 1 \
  --patience 2 \
  --train_size -1 \
  --seed 42 \
  --run_root "${RUN_ROOT}" \
  --run_name "${RUN_NAME}" \
  --compile "${COMPILE}"

# (B) If you prefer explicit train/val files, comment out (A) above and use:
# srun python "${SCRIPT}" \
#   --train_pickle "${TRAIN_PKL}" \
#   --val_pickle "${VAL_PKL}" \
#   --llama_model "${LLAMA_MODEL}" \
#   --use_structured "${USE_STRUCTURED}" \
#   --use_notes "${USE_NOTES}" \
#   --max_len "${MAX_LEN}" \
#   --tgt_reserve_tok "${TGT_RESERVE_TOK}" \
#   --epochs "${EPOCHS}" \
#   --per_device_train_batch_size "${PER_DEV_BS}" \
#   --per_device_eval_batch_size "${PER_DEV_BS}" \
#   --grad_accum "${GRAD_ACCUM}" \
#   --learning_rate "${LR}" \
#   --weight_decay 0.0 \
#   --warmup_ratio "${WARMUP}" \
#   --early_stop 1 \
#   --patience 2 \
#   --train_size -1 \
#   --seed 42 \
#   --run_root "${RUN_ROOT}" \
#   --run_name "${RUN_NAME}" \
#   --compile "${COMPILE}"

status=$?
end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) s"
echo "[INFO] Train exit code: $status"

# Where artifacts land (per pipelines/train_codegen.py)
echo "[INFO] Artifacts:"
echo "  - Run dir:        ${RUN_ROOT}/${RUN_NAME}"
echo "  - Adapter:        ${RUN_ROOT}/${RUN_NAME}/adapter_best"
echo "  - Tokenizer:      ${RUN_ROOT}/${RUN_NAME}/tokenizer"
echo "  - Label space:    ${RUN_ROOT}/${RUN_NAME}/label_space.json"
echo "  - Train summary:  ${RUN_ROOT}/${RUN_NAME}/train_summary.json"
exit $status