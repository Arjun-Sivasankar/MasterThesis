#!/bin/bash
#SBATCH --job-name=pipe_textgen_train
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen/train_textgen_%j.out

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

### 3) Virtualenv
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"

### 4) HPC / Torch runtime
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

# Prefer Torch-prefixed envs (NCCL_* variants now warn)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# If you have IB + Ethernet, list them; avoid loopback & docker
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

# Respect SLURM’s GPU allocation (do NOT hardcode CUDA_VISIBLE_DEVICES)
GPUS=${SLURM_GPUS_ON_NODE:-3}
echo "[INFO] Using $GPUS GPUs for training"

SCRIPT=gen/pipeline/train_textgen.py

### 5) Paths & args (edit as needed)
DATA_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl
# DATA_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/final_data/train_df.pkl
ICD_INDEX=./gen/TextGen/icd_index_v9
OUT_DIR=runs_textgen/checkpoints
ADAPTER_DIR=runs_textgen/adapter_v1

EPOCHS=10
echo "[INFO] Training for Epochs: ${EPOCHS}"

# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
BASE_LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
echo "[INFO] Using base LLM: ${BASE_LLM}"

### 6) Run: multi-GPU training (DDP) via torchrun
start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] Running Script: ${SCRIPT}"
echo "[INFO] Launching training…"

srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
  --data_pickle "${DATA_PKL}" \
  --llm "${BASE_LLM}" \
  --target_mode icd_titles \
  --icd_index_dir "${ICD_INDEX}" \
  --epochs ${EPOCHS} \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --grad_accum 16 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --N_max_terms 12 \
  --min_assistant_tokens 128 \
  --out_dir "${OUT_DIR}" \
  --save_adapter \
  --adapter_dir "${ADAPTER_DIR}"

status=$?
end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) s"
echo "[INFO] Train exit code: $status"
exit $status
