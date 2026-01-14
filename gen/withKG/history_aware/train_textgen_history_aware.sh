#!/bin/bash
#SBATCH --job-name=train_historyaware_rag
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/History_Aware/train_textgen_historyaware_%j.out

module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_DISABLE_PROGRESS_BAR=1
export TRANSFORMERS_VERBOSITY=error
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using ${GPUS} GPU(s) for training"

SCRIPT=gen/withKG/history_aware/train_textgen_history_aware.py
echo "[INFO] Training Script: ${SCRIPT}"

# Set these variables for your run:
MODE="baseline"   # options: baseline, h1_unweighted, h1_weighted, h2_unweighted, h2_weighted

# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct
BASE_LLM=models/Meditron3-8B

EPOCHS=10
PROMPT_DIR=dataset/history_aware_data/prompts

case $MODE in
  baseline)
    TRAIN_TSV=${PROMPT_DIR}/train_prompts.tsv
    VAL_TSV=${PROMPT_DIR}/val_prompts.tsv
    EXP_NAME="baseline"
    ;;
  h1_unweighted)
    TRAIN_TSV=${PROMPT_DIR}/train_prompts_h1_unweighted.tsv
    VAL_TSV=${PROMPT_DIR}/val_prompts_h1_unweighted.tsv
    EXP_NAME="h1_unweighted"
    ;;
  h1_weighted)
    TRAIN_TSV=${PROMPT_DIR}/train_prompts_h1_weighted.tsv
    VAL_TSV=${PROMPT_DIR}/val_prompts_h1_weighted.tsv
    EXP_NAME="h1_weighted"
    ;;
  h2_unweighted)
    TRAIN_TSV=${PROMPT_DIR}/train_prompts_h2_unweighted.tsv
    VAL_TSV=${PROMPT_DIR}/val_prompts_h2_unweighted.tsv
    EXP_NAME="h2_unweighted"
    ;;
  h2_weighted)
    TRAIN_TSV=${PROMPT_DIR}/train_prompts_h2_weighted.tsv
    VAL_TSV=${PROMPT_DIR}/val_prompts_h2_weighted.tsv
    EXP_NAME="h2_weighted"
    ;;
  *)
    echo "[ERROR] Unknown MODE: ${MODE}"
    exit 1
    ;;
esac

OUT_DIR=runs_historyaware/${EXP_NAME}/${BASE_LLM}/checkpoints_$EPOCHS
ADAPTER_DIR=runs_historyaware/${EXP_NAME}/${BASE_LLM}/adapter_$EPOCHS

srun torchrun \
  --standalone \
  --nproc_per_node=${GPUS} \
  ${SCRIPT} \
  --train_tsv "${TRAIN_TSV}" \
  --val_tsv "${VAL_TSV}" \
  --llm "${BASE_LLM}" \
  --max_len 5120 \
  --epochs ${EPOCHS} \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --grad_accum 16 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --out_dir "${OUT_DIR}" \
  --save_adapter \
  --adapter_dir "${ADAPTER_DIR}" \
  --experiment_name "${EXP_NAME}" \
  --early_stop 1 \
  --patience 2 \

status=$?
echo "[INFO] Training finished with exit code: $status"
exit $status