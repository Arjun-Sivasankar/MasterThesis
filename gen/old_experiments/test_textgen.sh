#!/bin/bash
#SBATCH --job-name=pipe_textgen_test
#SBATCH --partition=capella
#SBATCH --gres=gpu:1               # request enough if you want multi-GPU inference
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen/test_textgen_%j.out

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
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

### 5) Paths & args (edit as needed)
# SCRIPT=gen/TextGen/test_textgen.py
SCRIPT=gen/pipeline/test_textgen.py
# DATA_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl
DATA_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/final_data/test_df.pkl
ICD_INDEX=./gen/TextGen/icd_index_v9
# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct
# echo "[INFO] Using base LLM: ${BASE_LLM}"

# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
BASE_LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
# BASE_LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Meditron3-8B
echo "[INFO] Using base LLM: ${BASE_LLM}"

if [ ${BASE_LLM} == "meta-llama/Llama-3.2-1B-Instruct" ]; then
    ADAPTER_DIR=runs_textgen/llama3.2-1B/adapter_v1
    TMP_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/llama3.2-1B/adapter_v1/test_shards
    OUT_METRICS=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/llama3.2-1B/adapter_v1/test_metrics.json
elif [ ${BASE_LLM} == "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct" ]; then
    ADAPTER_DIR=runs_textgen/llama3.1-8B/adapter_v1
    TMP_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/llama3.1-8B/adapter_v1/test_shards
    OUT_METRICS=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/llama3.1-8B/adapter_v1/test_metrics.json
else
    ADAPTER_DIR=runs_textgen/meditron3-8B/adapter_v1
    TMP_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/meditron3-8B/adapter_v1/test_shards
    OUT_METRICS=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/meditron3-8B/adapter_v1/test_metrics.json
fi

# ADAPTER_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs/runs_textgen/checkpoints/checkpoint-11694
# ADAPTER_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/adapter_v1
# # TMP_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs/runs_textgen/checkpoints/checkpoint-11694/test_shards
# TMP_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/adapter_v1/test_shards
# OUT_METRICS=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_textgen/adapter_v1/test_metrics.json

# Decoding config
DECODING=greedy      # greedy | beam | sample
NUM_BEAMS=2
GEN_MAX_NEW=128
GEN_BS=8
NO_REPEAT_NGRAM=0
TEMP=1.0
TOPP=0.95
TOPK=50

# Label space
LABELS_SPACE=full    # full | head
HEAD_K=0             # only used if LABELS_SPACE=head

PRINT_SAMPLES=5
USE_BF16=1           # 1 to request bf16 if GPU allows it

TOP_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_codes.csv
BOT_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/bottom_50_codes.csv
TOP_PARENTS=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_category_levels.csv

## Mapper configs:
# original weights: w_cos=0.6, w_fuz=0.4
# new weights:
w_cos=1.0
w_fuz=0.0

### 6) Pick one launch mode

##############
# A) SINGLE-GPU quick test
##############
if true ; then
  echo "[INFO] Job started: $(date)"
  echo "[INFO] Running Script: ${SCRIPT}"
  echo "[INFO] Running SINGLE-GPU test on GPU 0"
  export CUDA_VISIBLE_DEVICES=0
  python ${SCRIPT} \
    --data_pickle "${DATA_PKL}" \
    --base_model "${BASE_LLM}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --icd_index_dir "${ICD_INDEX}" \
    --decoding "${DECODING}" \
    --num_beams ${NUM_BEAMS} \
    --gen_batch_size ${GEN_BS} \
    --gen_max_new ${GEN_MAX_NEW} \
    --no_repeat_ngram ${NO_REPEAT_NGRAM} \
    --temperature ${TEMP} \
    --top_p ${TOPP} \
    --top_k ${TOPK} \
    --labels_space ${LABELS_SPACE} \
    --labels_head_k ${HEAD_K} \
    --print_samples ${PRINT_SAMPLES} \
    --tmp_dir "${TMP_DIR}" \
    --out_metrics "${OUT_METRICS}" \
    --top_codes_csv "${TOP_CODES}" \
    --bottom_codes_csv "${BOT_CODES}" \
    --top_parent_csv "${TOP_PARENTS}" \
    --w_cos ${w_cos} \
    --w_fuz ${w_fuz} \
    --test_only \
    $( [[ "${USE_BF16}" == "1" ]] && echo --use_bf16 )
fi

##############
# B) MULTI-GPU sharded inference (uncomment to use)
##############
: '
GPUS=${SLURM_GPUS_ON_NODE:-3}
echo "[INFO] Running MULTI-GPU sharded inference on $GPUS GPUs"
srun torchrun --standalone --nproc_per_node=${GPUS} test_textgen.py \
  --distributed \
  --data_pickle "${DATA_PKL}" \
  --base_model "${BASE_LLM}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --icd_index_dir "${ICD_INDEX}" \
  --decoding "${DECODING}" \
  --num_beams ${NUM_BEAMS} \
  --gen_batch_size ${GEN_BS} \
  --gen_max_new ${GEN_MAX_NEW} \
  --no_repeat_ngram ${NO_REPEAT_NGRAM} \
  --temperature ${TEMP} \
  --top_p ${TOPP} \
  --top_k ${TOPK} \
  --labels_space ${LABELS_SPACE} \
  --labels_head_k ${HEAD_K} \
  --print_samples ${PRINT_SAMPLES} \
  --tmp_dir "${TMP_DIR}" \
  --out_metrics "${OUT_METRICS}" \
  --top_codes_csv "${TOP_CODES}" \
  --bottom_codes_csv "${BOT_CODES}" \
  --top_parent_csv "${TOP_PARENTS}" \
  $( [[ "${USE_BF16}" == "1" ]] && echo --use_bf16 )
'

status=$?
echo "[INFO] Test exit code: $status"
exit $status
