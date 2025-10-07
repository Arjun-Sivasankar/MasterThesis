#!/bin/bash
#SBATCH --job-name=icd_pipeline
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline/%x_%j.out

# =========================================================
# HOW TO USE
#   sbatch --export=ALL,TASK=textgen,STAGE=train run_pipeline.sh
#   sbatch --export=ALL,TASK=textgen,STAGE=test,DISTRIBUTED_TEST=1 run_pipeline.sh
#   sbatch --export=ALL,TASK=codegen,STAGE=train run_pipeline.sh
#   sbatch --export=ALL,TASK=codegen,STAGE=test run_pipeline.sh
#
# Overrides (optional): set any CONFIG var below via --export=ALL,VAR=...
# =========================================================

set -euo pipefail

############################
# 1) MODULES
############################
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

############################
# 2) PROJECT DIR
############################
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "${PROJECT_DIR}" || { echo "[ERR] Project dir not found: ${PROJECT_DIR}"; exit 1; }

############################
# 3) VENV
############################
VENV=/data/horse/ws/arsi805e-venv/venvs/finetune
source "${VENV}/bin/activate"
echo "[INFO] Virtual env: $VIRTUAL_ENV"

############################
# 4) RUNTIME ENVS
############################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

GPUS=${SLURM_GPUS_ON_NODE:-3}
echo "[INFO] GPUs from SLURM: $GPUS"

############################
# 5) CONFIG (edit or override via --export)
############################
# Which part of the pipeline?
TASK=${TASK:-textgen}            # textgen | codegen
STAGE=${STAGE:-train}            # train | test

# If you want the TextGen tester to shard across GPUs:
DISTRIBUTED_TEST=${DISTRIBUTED_TEST:-0}   # 0 | 1

# Data & resources
DATA_PKL=${DATA_PKL:-${PROJECT_DIR}/dataset/merged_icd9.pkl}
TEST_PKL=${TEST_PKL:-${DATA_PKL}}         # use same file by default; add --test_only inside test script if needed
ICD_INDEX=${ICD_INDEX:-./gen/TextGen/icd_index_v9}
BASE_LLM=${BASE_LLM:-meta-llama/Llama-3.2-1B-Instruct}

# Lists for buckets (optional; empty is fine)
TOP_CODES_CSV=${TOP_CODES_CSV:-lists/top_50_codes.csv}
BOTTOM_CODES_CSV=${BOTTOM_CODES_CSV:-lists/bottom_50_codes.csv}
TOP_PARENTS_CSV=${TOP_PARENTS_CSV:-lists/top_50_parents.csv}

# Output roots
TEXTGEN_OUT_ROOT=${TEXTGEN_OUT_ROOT:-runs_textgen}
CODEGEN_OUT_ROOT=${CODEGEN_OUT_ROOT:-runs_codegen}

# Script paths (point these to your integrated train/test scripts)
TRAIN_SCRIPT=${TRAIN_SCRIPT:-gen/pipeline/train.py}    # supports --task textgen|codegen
TEST_SCRIPT=${TEST_SCRIPT:-gen/pipeline/test.py}       # supports --task textgen|codegen

# Training/test hyperparams (override as desired)
EPOCHS_TEXTGEN=${EPOCHS_TEXTGEN:-4}
EPOCHS_CODEGEN=${EPOCHS_CODEGEN:-6}
LR=${LR:-2e-4}
GRAD_ACCUM=${GRAD_ACCUM:-16}
PER_DEV_BS=${PER_DEV_BS:-1}
WARMUP=${WARMUP:-0.03}
N_MAX_TERMS=${N_MAX_TERMS:-12}
MIN_ASSISTANT_TOK=${MIN_ASSISTANT_TOK:-128}
MAX_LEN=${MAX_LEN:-3072}

GEN_MAX_NEW=${GEN_MAX_NEW:-96}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-8}

# Where to save adapters/runs
TEXTGEN_OUT_DIR=${TEXTGEN_OUT_DIR:-${TEXTGEN_OUT_ROOT}/checkpoints}
TEXTGEN_ADAPTER_DIR=${TEXTGEN_ADAPTER_DIR:-${TEXTGEN_OUT_ROOT}/adapter_${SLURM_JOB_ID}}

CODEGEN_RUN_ROOT=${CODEGEN_RUN_ROOT:-${CODEGEN_OUT_ROOT}}
CODEGEN_RUN_NAME=${CODEGEN_RUN_NAME:-exp_${SLURM_JOB_ID}}

# If testing textgen or codegen, point to trained artifacts:
# For textgen: ADAPTER from training (change as needed)
TEXTGEN_ADAPTER_FOR_TEST=${TEXTGEN_ADAPTER_FOR_TEST:-${TEXTGEN_OUT_ROOT}/adapter_v1}
# For codegen: use a finished run dir's adapter + label space
CODEGEN_ADAPTER_FOR_TEST=${CODEGEN_ADAPTER_FOR_TEST:-${CODEGEN_RUN_ROOT}/${CODEGEN_RUN_NAME}/adapter_best}
CODEGEN_LABELS_JSON_FOR_TEST=${CODEGEN_LABELS_JSON_FOR_TEST:-${CODEGEN_RUN_ROOT}/${CODEGEN_RUN_NAME}/label_space.json}

# Output for tests
TEXTGEN_TEST_DIR=${TEXTGEN_TEST_DIR:-${TEXTGEN_OUT_ROOT}/test_${SLURM_JOB_ID}}
CODEGEN_TEST_DIR=${CODEGEN_TEST_DIR:-${CODEGEN_OUT_ROOT}/${CODEGEN_RUN_NAME}/eval_${SLURM_JOB_ID}}

############################
# 6) DISPATCH
############################
start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] TASK=$TASK, STAGE=$STAGE, DISTRIBUTED_TEST=$DISTRIBUTED_TEST"
echo "[INFO] DATA_PKL=$DATA_PKL"

mkdir -p logs "${TEXTGEN_OUT_ROOT}" "${CODEGEN_OUT_ROOT}"

# -------------------------------
# TEXTGEN: TRAIN
# -------------------------------
if [[ "$TASK" == "textgen" && "$STAGE" == "train" ]]; then
  echo "[INFO] TextGen TRAIN -> OUT=${TEXTGEN_OUT_DIR}  ADAPTER=${TEXTGEN_ADAPTER_DIR}"
  srun torchrun --standalone --nproc_per_node=${GPUS} "${TRAIN_SCRIPT}" \
    --task textgen \
    --data_pickle "${DATA_PKL}" \
    --llm "${BASE_LLM}" \
    --target_mode icd_titles \
    --icd_index_dir "${ICD_INDEX}" \
    --epochs "${EPOCHS_TEXTGEN}" \
    --per_device_train_batch_size "${PER_DEV_BS}" \
    --per_device_eval_batch_size "${PER_DEV_BS}" \
    --grad_accum "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP}" \
    --N_max_terms "${N_MAX_TERMS}" \
    --min_assistant_tokens "${MIN_ASSISTANT_TOK}" \
    --out_dir "${TEXTGEN_OUT_DIR}" \
    --save_adapter \
    --adapter_dir "${TEXTGEN_ADAPTER_DIR}"
fi

# -------------------------------
# TEXTGEN: TEST
# -------------------------------
if [[ "$TASK" == "textgen" && "$STAGE" == "test" ]]; then
  OUT_METRICS=${TEXTGEN_TEST_DIR}/test_metrics.json
  TMP_DIR=${TEXTGEN_TEST_DIR}/shards
  mkdir -p "${TEXTGEN_TEST_DIR}"

  echo "[INFO] TextGen TEST -> ADAPTER=${TEXTGEN_ADAPTER_FOR_TEST}"
  if [[ "${DISTRIBUTED_TEST}" == "1" ]]; then
    srun torchrun --standalone --nproc_per_node=${GPUS} "${TEST_SCRIPT}" \
      --task textgen \
      --data_pickle "${DATA_PKL}" \
      --base_model "${BASE_LLM}" \
      --adapter_dir "${TEXTGEN_ADAPTER_FOR_TEST}" \
      --icd_index_dir "${ICD_INDEX}" \
      --decoding greedy \
      --gen_batch_size "${GEN_BATCH_SIZE}" \
      --top_codes_csv "${TOP_CODES_CSV}" \
      --bottom_codes_csv "${BOTTOM_CODES_CSV}" \
      --top_parent_csv "${TOP_PARENTS_CSV}" \
      --distributed \
      --tmp_dir "${TMP_DIR}" \
      --out_metrics "${OUT_METRICS}"
  else
    srun python "${TEST_SCRIPT}" \
      --task textgen \
      --data_pickle "${DATA_PKL}" \
      --base_model "${BASE_LLM}" \
      --adapter_dir "${TEXTGEN_ADAPTER_FOR_TEST}" \
      --icd_index_dir "${ICD_INDEX}" \
      --decoding greedy \
      --gen_batch_size "${GEN_BATCH_SIZE}" \
      --top_codes_csv "${TOP_CODES_CSV}" \
      --bottom_codes_csv "${BOTTOM_CODES_CSV}" \
      --top_parent_csv "${TOP_PARENTS_CSV}" \
      --out_metrics "${OUT_METRICS}"
  fi
fi

# -------------------------------
# CODEGEN: TRAIN
# -------------------------------
if [[ "$TASK" == "codegen" && "$STAGE" == "train" ]]; then
  echo "[INFO] CodeGen TRAIN -> RUN_ROOT=${CODEGEN_RUN_ROOT} RUN_NAME=${CODEGEN_RUN_NAME}"
  srun python "${TRAIN_SCRIPT}" \
    --task codegen \
    --data_pickle "${DATA_PKL}" \
    --llama_model "${BASE_LLM}" \
    --use_structured 1 \
    --use_notes 1 \
    --epochs "${EPOCHS_CODEGEN}" \
    --per_device_train_batch_size "${PER_DEV_BS}" \
    --per_device_eval_batch_size "${PER_DEV_BS}" \
    --grad_accum "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP}" \
    --run_root "${CODEGEN_RUN_ROOT}" \
    --run_name "${CODEGEN_RUN_NAME}" \
    --tgt_reserve_tok 128 \
    --max_len "${MAX_LEN}"
fi

# -------------------------------
# CODEGEN: TEST
# -------------------------------
if [[ "$TASK" == "codegen" && "$STAGE" == "test" ]]; then
  mkdir -p "${CODEGEN_TEST_DIR}"
  echo "[INFO] CodeGen TEST -> ADAPTER=${CODEGEN_ADAPTER_FOR_TEST}"
  srun python "${TEST_SCRIPT}" \
    --task codegen \
    --test_pickle "${TEST_PKL}" \
    --llama_model "${BASE_LLM}" \
    --adapter_dir "${CODEGEN_ADAPTER_FOR_TEST}" \
    --labels_json "${CODEGEN_LABELS_JSON_FOR_TEST}" \
    --use_structured 1 \
    --use_notes 1 \
    --test_batch_size 16 \
    --gen_max_new "${GEN_MAX_NEW}" \
    --top_codes_csv "${TOP_CODES_CSV}" \
    --bottom_codes_csv "${BOTTOM_CODES_CSV}" \
    --top_parent_csv "${TOP_PARENTS_CSV}" \
    --out_dir "${CODEGEN_TEST_DIR}" \
    --test_examples 5
fi

end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) s"
echo "[INFO] Exit code: $?"
