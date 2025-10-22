#!/bin/bash
#SBATCH --job-name=pipe_codegen_test
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/Codegen/test_codegen_%j.out

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "${PROJECT_DIR}" || { echo "Project dir not found: ${PROJECT_DIR}"; exit 1; }

### 3) Virtualenv
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"

### 4) Runtime envs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib,eth,^lo,docker"
export NCCL_DEBUG=INFO

echo "[INFO] GPUs allocated by SLURM: ${SLURM_GPUS_ON_NODE:-1}"

### 5) Paths & args (edit as needed)
# Run the tester as a module so `from .util_codegen_core` works:
SCRIPT_MOD=gen/pipeline/test_codegen.py

# Data
TEST_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl
LLAMA_MODEL=meta-llama/Llama-3.2-1B-Instruct

# Point to a COMPLETED CodeGen run (change RUN_NAME or pass via sbatch --export)
RUN_ROOT=runs_codegen
RUN_NAME=${RUN_NAME:-exp_1013575}   # e.g., exp_12345678 from training

ADAPTER_DIR=${RUN_ROOT}/${RUN_NAME}/adapter_best
LABELS_JSON=${RUN_ROOT}/${RUN_NAME}/label_space.json
OUT_DIR=${RUN_ROOT}/${RUN_NAME}/eval_${SLURM_JOB_ID}

# Generation/eval settings
MAX_LEN=3072
GEN_MAX_NEW=96
TEST_BATCH_SIZE=16
USE_STRUCTURED=1
USE_NOTES=1
SEED=42

# Optional bucket files
TOP_CODES_CSV=lists/top_50_codes.csv
BOTTOM_CODES_CSV=lists/bottom_50_codes.csv
TOP_PARENTS_CSV=lists/top_50_parents.csv

### 6) Sanity checks
[[ -f "${TEST_PKL}" ]] || { echo "[ERR] TEST_PKL not found: ${TEST_PKL}"; exit 2; }
[[ -d "${ADAPTER_DIR}" ]] || { echo "[ERR] ADAPTER_DIR not found: ${ADAPTER_DIR}"; exit 2; }
[[ -f "${LABELS_JSON}" ]] || { echo "[ERR] LABELS_JSON not found: ${LABELS_JSON}"; exit 2; }
mkdir -p "${OUT_DIR}"

### 7) Launch tester (single process; no torchrun)
start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] Running: ${SCRIPT_MOD}"
echo "[INFO] CodeGen TEST -> ADAPTER=${ADAPTER_DIR}"
echo "[INFO] Outputs -> ${OUT_DIR}"

srun python ${SCRIPT_MOD} \
  --test_pickle "${TEST_PKL}" \
  --llama_model "${LLAMA_MODEL}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --labels_json "${LABELS_JSON}" \
  --max_len "${MAX_LEN}" \
  --gen_max_new "${GEN_MAX_NEW}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --use_structured "${USE_STRUCTURED}" \
  --use_notes "${USE_NOTES}" \
  --seed "${SEED}" \
  --top_codes_csv "${TOP_CODES_CSV}" \
  --bottom_codes_csv "${BOTTOM_CODES_CSV}" \
  --top_parent_csv "${TOP_PARENTS_CSV}" \
  --out_dir "${OUT_DIR}" \
  --test_examples 5

status=$?
end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) s"
echo "[INFO] Test exit code: $status"
echo "[INFO] Expected outputs:"
echo "  - ${OUT_DIR}/test_metrics.json"
echo "  - ${OUT_DIR}/test_metrics_buckets.json"
exit $status