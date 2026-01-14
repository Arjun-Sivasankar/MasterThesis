#!/bin/bash
#SBATCH --job-name=create_prompt
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/create_jsonl/create_prompt_%j.out

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

# --- 5) Create log directory ---
LOG_DIR="${PROJECT_DIR}/logs/create_jsonl"
mkdir -p ${LOG_DIR}
echo "[INFO] Log directory: ${LOG_DIR}"

JSONL_PATH=dataset/history_aware_data/jsonl_output_final/test_modular.jsonl
TSV_PATH=dataset/history_aware_data/prompts/test_prompts.tsv
echo "[INFO] JSONL Path: ${JSONL_PATH}"
echo "[INFO] TSV Path: ${TSV_PATH}"

# --- 6) Paths ---
SCRIPT="gen/withKG/history_aware/history_aware_prompt_builder.py"
echo "[INFO] Running Script: ${SCRIPT}"
python ${SCRIPT} --jsonl_path "${JSONL_PATH}" --tsv_path "${TSV_PATH}"

echo "[INFO] Successfully completed"