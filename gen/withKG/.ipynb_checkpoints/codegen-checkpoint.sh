#!/bin/bash
#SBATCH --job-name=KG_codegen
#SBATCH --account=p_scads_finetune
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/codegenKG/codegenKG_%j.out

# # --- 1) Modules ---
# module purge
# module load release/24.04 GCCcore/11.3.0
# module load Python/3.10.4

# # --- 2) Project dir ---
# PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
# cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

# # --- 3) Venv ---
# source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
# echo "[INFO] Virtual env: $VIRTUAL_ENV"
# python -V

# # --- 4) Environment ---
# export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPUS=${SLURM_GPUS_ON_NODE:-1}
# echo "[INFO] Using $GPUS GPU(s)"

# # --- 5) Paths ---
# SCRIPT="gen/withKG/codegen_withKG.py"

# ICD9_TRAIN=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/train_df.pkl
# ICD9_VAL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/val_df.pkl
# ICD9_ALLCODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl

# model_dir=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_gen/diffsize/20251015-132629_N54981_icd9_complete/adapter_best
# echo "[INFO] Using adapter model from: ${model_dir}"

# # LLM=meta-llama/Llama-3.2-1B-Instruct
# LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
# echo "[INFO] Using LLM model: ${LLM}"

# start=$(date +%s)
# echo "[INFO] Inference job started at $(date)"

# echo "[INFO] Starting ICD-9 inference with KG..."
# echo "[INFO] Inference script: $SCRIPT"

# python ${SCRIPT} \
#   --test_pickle /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl \
#   --base_model ${LLM} \
#   --adapter_dir ${model_dir} \
#   --kg_pkl /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/medical_knowledge_graph.pkl \
#   --icd9_dx_map_pkl /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/code2cui_icd9_dx.pkl \
#   --icd9_proc_map_pkl /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/code2cui_icd9_proc.pkl \
#   --loinc_map_pkl /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/code2cui_loinc.pkl \
#   --kg_hop 0 \
#   --kg_strategy hard_filter \
#   --batch_size 16

# TRAIN_EXIT_CODE=$?

# if [ $TRAIN_EXIT_CODE -eq 0 ]; then
#     echo "[INFO] Inference with KG completed successfully!"
# else
#     echo "[ERROR] Inference with KG failed with exit code: $TRAIN_EXIT_CODE"
# fi

# end=$(date +%s)
# echo "[INFO] Inference job finished at $(date)"
# echo "[TIME] Inference elapsed: $((end-start)) seconds"

# exit $TRAIN_EXIT_CODE



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
SCRIPT="gen/withKG/codegen_withKG_prompt.py"

TEST_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl
MERGED_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl
ICD9_ALLCODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl

TOP_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_codes.csv
BOT_CODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/bottom_50_codes.csv

ADAPTER_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_gen/diffsize/20251015-132629_N54981_icd9_complete/adapter_best
echo "[INFO] Using adapter model from: ${ADAPTER_DIR}"

LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct
echo "[INFO] Using LLM model: ${LLM}"

KG_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2

start=$(date +%s)
echo "[INFO] Inference job started at $(date)"
echo "[INFO] Starting ICD-9 inference with KG..."
echo "[INFO] Inference script: $SCRIPT"

# python ${SCRIPT} \
#   --test_pickle ${TEST_PKL} \
#   --data_pickle ${MERGED_PKL} \
#   --icd9_pickle ${ICD9_ALLCODES} \
#   --top_codes_csv ${TOP_CODES} \
#   --bot_codes_csv ${BOT_CODES} \
#   --base_model ${LLM} \
#   --adapter_dir ${ADAPTER_DIR} \
#   --kg_pkl ${KG_DIR}/medical_knowledge_graph.pkl \
#   --icd9_dx_map_pkl ${KG_DIR}/code2cui_icd9_dx.pkl \
#   --icd9_proc_map_pkl ${KG_DIR}/code2cui_icd9_proc.pkl \
#   --loinc_map_pkl ${KG_DIR}/code2cui_loinc.pkl \
#   --atc_map_pkl ${KG_DIR}/code2cui_atc.pkl \
#   --kg_hop 0 \
#   --kg_strategy hard_filter \
#   --batch_size 16 \
#   --show_examples 10 \
#   --examples_seed 123

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] Inference with KG completed successfully!"
else
    echo "[ERROR] Inference with KG failed with exit code: $EXIT_CODE"
fi

end=$(date +%s)
echo "[INFO] Inference job finished at $(date)"
echo "[TIME] Inference elapsed: $((end-start)) seconds"
exit $EXIT_CODE
