#!/bin/bash
#SBATCH --job-name=KG_codegen_prompt
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/codegenKGprompt/codegenKGprompt_%j.out

# module purge
# module load release/24.04 GCCcore/11.3.0
# module load Python/3.10.4

# PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
# cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

# source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
# echo "[INFO] Virtual env: $VIRTUAL_ENV"
# python -V

# export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SCRIPT="gen/withKG/codegen_withKG_prompt.py"
# TEST_PKL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl
# KG_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2
# ADAPTER_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/runs_gen/diffsize/20251015-132629_N54981_icd9_complete/adapter_best
# LLM=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/models/Llama-3.1-8B-Instruct

# # # Quick subset test: first 100 rows
# # SUBSET_N=100

# # start=$(date +%s)
# # echo "[INFO] Inference job started at $(date)"

# # python ${SCRIPT} \
# #   --test_pickle ${TEST_PKL} \
# #   --subset_n ${SUBSET_N} \
# #   --base_model ${LLM} \
# #   --adapter_dir ${ADAPTER_DIR} \
# #   --kg_pkl ${KG_DIR}/medical_knowledge_graph.pkl \
# #   --icd9_dx_map_pkl ${KG_DIR}/code2cui_icd9_dx.pkl \
# #   --icd9_proc_map_pkl ${KG_DIR}/code2cui_icd9_proc.pkl \
# #   --loinc_map_pkl ${KG_DIR}/code2cui_loinc.pkl \
# #   --atc_map_pkl ${KG_DIR}/code2cui_atc.pkl \
# #   --kg_hop 1 \
# #   --rel_whitelist "" \
# #   --rela_whitelist "" \
# #   --max_neighbors_show 24 \
# #   --max_candidates 32 \
# #   --max_len 3072 \
# #   --gen_max_new 96 \
# #   --batch_size 16 \
# #   --show_examples 5

# python ${SCRIPT} \
#   --test_pickle ${TEST_PKL} \
#   --base_model ${LLM} \
#   --adapter_dir ${ADAPTER_DIR} \
#   --kg_pkl ${KG_DIR}/medical_knowledge_graph.pkl \
#   --icd9_dx_map_pkl ${KG_DIR}/code2cui_icd9_dx.pkl \
#   --icd9_proc_map_pkl ${KG_DIR}/code2cui_icd9_proc.pkl \
#   --loinc_map_pkl ${KG_DIR}/code2cui_loinc.pkl \
#   --atc_map_pkl ${KG_DIR}/code2cui_atc.pkl \
#   --kg_hop 1 \
#   --max_len 3072 \
#   --assistant_reserve 128 \
#   --prompt_budget 2944 \
#   --notes_soft_budget 2718 \
#   --kg_soft_budget 226 \
#   --kg_lines_cap 64 \
#   --gen_max_new 96 \
#   --batch_size 16 \
#   --subset_n 100 \
#   --show_n 5

# EXIT_CODE=$?

# if [ $EXIT_CODE -eq 0 ]; then
#     echo "[INFO] Inference with KG hints completed successfully!"
# else
#     echo "[ERROR] Inference with KG hints failed with exit code: $EXIT_CODE"
# fi

# end=$(date +%s)
# echo "[INFO] Inference job finished at $(date)"
# echo "[TIME] Inference elapsed: $((end-start)) seconds"

# exit $EXIT_CODE

# SCRIPT="gen/withKG/trial.py"
# echo "[INFO] Running script: ${SCRIPT}"

# start=$(date +%s)
# echo "[INFO] Inference job started at $(date)"

# python ${SCRIPT}

# end=$(date +%s)
# echo "[INFO] Token budgetting job finished at $(date)"
# echo "[TIME] Token budgetting elapsed: $((end-start)) seconds"

# exit $EXIT_CODE


# 0) Modules/venv
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env: $VIRTUAL_ENV"
python -V

# 1) Paths
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

SCRIPT=gen/withKG/codegen_withKG_prompt.py
echo "[INFO] Running script: $SCRIPT"
mkdir -p logs/codegenKGprompt

TEST_PKL=$PROJECT_DIR/dataset/icd9/test_df.pkl

# Base + LoRA (adapter-only)
BASE_MODEL=$PROJECT_DIR/models/Llama-3.1-8B-Instruct
ADAPTER_DIR=$PROJECT_DIR/runs_gen/diffsize/20251015-132629_N54981_icd9_complete/adapter_best

# KG & maps (ATC fetched from 'ndc' col)
KG_DIR=$PROJECT_DIR/KG/kg_output2
KG_PKL=$KG_DIR/medical_knowledge_graph.pkl
DX_MAP=$KG_DIR/code2cui_icd9_dx.pkl
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
ATC_MAP=$KG_DIR/code2cui_atc.pkl

# 2) Runtime knobs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using $GPUS GPU(s)"

# Prompt budgets (new p95 split)
TOTAL_INPUT_BUDGET=3072
ASSISTANT_RESERVE=128
NOTES_SOFT_BUDGET=2307
KG_SOFT_BUDGET=637

# Generation
BATCH_SIZE=16
GEN_MAX_NEW=96

# KG expansion & filters
HOP=1                   # 0, 1, or 2
REL_WHITELIST=""        # e.g., "RO,RN"
RELA_WHITELIST=""       # e.g., "mapped_to"
MAX_NEIGHBORS_SHOW=24

# Quick test vs full
SUBSET_N=0              # 0 = all rows
SHOW_N=20

echo "[INFO] Inference started at $(date)"
echo "[INFO] Base model:      $BASE_MODEL"
echo "[INFO] Adapter dir:     $ADAPTER_DIR"
echo "[INFO] KG:              $KG_PKL"
echo "[INFO] Budgets:         total=$TOTAL_INPUT_BUDGET, reserve=$ASSISTANT_RESERVE, notes=$NOTES_SOFT_BUDGET, kg=$KG_SOFT_BUDGET"
echo "[INFO] Gen:             max_new=$GEN_MAX_NEW, batch_size=$BATCH_SIZE, hop=$HOP, neighbors_show=$MAX_NEIGHBORS_SHOW"
echo "[INFO] Subset:          $SUBSET_N rows (0=all), show_n=$SHOW_N"

start=$(date +%s)

python "$SCRIPT" \
  --test_pickle "$TEST_PKL" \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --kg_pkl "$KG_PKL" \
  --icd9_dx_map_pkl "$DX_MAP" \
  --icd9_proc_map_pkl "$PROC_MAP" \
  --loinc_map_pkl "$LOINC_MAP" \
  --atc_map_pkl "$ATC_MAP" \
  --hop $HOP \
  --rel_whitelist "$REL_WHITELIST" \
  --rela_whitelist "$RELA_WHITELIST" \
  --max_neighbors_show $MAX_NEIGHBORS_SHOW \
  --total_input_budget $TOTAL_INPUT_BUDGET \
  --assistant_reserve $ASSISTANT_RESERVE \
  --notes_soft_budget $NOTES_SOFT_BUDGET \
  --kg_soft_budget $KG_SOFT_BUDGET \
  --gen_max_new $GEN_MAX_NEW \
  --batch_size $BATCH_SIZE \
  --subset_n $SUBSET_N \
  --show_n $SHOW_N

EXIT_CODE=$?
end=$(date +%s)

echo "[INFO] Finished at $(date)"
echo "[TIME] Elapsed: $((end-start)) seconds"

if [ $EXIT_CODE -eq 0 ]; then
  echo "[INFO] Inference completed successfully."
else
  echo "[ERROR] Inference failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
