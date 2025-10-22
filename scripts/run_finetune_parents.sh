#!/bin/bash
#SBATCH --job-name=codegen_parent_leaf
#SBATCH --partition=capella
#SBATCH --gres=gpu:3                   # set to the number of GPUs you actually want
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_parent_ddp_%j.out

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

# --- 4) PyTorch / Tokenizers knobs ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# reduce CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 5) GPU selection ---
# If Slurm exports the GPU count, use it; else default to 2
GPUS=${SLURM_GPUS_ON_NODE:-2}
echo "[INFO] Using $GPUS GPU(s)"

# Optional manual pinning (uncomment & match --gres): 
# export CUDA_VISIBLE_DEVICES=0,1

# --- 6) Paths & script ---
# Save the single Python script I sent you as:
#   $PROJECT_DIR/gen/finetune_parent_leaf_lora.py
SCRIPT="gen/finetune_llama_gen_ddp_parents.py"
echo "[INFO] Script: $SCRIPT"

# Common data paths
ICD9_TRAIN=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/train_df.pkl
ICD9_VAL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/val_df.pkl
ICD9_TEST=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd9/test_df.pkl
ICD9_ALLCODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl

ICD10_TRAIN=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/train_df.pkl
ICD10_VAL=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/val_df.pkl
ICD10_TEST=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/icd10/test_df.pkl
ICD10_ALLCODES=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd10.pkl

# Your local model path (or HF Hub id if allowed)
LLM=meta-llama/Llama-3.2-1B-Instruct

mkdir -p logs runs_gen

start=$(date +%s)
echo "[INFO] Job started at $(date)"

#############################################
# RUN 1: ICD-9, PARENT training/eval (default)
#############################################
echo "[INFO] Starting ICD-9 PARENT training..."
srun torchrun --standalone --nproc_per_node=${GPUS} "${SCRIPT}" \
  --llama_model "${LLM}" \
  --train_pickle  "${ICD9_TRAIN}" \
  --val_pickle    "${ICD9_VAL}" \
  --test_pickle   "${ICD9_TEST}" \
  --subject_col   subject_id_x \
  --label_col     icd_code \
  --icd_scheme    icd9cm \
  --icd_level     parent \
  --codes_pickle  "${ICD9_ALLCODES}" \
  --use_complete_codes 1 \
  --train_size 54981 \
  --eval_sample_size 100 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --test_batch_size 16 \
  --grad_accum 16 \
  --epochs 1 \
  --run_root runs_gen/icd9_parent

#############################################
# RUN 2: (Optional) ICD-9, LEAF training/eval
#############################################
# echo "[INFO] Starting ICD-9 LEAF training..."
# srun torchrun --standalone --nproc_per_node=${GPUS} "${SCRIPT}" \
#   --llama_model "${LLM}" \
#   --train_pickle  "${ICD9_TRAIN}" \
#   --val_pickle    "${ICD9_VAL}" \
#   --test_pickle   "${ICD9_TEST}" \
#   --subject_col   subject_id_x \
#   --label_col     icd_code \
#   --icd_scheme    icd9cm \
#   --icd_level     leaf \
#   --codes_pickle  "${ICD9_ALLCODES}" \
#   --use_complete_codes 1 \
#   --train_size 54981 \
#   --eval_sample_size 100 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --test_batch_size 16 \
#   --grad_accum 16 \
#   --epochs 6 \
#   --run_root runs_gen/icd9_leaf

#############################################
# RUN 3: (Optional) ICD-10, PARENT training/eval
#############################################
# echo "[INFO] Starting ICD-10 PARENT training..."
# srun torchrun --standalone --nproc_per_node=${GPUS} "${SCRIPT}" \
#   --llama_model "${LLM}" \
#   --train_pickle  "${ICD10_TRAIN}" \
#   --val_pickle    "${ICD10_VAL}" \
#   --test_pickle   "${ICD10_TEST}" \
#   --subject_col   subject_id_x \
#   --label_col     icd_code \
#   --icd_scheme    icd10cm \
#   --icd_level     parent \
#   --codes_pickle  "${ICD10_ALLCODES}" \
#   --use_complete_codes 1 \
#   --train_size 40000 \
#   --eval_sample_size 100 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --test_batch_size 16 \
#   --grad_accum 16 \
#   --epochs 6 \
#   --run_root runs_gen/icd10_parent

end=$(date +%s)
echo "[INFO] Job finished at $(date)"
echo "[TIME] Elapsed: $((end-start)) seconds"

status=$?
echo "[INFO] Exit code: $status"
echo "[INFO] Script run: $SCRIPT"
exit $status