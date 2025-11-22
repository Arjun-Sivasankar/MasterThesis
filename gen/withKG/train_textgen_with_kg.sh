#!/bin/bash
#SBATCH --job-name=train_KG
#SBATCH --partition=capella
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/Textgen-withKG/train_textgen_KG_%j.out
#SBATCH --licenses=horse

### 1) Modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

### 2) Project dir
PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR" || { echo "Project dir not found"; exit 1; }

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

GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[INFO] Using $GPUS GPUs for training"

# --- Use the new script ---
SCRIPT=$PROJECT_DIR/gen/withKG/train_textgen_with_kg.py

### 5A) Common Paths
TRAIN_DATA_PKL=$PROJECT_DIR/dataset/final_data/train_df.pkl
VAL_DATA_PKL=$PROJECT_DIR/dataset/final_data/val_df.pkl
ICD_INDEX=./gen/TextGen/icd_index_v9

# --- KG Paths (Simplified) ---
KG_DIR=$PROJECT_DIR/KG/kg_output4
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl
ATC_MAP=$KG_DIR/code2cui_atc.pkl

# --- Model & Epochs ---
EPOCHS=10
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# BASE_LLM=models/Llama-3.1-8B-Instruct

# --- Debugging ---
SUBSET_N=10000  # Set to 0 for full run
VAL_N=100    # Set to 0 for full val set
LOGGING_STEPS=50 # Set to 50 for full run

# --- CRITICAL: Budgets & Static Config ---
MAX_LEN=5120
NOTES_SOFT_BUDGET=3008
N_MAX_TERMS=12 # Static cap for all prompts

#-----------------------------------------------------------------------
### 5B) ABLATION SETTINGS
#
# Choose ONE block to uncomment. This defines the model you are training.
#
# KG_SOFT_BUDGET: Set to 0 for "Baseline" (No KG)
#                 Set to >0 for "Method 1" (With KG)
#
# STRUCTURED_FORMAT: "names" (Natural Language) or "codes"
#-----------------------------------------------------------------------

# === OPTION 1: Method 1 (Names) ===
# (This is the recommended full model: KG + NL Names)
# echo "[INFO] TRAINING: Method 1 (KG + Names)"
# KG_SOFT_BUDGET=1500
# KG_BLOCK="both"
# KG_H2_RATIO=0.7
# STRUCTURED_FORMAT="names"
# OUT_DIR=runs_textgen/checkpoints_KG_Names
# ADAPTER_DIR=runs_textgen/adapter_KG_Names

# === OPTION 2: Method 1 (Codes) ===
# (Ablation: KG + Raw Codes)
echo "[INFO] TRAINING: Method 1 (KG + Codes)"
KG_SOFT_BUDGET=1500
KG_BLOCK="both"
KG_H2_RATIO=0.7
STRUCTURED_FORMAT="codes"
OUT_DIR=runs_textgen/checkpoints_KG_Codes_10000
ADAPTER_DIR=runs_textgen/adapter_KG_Codes_10000

# === OPTION 3: Baseline (Names) ===
# (Ablation: No KG + NL Names)
# echo "[INFO] TRAINING: Baseline (No KG + Names)"
# KG_SOFT_BUDGET=0
# KG_BLOCK="both" # (Ignored)
# KG_H2_RATIO=0.7 # (Ignored)
# STRUCTURED_FORMAT="names"
# OUT_DIR=runs_textgen/checkpoints_Baseline_Names
# ADAPTER_DIR=runs_textgen/adapter_Baseline_Names

# === OPTION 4: Baseline (Codes) ===
# (Ablation: No KG + Raw Codes)
# echo "[INFO] TRAINING: Baseline (No KG + Codes)"
# KG_SOFT_BUDGET=0
# KG_BLOCK="both" # (Ignored)
# KG_H2_RATIO=0.7 # (Ignored)
# STRUCTURED_FORMAT="codes"
# OUT_DIR=runs_textgen/checkpoints_Baseline_Codes
# ADAPTER_DIR=runs_textgen/adapter_Baseline_Codes

#-----------------------------------------------------------------------

echo "[INFO] Training for Epochs: ${EPOCHS}"
echo "[INFO] Using base LLM: ${BASE_LLM}"
echo "[INFO] Subset N: ${SUBSET_N}"
echo "[INFO] Static N_max_terms: ${N_MAX_TERMS}"
echo "[INFO] KG Budget: ${KG_SOFT_BUDGET}"
echo "[INFO] Structured Format: ${STRUCTURED_FORMAT}"
echo "[INFO] Adapter Dir: ${ADAPTER_DIR}"

### 6) Run: multi-GPU training (DDP) via torchrun
start=$(date +%s)
echo "[INFO] Job started: $(date)"
echo "[INFO] Running Script: ${SCRIPT}"
echo "[INFO] Launching training…"

srun torchrun --standalone --nproc_per_node=${GPUS} ${SCRIPT} \
  --train_data "${TRAIN_DATA_PKL}" \
  --val_data "${VAL_DATA_PKL}" \
  --subset_n ${SUBSET_N} \
  --val_n ${VAL_N} \
  --llm "${BASE_LLM}" \
  --target_mode icd_titles \
  --icd_index_dir "${ICD_INDEX}" \
  --epochs ${EPOCHS} \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --grad_accum 16 \
  --logging_steps ${LOGGING_STEPS} \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --N_max_terms ${N_MAX_TERMS} \
  --min_assistant_tokens 128 \
  --structured_format "${STRUCTURED_FORMAT}" \
  --save_adapter \
  \
  --out_dir "${OUT_DIR}" \
  --adapter_dir "${ADAPTER_DIR}" \
  \
  --max_len ${MAX_LEN} \
  --notes_soft_budget ${NOTES_SOFT_BUDGET} \
  --kg_soft_budget ${KG_SOFT_BUDGET} \
  --kg_block "${KG_BLOCK}" \
  --kg_h2_ratio ${KG_H2_RATIO} \
  --kg_k1 30 \
  --kg_k2 30 \
  \
  --kg_pkl "$KG_PKL" \
  --icd9_proc_map_pkl "$PROC_MAP" \
  --loinc_map_pkl "$LOINC_MAP" \
  --atc_map_pkl "$ATC_MAP"

status=$?
end=$(date +%s)
echo "[TIME] Elapsed: $((end-start)) s"
echo "[INFO] Train exit code: $status"
exit $status