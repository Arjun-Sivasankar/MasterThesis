#!/bin/bash
#SBATCH --job-name=retrieve_h1_train
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/History_Aware/prompts_kg_paths_%j.out

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

SCRIPT="gen/withKG/history_aware/build_history_aware_prompts_with_kg.py"
echo "[INFO] Running Script: ${SCRIPT}"
start_time=$(date +%s)

dataset_dir=dataset/history_aware_data

python ${SCRIPT} \
  --tsv ${dataset_dir}/prompts/train_prompts.tsv \
  --paths_jsonl ${dataset_dir}/jsonl_output_final/train_h2.jsonl \
  --modular_jsonl ${dataset_dir}/jsonl_output_final/train_modular.jsonl \
  --out_tsv ${dataset_dir}/prompts/train_prompts_h2_unweighted.tsv \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --max_kg_tokens 1500

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "[INFO] Retrieval completed for 'train' split in ${duration} seconds"
echo "[INFO] Successfully completed in ${duration} seconds"