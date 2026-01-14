#!/bin/bash
#SBATCH --job-name=Preproc
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=14             # Increased CPUs for faster data processing
#SBATCH --nodes=1
#SBATCH --mem=96G                      # Increased RAM for large dataframes
#SBATCH --time=24:00:00                # Max walltime
#SBATCH --output=logs/preprocess/struc_%j.out # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# --- 4) Create log directory if it doesn't exist ---
mkdir -p logs/preprocess

# --- 5) Create dataset directories if they don't exist ---
mkdir -p dataset/icd9
mkdir -p dataset/icd10

# --- 6) Check available disk space and memory ---
echo "[INFO] Available disk space: $(df -h . | awk 'NR==2 {print $4}') remaining"
echo "[INFO] Total memory: $(free -h | awk '/Mem:/ {print $2}') with $(free -h | awk '/Mem:/ {print $4}') available"

# --- 7) Run preprocessing for ICD-9 ---
echo "[INFO] Starting MIMIC-IV preprocessing with ICD-9 codes..."
start_icd9=$(date +%s)

python -u dataset/dataset_str.py \
  --data_path /data/horse/ws/arsi805e-finetune/Thesis/mimic-iv-3.1/hosp \
  --use_icd10 0 \
  --verbose 1 \
  --output_dir ./dataset/icd9

end_icd9=$(date +%s)
echo "[TIME] ICD-9 Preprocessing completed in $((end_icd9-start_icd9)) seconds"

# --- 8) Run preprocessing for ICD-10 ---
echo "[INFO] Starting MIMIC-IV preprocessing with ICD-10 codes..."
start_icd10=$(date +%s)

python -u dataset/dataset_str.py \
  --data_path /data/horse/ws/arsi805e-finetune/Thesis/mimic-iv-3.1/hosp \
  --use_icd10 1 \
  --verbose 1 \
  --output_dir ./dataset/icd10

end_icd10=$(date +%s)
echo "[TIME] ICD-10 Preprocessing completed in $((end_icd10-start_icd10)) seconds"

# --- 9) Check output file sizes ---
echo "[INFO] ICD-9 dataset size: $(du -sh dataset/icd9)"
echo "[INFO] ICD-10 dataset size: $(du -sh dataset/icd10)"

# --- 10) Compress timing reports ---
echo "[INFO] Compressing timing reports..."
find dataset -name "timing_report_*.json" -exec gzip {} \;

# --- 11) Exit status ---
status=$?
echo "[INFO] Job finished with exit code $status"
exit $status