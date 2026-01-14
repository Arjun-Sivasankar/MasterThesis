#!/bin/bash
#SBATCH --job-name=create_jsonl
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/create_jsonl/create_jsonl_%j.out

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

# --- 6) Paths ---
SCRIPT="gen/withKG/history_aware/history_aware_create_jsonl.py"

DATA_DIR="${PROJECT_DIR}/dataset/history_aware_data"
KG_PATH="${PROJECT_DIR}/KG/kg_output/medical_knowledge_graph2.pkl"
OUTPUT_DIR="${DATA_DIR}/jsonl_output_final"

echo "[INFO] Data directory: ${DATA_DIR}"
echo "[INFO] KG path: ${KG_PATH}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"

# --- 7) Verify input files exist ---
echo "[INFO] Verifying input files..."

if [ ! -f "${KG_PATH}" ]; then
    echo "[ERROR] Knowledge graph not found: ${KG_PATH}"
    exit 1
fi

TRAIN_PKL="${DATA_DIR}/train_df.pkl"
VAL_PKL="${DATA_DIR}/val_df.pkl"
TEST_PKL="${DATA_DIR}/test_df.pkl"

MISSING_FILES=0
for pkl_file in "${TRAIN_PKL}" "${VAL_PKL}" "${TEST_PKL}"; do
    if [ ! -f "${pkl_file}" ]; then
        echo "[WARNING] Input file not found: ${pkl_file}"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "[INFO] Found input file: ${pkl_file}"
    fi
done

if [ ${MISSING_FILES} -eq 3 ]; then
    echo "[ERROR] No input files found. Exiting."
    exit 1
fi

# --- 8) Create output directory ---
mkdir -p ${OUTPUT_DIR}

# --- 9) Run the script ---
start=$(date +%s)
echo "[INFO] JSONL creation job started at $(date)"
echo "[INFO] Script: ${SCRIPT}"
echo "[INFO] Processing splits: train, val, test"

python ${SCRIPT}

EXIT_CODE=$?

# --- 10) Check results ---
if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] JSONL creation completed successfully!"
    
    # Count lines in output files
    echo ""
    echo "[INFO] Output file statistics:"
    for split in train val test; do
        jsonl_file="${OUTPUT_DIR}/${split}_modular.jsonl"
        if [ -f "${jsonl_file}" ]; then
            line_count=$(wc -l < "${jsonl_file}")
            file_size=$(du -h "${jsonl_file}" | cut -f1)
            echo "  - ${split}_modular.jsonl: ${line_count} samples, ${file_size}"
            
            # Show first sample structure
            if [ ${line_count} -gt 0 ]; then
                echo ""
                echo "  Sample from ${split}_modular.jsonl:"
                head -n 1 "${jsonl_file}" | python -m json.tool 2>/dev/null | head -n 30
                echo "  ..."
            fi
        else
            echo "  - ${split}_modular.jsonl: NOT FOUND"
        fi
    done
else
    echo "[ERROR] JSONL creation failed with exit code: ${EXIT_CODE}"
fi

# --- 11) Summary ---
end=$(date +%s)
elapsed=$((end - start))
echo ""
echo "[INFO] JSONL creation job finished at $(date)"
echo "[TIME] Total elapsed time: ${elapsed} seconds ($((elapsed / 60)) minutes)"

# --- 12) Disk usage ---
echo ""
echo "[INFO] Output directory disk usage:"
du -sh ${OUTPUT_DIR}
echo ""
echo "[INFO] Individual file sizes:"
du -h ${OUTPUT_DIR}/*

exit $EXIT_CODE