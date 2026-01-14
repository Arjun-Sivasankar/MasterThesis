#!/bin/bash
#SBATCH --job-name=build_historyaware_indexes
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/History_Aware_RAG/build_historyaware_indexes_%j.out
#SBATCH --licenses=horse

set -e
set -u

echo "========================================"
echo "BUILDING HISTORY-AWARE MEDICAL FACT INDEXES"
echo "Sources: Medications, Labs, Procedures, AND Past Diagnoses"
echo "Targets: Current Diagnosis CUIs (from pkl)"
echo "Deduplication: Enabled"
echo "Limits: Per-Source AND Per-Target"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

PROJECT_DIR=/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis
cd "$PROJECT_DIR"

source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

### Paths
SCRIPT=$PROJECT_DIR/gen/withKG/history_aware/build_medical_fact_index_history_aware.py
echo "Running Script: $SCRIPT"

# ✓ CHANGED: Use history-aware dataset
TRAIN_DATA=$PROJECT_DIR/dataset/history_aware_data/train_df.pkl
VAL_DATA=$PROJECT_DIR/dataset/history_aware_data/val_df.pkl
TEST_DATA=$PROJECT_DIR/dataset/history_aware_data/test_df.pkl

# Code mappings (same as baseline)
KG_DIR=$PROJECT_DIR/KG/kg_output
ATC_MAP=$KG_DIR/code2cui_atc.pkl          # Source: Medications
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl      # Source: Labs  
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl   # Source: Procedures
DX_MAP=$KG_DIR/code2cui_icd9_dx.pkl       # Source: Past diagnoses + Target: Current diagnoses

# KG
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl

### Configuration - DUAL LIMITS (PER-SOURCE AND PER-TARGET)
# General knowledge paths (non-diagnosis targets)
MAX_H1_PER_SOURCE=50
MAX_H2_PER_SOURCE=10000

# Diagnosis-targeting paths - PER SOURCE LIMITS
MAX_H1_DIAGNOSIS_PER_SOURCE=None
MAX_H2_DIAGNOSIS_PER_SOURCE=None

# Diagnosis-targeting paths - PER TARGET LIMITS
MAX_H1_DIAGNOSIS_PER_TARGET=None
MAX_H2_DIAGNOSIS_PER_TARGET=None

SAPBERT_MODEL="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BATCH_SIZE=32

FACT_INDEX_DIR=$PROJECT_DIR/fact_indexes_history_aware
mkdir -p "$FACT_INDEX_DIR"
mkdir -p "$PROJECT_DIR/logs/History_Aware_RAG"

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "HISTORY-AWARE MODE:"
echo "  ✓ Past diagnosis codes are SOURCE CUIs (evidence)"
echo "  ✓ Current diagnosis codes are TARGET CUIs (predictions)"
echo ""
echo "SOURCE CUIs extracted from DATASET:"
echo "  - Medications (NDC→ATC): $ATC_MAP"
echo "  - Labs (LOINC): $LOINC_MAP"  
echo "  - Procedures (ICD9-PROC): $PROC_MAP"
echo "  - Past Diagnoses (ICD9-DX from 'past_icd' column): $DX_MAP"
echo ""
echo "TARGET DIAGNOSIS CUIs from PKL MAPPING:"
echo "  - Current Diagnosis CUIs: $DX_MAP (code2cui_icd9_dx.pkl)"
echo "  - NOT extracted from dataset current ICD codes"
echo ""
echo "KEY DIFFERENCE FROM BASELINE:"
echo "  - Past diagnoses (from 'past_icd') act as EVIDENCE"
echo "  - Current diagnoses (from pkl) remain PREDICTION TARGETS"
echo "  - Past diagnosis CUIs can appear in BOTH roles"
echo ""
echo "Dataset:"
echo "  - Train: $TRAIN_DATA"
echo "  - Val: $VAL_DATA" 
echo "  - Test: $TEST_DATA"
echo ""
echo "Knowledge Graph: $KG_PKL"
echo "Output: $FACT_INDEX_DIR"
echo ""
echo "PATH LIMITS:"
echo "  General paths (per source CUI):"
echo "    - Max H1: $MAX_H1_PER_SOURCE"
echo "    - Max H2: $MAX_H2_PER_SOURCE"
echo ""
echo "  Diagnosis paths (PER SOURCE CUI):"
echo "    - Max H1 per source: $MAX_H1_DIAGNOSIS_PER_SOURCE"
echo "    - Max H2 per source: $MAX_H2_DIAGNOSIS_PER_SOURCE"
echo ""
echo "  Diagnosis paths (PER TARGET CUI):"
echo "    - Max H1 per target: $MAX_H1_DIAGNOSIS_PER_TARGET"
echo "    - Max H2 per target: $MAX_H2_DIAGNOSIS_PER_TARGET"
echo "========================================"

# Verify all required files exist
echo "Verifying input files..."
for file in "$TRAIN_DATA" "$VAL_DATA" "$TEST_DATA" "$ATC_MAP" "$LOINC_MAP" "$PROC_MAP" "$DX_MAP" "$KG_PKL"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
done
echo "✓ All input files found"

# Build command
CMD="python $SCRIPT \
  --train_data $TRAIN_DATA \
  --val_data $VAL_DATA \
  --test_data $TEST_DATA \
  --atc_map $ATC_MAP \
  --loinc_map $LOINC_MAP \
  --proc_map $PROC_MAP \
  --dx_map $DX_MAP \
  --kg_pkl $KG_PKL \
  --output_dir $FACT_INDEX_DIR \
  --sapbert_model $SAPBERT_MODEL \
  --batch_size $BATCH_SIZE"

# Add limits if not "None"
if [ "$MAX_H1_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h1_per_source $MAX_H1_PER_SOURCE"
fi

if [ "$MAX_H2_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h2_per_source $MAX_H2_PER_SOURCE"
fi

if [ "$MAX_H1_DIAGNOSIS_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h1_diagnosis_per_source $MAX_H1_DIAGNOSIS_PER_SOURCE"
fi

if [ "$MAX_H2_DIAGNOSIS_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h2_diagnosis_per_source $MAX_H2_DIAGNOSIS_PER_SOURCE"
fi

if [ "$MAX_H1_DIAGNOSIS_PER_TARGET" != "None" ]; then
    CMD="$CMD --max_h1_diagnosis_per_target $MAX_H1_DIAGNOSIS_PER_TARGET"
fi

if [ "$MAX_H2_DIAGNOSIS_PER_TARGET" != "None" ]; then
    CMD="$CMD --max_h2_diagnosis_per_target $MAX_H2_DIAGNOSIS_PER_TARGET"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD

status=$?

echo "========================================"
echo "COMPLETE (Exit code: $status)"
echo "End time: $(date)"
echo ""

if [ $status -eq 0 ]; then
    echo " SUCCESS!"
    echo ""
    echo "Output structure:"
    echo "  $FACT_INDEX_DIR/"
    echo "  ├── h1_index/"
    echo "  │   ├── facts.json"
    echo "  │   ├── relationships.json"
    echo "  │   ├── diagnosis_flags.json"
    echo "  │   ├── embeddings.npy"
    echo "  │   └── faiss_index.bin"
    echo "  ├── h2_index/"
    echo "  │   ├── facts.json"
    echo "  │   ├── relationships.json"
    echo "  │   ├── diagnosis_flags.json"
    echo "  │   ├── embeddings.npy"
    echo "  │   └── faiss_index.bin"
    echo "  └── metadata.json"
    echo ""
    echo "Check metadata.json for:"
    echo "  - history_aware: true"
    echo "  - past_diagnoses_as_sources: true"
    echo "  - source_statistics: Includes past_diagnoses mapping stats"
else
    echo " FAILED!"
    echo "Check the log for errors: logs/History_Aware_RAG/build_historyaware_indexes_${SLURM_JOB_ID}.out"
fi

echo "========================================"
exit $status