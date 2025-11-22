#!/bin/bash
#SBATCH --job-name=build_dedup_indexes
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Textgen-withKG-withRAG/build_dedup_indexes_%j.out
#SBATCH --licenses=horse

set -e
set -u

echo "========================================"
echo "BUILDING DEDUPLICATED SOURCE→TARGET INDEXES"
echo "Sources: Medications, Labs, Procedures (from dataset)"
echo "Targets: Diagnosis CUIs (from pkl) + General Knowledge"
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
SCRIPT=$PROJECT_DIR/gen/withKG/withRAG/build_medical_fact_index_new.py
echo "Running Script: $SCRIPT"

# Dataset (ICD codes will be EXCLUDED as sources)
TRAIN_DATA=$PROJECT_DIR/dataset/final_data/train_df.pkl
VAL_DATA=$PROJECT_DIR/dataset/final_data/val_df.pkl
TEST_DATA=$PROJECT_DIR/dataset/final_data/test_df.pkl

# Code mappings
KG_DIR=$PROJECT_DIR/KG/kg_output4
ATC_MAP=$KG_DIR/code2cui_atc.pkl          # Source: Medications
LOINC_MAP=$KG_DIR/code2cui_loinc.pkl      # Source: Labs  
PROC_MAP=$KG_DIR/code2cui_icd9_proc.pkl   # Source: Procedures
DX_MAP=$KG_DIR/code2cui_icd9_dx.pkl       # TARGET ONLY: Diagnosis CUIs

# KG
KG_PKL=$KG_DIR/medical_knowledge_graph2.pkl

### Configuration - DUAL LIMITS (PER-SOURCE AND PER-TARGET)
# First run: Set to None to see path statistics
# Subsequent runs: Set limits based on statistics and memory constraints

# General knowledge paths (non-diagnosis targets)
MAX_H1_PER_SOURCE=50         # e.g., 1000 - limits general H1 paths per source CUI
MAX_H2_PER_SOURCE=30000         # e.g., 500  - limits general H2 paths per source CUI

# Diagnosis-targeting paths - PER SOURCE LIMITS
# Controls how many diagnosis paths each source CUI (med/lab/proc) can have
MAX_H1_DIAGNOSIS_PER_SOURCE=None   # e.g., 500 - each source can connect to max 500 diagnoses
MAX_H2_DIAGNOSIS_PER_SOURCE=None   # e.g., 250 - each source can have max 250 2-hop diagnosis paths

# Diagnosis-targeting paths - PER TARGET LIMITS
# Controls how many incoming paths each diagnosis CUI can receive
# Useful to prevent highly-connected diagnosis concepts from dominating
MAX_H1_DIAGNOSIS_PER_TARGET=None   # e.g., 1000 - each diagnosis can receive max 1000 incoming H1 paths
MAX_H2_DIAGNOSIS_PER_TARGET=None   # e.g., 500  - each diagnosis can receive max 500 incoming H2 paths

SAPBERT_MODEL="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BATCH_SIZE=64

# Output directory
FACT_INDEX_DIR=$PROJECT_DIR/fact_indexes
mkdir -p "$FACT_INDEX_DIR"
mkdir -p "$PROJECT_DIR/logs/Textgen-withKG-withRAG"

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "PRINCIPLE: Diagnosis CUIs from pkl mapping (NOT dataset)"
echo ""
echo "SOURCE CUIs extracted from DATASET:"
echo "  - Medications (NDC→ATC): $ATC_MAP"
echo "  - Labs (LOINC): $LOINC_MAP"  
echo "  - Procedures (ICD9-PROC): $PROC_MAP"
echo ""
echo "TARGET DIAGNOSIS CUIs from PKL MAPPING:"
echo "  - Diagnosis CUIs: $DX_MAP (code2cui_icd9_dx.pkl)"
echo "  - NOT extracted from dataset ICD codes"
echo ""
echo "GENERAL KNOWLEDGE TARGETS:"
echo "  - Any other CUI in KG"
echo ""
echo "EXCLUDED from sources:"
echo "  - ICD diagnosis codes from dataset (they are prediction targets)"
echo ""
echo "DEDUPLICATION:"
echo "  - Enabled: No path appears twice"
echo "  - H1: Tracks (source, target) pairs"
echo "  - H2: Tracks (source, intermediate, target) triples"
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
echo "    → Limits how many diagnosis paths each source CUI can have"
echo ""
echo "  Diagnosis paths (PER TARGET CUI):"
echo "    - Max H1 per target: $MAX_H1_DIAGNOSIS_PER_TARGET"
echo "    - Max H2 per target: $MAX_H2_DIAGNOSIS_PER_TARGET"
echo "    → Limits how many incoming paths each diagnosis CUI can receive"
echo ""
echo "NOTES:"
echo "  - Set to 'None' for unlimited (first run to see statistics)"
echo "  - Per-source limits: Control evidence spreading from meds/labs/procs"
echo "  - Per-target limits: Control popular diagnoses from dominating index"
echo "  - Both can be used together for fine-grained control"
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

# Add general path limits if not "None"
if [ "$MAX_H1_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h1_per_source $MAX_H1_PER_SOURCE"
fi

if [ "$MAX_H2_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h2_per_source $MAX_H2_PER_SOURCE"
fi

# Add diagnosis per-source limits if not "None"
if [ "$MAX_H1_DIAGNOSIS_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h1_diagnosis_per_source $MAX_H1_DIAGNOSIS_PER_SOURCE"
fi

if [ "$MAX_H2_DIAGNOSIS_PER_SOURCE" != "None" ]; then
    CMD="$CMD --max_h2_diagnosis_per_source $MAX_H2_DIAGNOSIS_PER_SOURCE"
fi

# Add diagnosis per-target limits if not "None"
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
    echo "✅ SUCCESS!"
    echo ""
    echo "Output structure:"
    echo "  $FACT_INDEX_DIR/"
    echo "  ├── h1_index/"
    echo "  │   ├── facts.json              # H1 linearized facts"
    echo "  │   ├── relationships.json      # H1 relationship types"
    echo "  │   ├── diagnosis_flags.json    # H1 diagnosis target flags"
    echo "  │   ├── embeddings.npy          # H1 SapBERT embeddings"
    echo "  │   └── faiss_index.bin         # H1 FAISS search index"
    echo "  ├── h2_index/"
    echo "  │   ├── facts.json              # H2 linearized facts"
    echo "  │   ├── relationships.json      # H2 relationship types"
    echo "  │   ├── diagnosis_flags.json    # H2 diagnosis target flags"
    echo "  │   ├── embeddings.npy          # H2 SapBERT embeddings"
    echo "  │   └── faiss_index.bin         # H2 FAISS search index"
    echo "  └── metadata.json               # Combined metadata"
    echo ""
    echo "Check metadata.json for:"
    echo "  - diagnosis_statistics: Info about diagnosis CUIs from pkl"
    echo "  - h1/h2_statistics:"
    echo "      * duplicates_skipped"
    echo "      * skipped_by_source_limit (per-source limits applied)"
    echo "      * skipped_by_target_limit (per-target limits applied)"
    echo "      * diagnosis_targets_reached (unique diagnosis CUIs with paths)"
    echo ""
    echo "Understanding the limits:"
    echo "  PER-SOURCE limits: Each source CUI has max N diagnosis paths"
    echo "    → Use to prevent single meds/labs/procs from dominating"
    echo "  PER-TARGET limits: Each diagnosis CUI receives max N incoming paths"
    echo "    → Use to prevent popular diagnoses from causing OOM"
    echo ""
    echo "Next steps:"
    echo "  1. Review statistics in metadata.json"
    echo "  2. Check if OOM occurred"
    echo "  3. If OOM:"
    echo "     - First try per-target limits (control popular diagnoses)"
    echo "     - Then try per-source limits (control prolific evidence)"
    echo "  4. Balance diagnosis vs general knowledge paths"
else
    echo "❌ FAILED!"
    echo "Check the log for errors: logs/Textgen-withKG-withRAG/build_dedup_indexes_${SLURM_JOB_ID}.out"
fi

echo "========================================"
exit $status