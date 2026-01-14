#!/bin/bash
#SBATCH --job-name=SubgraphViz
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/KGView/%j.out

# Load modules
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# Navigate to project directory
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG || { echo "Project dir not found"; exit 1; }

# Activate virtual environment
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# Create output directory
mkdir -p subgraphs/

# Example 1: 1-hop subgraph with names
python subgraph_view.py \
    --pkl ./KG/kg_output4/medical_knowledge_graph2.pkl \
    --seed C0178237 \
    --radius 1 \
    --direction both \
    --label-type cui \
    --edge-label rela_canon \
    --layout spring \
    --k 0.5 \
    --max-edges 50 \
    --edge-sample-method first \
    --dpi 300 \
    --out subgraphs/745.4/subgraph_C0178237_r1_cui.png

# Example 2: 2-hop subgraph
python subgraph_view.py \
    --pkl /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/medical_knowledge_graph2.pkl \
    --seed C0178237 \
    --radius 2 \
    --direction both \
    --label-type cui \
    --edge-label rela_canon \
    --layout spring \
    --k 0.5 \
    --max-edges 10 \
    --edge-sample-method first \
    --dpi 300 \
    --figsize 16 12 \
    --out subgraphs/745.4/subgraph_C0178237_r2_cui.png

status=$?
echo "[INFO] Job finished with exit code $status"
exit $status