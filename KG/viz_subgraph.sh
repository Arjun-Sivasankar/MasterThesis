#!/bin/bash
#SBATCH --job-name=Subgraph
#SBATCH --partition=capella            # GPU partition
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores for data loading
#SBATCH --nodes=1
#SBATCH --mem=64G                      # RAM
#SBATCH --time=03:00:00                # Max walltime
#SBATCH --output=logs/KGView/%j.out    # stdout+stderr log

# --- 1) Load modules ---
module purge
module load release/24.04 GCCcore/11.3.0
module load Python/3.10.4

# --- 2) Move to project directory ---
cd /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG || { echo "Project dir not found"; exit 1; }

# --- 3) Activate virtual environment ---
source /data/horse/ws/arsi805e-venv/venvs/finetune/bin/activate
echo "[INFO] Virtual env loaded: $VIRTUAL_ENV"

# 1-hop, both directions, default labels (CUI) and auto edge labels:
# python viz_subgraph.py \
#     --seed C0152602 \
#     --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_nodes.csv \
#     --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_edges.csv \
#     --radius 1 \
#     --direction both \
#     --edge-label none \
#     --out subgraphs/subgraph_C0152602_r1_dirboth.png

# python viz_subgraph.py \
#     --seed C0152602 \
#     --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_nodes.csv \
#     --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_edges.csv \
#     --radius 1 \
#     --direction in \
#     --edge-label none \
#     --out subgraphs/subgraph_C0152602_r1_dirin.png

# ----------------------- kg output2 ----------------------- #
# python viz_subgraph.py \
#     --seed C0152602 \
#     --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/kg_nodes.csv \
#     --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/kg_edges.csv \
#     --radius 1 \
#     --direction both \
#     --edge-label none \
#     --out subgraphs/new/subgraph_C0152602_r1_dirboth.png

# python viz_subgraph.py \
#     --seed C0152602 \
#     --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/kg_nodes.csv \
#     --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output2/kg_edges.csv \
#     --radius 2 \
#     --direction both \
#     --edge-label none \
#     --out subgraphs/new/subgraph_C0152602_r2_dirboth.png

# ----------------------- kg output3 ----------------------- #
python viz_subgraph.py \
    --seed C1704311 \
    --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_nodes.csv \
    --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_edges.csv \
    --radius 1 \
    --direction both \
    --edge-label none \
    --out subgraphs/with_kgoutput4/subgraph_C1704311_r1_dirboth.png

python viz_subgraph.py \
    --seed C1704311 \
    --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_nodes.csv \
    --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_edges.csv \
    --radius 2 \
    --direction both \
    --edge-label none \
    --out subgraphs/with_kgoutput4/subgraph_C1704311_r2_dirboth.png

python viz_subgraph.py \
    --seed C1704311 \
    --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_nodes.csv \
    --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_edges.csv \
    --radius 3 \
    --direction both \
    --edge-label none \
    --out subgraphs/with_kgoutput4/subgraph_C1704311_r3_dirboth.png

python viz_subgraph.py \
    --seed C1704311 \
    --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_nodes.csv \
    --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/kg_edges.csv \
    --radius 4 \
    --direction both \
    --edge-label none \
    --out subgraphs/with_kgoutput4/subgraph_C1704311_r4_dirboth.png

## DOESN't WORK DUE TO MEMORY LIMITS ##
# python viz_subgraph.py \
#     --seed C1704311 \
#     --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output3/kg_nodes.csv \
#     --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output3/kg_edges.csv \
#     --radius 5 \
#     --direction both \
#     --edge-label none \
#     --out subgraphs/with_kgoutput3/subgraph_C1704311_r5_dirboth.png


# 1-hop, hide edge labels if the figure is cluttered:
# python viz_subgraph.py \
#     --seed C0011849 \
#     --edge-label none \
#     --out subgraphs/subgraph_no_edgelabels.png

# 2-hop outbound only, show names as labels, only CHD/PAR relations:
# python viz_subgraph.py \
#     --seed C0011849 \
#     --radius 2 \
#     --direction out \
#     --label-type name \
#     --rel-allow CHD,PAR \
#     --nodes /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_nodes.csv \
#     --edges /data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output/kg_edges.csv \
#     --out subgraphs/subgraph_C0011849_r2_out.png

status=$?
echo "[INFO] Job finished with exit code $status"
exit $status