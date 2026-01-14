#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build medical_knowledge_graph2.pkl from kg_edges_canon.csv
This creates the enriched graph with canonicalized relations.
"""

import os
import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

# Configuration
KG_OUTPUT_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4'
NODES_CSV = os.path.join(KG_OUTPUT_DIR, 'kg_nodes_aggregated.csv')  # or kg_nodes.csv
EDGES_CANON_CSV = os.path.join(KG_OUTPUT_DIR, 'kg_edges_canon.csv')
OUTPUT_PKL = os.path.join(KG_OUTPUT_DIR, 'medical_knowledge_graph2.pkl')

def main():
    print("=" * 80)
    print("Building Graph 2 from canonicalized edges")
    print("=" * 80)
    
    # Step 1: Load nodes to get node attributes
    print("\n[1/4] Loading nodes...")
    nodes_df = pd.read_csv(NODES_CSV)
    print(f"Loaded {len(nodes_df):,} nodes")
    
    # Step 2: Create graph and add nodes with attributes
    print("\n[2/4] Creating graph and adding nodes...")
    G = nx.DiGraph()
    
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Adding nodes"):
        cui = row['cui']
        
        # Parse multi-value fields
        sab = row['sab'].split('|') if pd.notna(row['sab']) and row['sab'] else []
        code = row['code'].split('|') if pd.notna(row['code']) and row['code'] else []
        semantic_type = row['semantic_type'].split('|') if pd.notna(row['semantic_type']) and row['semantic_type'] else []
        
        G.add_node(cui,
                   name=row['name'] if pd.notna(row['name']) else '',
                   sab=sab,
                   code=code,
                   semantic_type=semantic_type)
    
    print(f"Added {G.number_of_nodes():,} nodes to graph")
    
    # Step 3: Load canonicalized edges
    print("\n[3/4] Loading canonicalized edges...")
    edges_df = pd.read_csv(EDGES_CANON_CSV)
    print(f"Loaded {len(edges_df):,} edges")
    
    # Step 4: Add edges with all attributes including canonicalized ones
    print("\n[4/4] Adding edges to graph...")
    
    # Group by CUI pair to avoid duplicates (keep first occurrence of each CUI pair)
    # This mimics the original graph structure where each CUI pair has one edge
    edges_grouped = edges_df.groupby(['cui_start', 'cui_target']).first().reset_index()
    print(f"After deduplication: {len(edges_grouped):,} unique CUI pairs")
    
    for _, row in tqdm(edges_grouped.iterrows(), total=len(edges_grouped), desc="Adding edges"):
        u = row['cui_start']
        v = row['cui_target']
        
        if u not in G or v not in G:
            continue
        
        # Add edge with all attributes (both original and canonicalized)
        edge_attrs = {
            'rel': row['rel'] if pd.notna(row['rel']) else '',
            'rela': row['rela'] if pd.notna(row['rela']) else '',
            'sab_relation': row['sab_relation'] if pd.notna(row['sab_relation']) else '',
            'rela_raw': row['rela_raw'] if pd.notna(row['rela_raw']) else '',
            'rela_final': row['rela_final'] if pd.notna(row['rela_final']) else '',
            'rela_canon': row['rela_canon'] if pd.notna(row['rela_canon']) else '',
            'rela_score': float(row['rela_score']) if pd.notna(row['rela_score']) else 0.0,
            'name_start': row['name_start'] if pd.notna(row['name_start']) else '',
            'name_target': row['name_target'] if pd.notna(row['name_target']) else '',
            'sab_start': row['sab_start'] if pd.notna(row['sab_start']) else '',
            'sab_target': row['sab_target'] if pd.notna(row['sab_target']) else '',
            'codes_start': row['codes_start'] if pd.notna(row['codes_start']) else '',
            'codes_target': row['codes_target'] if pd.notna(row['codes_target']) else ''
        }
        
        G.add_edge(u, v, **edge_attrs)
    
    print(f"\nFinal graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Step 5: Save the graph
    print(f"\n[5/5] Saving graph to {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\n" + "=" * 80)
    print("SUCCESS: Graph 2 created successfully!")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_PKL}")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    
    # Show sample edge attributes
    if G.number_of_edges() > 0:
        sample_edge = list(G.edges(data=True))[0]
        print(f"\nSample edge attributes:")
        print(f"  {sample_edge[0]} -> {sample_edge[1]}")
        for key, value in sorted(sample_edge[2].items()):
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()