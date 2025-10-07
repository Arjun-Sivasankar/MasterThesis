import pickle
import networkx as nx
import matplotlib.pyplot as plt

DATA_DIR = 'KG/data_files'

# Load the full knowledge graph
with open(f"{DATA_DIR}/clinical_kg_networkx.pkl", "rb") as f:
    full_graph = pickle.load(f)

# Load the connected component subgraph
with open(f"{DATA_DIR}/clinical_kg_connected_networkx.pkl", "rb") as f:
    connected_graph = pickle.load(f)

# Load the lookup dictionaries
with open(f"{DATA_DIR}/kg_lookups.pkl", "rb") as f:
    lookups = pickle.load(f)

# Check basic graph properties
print(f"Full graph: {full_graph.number_of_nodes()} nodes, {full_graph.number_of_edges()} edges")
print(f"Connected component: {connected_graph.number_of_nodes()} nodes, {connected_graph.number_of_edges()} edges")

# Check what's in the lookups dictionary
print(f"Lookup keys: {list(lookups.keys())}")

# Example of checking node types
node_types = {}
for node in list(connected_graph.nodes())[:10]:  # Check first 10 nodes
    node_data = connected_graph.nodes[node]
    print(f"Node {node}: {node_data}")