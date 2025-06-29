import osmnx as ox
import networkx as nx
import torch
import pandas as pd
import os
from torch_geometric.data import Data

# Define data paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
traffic_data_path = os.path.join(data_dir, "synthetic_traffic_data.csv")
road_network_path = os.path.join(data_dir, "road_network.graphml")

# Load synthetic traffic data
traffic_df = pd.read_csv(traffic_data_path)

# Convert traffic data into a dictionary for **faster lookup**
traffic_dict = {(row["start_node"], row["end_node"]): row for _, row in traffic_df.iterrows()}

def load_osm_graph():
    """Load road network from file."""
    G = ox.load_graphml(road_network_path)

    # Convert to undirected if necessary
    if nx.is_directed(G):
        print("🔄 Converting OSM Graph to an undirected graph...")
        G = G.to_undirected()

    print(f"✅ Loaded OSM Graph with {len(G.nodes)} nodes & {len(G.edges)} edges")  
    return G

def convert_graph_to_pyg(G, priority="yellow"):
    """
    Convert OSMnx Graph to PyTorch Geometric Graph with dynamic weights.
    
    Priorities:
    - "red" -> Fast Reach (Prefers speed)
    - "yellow" -> Balanced (Mix of speed, congestion)
    - "green" -> Safe & Slow (Low congestion)
    """

    # Map node IDs to sequential indices
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    edge_index = []
    edge_weights = []
    added_edges = set()  # Track added edges to prevent duplicates

    missing_edges = 0  # Track missing traffic data

    for u, v, data in G.edges(data=True):
        edge_key = (u, v)

        # Lookup traffic data (check both directions) and ensure a single row is extracted
        traffic_row = traffic_dict.get(edge_key, None)
    
        if traffic_row is None:
         traffic_row = traffic_dict.get((v, u), None)

        if traffic_row is None:
         missing_edges += 1
         continue  # Skip if missing

        if isinstance(traffic_row, pd.Series):
         traffic_row = traffic_row.to_dict()  # Convert to dictionary if it's a Series


        # Extract values directly using column names
        length = float(traffic_row["length"])
        speed = max(float(traffic_row["speed"]), 1)  # Avoid zero speed
        congestion = float(traffic_row["congestion"])
        base_weight = float(traffic_row["base_weight"])

        # Adjust weights based on priority
        if priority == "red":
            weight = base_weight / speed  # Prioritize high speed
        elif priority == "yellow":
            weight = base_weight * (0.7 + 0.3 * congestion)  # Balance speed & congestion
        else:  # Green
            weight = base_weight * (1 + congestion)  # Avoid high congestion

        # Add edge only if not already added
        if (u, v) not in added_edges and (v, u) not in added_edges:
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                edge_weights.append(weight)

                # Mark both directions as added
                added_edges.add((u, v))
                added_edges.add((v, u))

    print(f"⚠️ Skipped {missing_edges} edges due to missing traffic data.")

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # Create PyG Data object
    return Data(edge_index=edge_index, edge_attr=edge_weights)

def save_pyg_graph(priority):
    """
    Convert OSM graph to PyG format and save it as a file.
    """
    G = load_osm_graph()
    G_pyg = convert_graph_to_pyg(G, priority)
    
    save_path = os.path.join(data_dir, f"graph_{priority}.pt")
    torch.save(G_pyg, save_path)
    print(f"📁 Saved PyG graph for '{priority}' at: {save_path}")

def load_pyg_graph(priority):
    """
    Load precomputed PyG graph from file.
    """
    load_path = os.path.join(data_dir, f"graph_{priority}.pt")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ PyG graph not found: {load_path}")
    return torch.load(load_path)

if __name__ == "__main__":
    for priority in ["red", "yellow", "green"]:
        save_pyg_graph(priority)  # Precompute and save graphs
