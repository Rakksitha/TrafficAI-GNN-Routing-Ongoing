import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from traffic_gnn import TrafficGNN  # Import GNN model
import osmnx as ox
import networkx as nx
import pandas as pd

# Define data paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
traffic_data_path = os.path.join(data_dir, "synthetic_traffic_data.csv")
road_network_path = os.path.join(data_dir, "road_network.graphml")

# Load synthetic traffic data
traffic_df = pd.read_csv(traffic_data_path)

# Convert traffic data into a dictionary for faster lookup
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

        # Lookup traffic data (check both directions)
        traffic_row = traffic_dict.get(edge_key, traffic_dict.get((v, u), None))

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

    # 🔴 **Fix: Add node features (dummy values for now)**
    num_nodes = len(node_mapping)
    node_features = torch.rand((num_nodes, 3))  # Random features for now

    # Create PyG Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

# Train the GNN Model
def train_traffic_gnn(priority, epochs=100, lr=0.01):
    print(f"🟢 Training GNN model for priority: {priority}...")
    
    # ✅ **Fix: Properly call `convert_graph_to_pyg`**
    G_pyg = convert_graph_to_pyg(load_osm_graph(), priority)

    num_nodes = G_pyg.num_nodes
    model = TrafficGNN(num_nodes=num_nodes, in_feats=3, hidden_size=16, out_feats=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 🔴 **Fix: Handle missing `G_pyg.x` (use random values)**
    target = G_pyg.x[:, 0].view(-1, 1) if G_pyg.x is not None else torch.zeros((num_nodes, 1))
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(G_pyg.x, G_pyg.edge_index, edge_weight=G_pyg.edge_attr if G_pyg.edge_attr is not None else None)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"📌 Priority: {priority} | Epoch {epoch}: Loss {loss.item():.4f}")
    
    model_path = os.path.join(data_dir, f"gnn_model_{priority}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ GNN Model for '{priority}' saved at: {model_path}")

# Train models for all priorities
if __name__ == "__main__":
    for priority in ["red", "yellow", "green"]:
        train_traffic_gnn(priority)
