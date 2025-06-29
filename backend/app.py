import os
import torch
import networkx as nx
import osmnx as ox
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from torch_geometric.utils import from_networkx
from traffic_gnn import TrafficGNN  
import uvicorn

app = FastAPI(title="AI Traffic Routing API", description="API for AI-based Traffic Routing using A* and GNN")

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
traffic_data_path = os.path.join(data_dir, "synthetic_traffic_data.csv")
road_network_path = os.path.join(data_dir, "road_network.graphml")
output_csv_path = os.path.join(data_dir, "route_results.csv")

# Load Traffic Data
traffic_df = pd.read_csv(traffic_data_path) if os.path.exists(traffic_data_path) else None
if traffic_df is not None:
    print("✅ Traffic data loaded successfully!")
    print(traffic_df.head())  # Print first few rows to check columns
else:
    raise FileNotFoundError(f"❌ Traffic data file not found: {traffic_data_path}")

# Load Road Network
G = ox.load_graphml(road_network_path) if os.path.exists(road_network_path) else None
if G is not None:
    print("✅ Road network loaded successfully!")
else:
    raise FileNotFoundError(f"❌ Road network file not found: {road_network_path}")

# Load trained models
models = {}
for priority in ["red", "yellow", "green"]:
    model_path = os.path.join(data_dir, f"gnn_model_{priority}.pth")
    if os.path.exists(model_path):
        hidden_size = 16
        num_nodes = len(G.nodes)  
        model = TrafficGNN(in_feats=3, hidden_feats=hidden_size, out_feats=1, num_nodes=num_nodes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Ensure loading on CPU
        model.eval()
        models[priority] = model
        print(f"✅ Model for {priority} priority loaded successfully!")
    else:
        print(f"⚠️ Warning: Model file missing for {priority}")

# API Model for input
class RouteRequest(BaseModel):
    priority: str
    orig_x: float
    orig_y: float
    dest_x: float
    dest_y: float

# Modify graph weights based on priority
def modify_graph_weights(G, priority):
    G_nx = nx.DiGraph()

    # Correct the traffic dictionary without 'road_id'
    traffic_dict = {f"{row['start_node']}_{row['end_node']}": row for _, row in traffic_df.iterrows()}

    for u, v, data in G.edges(data=True):
        edge_id = f"{u}_{v}"  # Fix key format
        traffic_row = traffic_dict.get(edge_id)
        base_weight = float(data.get("length", 100))  
        congestion = float(traffic_row["congestion"]) if traffic_row is not None else 0

        if priority == "red":
            weight = base_weight / max(float(traffic_row["speed"]) if traffic_row is not None else 1, 1)
        elif priority == "yellow":
            weight = base_weight * (0.7 + 0.3 * congestion)
        else:
            weight = base_weight * (1 + congestion)

        G_nx.add_edge(u, v, weight=weight)

    print(f"✅ Graph modified for priority: {priority}")
    return G_nx

def update_graph_with_predictions(G, model):
    G_pyg = from_networkx(G, group_edge_attrs=["weight"])
    G_pyg.x = torch.ones((G_pyg.num_nodes, 3))  # Placeholder node features
    
    # Convert to adjacency matrix
    adj = torch.tensor(nx.to_numpy_array(G)).float()

    # 🚨 Debugging Statements
    print(f"Number of nodes: {G_pyg.num_nodes}, Number of edges: {G_pyg.num_edges}")
    print(f"Edge index shape before passing to model: {G_pyg.edge_index.shape}")

    predicted_congestion = model(adj, G_pyg.x).detach().numpy()
    print(f"Predicted Congestion Shape: {predicted_congestion.shape}")

    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    
    edge_congestion = [
        (predicted_congestion[node_to_index[u]] + predicted_congestion[node_to_index[v]]) / 2
        if u in node_to_index and v in node_to_index else 0
        for u, v in G.edges()
    ]

    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] *= (1 + edge_congestion[i] / 100)
    
    return G

# Run A* Algorithm
def get_route_from_input(priority, orig_x, orig_y, dest_x, dest_y):
    if priority not in models:
        return {"error": "Invalid priority. Choose from red, yellow, green"}
    
    model = models[priority]

    try:
        orig_node = ox.distance.nearest_nodes(G, orig_x, orig_y)
        dest_node = ox.distance.nearest_nodes(G, dest_x, dest_y)
        print(f"📍 Origin node: {orig_node}, Destination node: {dest_node}")
    except Exception as e:
        return {"error": f"Could not find nearest nodes: {str(e)}"}

    G_nx = modify_graph_weights(G, priority)
    G_updated = update_graph_with_predictions(G_nx, model)

    try:
        route_without_gnn = nx.astar_path(G_nx, orig_node, dest_node, weight="weight")
        route_with_gnn = nx.astar_path(G_updated, orig_node, dest_node, weight="weight")
    except nx.NetworkXNoPath:
        return {"error": "No valid path found!"}

    print(f" Route without GNN: {route_without_gnn}")
    print(f" Route with GNN: {route_with_gnn}")

    route_data = [{"from": u, "to": v, "weight": G_updated[u][v]["weight"], "length": G_updated[u][v].get("length", 100), "priority": priority, "predicted_congestion": G_updated[u][v].get("predicted_congestion", 0)} for u, v in zip(route_with_gnn[:-1], route_with_gnn[1:])]
    
    pd.DataFrame(route_data).to_csv(output_csv_path, index=False)
    print(f" Route results saved successfully to {output_csv_path}!")

    return {"route": route_data, "message": "Optimized route found and saved!"}

if __name__ == "__main__":
    priority = "yellow"
    orig_x, orig_y = 78.144718, 11.6082422
    dest_x, dest_y = 77.8387854, 11.5788835
    result = get_route_from_input(priority, orig_x, orig_y, dest_x, dest_y)
    print(result)
    uvicorn.run(app, host="0.0.0.0", port=8000)
