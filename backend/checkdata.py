import pandas as pd
import osmnx as ox

# Load the road network
graph_path = "data/road_network.graphml"
G = ox.load_graphml(graph_path)

# Extract all valid nodes from the road network
valid_nodes = set(G.nodes)

# Load synthetic traffic data
traffic_data_path = "data/synthetic_traffic_data.csv"
df = pd.read_csv(traffic_data_path)

# Extract start_node and end_node
start_nodes = set(df["start_node"])
end_nodes = set(df["end_node"])

# Identify missing nodes
missing_start_nodes = start_nodes - valid_nodes
missing_end_nodes = end_nodes - valid_nodes

# Print results
print(f"Total start nodes in data: {len(start_nodes)}")
print(f"Total end nodes in data: {len(end_nodes)}")
print(f"Total valid nodes in road network: {len(valid_nodes)}")

print(f"Missing start nodes: {len(missing_start_nodes)}")
print(f"Missing end nodes: {len(missing_end_nodes)}")

if missing_start_nodes:
    print("⚠️ The following start_nodes are missing in the road network:")
    print(missing_start_nodes)

if missing_end_nodes:
    print("⚠️ The following end_nodes are missing in the road network:")
    print(missing_end_nodes)
