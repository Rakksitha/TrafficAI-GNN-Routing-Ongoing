import osmnx as ox
import random
import json

# Load road network
G = ox.load_graphml("data/road_network.graphml")  # Adjust path if needed

# Extract nodes with longitude-latitude
nodes = [(node, data["x"], data["y"]) for node, data in G.nodes(data=True)]

# Select random N nodes
N = 10  # Adjust count
random_pairs = random.sample(nodes, N)

# Convert to JSON format
coordinate_pairs = [
    {"id": node, "longitude": lon, "latitude": lat} for node, lon, lat in random_pairs
]

# Save JSON for frontend
with open("data/frontend_coords.json", "w") as f:
    json.dump(coordinate_pairs, f, indent=2)

print("✅ Coordinates saved to frontend_coords.json")
