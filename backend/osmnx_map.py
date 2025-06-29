import osmnx as ox
import networkx as nx
import os
import json

def get_road_network(city_name="Salem, India"):
    """Download and process the road network for MARL."""
    print(f"Downloading road network for {city_name}...")

    # Download road network (only major roads)
    G = ox.graph_from_place(city_name, network_type="drive")

    # Filter out minor roads
    major_highways = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    edges_to_remove = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if "highway" in data and isinstance(data["highway"], str) and data["highway"] not in major_highways
    ]

    G.remove_edges_from(edges_to_remove)

    # Ensure 'data' folder exists
    os.makedirs("data", exist_ok=True)

    # Save reduced road network
    ox.save_graphml(G, "data/road_network.graphml")

    # Identify intersections (nodes with degree ≥ 3)
    intersections = {str(node): G.degree[node] for node in G.nodes if G.degree[node] >= 3}

    # Save intersections (agent positions) for MARL
    with open("data/intersections.json", "w") as f:
        json.dump(intersections, f, indent=4)

    print(f"Filtered road network saved as 'data/road_network.graphml'!")
    print(f"Intersections (MARL agents) saved in 'data/intersections.json'!")

    return G

if __name__ == "__main__":
    G = get_road_network()
