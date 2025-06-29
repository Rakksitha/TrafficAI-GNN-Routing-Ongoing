import os
import random
import pandas as pd
import osmnx as ox

def generate_synthetic_traffic(graph_path="data/road_network.graphml", save_path="data/synthetic_traffic_data.csv"):
    """Generates synthetic traffic data for all edges in the road network, with a separate source column."""

    # Load road network
    G = ox.load_graphml(graph_path)

    # Extract edges (roads)
    traffic_data = []

    for u, v, data in G.edges(data=True):
        if "length" not in data or "highway" not in data:
            continue  # Skip edges with missing length or highway type

        length = float(data["length"])  # Convert to float
        speed = data.get("maxspeed", 30)  # Default to 30 km/h if missing

        # Handle cases where 'maxspeed' is a list
        if isinstance(speed, list):
            speed = float(speed[0]) if speed else 30  # Take first value

        speed = float(speed)  # Ensure numeric type

        congestion = random.uniform(0, 1)  # Random congestion (0-1)
        vehicle_count = random.randint(5, 500)  # Random vehicle count

        base_weight = length / max(speed, 1)  # Avoid division by zero

        # Store source separately
        traffic_data.append([u, v, u, length, speed, congestion, vehicle_count, base_weight])

    # Convert to DataFrame and save
    df = pd.DataFrame(traffic_data, columns=["start_node", "end_node", "source", "length", "speed", "congestion", "vehicle_count", "base_weight"])
    os.makedirs("data", exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Synthetic traffic data saved at {save_path}")
    print(f"Total synthetic roads: {len(df)}")
    print(f"Sample data:\n{df.head()}")

if __name__ == "__main__":
    generate_synthetic_traffic()
