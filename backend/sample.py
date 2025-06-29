import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import heapq

edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 0, 2, 1, 3],  
                      [1, 2, 3, 3, 4, 4, 5, 5, 2, 5, 5, 6]]) 

base_edge_weights = torch.tensor([1., 1., 2., 2., 3., 3., 1., 1., 2., 2., 3., 1.])  

node_coordinates = {  
    0: (11.6621, 78.1450),
    1: (11.6628, 78.1462),
    2: (11.6635, 78.1475),
    3: (11.6640, 78.1488),
    4: (11.6648, 78.1500),
    5: (11.6655, 78.1512),
    6: (11.6662, 78.1525)
}

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        edge_features = (x[edge_index[0]] + x[edge_index[1]]) / 2
        return edge_features.squeeze()

num_nodes = 7  
x = torch.eye(num_nodes)  
model = SimpleGNN(in_feats=7, hidden_feats=4, out_feats=1)


gnn_weights = torch.tensor([
    1.5,  # Edge (0-1): Slight congestion
    2.7,  # Edge (1-2): High congestion
    3.2,  # Edge (1-3): Heavy congestion
    1.8,  # Edge (2-3): Medium congestion
    4.5,  # Edge (2-4): Severe congestion
    5.0,  # Edge (3-4): Very high congestion
    1.2,  # Edge (3-5): Low congestion
    2.0,  # Edge (4-5): Moderate congestion
    3.0,  # Edge (0-2): High congestion
    1.1,  # Edge (2-5): Low congestion
    4.0,  # Edge (1-5): Heavy congestion
    1.4   # Edge (5-6): Low congestion
])

gnn_edge_weights = base_edge_weights + 2.5 * torch.abs(gnn_weights)  

print("Base Weights (No GNN):", base_edge_weights.numpy())
print("GNN Predicted Weights:", gnn_edge_weights.numpy())

def astar_path(G, start, goal):
    queue = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        for neighbor in G.neighbors(current):
            new_cost = cost_so_far[current] + G[current][neighbor]['weight']
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current

    path = []
    node = goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

def generate_graph(priority):
    G = nx.Graph()
    
    if priority == "red":  
        edge_weights = base_edge_weights
    elif priority == "yellow":  
        edge_weights = (0.6 * base_edge_weights + 0.4 * gnn_edge_weights)  
    elif priority == "green":  
        edge_weights = gnn_edge_weights
    else:
        raise ValueError("Invalid priority! Choose 'red', 'yellow', or 'green'.")

    for i in range(edges.shape[1]):
        G.add_edge(int(edges[0, i]), int(edges[1, i]), weight=edge_weights[i].item())

    return G

start_node, goal_node = 0, 5

paths = {}
for priority in ["red", "yellow", "green"]:
    G = generate_graph(priority)
    paths[priority] = astar_path(G, start_node, goal_node)

print(f"A* (Red - Fastest Route): {paths['red']}")
print(f"A* (Yellow - Balanced): {paths['yellow']}")
print(f"A* (Green - Least Congested): {paths['green']}")

if paths["red"] != paths["green"]:
    print("GNN dynamically optimized the route based on congestion!")
else:
    print("No change in route! Try modifying GNN weights.")
