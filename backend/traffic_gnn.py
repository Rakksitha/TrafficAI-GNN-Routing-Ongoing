import osmnx as ox 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_nodes):
        super(TrafficGNN, self).__init__()

        # Learnable Node Embeddings
        self.node_embeddings = nn.Embedding(num_nodes, in_feats)

        # GCN Layers
        self.conv1 = GCNConv(in_feats, hidden_feats, normalize=True)
        self.conv2 = GCNConv(hidden_feats, out_feats, normalize=True)

        # Batch Normalization & Dropout
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x, edge_index, edge_weight=None):
      if x is None:
        x = self.node_embeddings.weight  # Ensure embeddings exist

    # Ensure edge_weight is not None
      if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)  

    # Debugging prints
      print("Edge index shape:", edge_index.shape)  # Should be [2, num_edges]
      print("Edge weight shape:", edge_weight.shape if edge_weight is not None else "None")  # Should be [num_edges]
      print("Input feature shape:", x.shape)  # Should be [num_nodes, num_features]

      x = self.conv1(x, edge_index, edge_weight=edge_weight)
      x = self.bn1(x)  
      x = F.relu(x)
      x = self.dropout(x)

      x = self.conv2(x, edge_index, edge_weight=edge_weight)

      if x.shape[1] > 1:  
        x = F.log_softmax(x, dim=1)  
      else:
        x = torch.tanh(x)  

      return x


# ✅ Load the graph outside the class
graph = ox.load_graphml("E:/TrafficAI/data/road_network.graphml")
num_nodes = len(graph.nodes)

# ✅ Initialize model correctly
model = TrafficGNN(in_feats=3, hidden_feats=32, out_feats=1, num_nodes=num_nodes)
