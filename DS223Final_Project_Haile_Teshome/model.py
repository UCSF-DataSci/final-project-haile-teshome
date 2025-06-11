import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig

# ---------------- GNN Model ---------------- #
class GNNClassifier(nn.Module):
    def __init__(self, num_node_features=21, hidden_dim=64, num_classes=6):
        super(GNNClassifier, self).__init__()
        self.node_emb = nn.Embedding(num_node_features, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = 0.3

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

    def embed(self, x, edge_index, batch):
        x = self.node_emb(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)

# ---------------- Explainer-Compatible Wrapper ---------------- #
class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, batch):
        logits = self.base_model(x, edge_index, batch)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, None


# ---------------- GNN Explanation ---------------- #
def explain_prediction(model, data):
    """
    Generates node-level importance scores using GNNExplainer.
    Compatible with PyTorch Geometric 2.6.1 Explainer API.
    """
    model.eval()
    wrapped_model = WrappedModel(model)

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs'
        ),
        node_mask_type='attributes',
        edge_mask_type='object',
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        batch=torch.zeros(data.x.size(0), dtype=torch.long)
    )

    return explanation
