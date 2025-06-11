import torch
import umap
import pandas as pd
import numpy as np

from data import load_domain_graphs, DOMAIN_FAMILIES
from model import GNNClassifier

# Load data and model
families = DOMAIN_FAMILIES
data_list, id_list = load_domain_graphs(families=families, max_per_family=100)
model = GNNClassifier(num_node_features=21, hidden_dim=64, num_classes=len(families))
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Get embeddings for each graph
embeddings = []
labels = []
for data, uid in zip(data_list, id_list):
    # Create batch vector of zeros (single graph)
    batch_vec = torch.zeros(data.x.size(0), dtype=torch.long)
    with torch.no_grad():
        emb = model.embed(data.x, data.edge_index, batch_vec)
    embeddings.append(emb.numpy())
    labels.append(families[data.y.item()])

embeddings = np.vstack(embeddings)  # shape [N, 64]
# Apply UMAP to reduce to 2 dimensions
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Save to CSV
df = pd.DataFrame({
    'id': id_list,
    'family': labels,
    'x': embedding_2d[:, 0],
    'y': embedding_2d[:, 1]
})
df.to_csv("embeddings2D.csv", index=False)
print("Saved embeddings2D.csv with 2D projection of embeddings.")
