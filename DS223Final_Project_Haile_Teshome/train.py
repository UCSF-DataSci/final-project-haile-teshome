import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from data import load_domain_graphs, DOMAIN_FAMILIES
from model import GNNClassifier

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
torch.manual_seed(42)

# Load dataset
families = DOMAIN_FAMILIES  # list of Pfam IDs
data_list, id_list = load_domain_graphs(families=families, max_per_family=100)
num_classes = len(families)

# Split into train and test sets (stratified by family labels)
labels = [int(data.y) for data in data_list]
train_indices, test_indices = train_test_split(
    list(range(len(data_list))), test_size=0.2, stratify=labels, random_state=42
)
train_data = [data_list[i] for i in train_indices]
test_data = [data_list[i] for i in test_indices]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, optimizer, loss
model = GNNClassifier(num_node_features=21, hidden_dim=64, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
best_test_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)  # forward pass
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Evaluate on training and test sets
    model.eval()
    def evaluate(loader):
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for batch in loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss_sum += loss.item()
                # Predicted class = argmax of logits
                pred = out.argmax(dim=1)
                correct += int((pred == batch.y).sum())
                total += batch.y.size(0)
        return loss_sum / len(loader), correct / total

    train_loss, train_acc = evaluate(train_loader)
    test_loss, test_acc = evaluate(test_loader)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "model.pth")  # save best model
    print(f"Epoch {epoch:02d}: "
          f"Train loss={train_loss:.4f}, acc={train_acc*100:.1f}%;  "
          f"Test loss={test_loss:.4f}, acc={test_acc*100:.1f}%")

print(f"Training complete. Best test accuracy: {best_test_acc*100:.1f}%")
