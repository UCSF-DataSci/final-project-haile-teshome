# Protein Binding Prediction — Project Overview

This repository contains a self-contained machine learning project designed to predict protein–drug binding sites and Pfam domain families using a Graph Neural Network (GNN) trained on AlphaFold-predicted 3D protein structures. A visual and interactive Streamlit dashboard is also provided to explore and interpret the model's predictions.
<img width="1639" alt="Screenshot 2025-06-11 at 3 06 07 AM" src="https://github.com/user-attachments/assets/ba06ed6e-ed13-4891-a210-4f6f289e35c4" />
<img width="1644" alt="Screenshot 2025-06-11 at 3 06 20 AM" src="https://github.com/user-attachments/assets/f7eaf3f2-f092-4e4c-b960-9eecccba9555" />

---

## 1. Project Summary

Our goal was to build a pipeline that can:

1. Automatically download AlphaFold-predicted protein structures.
2. Convert them into graph representations suitable for GNNs.
3. Train a model to classify each protein domain’s Pfam family and identify likely binding-site residues.
4. Visualize both the embeddings and the predictions interactively.

---

## 2. Dataset Description

- **Source:** AlphaFold DB
- **Selection:** Proteins from 10 Pfam families, filtered to ensure:
  - Length < 300 residues
  - Chains with a single model
- **Graph format:**
  - **Nodes:** amino acid residues
  - **Edges:** residues whose alpha-carbon (Cα) atoms are < 8 Å apart
  - **Node features:** amino acid identity, 3D coordinates
- **Labels:**
  - Binary indicator for whether a residue is in a known binding site
  - One-hot encoded Pfam ID for multi-class classification

---

## 3. Model Architecture

- Implemented in PyTorch Geometric
- **Three-layer GCN**
  - Each layer updates node features via message passing
- **Global Mean Pooling** to collapse node-level embeddings to a single graph-level vector
- **Multi-head output**:
  - One head for Pfam classification (softmax)
  - One head for per-residue binding prediction (sigmoid)

---

## 4. Repository Layout

```
protein_binding_prediction/
├── model.py                # GCN architecture
├── data.py                 # Download & process AlphaFold structures → PyG graphs
├── train.py                # Training / validation loop (50 epochs)
├── visualize.py            # UMAP projections of embeddings
├── app.py                  # Streamlit inference & dashboard
├── drug_binding_site_predictor_streamlit.py  # legacy dashboard alias
├── model.pth               # Trained model weights
├── embeddings2D.csv        # 2D UMAP projection of graph embeddings
├── data/                   # Cached graph objects & downloaded PDBs
└── README.md               # This file
```

---

## 5. Installation & Setup

```bash
# Clone the repo
git clone https://github.com/your-handle/protein_binding_prediction.git
cd protein_binding_prediction

# Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
# OR manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric pandas scikit-learn umap-learn biopython streamlit matplotlib tqdm
```

# Create a .env file
Include an openai api key in .env file in order to use virtual assistant


---

## 6. Running the Project

| Step             | Command                                 | Description                                                                      |
| ---------------- | --------------------------------------- | -------------------------------------------------------------------------------- |
| Build dataset    | `python data.py --pfam PF00002 PF00073` | Downloads AlphaFold PDBs, parses residues, builds PyG `.pt` graphs               |
| Train model      | `python train.py --epochs 50 --lr 1e-3` | Trains GCN, saves `model.pth`                                                    |
| Visualize output | `python visualize.py`                   | UMAP projection of graph embeddings, saves `embeddings2D.csv` + `umap.png`       |
| Launch UI        | `streamlit run app.py`                  | Launches Streamlit dashboard for uploading structures or exploring test proteins |

To test it all together:

```bash
python data.py --pfam PF00002 PF00073
python train.py
python visualize.py
streamlit run app.py
```

The application should pop up in your browser after running streamline command.

---

## 7. Example Output

When given a chain like **P53\_HUMAN** (P04637):

- **Pfam classification**: predicted PF00870 with 94% confidence
- **Binding-site prediction**: residues 273–281 and 300–312 flagged as binding
- **Visualization**: UMAP plot shows clear separation by Pfam class

---

## 8. Decisions and Tradeoffs

### Dataset Constraints

- **Pfam Family Selection**: We focused on 10 families to balance diversity with tractable runtime. Including more would increase prediction utility but require larger models and more memory.
- **Residue Cutoff (<300)**: Shorter sequences make graphs more compact and speed up training, but exclude biologically important large proteins.
- **AlphaFold Filtering**: Only structures with a single model were used to avoid ambiguity. This excludes proteins with conformational diversity.

### Graph Construction Choices

- **Distance Threshold (8 Å)**: Chosen empirically to create enough edges without overwhelming sparsity or density. Other thresholds were tested but gave noisier graphs.
- **Only Alpha-Carbon (Cα)**: Simplifies graph topology. Using full backbone or sidechain atoms was deemed too computationally expensive.
- **Node Features**: Used simple residue identities and positions rather than richer biochemical features for speed.

### Model Architecture

- **GCN over GAT/GIN**: GCN was chosen for interpretability and lower parameter count. GAT and GIN were considered but more sensitive to hyperparameters.
- **Global Pooling**: GlobalMeanPooling was chosen over max or attention for its stability and ease of training.
- **Multitask Head**: Separating Pfam classification and binding prediction allowed the model to learn distinct representations.

### Development Constraints

- **No Hyperparameter Grid Search**: Manual tuning due to compute limitations. Default learning rate and layer sizes were sufficient.
- **Framework Choice**: PyTorch Geometric was selected over DGL due to documentation quality and integration with PyTorch.
- **Streamlit UI**: Chosen for simplicity and speed of deployment, despite limited control over UI layout.

### Challenges & Fixes

- **Memory Spikes**: Handled by using PyG’s `InMemoryDataset` and pre-caching graph objects.
- **Corrupted PDBs**: Filtered with Biopython and added try/except to ignore malformed entries.
- **Label Imbalance**: Addressed binding-site sparsity by weighting the loss function.

---


