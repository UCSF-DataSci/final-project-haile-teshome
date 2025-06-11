import os
import torch
import streamlit as st
import pandas as pd
import py3Dmol
import plotly.express as px
import openai
from sklearn.metrics.pairwise import euclidean_distances  # ✅ add this
from data import structure_to_graph, DOMAIN_FAMILIES
from model import GNNClassifier
st.set_page_config(page_title="Protein Dashboard", layout="wide")



openai.api_key = os.getenv("OPENAI_API_KEY")
# ---------------- OpenAI Key ---------------- #
if not openai.api_key:
    st.warning("🔐 OpenAI API key not set. Please configure it to enable chat.")
else:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.chat_history,
            temperature=0.5,
            max_tokens=400,
        )
        reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Error: {e}"

# ------------------------------------------------------------------
#  GLOBAL CSS  (you already have most of this; keep / merge as needed)
# ------------------------------------------------------------------
st.markdown("""
<style>
.stApp            { background-color: white; }
.block-container   { padding: 2rem 3rem; }
.stylable {
    background-color: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}
.subtitle { font-size: 1rem;  color: #666;   margin-bottom: 1rem; }
.metric   { font-size: 1.5rem; color: #379683; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
#  Context-manager helper so you can write `with stylable_card(): ...`
# ------------------------------------------------------------------
from contextlib import contextmanager
@contextmanager
def stylable_card():
    st.markdown("<div class='stylable'>", unsafe_allow_html=True)
    yield          # ← everything after this line is INSIDE the card
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Styling ---------------- #
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #001f3f;
}

/* Keep sidebar headers/labels white */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label {
    color: white !important;
}

/* Input elements: black text on white background */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select {
    color: black !important;
    background-color: white !important;
}

/* Other layout */
.stApp {
    background-color: white;
}
.block-container {
    padding: 2rem 3rem;
}
.stylable {
    background-color: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}
.title {
    font-size: 2rem;
    font-weight: 700;
    color: #05386B;
    margin-bottom: 0.5rem;
}
.subtitle {
    font-size: 1rem;
    color: #666;
    margin-bottom: 1rem;
}
.metric {
    font-size: 1.5rem;
    color: #379683;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Header ---------------- #
st.markdown("""
<div class="stylable">
    <div class="title">🧬 Protein Domain Family Classifier</div>
    <div class="subtitle">Search, visualize, and classify protein structures using a GNN trained on AlphaFold domains.</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Data ---------------- #
if not os.path.exists("embeddings2D.csv"):
    st.error("Missing embeddings2D.csv. Run visualize.py first.")
    st.stop()

df = pd.read_csv("embeddings2D.csv")

# ---------------- Sidebar with Chat ---------------- #
with st.sidebar:
    # Simple dropdown from internal dataset
    st.markdown("### Select a Protein from Dataset")
    protein_id = st.selectbox("Protein ID", sorted(df["id"].unique()))
    pdb_path = os.path.join("data", "structures", f"{protein_id}.pdb")
    actual_family = df[df["id"] == protein_id]["family"].values[0]

    st.markdown("---")
    st.markdown("### 💬 GenAI Assistant")
    st.markdown("Ask about the selected protein")

    # ---------------- Set up session state ---------------- #
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "system",
                "content": f"You are an expert on protein biology. The user is asking about protein ID '{protein_id}' from family '{actual_family}'."
            }
        ]

    # ---------------- Display previous messages ---------------- #
    for msg in st.session_state.chat_history[1:]:
        role = "🧠 Assistant" if msg["role"] == "assistant" else "🙋 You"
        st.markdown(f"**{role}:** {msg['content']}")

    # ---------------- Input + Submit ---------------- #
    user_prompt = st.text_input("Ask a question...", key="chat_input")

    if st.button("Submit Question"):
        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            if not openai.api_key:
                reply = "🔐 OpenAI API key not set."
            else:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.chat_history,
                        temperature=0.5,
                        max_tokens=400,
                    )
                    reply = response["choices"][0]["message"]["content"]
                except Exception as e:
                    reply = f"Error: {e}"

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.experimental_rerun()


# ---------------- Prediction ---------------- #
pdb_path = os.path.join("data", "structures", f"{protein_id}.pdb")
predicted_family = "N/A"
confidence_score = ""
top_k_families = []

if os.path.exists(pdb_path):
    model = GNNClassifier(num_node_features=21, hidden_dim=64, num_classes=len(DOMAIN_FAMILIES))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    
    graph = structure_to_graph(pdb_path)
    batch_vec = torch.zeros(graph.x.size(0), dtype=torch.long)

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index, batch_vec)
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        predicted_family = DOMAIN_FAMILIES[pred_idx]
        confidence_score = f"{probs[pred_idx].item() * 100:.1f}%"

        top_k = torch.topk(probs, k=3)
        top_k_families = [(DOMAIN_FAMILIES[i], f"{probs[i].item()*100:.1f}%") for i in top_k.indices]

# ---------------- Metrics with Descriptions + Top-K in Confidence Box ---------------- #
pfam_descriptions = {
    "PF00042": "ATP-powered molecule transport (ABC transporter)",
    "PF00072": "Bacterial signal relay (response regulator)",
    "PF00069": "Protein phosphorylation (kinase domain)",
    "PF00017": "Immune recognition (Ig-like domain)",
    "PF00089": "RNA binding and splicing (RRM)"
}

actual_description = pfam_descriptions.get(actual_family, "")
predicted_description = pfam_descriptions.get(predicted_family, "")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='stylable'>
        <div class='subtitle'>Actual Family</div>
        <div class='metric'>{actual_family}</div>
        <div style='font-size:0.9rem;color:#555;margin-top:0.5rem;'>{actual_description}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='stylable'>
        <div class='subtitle'>Predicted Family</div>
        <div class='metric'>{predicted_family}</div>
        <div style='font-size:0.9rem;color:#555;margin-top:0.5rem;'>{predicted_description}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='stylable'>
        <div class='subtitle'>Confidence</div>
        <div class='metric'>{confidence_score}</div>
        <div style='font-size:0.9rem;color:#555;margin-top:1rem;'>
            <b>Top 3 Predictions:</b>
            <table style='width:100%; margin-top:0.5rem; font-size:0.85rem;'>
                <tr><th style='text-align:left;'>Family</th><th style='text-align:right;'>Confidence</th></tr>
                {''.join(f"<tr><td>{fam}</td><td style='text-align:right;'>{conf}</td></tr>" for fam, conf in top_k_families)}
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- Similar Proteins by UMAP Distance ---------------- #

                    
st.markdown("#### Similar Proteins Nearby by UMAP space")
nearest_display = None
if protein_id in df["id"].values:
    current_vec = df.loc[df["id"] == protein_id, ["x", "y"]].values
    others      = df[df["id"] != protein_id].copy()

    if not others.empty:
        dists = euclidean_distances(current_vec, others[["x", "y"]].values)[0]
        nearest_display = (
            others.assign(distance=dists)
                    .sort_values("distance")
                    .head(5)[["id", "family", "distance"]]
                    .rename(columns={
                        "id":      "Protein ID",
                        "family":  "Family",
                        "distance":"UMAP Distance"
                    })
                    .assign(**{"UMAP Distance": lambda d: d["UMAP Distance"].round(4)})
        )
if nearest_display is not None:
    st.dataframe(nearest_display, use_container_width=True, hide_index=True)
else:
    st.info("No neighbouring proteins available to display.")


# ---------------- Side-by-Side Layout: UMAP + 3D Viewer ---------------- #
left_col, right_col = st.columns([2, 1])

with left_col:
    with st.container():
        st.markdown("""### UMAP Projection of Protein Embeddings
        """, unsafe_allow_html=True)

        fig = px.scatter(
            df, x="x", y="y", color="family", hover_name="id",
            labels={"x": "UMAP 1", "y": "UMAP 2"}
        )
        fig.update_traces(marker=dict(size=8))

        st.plotly_chart(fig, use_container_width=True)


with right_col:
    if os.path.exists(pdb_path):
        with st.container():
            st.markdown("""#### 3D Structure Viewer
            """, unsafe_allow_html=True)

            with open(pdb_path) as f:
                pdb_data = f.read()

            viewer = py3Dmol.view(width=400, height=350)
            viewer.addModel(pdb_data, "pdb")
            viewer.setStyle({"cartoon": {"color": "spectrum"}})
            viewer.zoomTo()

            # Also inside the card
            st.components.v1.html(viewer._make_html(), height=380, scrolling=False)

            st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Protein Comparison ---------------- #
st.markdown(""" ### Compare Two Proteins</div>
""", unsafe_allow_html=True)

# Select Protein A and B
compare_col1, compare_col2 = st.columns(2)
with compare_col1:
    protein_a = st.selectbox("Protein A", sorted(df["id"].unique()), key="comp_a")
with compare_col2:
    protein_b = st.selectbox("Protein B", sorted(df["id"].unique()), key="comp_b")

def get_prediction_info(pid):
    path = os.path.join("data", "structures", f"{pid}.pdb")
    if not os.path.exists(path):
        return {"actual": "N/A", "predicted": "N/A", "confidence": "N/A"}

    g = structure_to_graph(path)
    batch = torch.zeros(g.x.size(0), dtype=torch.long)
    with torch.no_grad():
        logits = model(g.x, g.edge_index, batch)
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        pred_fam = DOMAIN_FAMILIES[pred_idx]
        confidence = f"{probs[pred_idx].item() * 100:.1f}%"
    actual = df[df["id"] == pid]["family"].values[0]
    return {"actual": actual, "predicted": pred_fam, "confidence": confidence}

info_a = get_prediction_info(protein_a)
info_b = get_prediction_info(protein_b)

# UMAP distance
vec_a = df[df["id"] == protein_a][["x", "y"]].values
vec_b = df[df["id"] == protein_b][["x", "y"]].values
umap_dist = euclidean_distances(vec_a, vec_b)[0][0]
umap_dist = f"{umap_dist:.4f}"

# Show comparison table
compare_df = pd.DataFrame({
    "Metric": ["Actual Family", "Predicted Family", "Confidence", "UMAP Distance"],
    "Protein A": [info_a["actual"], info_a["predicted"], info_a["confidence"], "--"],
    "Protein B": [info_b["actual"], info_b["predicted"], info_b["confidence"], umap_dist]
})
st.table(compare_df)

