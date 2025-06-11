import os
import math
import random
import requests
import numpy as np
from Bio.PDB import PDBParser, Polypeptide
import torch
from torch_geometric.data import Data
from Bio.Data.IUPACData import protein_letters_3to1

# Pre-defined Pfam families to use (Pfam IDs)
DOMAIN_FAMILIES = ['PF00017', 'PF00018', 'PF00042', 'PF00069', 'PF00072', 'PF00089']

# Optional length thresholds for filtering (family-specific)
LENGTH_THRESHOLDS = {
    'PF00017': 200,  # SH2 domains ~100 aa; filter out sequences >200 aa (likely multi-domain)
    'PF00018': 150,  # SH3 ~60 aa; filter out >150 aa
    'PF00042': 250,  # Globin ~150 aa; filter out >250 aa
    'PF00069': 400,  # Kinase ~300 aa; filter out >400 aa
    'PF00072': 200,  # Receiver ~120 aa; filter out >200 aa (to avoid attached output domain)
    'PF00089': 350   # Trypsin ~240 aa; filter out >350 aa (to avoid multi-domain proteases)
}


# Map 3-letter amino acid codes to 1-letter codes (for standard amino acids)
def safe_three_to_one(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except KeyError:
        if resname == "SEC":
            return "U"
        elif resname == "PYL":
            return "O"
        else:
            return "X"

# Create a mapping from 1-letter AA to index 0-19, and index 20 for unknown
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
AA_TO_INDEX['X'] = len(AA_TO_INDEX)  # 'X' or any unknown = last index (20)

def fetch_family_members(pfam_id, max_members=100, reviewed=True):
    """Query UniProt to get a list of protein accession IDs for a given Pfam family."""
    query = f"database:Pfam {pfam_id}"
    if reviewed:
        query += " AND reviewed:true"  # limit to Swiss-Prot (reviewed) proteins if desired
    url = ("https://rest.uniprot.org/uniprotkb/search"
           f"?query={query}&format=tsv&fields=accession,length&size={max_members}")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"UniProt API query failed for {pfam_id}: {response.status_code}")
    lines = response.text.strip().split('\n')
    # TSV header is "Entry\tLength"
    members = []
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        acc, length_str = parts[0], parts[1]
        try:
            length = int(length_str)
        except:
            length = None
        # Apply length filter if threshold is set for this family
        if pfam_id in LENGTH_THRESHOLDS and length is not None:
            if length > LENGTH_THRESHOLDS[pfam_id]:
                continue
        members.append(acc)
    return members

def download_structure(uniprot_id, out_dir):
    """Download AlphaFold structure for a UniProt ID. Saves as PDB file if not already present."""
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f"{uniprot_id}.pdb")
    if os.path.isfile(pdb_path):
        return pdb_path  # already downloaded
    # Construct URL for AlphaFold model (using current version 4)
    af_id = f"AF-{uniprot_id}-F1"
    url = f"https://alphafold.ebi.ac.uk/files/{af_id}-model_v4.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        with open(pdb_path, 'wb') as f:
            f.write(r.content)
        return pdb_path
    else:
        raise Exception(f"Failed to download structure for {uniprot_id} (status {r.status_code})")

def structure_to_graph(pdb_file):
    """Parse a PDB file and convert to a PyG Data object (residue graph)."""
    parser = PDBParser(QUIET=True)  # QUIET=True to suppress warnings
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    coords = []
    types = []
    for res in structure.get_residues():
        # Consider only standard amino acid residues
        if not Polypeptide.is_aa(res, standard=True):
            continue
        res_name = res.get_resname()
        # Convert 3-letter to 1-letter code, default 'X' if unknown
        try:
            aa_letter = safe_three_to_one(res_name)
        except KeyError:
            aa_letter = 'X'
        coords.append(list(res['CA'].get_coord()))  # C-alpha coordinate
        types.append(AA_TO_INDEX.get(aa_letter, AA_TO_INDEX['X']))
    num_nodes = len(types)
    if num_nodes == 0:
        raise Exception(f"No amino acids parsed from structure {pdb_file}")
    # Compute edges based on distance threshold
    coord_array = np.array(coords)
    # Pairwise distance matrix
    dist_mat = np.linalg.norm(coord_array[:, None, :] - coord_array[None, :, :], axis=-1)
    src_indices, dst_indices = np.where((dist_mat < 8.0) & (dist_mat > 0.0))
    # Use only each pair once (undirected graph) by enforcing src < dst
    edges = set()
    for i, j in zip(src_indices, dst_indices):
        if i < j:
            edges.add((i, j))
            edges.add((j, i))
    if len(edges) == 0:
        # If no edges (very small protein), connect sequentially as fallback
        for i in range(num_nodes - 1):
            edges.add((i, i+1))
            edges.add((i+1, i))
    # Convert edge list to tensor
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    # edge_index.shape should be [2, E]
    # Node feature tensor
    x = torch.tensor(types, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data

def load_domain_graphs(families=None, max_per_family=100, data_dir="data"):
    """Fetch data for the specified families and return list of graphs with labels and corresponding IDs."""
    if families is None:
        families = DOMAIN_FAMILIES
    data_list = []
    id_list = []
    class_index = {fam: idx for idx, fam in enumerate(families)}
    struct_dir = os.path.join(data_dir, "structures")
    os.makedirs(struct_dir, exist_ok=True)
    for fam in families:
        members = fetch_family_members(fam, max_members=max_per_family)
        for uid in members:
            try:
                pdb_file = download_structure(uid, struct_dir)
            except Exception as e:
                print(f"Warning: could not download structure for {uid} ({e})")
                continue
            try:
                graph = structure_to_graph(pdb_file)
            except Exception as e:
                print(f"Warning: failed to parse structure for {uid} ({e})")
                continue
            # Assign label
            label_idx = class_index[fam]
            graph.y = torch.tensor(label_idx, dtype=torch.long)
            data_list.append(graph)
            id_list.append(uid)
    return data_list, id_list
