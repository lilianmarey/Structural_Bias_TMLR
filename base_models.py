import networkx as nx
from node2vec import Node2Vec
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from tqdm import tqdm

################################################################################


def N2V_embedding(G, d):
    nodes = list(G.nodes())
    n2v_model = Node2Vec(
        G,
        dimensions=d,
        walk_length=80,
        num_walks=20,
        p=2,
        q=2,
        workers=10,
        quiet=True,
    ).fit()
    return dict([(node, n2v_model.wv[str(node)]) for node in nodes])


################################################################################


def SVD_embedding(G, d=32):
    nodes = list(G.nodes())
    L = nx.normalized_laplacian_matrix(G).toarray()
    L_shifted = L - L.min() + 1e-6
    svd_model = TruncatedSVD(n_components=d, random_state=13)
    W = svd_model.fit_transform(L_shifted)
    W_normalized = normalize(W)

    return {node: W_normalized[i].tolist() for i, node in enumerate(nodes)}


################################################################################


def NMF_embedding(G, d):
    nodes = list(G.nodes())

    L = nx.normalized_laplacian_matrix(G).toarray()
    L_shifted = L - L.min() + 1e-6
    nmf_model = NMF(n_components=d, init="random", random_state=13, max_iter=10000)
    W = nmf_model.fit_transform(L_shifted)
    W_normalized = normalize(W)

    return {node: W_normalized[i].tolist() for i, node in enumerate(nodes)}


################################################################################


class GCN(torch.nn.Module):
    """Two-layer Graph Convolutional Network for link-prediction embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def _graph_to_pyg(G) -> Data:
    """Convert an undirected NetworkX graph to a PyG Data object (identity features)."""
    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x = torch.eye(len(nodes))
    return Data(x=x, edge_index=edge_index)


def train_gcn(
    model: GCN,
    data: Data,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cpu",
    verbose: bool = True,
) -> GCN:

    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        iterations = tqdm(range(epochs))
    else:
        iterations = range(epochs)

    for _ in iterations:
        model.train()
        optimizer.zero_grad()

        z = model(data.x, data.edge_index)

        # --- link-prediction loss ---
        pos_edge_index = data.edge_index
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1) * 20,  # 20:1
        )
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()

        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

    return model


def GCN_embedding(G, d, device="cpu"):
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    data = _graph_to_pyg(G)
    model = GCN(in_dim=G.number_of_nodes(), hidden_dim=16, out_dim=d)

    train_gcn(
        model,
        data,
        epochs=200,
        lr=0.01,
        device=device,
        verbose=False,
    )

    model.eval()
    with torch.no_grad():
        z = model(data.x.to(device), data.edge_index.to(device)).cpu()

    return {node: list(z[node_idx[node]].numpy()) for node in nodes}
