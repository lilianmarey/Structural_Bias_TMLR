import pickle
import numpy as np
import networkx as nx

from collections import Counter
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def make_parameter_grid(usecase, n_parameter_sample):

    if usecase == "friendship":
        param_values = {
            "alpha": [round(i, 3) for i in np.linspace(0.5, 0.9, n_parameter_sample)],
            "beta": [round(i, 3) for i in np.linspace(0, 3.8, n_parameter_sample)],
        }
    elif usecase in ["collab", "opinion"]:
        param_values = {
            "alpha": [round(i, 3) for i in np.linspace(0.5, 0.9, n_parameter_sample)],
            "beta": [round(i, 3) for i in np.linspace(0, 8, n_parameter_sample)],
        }
    else:
        raise ValueError

    params = list(
        product(
            param_values["alpha"],
            param_values["beta"],
        )
    )
    return params


def split_graph(G_base, test_size=.2, seed=13):
    edges = list(G_base.edges())
    _, test_edges = train_test_split(edges, test_size=test_size, random_state=seed)
    G_train = G_base.copy()
    G_train.remove_edges_from(test_edges)
    return G_train, test_edges


def save_pkl(object, path):
    with open(path, "wb") as f:
        pickle.dump(object, f)


def load_pickle(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x


def compute_topk_reco(embeddings, G, max_k=50):
    """
    Computing top_k recommendations from embeddings on all nodes of input graph.
    Based on cosine similarity from node embedding pairs.
    output :
        top_k_recommendations (dict) : {node_i : reco_node_i, ...}
        with reco_node_i being a list of nodes, being the ordered
            top 50 recommendations for node_i, the first index being the best reco.
        reco_node_i must not contain node_i and nodes already being neighbors of node_i in G_train
    """
    node_ids = []
    emb = []
    for key, val in embeddings.items():
        node_ids.append(key)
        emb.append(val)

    similarity_matrix = cosine_similarity(emb)
    train_edge_set = set([frozenset(edge) for edge in list(G.edges)])

    top_k_recommendations = {}
    for i, node in enumerate(node_ids):
        sim_scores = similarity_matrix[i]
        top_k_indices = np.argsort(sim_scores)[::-1]
        top_k_recommendations[node] = [
            node_ids[idx]
            for idx in top_k_indices

            if not (frozenset((node, node_ids[idx])) in train_edge_set) and idx != i
        ][:max_k]
    return top_k_recommendations


def compute_pred_metrics(G, reco, test_edges, topk):
    """
    Computing predictive metrics from recommendation results, for a specific top_k
    """
    ####################################  formatting #################################################
    sensitive_attr = {node: G.nodes[node]["sensitive"] for node in G.nodes()}
    test_edges_set = set(
        [frozenset((int(edge[0]), int(edge[1]))) for edge in test_edges]
    )

    # Gathering all recommendations in a single set
    predicted_edges = []
    for node, reco_node in reco.items():
        predicted_edges.extend(
            [frozenset((int(node), int(node_))) for node_ in reco_node[:topk]]
        )
    predicted_edges = set(predicted_edges)

    ####################################  Computing metrics #################################################

    ap, hits = [], []

    for node, reco_nodes in reco.items():
        predictable = [edge for edge in test_edges_set if node in edge]
        if len(predictable) == 0:
            continue

        # AP
        h, score = 0, 0
        for i, n in enumerate(reco_nodes[:topk]):
            if frozenset((node, n)) in predictable:
                h += 1
                score += h / (i + 1)
        ap.append(score / len(predictable))

        # Hit@k
        hits.append(
            int(any(frozenset((node, n)) in predictable for n in reco_nodes[:topk]))
        )

    avg_precision = np.mean(ap)
    hitrate = np.mean(hits)

    # Statistical Parity
    SP = np.abs(
        len(
            [
                reco_edge
                for reco_edge in predicted_edges
                if sensitive_attr[list(reco_edge)[0]]
                != sensitive_attr[list(reco_edge)[1]]
            ]
        )
        / len(predicted_edges)
        - len(
            [
                reco_edge
                for reco_edge in predicted_edges
                if sensitive_attr[list(reco_edge)[0]]
                == sensitive_attr[list(reco_edge)[1]]
            ]
        )
        / len(predicted_edges)
    )

    # Equal Opportunity
    true_sensitive = [
        e
        for e in test_edges_set
        if sensitive_attr[list(e)[0]] != sensitive_attr[list(e)[1]]
    ]

    true_nonsensitive = [
        e
        for e in test_edges_set
        if sensitive_attr[list(e)[0]] == sensitive_attr[list(e)[1]]
    ]

    # true positives
    tp_sensitive = sum(1 for e in predicted_edges if e in true_sensitive)
    tp_nonsensitive = sum(1 for e in predicted_edges if e in true_nonsensitive)

    # recall per group
    recall_sensitive = tp_sensitive / max(1, len(true_sensitive))
    recall_nonsensitive = tp_nonsensitive / max(1, len(true_nonsensitive))

    EO = np.abs(recall_sensitive - recall_nonsensitive)

    ####################################  Formatting results #################################################

    result = {
        "Hit": hitrate,
        "AP": avg_precision,
        "EO": EO,
        "SP": SP,
    }

    return result