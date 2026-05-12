from base_models import *
from globals import *


def compute_node_embeddings(G, d=64):
    """
    Computes base models embeddings from graph
    """
    return {
        base_model: embedding_func(G, d)
        for base_model, embedding_func in EMBEDDING_MODELS.items()
    }