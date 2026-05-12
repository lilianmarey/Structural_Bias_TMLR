import numpy as np
import networkx as nx


def generator(usecase, alpha, beta, seed=13):
    """
    Generate a synthetic graph based on the specified usecase.

    This function creates a graph where nodes are added iteratively as in Barabasi-Albert's method,
    with edge creation influenced by the given usecase (opinion, friendship, or collaborative), homophily
    parameter beta, and other parameters. Each node is assigned a "sensitive attribute"
    used to guide edge formation probabilities.

    Parameters:
    ----------
    usecase : str
        Specifies the type of graph generation usecase.
    alpha : float
        Proportion of non-sensitive nodes.
    beta : float
        Parameter controlling the tendency of nodes to connect with others of the same
        sensitive attribute. Higher values imply stronger homophily.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    """

    np.random.seed(seed)

    if usecase == "friendship":
        n = 1034
        m = 13
        scale = None

    elif usecase == "collab":
        n = 860
        m = 3
        scale = 1
    elif usecase == "opinion":
        n = 1222
        m = 14
        scale = 0.08
    else:
        raise ValueError

    n_sensistive = round(n * (1 - alpha))
    attributes_list = [1 for _ in range(n_sensistive)] + [
        0 for _ in range(n - n_sensistive)
    ]
    np.random.shuffle(attributes_list)

    sensitive_attribute_dict = dict()
    G = nx.star_graph(m + 1)

    for node in G:
        sensitive_attribute_dict[node] = attributes_list.pop()

    while len(attributes_list) > 0:
        new_node_index = len(G)
        new_node_attribute = attributes_list.pop()
        sensitive_attribute_dict[new_node_index] = new_node_attribute
        degrees = np.array(list(dict(G.degree).values()))
        same_sensitive_attribute_mask = (
            np.array(list(sensitive_attribute_dict.values())[:-1]) == new_node_attribute
        ).astype(int)
        adjusted_degrees = degrees + (
            (np.exp(beta) - 1) * same_sensitive_attribute_mask
        )
        attachment_probas = adjusted_degrees / np.sum(adjusted_degrees)

        if usecase == "opinion":

            new_m = max(1, round(np.random.gamma(shape=scale, scale=(1 / scale) * m)))
            new_m = min(new_m, len(attachment_probas))

            chosen_nodes = list(
                np.random.choice(
                    list(range(len(attachment_probas))),
                    replace=False,
                    size=new_m,
                    p=attachment_probas,
                )
            )
        elif usecase == "friendship":

            adjusted_degrees = 1 + ((np.exp(beta) - 1) * same_sensitive_attribute_mask)
            attachment_probas = adjusted_degrees / np.sum(adjusted_degrees)

            anchor = np.random.choice(
                list(range(len(attachment_probas))),
                p=attachment_probas,
            )

            neighs = list(nx.neighbors(G, anchor))
            neighs_of_neighs = [
                i for j in [list(nx.neighbors(G, neigh)) for neigh in neighs] for i in j
            ]
            new_m = min(
                round(len(attachment_probas)),
                round(0.55 * len(neighs) + 3),
            )
            a, b, c = 10e3, 1, 1
            neighbor_mask = np.array(
                [a if node in neighs else 0 for node in list(G.nodes())]
            )
            neighbor_2_mask = np.array(
                [b if node in neighs_of_neighs else 0 for node in list(G.nodes())]
            )
            post_anchor_attachment_probas = c + neighbor_mask + neighbor_2_mask
            post_anchor_attachment_probas = post_anchor_attachment_probas / np.sum(
                post_anchor_attachment_probas
            )

            chosen_nodes = [anchor] + list(
                np.random.choice(
                    list(range(len(attachment_probas))),
                    replace=False,
                    size=new_m - 1,
                    p=post_anchor_attachment_probas,
                )
            )
        elif usecase == "collab":

            anchor = np.random.choice(
                list(range(len(attachment_probas))),
                p=attachment_probas,
            )

            neighs = list(nx.neighbors(G, anchor))
            new_m = max(
                1, round(np.random.gamma(shape=scale, scale=(1 / scale) * m) + 1)
            )
            new_m = min(new_m, len(neighs) + 1)
            a, b = np.exp(10), 10e-15
            neighbor_mask = np.array(
                [a if node in neighs else 0 for node in list(G.nodes())]
            )
            post_anchor_attachment_probas = b + neighbor_mask
            post_anchor_attachment_probas = post_anchor_attachment_probas / np.sum(
                post_anchor_attachment_probas
            )

            chosen_nodes = [anchor] + list(
                np.random.choice(
                    list(range(len(attachment_probas))),
                    replace=False,
                    size=new_m - 1,
                    p=post_anchor_attachment_probas,
                )
            )
        new_edges = list(
            zip(
                [new_node_index] * new_m,
                chosen_nodes,
            )
        )
        G.add_edges_from(new_edges)

    nx.set_node_attributes(G, sensitive_attribute_dict, "sensitive")

    return G
