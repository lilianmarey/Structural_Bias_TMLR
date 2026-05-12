import numpy as np
import networkx as nx
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")


class Bias:
    """
    Implementation of structural bias measures
    """

    def __init__(self, G):
        self.G = G
        self.nodes = list(G.nodes())
        self.neighbors_dict = {n: list(G.neighbors(n)) for n in G.nodes()}
        self.sensitive_nodes = [i for i in self.nodes if G.nodes[i]["sensitive"] == 1]
        self.non_sensitive_nodes = [
            i for i in self.nodes if G.nodes[i]["sensitive"] == 0
        ]

        self.betweeness_centrality = nx.betweenness_centrality(self.G)
        self.closeness_centrality = nx.closeness_centrality(self.G)
        self.prestige_centrality = nx.eigenvector_centrality(G, max_iter=int(10e6))
        self.shortest_paths = dict(nx.shortest_path(G))
        self.resistance_distances = nx.resistance_distance(
            G.subgraph(list(nx.connected_components(G))[0])
        )

    def aggregate(self, f):
        sensitive_values = [f(v) for v in self.sensitive_nodes]
        non_sensitive_values = [f(v) for v in self.non_sensitive_nodes]

        mean_sensitive = np.nanmean(sensitive_values)
        mean_non_sensitive = np.nanmean(non_sensitive_values)

        return (mean_non_sensitive - mean_sensitive) / np.nanmean(
            mean_non_sensitive + mean_sensitive
        )

    def aggregate_effective_resistance(self, f):
        sensitive_values = [f(v) for v in self.sensitive_nodes]
        non_sensitive_values = [f(v) for v in self.non_sensitive_nodes]

        mean_sensitive = np.nanmean(sensitive_values)
        mean_non_sensitive = np.nanmean(non_sensitive_values)

        return np.abs(mean_sensitive - mean_non_sensitive)

    #########################################################

    def degree_(self, node):
        return len(self.neighbors_dict[node])

    def degree(self):
        return self.aggregate(self.degree_)

    ###########################

    def constraint_(self, node):
        return np.sum([len(self.neighbors_dict[v]) for v in self.neighbors_dict[node]])

    def constraint(self):
        return self.aggregate(self.constraint_)

    ###########################

    def density_(self, node):
        sub_G = self.G.subgraph(self.neighbors_dict[node] + [node])
        return 1 - nx.density(sub_G)

    def density(self):
        return self.aggregate(self.density_)

    ###########################

    def prestige_(self, node):
        return self.prestige_centrality[node]

    def prestige(self):
        return self.aggregate(self.prestige_)

    ###########################

    def get_power_law_coeff_from_deg_list(self, deg_list):

        deg_values = list(range(np.min(deg_list) + 1, np.max(deg_list) + 1))
        counts = [deg_list.count(i + 1) for i in deg_values]

        deg_points = [i for i in list(zip(deg_values, counts)) if i[1] > 0]
        log_deg_x = [[np.log(i[0])] for i in deg_points]
        log_deg_y = [np.log(i[1]) for i in deg_points]

        lin_reg = LinearRegression()
        lin_reg.fit(log_deg_x, log_deg_y)
        return -lin_reg.coef_[0]

    def power_law_exponent(self):
        degrees = dict(nx.degree(self.G))
        sensitive_degrees_list = [
            degrees[node] for node in self.nodes if self.G.nodes[node]["sensitive"] == 1
        ]
        non_sensitive_degrees_list = [
            degrees[node] for node in self.nodes if self.G.nodes[node]["sensitive"] == 0
        ]

        return self.get_power_law_coeff_from_deg_list(
            sensitive_degrees_list
        ) / self.get_power_law_coeff_from_deg_list(non_sensitive_degrees_list)

    ###########################

    def betweeness_(self, node):
        return self.betweeness_centrality[node]

    def betweeness(self):
        return self.aggregate(self.betweeness_)

    ###########################

    def closeness_(self, node):
        return self.closeness_centrality[node]

    def closeness(self):
        return self.aggregate(self.closeness_)

    ###########################

    def heterogeneity_(self, node):
        values = [self.G.nodes[v]["sensitive"] for v in self.neighbors_dict[node]]
        if values == []:
            return np.nan
        else:
            return 1 - 2 * np.abs(np.mean(values) - 0.5)

    def heterogeneity(self):
        return self.aggregate(self.heterogeneity_)

    ###########################
    def assortativity(self):
        return nx.attribute_assortativity_coefficient(self.G, "sensitive")

    ###########################

    def avg_mixed_distance(self):

        distances_between_groups = []
        for u in self.sensitive_nodes:
            for v in self.non_sensitive_nodes:
                try:
                    distances_between_groups.append(len(self.shortest_paths[u][v]))
                except KeyError:
                    pass

        return np.mean(distances_between_groups)

    ###########################
    # Effective Resistance measures

    def isolation_(self, v):
        values = []
        for u in self.G.nodes:
            try:
                values.append(self.resistance_distances[v][u])
            except:
                pass
        if values == []:
            return np.nan
        else:
            return np.mean(values)

    def isolation(self):
        return self.aggregate_effective_resistance(self.isolation_)

    def diameter_(self, v):
        values = []
        for u in self.G.nodes:
            try:
                values.append(self.resistance_distances[v][u])
            except:
                pass
        if values == []:
            return np.nan
        else:
            return np.max(values)

    def diameter(self):
        return self.aggregate_effective_resistance(self.diameter_)

    def control_(self, v):
        values = []
        for u in list(nx.neighbors(self.G, v)):
            try:
                values.append(self.resistance_distances[v][u])
            except:
                pass
        if values == []:
            return np.nan
        else:
            return np.sum(values)

    def control(self):
        return self.aggregate_effective_resistance(self.control_)

    ###########################
    # information unfairness

    def max_distance(self, a, b, c):
        distance_1 = abs(a - b)
        distance_2 = abs(a - c)
        distance_3 = abs(b - c)
        return max(distance_1, distance_2, distance_3)

    def information_unfairness(self):

        adjacency_matrix = nx.adjacency_matrix(self.G).toarray()
        accessibility_matrix = (
            (1 / 2) * adjacency_matrix
            + (1 / 3) * (adjacency_matrix.dot(adjacency_matrix))
            + (1 / 4) * (adjacency_matrix.dot(adjacency_matrix).dot(adjacency_matrix))
        )
        np.fill_diagonal(accessibility_matrix, 0)

        #############################################################
        intra_category_mask = np.zeros_like(accessibility_matrix)
        for node_i in self.sensitive_nodes:
            for node_j in self.sensitive_nodes:
                intra_category_mask[node_i, node_j] = 1
        intra_category_accessibility_1 = np.multiply(
            accessibility_matrix, intra_category_mask
        )
        average_intra_category_accessibility_1 = np.mean(intra_category_accessibility_1)

        intra_category_mask = np.zeros_like(accessibility_matrix)
        for node_i in self.non_sensitive_nodes:
            for node_j in self.non_sensitive_nodes:
                intra_category_mask[node_i, node_j] = 1
        intra_category_accessibility_2 = np.multiply(
            accessibility_matrix, intra_category_mask
        )
        average_intra_category_accessibility_2 = np.mean(intra_category_accessibility_2)

        intra_category_mask = np.zeros_like(accessibility_matrix)
        for node_i in self.sensitive_nodes:
            for node_j in self.non_sensitive_nodes:
                intra_category_mask[node_i, node_j] = 1
        intra_category_accessibility_1_2 = np.multiply(
            accessibility_matrix, intra_category_mask
        )
        average_intra_category_accessibility_1_2 = np.mean(
            intra_category_accessibility_1_2
        )

        value = self.max_distance(
            average_intra_category_accessibility_1,
            average_intra_category_accessibility_2,
            average_intra_category_accessibility_1_2,
        )
        return value

    #############################################################

    def compute_bias_values(self):

        values = dict()

        values["closeness"] = self.aggregate(self.closeness_)
        values["betweeness"] = self.aggregate(self.betweeness_)
        values["prestige"] = self.aggregate(self.prestige_)
        values["degree"] = self.aggregate(self.degree_)
        values["constraint"] = self.aggregate(self.constraint_)
        values["density"] = self.aggregate(self.density_)
        values["heterogeneity"] = self.aggregate(self.heterogeneity_)

        values["isolation"] = self.isolation()
        values["diameter"] = self.diameter()
        values["control"] = self.control()
        values["avg_mixed_distance"] = self.avg_mixed_distance()
        values["assortativity"] = self.assortativity()
        values["power_law_exponent"] = self.power_law_exponent()
        values["information_unfairness"] = self.information_unfairness()

        return values
