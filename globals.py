"""
Defining constant variables
"""

from base_models import *

USECASES = ["collab", "friendship", "opinion"]

K_VALUES = [10]

BASELINE_MODELS = []

BASE_PERF_METRICS = ["Hit", "AP"]
BASE_FAIRNESS_METRICS = ["EO", "SP"]

PRED_METRICS = BASE_PERF_METRICS + BASE_FAIRNESS_METRICS

ALL_METRICS = [
    f"{m}@{k}" for m in PRED_METRICS for k in K_VALUES
] 
FAIRNESS_METRICS = [
    f"{m}@{k}" for m in BASE_FAIRNESS_METRICS for k in K_VALUES
] 

EMBEDDING_MODELS = {
    "n2v": N2V_embedding,
    "svd": SVD_embedding,
    "nmf": NMF_embedding,
    "GCN": GCN_embedding,
}

# Managing bias measures

BIAS_MEASURES = [
    "closeness",
    "betweeness",
    "prestige",
    "degree",
    "constraint",
    "density",
    "heterogeneity",
    "isolation",
    "diameter",
    "control",
    "assortativity",
    "avg_mixed_distance",
    "power_law_exponent",
    "information_unfairness",
]

SCALED_BIAS = [
    "closeness",
    "betweeness",
    "prestige",
    "degree",
    "constraint",
    "heterogeneity",
    "density",
    "assortativity",
]

FROM_0_BIAS = [
    "isolation",
    "diameter",
    "control",
    "information_unfairness",
]

NO_SCALE_BIAS = [
    "power_law_exponent",
    "avg_mixed_distance",
]
