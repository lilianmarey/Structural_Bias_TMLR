"""
Script for computing and saving graph generation, graph bias, GCN, SVD and N2V embeddings, and top-k recommendation results.

split_seed : seed to use for splitting train and test edges

test_size : size of test edges set

n_parameter_sample : number of parameter to compute per axis (beta for homophily parameter, alpha for size_ratio parameter).
                    The parameter bounds are defined according to the usecase at hand

n_runs : Number of runs to perform per usecase

folder_name : name of the folder where results are saved
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
from helpers import (
    make_parameter_grid,
    split_graph,
    save_pkl,
    compute_topk_reco,
)
from embedding_computer import compute_node_embeddings
from graph_generator import generator
from bias_measures import Bias
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from globals import *

#####################################################################################

split_seed = 13
test_size = 0.2
n_parameter_sample = 4
n_runs = 1

folder_name = "test"

#####################################################################################

top_folder = f"results/{folder_name}"


def f(args):
    parameter, usecase, top_folder, run, test_size, split_seed = args
    alpha, beta = parameter
    folder = f"{top_folder}/{usecase}/run{run}/{alpha}_{beta}"
    os.makedirs(folder, exist_ok=True)

    bias_path = f"{folder}/bias.pkl"
    graph_path = f"{folder}/G_train.pkl"
    test_edges_path = f"{folder}/test_edges.pkl"
    embeddings_path = f"{folder}/embeddings.pkl"

    G_base = generator(usecase, alpha, beta, run)
    G_train, test_edges = split_graph(G_base, test_size, split_seed)
    save_pkl(G_train, graph_path)
    save_pkl(test_edges, test_edges_path)

    bias_computer = Bias(G_train)
    bias_values = bias_computer.compute_bias_values()
    save_pkl(bias_values, path=bias_path)

    node_embeddings = compute_node_embeddings(G_train)
    save_pkl(node_embeddings, path=embeddings_path)

    for method, embeddings in node_embeddings.items():
        reco_path = f"{folder}/reco_{method}.pkl"
        reco = compute_topk_reco(embeddings, G_train)
        save_pkl(reco, path=reco_path)


#####################################################################################

os.makedirs(top_folder, exist_ok=True)

for usecase in USECASES:
    print(f"Running {usecase} usecase...")
    parameter_grid = make_parameter_grid(usecase, n_parameter_sample)
    for run in range(1, n_runs + 1):
        print(f"Running run {run}...")
        args_list = [
            (parameter, usecase, top_folder, run, test_size, split_seed)
            for parameter in parameter_grid
        ]
        with tqdm_joblib(tqdm(desc="Processing", total=len(args_list))):
            results = Parallel(n_jobs=-1)(
                delayed(f)(arg) for arg in args_list
            )