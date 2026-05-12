"""
Script for computing and saving baselines results from amready computed results from "main.py".

folder_name : name of the folder the results have been saved in
"""

import os
from helpers import load_pickle, save_pkl
from baselines import compute_fairwalk, compute_debayes, compute_crosswalk
from joblib import Parallel, delayed

from globals import *

#####################################################################################

folder_name = "final"

#####################################################################################


def f(args):
    G_path, params_folder = args
    G_train = load_pickle(G_path)

    fairwalk_name = "fairwalk"
    fairwalk_path = f"{params_folder}/reco_{fairwalk_name}.pkl"
    save_pkl(compute_fairwalk(G_train), path=fairwalk_path)

    debayes_name = "debayes"
    debayes_path = f"{params_folder}/reco_{debayes_name}.pkl"
    save_pkl(compute_debayes(G_train), path=debayes_path)

    crosswalk_name = "crosswalk"
    crosswalk_path = f"{params_folder}/reco_{crosswalk_name}.pkl"
    save_pkl(compute_crosswalk(G_train), path=crosswalk_path)


#####################################################################################

folder = f"results/{folder_name}"

for usecase in USECASES:
    print(f"Running {usecase} usecase...")
    usecase_folder = f"{folder}/{usecase}"
    for run in [f for f in os.listdir(usecase_folder) if f != ".DS_Store"]:
        print(f"Running {run}...")
        run_folder = f"{usecase_folder}/{run}"
        args_list = [
            (f"{run_folder}/{params}/G_train.pkl", f"{run_folder}/{params}")
            for params in [f for f in os.listdir(run_folder) if f != ".DS_Store"]
        ]
        Parallel(n_jobs=-1)(delayed(f)(arg) for arg in args_list)
