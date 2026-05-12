from joblib import Parallel, delayed
import os
from globals import *
from helpers_analyze import *

#####################################################################################

folder_name = "test"

#####################################################################################

folder = f"results/{folder_name}"
save_folder = f"analysis/{folder_name}"
os.makedirs(save_folder, exist_ok=True)

try:
    df_results = {
    usecase: pd.read_csv(f"analysis/{folder_name}/{usecase}/results.csv")
    for usecase in USECASES
}
except:
    df_results = results_summary_dataframes(folder)

def process_usecase(usecase, df, save_folder):
    usecase_folder = f"{save_folder}/{usecase}"
    os.makedirs(usecase_folder, exist_ok=True)
    df.to_csv(f"{usecase_folder}/results.csv", index=False)

Parallel(n_jobs=-1)(
    delayed(process_usecase)(usecase, df, save_folder)
    for usecase, df in df_results.items()
)

Parallel(n_jobs=-1)(
    delayed(func)(df_results, save_folder)
    for func in [
        save_bias_heatmaps,
        save_pred_heatmaps,
        save_correlations,
        save_RF_importance,
        save_regression_results,
    ]
)

save_aggregate_regression_results(save_folder)
