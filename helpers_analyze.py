import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error

from helpers import load_pickle, compute_pred_metrics

from globals import *


def results_summary_dataframes(folder, models=list(EMBEDDING_MODELS.keys())):
    df_results = dict()
    for usecase in USECASES:
        results = []
        usecase_folder = f"{folder}/{usecase}"
        for run in [f for f in os.listdir(usecase_folder) if f != ".DS_Store"]:
            run_folder = f"{usecase_folder}/{run}"
            for params in tqdm([f for f in os.listdir(run_folder) if f != ".DS_Store"]):

                result_line = [usecase, run] + [float(i) for i in params.split("_")]

                params_folder = f"{run_folder}/{params}"

                test_edges = load_pickle(f"{params_folder}/test_edges.pkl")
                G_train = load_pickle(f"{params_folder}/G_train.pkl")

                bias = load_pickle(f"{params_folder}/bias.pkl")
                result_line += [bias[measure] for measure in BIAS_MEASURES]

                for model in models:
                    reco = load_pickle(f"{params_folder}/reco_{model}.pkl")
                    for k in K_VALUES:
                        pred_metrics = compute_pred_metrics(
                            G_train, reco, test_edges, topk=k
                        )
                        result_line += [pred_metrics[metric] for metric in PRED_METRICS]
                results.append(result_line)

        columns = (
            ["usecase", "run", "beta", "alpha"]
            + BIAS_MEASURES
            + [
                f"{model}_{m}@{k}"
                for model in models
                for k in K_VALUES
                for m in PRED_METRICS
            ]
        )
        df = pd.DataFrame(results, columns=columns)
        df_results[usecase] = df
    return df_results


def save_bias_heatmaps(df_results, save_folder):
    for usecase in USECASES:
        bias_heatmaps_folder = f"{save_folder}/{usecase}/heatmaps/bias"
        os.makedirs(bias_heatmaps_folder, exist_ok=True)
        for measure in BIAS_MEASURES:
            plot_df = (
                df_results[usecase][["beta", "alpha", measure]]
                .groupby(["beta", "alpha"])
                .mean()
                .reset_index()
            )
            heatmap_data = plot_df.pivot(index="alpha", columns="beta", values=measure)

            vmax_abs = np.abs(heatmap_data.values).max()
            vmax = heatmap_data.values.max()
            vmin = heatmap_data.values.min()

            if measure in SCALED_BIAS:
                cmap = LinearSegmentedColormap.from_list(
                    "custom_diverging", ["tab:blue", "whitesmoke", "tab:red"]
                )
                norm = Normalize(vmin=-vmax_abs, vmax=vmax_abs)
            elif measure in FROM_0_BIAS:
                cmap = LinearSegmentedColormap.from_list(
                    "custom_diverging", ["whitesmoke", "tab:red"]
                )
                norm = Normalize(vmin=0, vmax=vmax_abs)
            elif measure in NO_SCALE_BIAS:
                cmap = LinearSegmentedColormap.from_list(
                    "custom_diverging", ["whitesmoke", "tab:red"]
                )
                norm = Normalize(vmin=vmin, vmax=vmax)

            plt.figure(figsize=(4, 3))
            ax = sns.heatmap(
                heatmap_data,
                cmap=cmap,
                norm=norm,
                annot=False,
                fmt=".2f",
                xticklabels=3,
                yticklabels=3,
            )
            cbar = ax.collections[0].colorbar
            cbar_ticks = np.linspace(cbar.vmin, cbar.vmax, 3)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([f"{t:.2f}" for t in cbar_ticks])
            cbar.ax.tick_params(labelsize=16)
            ax.set_xticks([0, len(heatmap_data.columns) - 1])
            ax.set_xticklabels(
                [f"{heatmap_data.columns[0]:.2f}", f"{heatmap_data.columns[-1]:.2f}"]
            )

            ax.set_yticks([0, len(heatmap_data.index) - 1])
            ax.set_yticklabels(
                [
                    f"{heatmap_data.index[0]:.2f}",
                    f"{heatmap_data.index[-1]:.2f}",
                ]
            )
            ax.tick_params(axis="both", labelsize=13)
            ax.xaxis.set_tick_params(pad=10)
            plt.gca().invert_yaxis()
            plt.xlabel(r"class imbalance ($\alpha$)", fontsize=16)
            plt.ylabel(r"homophily ($\beta$)", fontsize=16)
            plt.tight_layout()

            plt.savefig(
                f"{bias_heatmaps_folder}/{measure}.pdf",
                format="pdf",
                bbox_inches="tight",
            )
            plt.close()


def save_pred_heatmaps(df_results, save_folder, models=list(EMBEDDING_MODELS.keys())):

    for usecase in USECASES:
        metrics_heatmaps_folder = f"{save_folder}/{usecase}/heatmaps/metrics"
        os.makedirs(metrics_heatmaps_folder, exist_ok=True)
        for model in models:
            model_heatmaps_folder = f"{save_folder}/{usecase}/heatmaps/metrics/{model}"
            os.makedirs(model_heatmaps_folder, exist_ok=True)
            for metric in ALL_METRICS:

                plot_df = (
                    df_results[usecase][["beta", "alpha", f"{model}_{metric}"]]
                    .groupby(["beta", "alpha"])
                    .mean()
                    .reset_index()
                )
                heatmap_data = plot_df.pivot(
                    index="alpha", columns="beta", values=f"{model}_{metric}"
                )

                vmax = heatmap_data.values.max()
                vmin = heatmap_data.values.min()

                cmap = LinearSegmentedColormap.from_list(
                    "custom_diverging", ["whitesmoke", "tab:red"]
                )
                norm = Normalize(vmin=vmin, vmax=vmax)

                plt.figure(figsize=(4, 3))
                ax = sns.heatmap(
                    heatmap_data,
                    cmap=cmap,
                    norm=norm,
                    annot=False,
                    fmt=".2f",
                    xticklabels=3,
                    yticklabels=3,
                )
                cbar = ax.collections[0].colorbar
                cbar_ticks = np.linspace(cbar.vmin, cbar.vmax, 3)
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([f"{t:.2f}" for t in cbar_ticks])
                cbar.ax.tick_params(labelsize=16)
                ax.set_xticks([0, len(heatmap_data.columns) - 1])
                ax.set_xticklabels(
                    [
                        f"{heatmap_data.columns[0]:.2f}",
                        f"{heatmap_data.columns[-1]:.2f}",
                    ]
                )

                ax.set_yticks([0, len(heatmap_data.index) - 1])
                ax.set_yticklabels(
                    [
                        f"{heatmap_data.index[0]:.2f}",
                        f"{heatmap_data.index[-1]:.2f}",
                    ]
                )
                ax.tick_params(axis="both", labelsize=13)
                ax.xaxis.set_tick_params(pad=10)
                plt.gca().invert_yaxis()
                plt.xlabel(r"class imbalance ($\alpha$)", fontsize=16)
                plt.ylabel(r"homophily ($\beta$)", fontsize=16)
                plt.tight_layout()

                plt.savefig(
                    f"{model_heatmaps_folder}/{metric}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
                plt.close()


def save_correlations(df_results, save_folder):
    for usecase in USECASES:
        plot_df = (
            df_results[usecase]
            .drop(columns=["run", "usecase"])
            .groupby(["beta", "alpha"])
            .mean()
            .reset_index()
            .drop(columns=["beta", "alpha"])[BIAS_MEASURES]
        )
        cols = plot_df[BIAS_MEASURES].columns
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for col1 in cols:
            for col2 in cols:
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                    pval_matrix.loc[col1, col2] = 0.0
                else:
                    corr, pval = pearsonr(
                        plot_df[BIAS_MEASURES][col1], plot_df[BIAS_MEASURES][col2]
                    )
                    corr_matrix.loc[col1, col2] = corr
                    pval_matrix.loc[col1, col2] = pval

        beta = 0.01
        significant_corr = corr_matrix.mask(pval_matrix > beta, other=0.0)

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            significant_corr,
            annot=False,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
        )
        plt.tight_layout()
        cbar = ax.collections[0].colorbar
        cbar_ticks = np.linspace(cbar.vmin, cbar.vmax, 3)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in cbar_ticks], size=16)
        ax.tick_params(axis="both", labelsize=16)
        plt.savefig(
            f"{save_folder}/{usecase}/correlations.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.savefig(
            f"{save_folder}/{usecase}/correlations.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()


def save_RF_importance(df_results, save_folder, models=list(EMBEDDING_MODELS.keys())):

    for usecase in USECASES:
        importance_folder = f"{save_folder}/{usecase}/importance/RF"
        os.makedirs(importance_folder, exist_ok=True)
        for model in models:
            model_importance_folder = f"{importance_folder}/{model}"
            os.makedirs(model_importance_folder, exist_ok=True)
            for metric in FAIRNESS_METRICS:
                data = df_results[usecase].drop(
                    columns=["usecase", "beta", "alpha", "run"]
                )
                X = data[BIAS_MEASURES].to_numpy()
                y = data[[f"{model}_{metric}"]].to_numpy().ravel()

                reg_model = RandomForestRegressor()
                reg_model.fit(X, y)
                coeffs_df = pd.DataFrame(
                    list(zip(BIAS_MEASURES, list(reg_model.feature_importances_))),
                    columns=["measure", "coeff"],
                )
                fig, axes = plt.subplots(
                    1,
                    1,
                    figsize=(5, 8),
                )
                sns.barplot(
                    data=coeffs_df,
                    y="measure",
                    x="coeff",
                    hue="coeff",
                    palette="dark:#5A9_r",
                    edgecolor="black",
                    legend=False,
                )
                axes.set_ylabel("", fontsize=1)
                axes.set_xlabel("", fontsize=1)
                max_val = coeffs_df["coeff"].max()
                xticks = np.linspace(0, np.floor(max_val * 10) / 10, 3)
                plt.xticks(xticks, fontsize=25)
                plt.yticks(fontsize=25)
                plt.xticks(fontsize=25)
                plt.savefig(
                    f"{model_importance_folder}/{metric}.svg",
                    format="svg",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{model_importance_folder}/{metric}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
                plt.close()


def save_regression_results(df_results, save_folder, models=list(EMBEDDING_MODELS.keys())):
    for usecase in USECASES:
        usecase_reg_folder = f"{save_folder}/{usecase}/regression"
        os.makedirs(usecase_reg_folder, exist_ok=True)
        for model in models:
            model_reg_folder = f"{usecase_reg_folder}/{model}"
            os.makedirs(model_reg_folder, exist_ok=True)
            reg_results = []
            for metric in ALL_METRICS:
                data = df_results[usecase].drop(columns=["usecase", "beta", "alpha"])
                for run in data.run.unique():
                    data_run = data[data["run"] == run]
                    X = data_run[BIAS_MEASURES].to_numpy()
                    X_assort = data_run[["assortativity"]].to_numpy()
                    X_other = data_run[
                        ["control", "diameter", "isolation", "information_unfairness"]
                    ].to_numpy()
                    y = data_run[[f"{model}_{metric}"]].to_numpy()

                    for name, X in [
                        ("full", X),
                        ("assortativity", X_assort),
                        ("other", X_other),
                    ]:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=13
                        )

                        reg_model = Ridge()
                        reg_model.fit(X_train, y_train)
                        train_R2 = reg_model.score(X_train, y_train)
                        test_R2 = reg_model.score(X_test, y_test)
                        train_RMSE = root_mean_squared_error(
                            reg_model.predict(X_train), y_train
                        )
                        test_RMSE = root_mean_squared_error(
                            reg_model.predict(X_test), y_test
                        )
                        reg_results.append(
                            [
                                "ridge",
                                metric,
                                run,
                                name,
                                train_R2,
                                test_R2,
                                train_RMSE,
                                test_RMSE,
                            ]
                        )

                        reg_model = RandomForestRegressor()
                        reg_model.fit(X_train, y_train.ravel())
                        train_R2 = reg_model.score(X_train, y_train.ravel())
                        test_R2 = reg_model.score(X_test, y_test.ravel())
                        train_RMSE = root_mean_squared_error(
                            reg_model.predict(X_train), y_train
                        )
                        test_RMSE = root_mean_squared_error(
                            reg_model.predict(X_test), y_test
                        )
                        reg_results.append(
                            [
                                "RF",
                                metric,
                                run,
                                name,
                                train_R2,
                                test_R2,
                                train_RMSE,
                                test_RMSE,
                            ]
                        )
            pd.DataFrame(
                reg_results,
                columns=[
                    "reg_model",
                    "metric",
                    "run",
                    "type",
                    "train_R2",
                    "test_R2",
                    "train_RMSE",
                    "test_RMSE",
                ],
            ).to_csv(f"{model_reg_folder}/raw_results.csv", index=False)


def save_aggregate_regression_results(save_folder, models=list(EMBEDDING_MODELS.keys())):
    for usecase in USECASES:
        usecase_reg_folder = f"{save_folder}/{usecase}/regression"
        os.makedirs(usecase_reg_folder, exist_ok=True)
        for model in models:
            model_reg_folder = f"{usecase_reg_folder}/{model}"
            os.makedirs(model_reg_folder, exist_ok=True)

            df = pd.read_csv(f"{model_reg_folder}/raw_results.csv")
            df.drop(columns=["run"]).groupby(
                ["reg_model", "metric", "type"]
            ).mean().reset_index().to_csv(
                f"{model_reg_folder}/avg_results.csv", index=False
            )
            df.drop(columns=["run"]).groupby(
                ["reg_model", "metric", "type"]
            ).std().reset_index().to_csv(
                f"{model_reg_folder}/std_results.csv", index=False
            )
