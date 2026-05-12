import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from globals import *
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

def plot_real_graph(G, save=None):

    node_colors = [
        "tab:red" if G.nodes[node]["sensitive"] else "tab:blue" for node in G.nodes()
    ]

    plt.figure(figsize=(6, 4))
    nx.draw(G, node_color=node_colors, node_size=50, alpha=0.3)
    if save != None:
        plt.savefig(f"{save}.png", format="png")
    plt.show()
    plt.close()


def plot_graph_grid(graphs_dict, figsize_per_plot=(4, 4)):

    alpha_beta_values = list(graphs_dict.keys())
    alpha_values = sorted(list(set([alpha for alpha, _ in alpha_beta_values])))
    beta_values = sorted(list(set([beta for _, beta in alpha_beta_values])))

    n_alpha = len(alpha_values)
    n_beta = len(beta_values)

    fig, axes = plt.subplots(
        n_alpha,
        n_beta,
        figsize=(figsize_per_plot[0] * n_beta, figsize_per_plot[1] * n_alpha),
    )

    if n_alpha == 1:
        axes = axes[None, :]
    if n_beta == 1:
        axes = axes[:, None]

    for i, alpha in tqdm(list(enumerate(alpha_values))):
        for j, beta in enumerate(beta_values):

            ax = axes[i, j]
            G = graphs_dict[(alpha, beta)]

            node_colors = [
                "tab:red" if G.nodes[node]["sensitive"] else "tab:blue"
                for node in G.nodes()
            ]

            nx.draw(
                G,
                ax=ax,
                node_color=node_colors,
                node_size=50,
                alpha=0.3,
                with_labels=False,
            )
            ax.set_axis_on()
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(rf"$\beta={beta}$", fontsize=10)

            if j == 0:
                ax.set_ylabel(rf"$\alpha={alpha}$", fontsize=10)

    plt.tight_layout()
    plt.show()


def show_heatmaps(df, metrics):

    n = len(metrics)

    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging", ["whitesmoke", "tab:red"]
    )

    for i, metric in enumerate(metrics):
        ax = axes[i]

        heatmap_data = df.pivot(index="alpha", columns="beta", values=metric)

        sns.heatmap(
            heatmap_data,
            ax=ax,
            annot=False,
            cmap=cmap,
            fmt=".2f",
            xticklabels=3,
            yticklabels=3,
        )

        ax.set_title(metric)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\alpha$")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def regression_results(df, metric):
    df_reg = []

    X = df[BIAS_MEASURES].to_numpy()
    X_assort = df[["assortativity"]].to_numpy()
    y = df[[metric]].to_numpy()

    for name, X in [
        ("all_bias", X),
        ("assortativity", X_assort),
    ]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=13
        )

        reg_model = RandomForestRegressor()
        reg_model.fit(X_train, y_train.ravel())
        train_R2 = reg_model.score(X_train, y_train.ravel())
        test_R2 = reg_model.score(X_test, y_test.ravel())
        train_RMSE = root_mean_squared_error(reg_model.predict(X_train), y_train)
        test_RMSE = root_mean_squared_error(reg_model.predict(X_test), y_test)
        df_reg.append([name, abs(train_R2), abs(test_R2), train_RMSE, test_RMSE])

    df_reg = pd.DataFrame(
        df_reg,
        columns=[
            "input_bias_measures",
            "train_R2",
            "test_R2",
            "train_RMSE",
            "test_RMSE",
        ],
    )
    return df_reg


def importance_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(
        'Bias importance plots in random forest regression on predictive metrics',
    )
    for i, m in enumerate(["SP", "EO"]):

        X = df[BIAS_MEASURES].to_numpy()
        y = df[[m]].to_numpy().ravel()

        reg_model = RandomForestRegressor()
        reg_model.fit(X, y)

        coeffs_df = pd.DataFrame(
            list(zip(BIAS_MEASURES, reg_model.feature_importances_)),
            columns=["measure", "coeff"],
        )

        sns.barplot(
            data=coeffs_df,
            y="measure",
            x="coeff",
            hue="coeff",
            palette="dark:#5A9_r",
            edgecolor="black",
            legend=False,
            ax=axes[i],
        )

        axes[i].set_title(m)
        axes[i].set_ylabel("")
        axes[i].set_xlabel("")

        max_val = coeffs_df["coeff"].max()
        xticks = np.linspace(0, np.floor(max_val * 10) / 10, 3)
        axes[i].set_xticks(xticks)

        axes[i].tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_correlations(df, metrics):
    sns.heatmap(
        df[metrics].corr(),
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    plt.title("Bias and fairness metrics correlations")
