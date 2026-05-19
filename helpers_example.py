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
from scipy.stats import spearmanr
from scipy.stats import rankdata as _rankdata
from numpy.linalg import lstsq

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

def plot_metric_pairs(models, metric_pairs, title="Node2Vec vs FairWalk", figsize=(10, 4)):
    fig, axes = plt.subplots(1, len(metric_pairs), figsize=figsize, sharey=False)
    if len(metric_pairs) == 1:
        axes = [axes]

    for ax, (x_col, x_label, y_col, y_label) in zip(axes, metric_pairs):
        plotted = False

        for label, df_model, color in models:
            if x_col not in df_model.columns or y_col not in df_model.columns:
                continue

            x_mean, x_std = df_model[x_col].mean(), df_model[x_col].std()
            y_mean, y_std = df_model[y_col].mean(), df_model[y_col].std()

            ax.errorbar(
                x_mean, y_mean,
                xerr=x_std, yerr=y_std,
                fmt="o", color=color,
                markersize=8, capsize=4,
                capthick=1.2, elinewidth=1.0,
                label=label
            )
            plotted = True

        if not plotted:
            ax.set_visible(False)
            continue

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_partial_spearman(
    models_data,
    model_labels,
    metrics=["Hit", "EO", "SP"],
    always_control=["assortativity", "AP"],
    metric_labels=None,
    model_colors=None,
    model_markers=None,
    top=3
    ):

    ref_df   = next(iter(models_data.values()))
    bias_cols = [c for c in ref_df.columns
                 if c not in ["alpha", "beta"] + metrics + always_control]
    models    = list(models_data.keys())

    if metric_labels is None:
        metric_labels = {m: m for m in metrics}
    default_colors  = ['#2c7bb6', '#d7191c', '#1a9641', '#fdae61']
    default_markers = ['o', 's', '^', 'D']
    if model_colors is None:
        model_colors  = {m: default_colors[i]  for i, m in enumerate(models)}
    if model_markers is None:
        model_markers = {m: default_markers[i] for i, m in enumerate(models)}


    def partial_spearman(df, x_col, y_col, controls):
        data = df[[x_col, y_col] + controls].dropna()
        if len(data) < 10:
            return np.nan, np.nan
        def resid(y, X):
            X_ = np.column_stack([np.ones(len(y)), X])
            return y - X_ @ lstsq(X_, y, rcond=None)[0]
        rx = _rankdata(data[x_col].values).astype(float)
        ry = _rankdata(data[y_col].values).astype(float)
        if controls:
            rc = np.column_stack([_rankdata(data[c].values) for c in controls])
            rx, ry = resid(rx, rc), resid(ry, rc)
        return spearmanr(rx, ry)

    rows = []
    for model, df in models_data.items():
        for bias in bias_cols:
            ctrl = [b for b in bias_cols if b != bias] + always_control
            for metric in metrics:
                rho, pval = partial_spearman(df, bias, metric, ctrl)
                rows.append(dict(model=model, bias=bias, metric=metric,
                                 rho_obs=rho,
                                 signif=(not np.isnan(pval) and pval < 0.05)))
    df_rho = pd.DataFrame(rows)


    top_biases = (df_rho[df_rho['metric'] == 'SP']
                  .groupby('bias')['rho_obs']
                  .apply(lambda x: x.abs().mean())
                  .sort_values(ascending=False)
                  .index[:top].tolist())
    bias_labels = {b: b for b in bias_cols}


    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.linewidth': 0.8, 'pdf.fonttype': 42,
    })

    n_bias     = len(top_biases)
    n_models   = len(models)
    model_step = 1.2
    group_gap  = 2.2
    group_h    = n_models * model_step + group_gap
    col_center = len(metrics) // 2

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.8 * len(metrics),
                                      n_bias * group_h * 0.38 + 2.0),
                             sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for col_i, metric in enumerate(metrics):
        ax = axes[col_i]

        all_rho = df_rho[(df_rho['bias'].isin(top_biases)) &
                         (df_rho['metric'] == metric)]['rho_obs'].dropna()
        margin = 0.15
        x_min = all_rho.min() - margin if len(all_rho) else -0.8
        x_max = all_rho.max() + margin if len(all_rho) else  0.8

        for b_i, bias in enumerate(top_biases):
            y_base        = b_i * group_h
            y_block_start = y_base - model_step * 0.6
            y_block_end   = y_base + (n_models - 1) * model_step + model_step * 0.6

            ax.fill_betweenx([y_block_start, y_block_end], -0.1, 0.1,
                             alpha=0.05, color='grey', zorder=0)
            ax.plot([0, 0], [y_block_start, y_block_end],
                    color='#555555', lw=0.8, ls='--', alpha=0.6, zorder=1)

            if col_i == col_center:
                ax.text((x_min + x_max) / 2, y_block_start, bias_labels[bias],
                        va='bottom', ha='center', fontsize=14,
                        fontweight='bold', color='#222222')

            for m_i, model in enumerate(models):
                y_center = y_base + m_i * model_step

                if m_i % 2 == 0:
                    ax.fill_betweenx([y_center - model_step * 0.45,
                                      y_center + model_step * 0.45],
                                     x_min, x_max, alpha=0.07, color='#333333', zorder=0)

                row = df_rho[(df_rho['bias']   == bias)  &
                             (df_rho['model']  == model) &
                             (df_rho['metric'] == metric)]
                if not row.empty:
                    rho = row.iloc[0]['rho_obs']
                    sig = row.iloc[0]['signif']
                    ax.scatter(rho, y_center,
                               marker=model_markers[model], color=model_colors[model],
                               s=85 if sig else 35, alpha=1.0 if sig else 0.28,
                               zorder=3, linewidths=0.5, edgecolors='white')

                if col_i == 0:
                    ax.text(x_min - 0.05, y_center, model_labels[model],
                            va='center', ha='right', fontsize=13, color='#444444')

        total_h_max = (n_bias - 1) * group_h + (n_models - 1) * model_step + model_step
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(total_h_max, -model_step)
        ax.set_title(metric_labels[metric], fontsize=18, fontweight='bold', pad=15)
        ax.set_yticks([])
        ax.xaxis.grid(True, alpha=0.2, lw=0.5)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

  
    fig.text(0.5, 0.02, r'Partial Spearman $\rho$', ha='center', fontsize=18)

    handles = [plt.scatter([], [], marker=model_markers[m], color=model_colors[m],
                           s=60, label=model_labels[m]) for m in models]
    handles += [
        plt.scatter([], [], marker='o', color='#777777', s=60, alpha=1.0, label='$p<0.05$'),
        plt.scatter([], [], marker='o', color='#777777', s=25, alpha=0.3,  label='n.s.'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, -0.05), fontsize=15, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1.0, 1.0])
    plt.show()