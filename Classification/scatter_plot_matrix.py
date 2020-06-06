import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
from Util import compose_filename


def scatter_plot_matrix(X, figsize, names, alpha=1.0, **kwargs):
    num_examples, num_features = X.shape

    fig, axes = plt.subplots(nrows=num_features,
                             ncols=num_features,
                             figsize=figsize)

    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        x, y = X[:, j], X[:, i]
        r, p = spearmanr(x, y)
        axes[j, i].scatter(x, y, alpha=alpha, edgecolors='none', **kwargs)
        axes[j, i].set_ylim(top=max(y) + (max(y)-min(y))*0.25)
        axes[j, i].text(0.03, 0.97, f"r = {round(r, 2)}", va='top', transform=axes[j, i].transAxes)
        axes[j, i].set_xlabel(names[j])
        axes[j, i].set_ylabel(names[i])
        axes[i, j].set_axis_off()

    for i in range(num_features):
        axes[i, i].hist(X[:, i])
        axes[i, i].set_ylabel('Count')
        axes[i, i].set_xlabel(names[i])

    return fig, axes


class ScatterPlotMatrixPlotter:
    def __init__(self, config_filepath):
        self.config = None
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

    def plot_correlation_matrix(self):
        feature_file = compose_filename(self.config['input_data_folder'], True, False, 'no-pca', 'features', 'original',
                                        'csv')
        df = pd.read_csv(feature_file, index_col=0)
        columns = [col for col in df.columns if col not in self.config['skip-features']]
        df = df[columns]

        corr = df.corr(method='spearman')
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()
        im = ax.matshow(corr)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)

        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, fontsize=12, rotation=-45, rotation_mode='anchor', ha='right')
        ax.set_yticks(range(df.shape[1]))
        ax.set_yticklabels(df.columns, fontsize=12)

        # plt.title('Spearman Feature Correlation Matrix', y=-0.07, x=-11, fontsize=20)
        fig.tight_layout(pad=2)
        plt.savefig('Classification/Output/plots/scatter_plot_matrix/feature_correlation_matrix', dpi=150)

    def plot_scatter_matrix(self):
        feature_file = compose_filename(self.config['input_data_folder'], True, False, 'no-pca', 'features', 'original',
                                        'csv')
        df = pd.read_csv(feature_file, index_col=0)
        columns = [col for col in df.columns if col not in self.config['skip-features']]
        df = df[columns]
        scatter_plot_matrix(df.to_numpy(), figsize=(60, 60), names=columns, alpha=0.3)
        print('Done plotting')
        plt.tight_layout(h_pad=0, w_pad=0)
        print('Layout done')
        plt.savefig('Classification/Output/plots/scatter_plot_matrix/scatter_plot_matrix', dpi=150)
        print('Saving done')
