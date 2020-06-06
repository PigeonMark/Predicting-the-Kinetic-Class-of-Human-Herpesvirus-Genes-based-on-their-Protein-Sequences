import json
import pickle
from operator import add

from Util import compose_filename, compose_configuration
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

color_dict = {'green': '#5cb85c', 'blue': '#5bc0de', 'orange': '#f0ad4e', 'red': '#d9534f'}
color_phase_dict = {'immediate-early': color_dict['blue'], 'early': color_dict['green'], 'late': color_dict['orange'],
                    'latent': color_dict['red']}


def _plot_pca_fi_barchart(relative, original_features, pca):
    plt.figure(figsize=(12, 8))
    bottom = [0 for _ in original_features]
    for comp in range(2):
        y = abs(pca.components_[comp])
        if relative:
            y = [el * pca.explained_variance_ratio_[comp] for el in y]
        plt.bar(original_features, y, bottom=bottom, label=f"Component {comp}")
        bottom = list(map(add, bottom, y))

    plt.xticks(rotation=45, rotation_mode='anchor', ha='right')
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    if relative:
        plt.title("PCA feature importance, relative to Explained Variance Ratio of the components", wrap=True)
    else:
        plt.title("PCA feature importance", wrap=True)
    plt.legend()
    plt.tight_layout()


class PCAPlotter:
    def __init__(self, config_filepath):
        self.config = None
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

    def plot_explained_variance(self, name):
        pca_file = f"{self.config['input_pca_folder']}None-pca.p"
        pca = pickle.load(open(pca_file, 'rb'))  # type: PCA

        y = [ev for ev in pca.explained_variance_ratio_[:10]] + [sum(pca.explained_variance_ratio_[10:])]

        plt.figure(figsize=(5, 4))
        x = [f"PC {i + 1}" for i, _ in enumerate(y[:-1])] + [f"PC 11 - 28"]
        plt.bar(x, y)

        # for i, v in enumerate(y):
        #     plt.text(i, v, str(round(v * 100, 2)) + '%', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45, rotation_mode='anchor', ha='right')
        plt.xlabel('PCA Component')
        plt.ylabel('Explained Variance Ratio')
        plt.tight_layout()
        plt.savefig(f"{self.config['output_pca_variance_plot_directory']}pca_explained_variance", dpi=300)
        plt.close()

    def plot_feature_importance(self, name):
        pca_file = f"{self.config['input_pca_folder']}None-pca.p"
        pca = pickle.load(open(pca_file, 'rb'))  # type: PCA

        features_file = compose_filename(self.config["input_data_folder"], self.config['filter_latent'],
                                         self.config['standardization'], 'no-pca', 'features', name, 'csv')

        original_features = pd.read_csv(features_file, index_col=0).columns
        original_features = [col for col in original_features if col not in self.config['skip-features']]

        _plot_pca_fi_barchart(False, original_features, pca)
        plt.savefig(f"{self.config['output_pca_variance_plot_directory']}PCA_Features_Importance_{name}")
        plt.close()

        _plot_pca_fi_barchart(True, original_features, pca)
        plt.savefig(
            f"{self.config['output_pca_variance_plot_directory']}PCA_Features_Importance_Relative_to_variance{name}")
        plt.close()

    def plot(self, name):
        # title = compose_configuration(f'PCA scatter plot', self.config['filter_latent'],
        #                               self.config['standardization'], 2, name)

        title = ""
        print(f"Plotting {title}")

        features_file = compose_filename(self.config["input_data_folder"], self.config['filter_latent'],
                                         self.config['standardization'], 'no-pca', 'features', name, 'csv')

        original_features = pd.read_csv(features_file, index_col=0).columns
        original_features = [col for col in original_features if col not in self.config['skip-features']]

        pca_object_file = f"{self.config['input_pca_folder']}None-pca.p"
        pca = pickle.load(open(pca_object_file, 'rb'))  # type: PCA

        pca_file = compose_filename(self.config["input_data_folder"], self.config['filter_latent'],
                                    self.config['standardization'], 2, 'features', name, 'csv')
        data = pd.read_csv(pca_file)
        plt.figure(figsize=(5, 4))

        print(pca.components_[0])
        print(pca.explained_variance_)
        print(pca.singular_values_)

        for i, feature in enumerate(original_features):
            plt.arrow(0, 0, pca.components_[0][i], pca.components_[1][i], alpha=0.5, length_includes_head=True, head_width=0.005)
            plt.text(pca.components_[0][i]*1.1, pca.components_[1][i]*1.1, feature, fontsize=8)

        xs = []
        ys = []
        for label in ['immediate-early', 'early', 'late']:
            label_data = data[data.label.eq(label)]
            xs.extend(label_data['comp_0'])
            ys.extend(label_data['comp_1'])
        scalex = 0.75 / (max(xs) - min(xs))
        scaley = 0.75 / (max(ys) - min(ys))

        for label in ['immediate-early', 'early', 'late']:
            label_data = data[data.label.eq(label)]
            x = label_data['comp_0']
            y = label_data['comp_1']
            plt.scatter(x * scalex, y * scaley, label=label, s=40, alpha=0.65, edgecolors='none',
                        color=color_phase_dict[label])
        plt.legend()
        plt.title(title, wrap=True)
        plt.xlim(-.3, .8)
        plt.ylim(-.6, .5)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.tight_layout()
        filename = compose_filename(self.config['output_pca_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], 'no-pca', '2-pca', 'scatter', 'png')
        plt.savefig(filename, dpi=300)
        plt.close()
