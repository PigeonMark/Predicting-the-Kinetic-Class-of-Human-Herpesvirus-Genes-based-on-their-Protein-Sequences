import json
from Util import compose_filename, compose_configuration
import pandas as pd
import matplotlib.pyplot as plt


class PCAPlotter:
    def __init__(self, config_filepath):
        self.config = None
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

    def plot(self, name):
        title = compose_configuration(f'PCA scatter plot', self.config['filter_latent'],
                                      self.config['standardization'], 2, name)
        print(f"Plotting {title}")

        pca_file = compose_filename(self.config["input_data_folder"], self.config['filter_latent'],
                                    self.config['standardization'], 2, 'features', name, 'csv')
        data = pd.read_csv(pca_file)
        plt.figure()
        for label in ['immediate-early', 'early', 'late']:
            label_data = data[data.label.eq(label)]
            x = label_data['comp_0']
            y = label_data['comp_1']
            plt.scatter(x, y, label=label, s=60, alpha=0.6, edgecolors='none')
        plt.legend()
        plt.title(title, wrap=True)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.tight_layout()
        filename = compose_filename(self.config['output_pca_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], 'no-pca', f'pca', name, '')
        plt.savefig(filename)
        plt.clf()
