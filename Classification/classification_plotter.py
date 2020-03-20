import json
import pickle
import matplotlib.pyplot as plt
import numpy as np


class ClassificationPlotter:
    def __init__(self, config_filepath):
        self.config = None
        self.results = {}
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

    def load_results(self):
        self.results = pickle.load(
            open(f"{self.config['output_result_directory']}classification_results_{self.config['k-fold-splits']}.p",
                 'rb'))

    def plot_ba(self):
        plt.figure()
        for ml_method, classifier in self.results.items():
            method_data = [np.mean([result.ba for result in results]) for classifier_name, results in
                           classifier.items()]
            plt.plot(classifier.keys(), method_data, 'o-', label=ml_method)

        plt.title('Balanced Accuracy')
        plt.xlabel('Classifier')
        plt.ylabel('Accuracy')
        plt.legend(title='Multi-label Method')
        plt.savefig(f"{self.config['output_ba_plot_directory']}BalancedAccuracy_{self.config['k-fold-splits']}")
        plt.clf()

    def plot_adjusted_ba(self):
        plt.figure()
        for ml_method, classifier in self.results.items():
            method_data = [np.mean([result.adjusted_ba for result in results]) for classifier_name, results in
                           classifier.items()]
            plt.plot(classifier.keys(), method_data, 'o-', label=ml_method)

        plt.title('Adjusted Balanced Accuracy')
        plt.xlabel('Classifier')
        plt.ylabel('Accuracy')
        plt.legend(title='Multi-label Method')
        plt.savefig(f"{self.config['output_ba_plot_directory']}AdjustedBalancedAccuracy_{self.config['k-fold-splits']}")
        plt.clf()
