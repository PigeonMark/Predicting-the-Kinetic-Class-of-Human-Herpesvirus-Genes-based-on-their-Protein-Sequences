import json
import pickle

from Classification import Classifier
from Util import compose_configuration, compose_filename


class Classification:
    def __init__(self, config_filepath, name):
        self.config = None
        self.name = name
        self.config_filepath = config_filepath
        self.results = {}
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

    def save_scores(self, n_pca):
        filename = compose_filename(self.config['output_result_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca, 'classification_results', self.name, 'p')
        pickle.dump(self.results, open(filename, 'wb'))

    def grid_search(self, ml_method, classifier_name, grid, splits, n_pca):
        print(compose_configuration('Grid Search', self.config['filter_latent'], self.config['standardization'], n_pca,
                                    self.name))
        print(ml_method, classifier_name)
        classifier = Classifier(self.config_filepath, ml_method, classifier_name)
        classifier.grid_search(self.name, grid, splits, n_pca)

    def _fit_all(self, n_pca):
        print(compose_configuration('Fitting features', self.config['filter_latent'], self.config['standardization'],
                                    n_pca, self.name))
        for ml_method in self.config['ML-method-options']:
            for classifier_name in self.config['Classifier-options']:
                classifier = Classifier(self.config_filepath, ml_method, classifier_name)
                scores = classifier.fit(self.name, n_pca)
                if ml_method in self.results:
                    self.results[ml_method][classifier_name] = scores
                else:
                    self.results[ml_method] = {classifier_name: scores}

    def fit_all(self):
        if self.config['pca_features']:
            for n in self.config['n-pca']:
                self._fit_all(n)
                self.save_scores(n)
        else:
            self._fit_all('no-pca')
            self.save_scores('no-pca')
