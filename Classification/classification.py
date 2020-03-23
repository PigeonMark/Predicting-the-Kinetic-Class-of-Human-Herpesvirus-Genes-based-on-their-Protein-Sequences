import json
import pickle

from Classification import Classifier


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

    def save_scores(self):
        pickle.dump(self.results, open(
            f"{self.config['output_result_directory']}classification_results_{self.name}.p", 'wb'))

    def grid_search(self, ml_method, classifier_name, grid, splits):
        classifier = Classifier(self.config_filepath, ml_method, classifier_name)
        classifier.grid_search(self.name, grid, splits)

    def fit_all(self):
        print(f"Fitting {self.name} features")
        for ml_method in self.config['ML-method-options']:
            for classifier_name in self.config['Classifier-options']:
                classifier = Classifier(self.config_filepath, ml_method, classifier_name)
                scores = classifier.fit(self.name)
                if ml_method in self.results:
                    self.results[ml_method][classifier_name] = scores
                else:
                    self.results[ml_method] = {classifier_name: scores}
