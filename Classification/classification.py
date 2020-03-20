import json
import pickle

from Classification import Classifier


class Classification:
    def __init__(self, config_filepath):
        self.config = None
        self.config_filepath = config_filepath
        self.results = {}
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

    def save(self):
        pickle.dump(self.results, open(
            f"{self.config['output_result_directory']}classification_results_{self.config['k-fold-splits']}.p", 'wb'))

    def classify_all(self):
        for ml_method in self.config['ML-method-options']:
            for classifier_name in self.config['Classifier-options']:
                classifier = Classifier(self.config_filepath, ml_method, classifier_name)
                result = classifier.run()
                if ml_method in self.results:
                    self.results[ml_method][classifier_name] = result
                else:
                    self.results[ml_method] = {classifier_name: result}
