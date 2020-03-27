import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

from Util import compose_filename, compose_configuration


class ClassificationPlotter:
    def __init__(self, config_filepath, name):
        self.config = None
        self.results = {}
        self.name = name
        self.read_config(config_filepath)

        self.TITLE = {
            "ba": f'Balanced Accuracy',
            "a_ba": f'Adjusted Balanced',
            "roc_auc_ovo": f'ROC AUC (ovo)',
            "roc_auc_ovr": f'ROC AUC (ovr)'
        }
        self.SCORE_NAME = {
            "ba": 'test_balanced_accuracy',
            "a_ba": 'test_adjusted_balanced_accuracy',
            "roc_auc_ovo": 'test_roc_auc_ovo_score',
            "roc_auc_ovr": 'test_roc_auc_ovr_score'
        }
        self.SAVE_TITLE = {
            "ba": f"BalancedAccuracy",
            "a_ba": f"AdjustedBalancedAccuracy",
            "roc_auc_ovo": f"ROC_AUC_ovo",
            "roc_auc_ovr": f"ROC_AUC_ovr"
        }
        self.YLABEL = {
            "ba": 'Accuracy',
            "a_ba": 'Accuracy',
            "roc_auc_ovo": 'AUC',
            "roc_auc_ovr": 'AUC'
        }

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            self.config = json.load(config_file)

    def load_results(self):
        filename = compose_filename(self.config['output_result_directory'], self.config['filter_latent'],
                                    self.config['standardization'], 'classification_results', self.name, 'p')
        self.results = pickle.load(open(filename, 'rb'))

    def plot(self, score_metric):
        plt.figure()
        title = compose_configuration(self.TITLE[score_metric], self.config['filter_latent'],
                                      self.config['standardization'], self.name)
        print(f"Plotting {title}")

        bar_width = 0.25
        n_groups = len(self.results)
        index = np.arange(n_groups)

        for i, (ml_method, classifier) in enumerate(self.results.items()):
            classifier_tuples = list(classifier.items())
            score_name = self.SCORE_NAME[score_metric]
            x = []
            y = []
            error = []
            for cl, res in classifier_tuples:
                if score_name in res:
                    x.append(cl)
                    y.append(np.mean(res.get(score_name, None)))
                    error.append(np.std(res.get(score_name), None))

            if len(x) > 1:
                plt.bar(index + (i * bar_width), y, yerr=error, width=bar_width, label=ml_method, capsize=5)

                print(f"\t{ml_method}")
                for j, mean in enumerate(y):
                    print(f"\t\t{x[j]}: {100 * mean:.2f}%")

        plt.title(title)
        plt.xticks(index + bar_width, list(self.results.values())[0].keys())
        plt.xlabel('Classifier')
        plt.ylabel(self.YLABEL[score_metric])
        plt.legend(title='Multi-label Method')
        filename = compose_filename(self.config['output_bar_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], self.SAVE_TITLE[score_metric], self.name, '')
        plt.savefig(filename)
        plt.clf()
        print()

    def plot_all(self):
        self.plot('ba')
        self.plot('a_ba')
        self.plot('roc_auc_ovo')
        self.plot('roc_auc_ovr')

    def plot_feature_importance(self, ml_method, classifier_name):
        features = self.results[ml_method][classifier_name]['features']
        f_imps = self.results[ml_method][classifier_name]['feature_importance']

        feature_importances = {f: np.mean([f_imp[i] for f_imp in f_imps]) for i, f in enumerate(features)}
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:30]}

        x_size = len(feature_importances) / 2.5
        y_size = x_size / 1.375
        plt.figure(figsize=(x_size, y_size))
        plt.bar(feature_importances.keys(), feature_importances.values(), width=1)
        plt.xticks(rotation=45, rotation_mode='anchor', ha='right')
        plt.ylabel('Feature Importance')
        plt.xlabel('Feature')
        plt.tight_layout()
        plt.savefig(
            f"{self.config['output_fi_plot_directory']}FeatureImportance_{ml_method}_{classifier_name}_{self.name}")
        plt.clf()
