import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

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
            "ba": 'balanced_accuracy',
            "a_ba": 'adjusted_balanced_accuracy',
            "roc_auc_ovo": 'roc_auc_ovo_score',
            "roc_auc_ovr": 'roc_auc_ovr_score'
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

    def load_results(self, n_pca):
        filename = compose_filename(self.config['output_result_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca, 'classification_results', self.name, 'p')
        self.results = pickle.load(open(filename, 'rb'))

    def plot(self, score_metric, n_pca):
        plt.figure()
        title = compose_configuration(self.TITLE[score_metric], self.config['filter_latent'],
                                      self.config['standardization'], n_pca, self.name)
        print(f"Plotting {title}")
        max_score = 0
        max_configuration = None

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
                    score = np.mean(res.get(score_name, None))
                    y.append(score)
                    error.append(np.std(res.get(score_name), None))
                    if score >= max_score:
                        max_score = score
                        max_configuration = (ml_method, cl)

            if len(x) > 0:
                plt.bar(index + (i * bar_width), y, yerr=error, width=bar_width, label=ml_method, capsize=5)

                # print(f"\t{ml_method}")
                # for j, mean in enumerate(y):
                #     print(f"\t\t{x[j]}: {100 * mean:.2f}%")
        print(f"Maximum score: {max_configuration[0]}, {max_configuration[1]}: {100 * max_score:.2f}%")

        plt.title(title, wrap=True)
        plt.xticks(index + bar_width, list(self.results.values())[0].keys())
        plt.xlabel('Classifier')
        plt.ylabel(self.YLABEL[score_metric])
        plt.legend(title='Multi-label Method')
        filename = compose_filename(self.config['output_bar_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca, self.SAVE_TITLE[score_metric], self.name, '')
        plt.savefig(filename)
        plt.clf()
        print()

    def plot_roc_class(self, ml_method, classifier, class_, label, ls):
        _fprs = self.results[ml_method][classifier]['fpr'][class_]
        _tprs = self.results[ml_method][classifier]['tpr'][class_]
        _aucs = self.results[ml_method][classifier]['roc_auc'][class_]

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, fpr in enumerate(_fprs):
            interp_tpr = np.interp(mean_fpr, fpr, _tprs[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(_aucs[i])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, ls, lw=3, label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc))

    def plot_roc(self, ml_method, classifier, n_pca):
        title = compose_configuration(f'ROC curves of {ml_method} {classifier}', self.config['filter_latent'],
                                      self.config['standardization'], n_pca, self.name)
        print(f"Plotting {title}")

        plt.figure(figsize=(8, 6))

        self.plot_roc_class(ml_method, classifier, 'micro', f'micro-average ROC curve', ':')
        self.plot_roc_class(ml_method, classifier, 'macro', f'macro-average ROC curve', ':')

        for class_ in ['early', 'immediate-early', 'late']:
            self.plot_roc_class(ml_method, classifier, class_, f'Mean ROC curve of {class_}', '-')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        filename = compose_filename(self.config['output_roc_curves_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca, f'ROC_curves_{ml_method}_{classifier}',
                                    self.name, '')
        plt.savefig(filename)
        plt.clf()

    def _plot_all(self, n_pca):
        self.plot('ba', n_pca)
        self.plot('a_ba', n_pca)
        self.plot('roc_auc_ovo', n_pca)
        self.plot('roc_auc_ovr', n_pca)

    def _plot_roc(self, n_pca):
        for ml_method, classifiers in self.results.items():
            for classifier, result in classifiers.items():
                if len(result['fpr']['early']) > 0:
                    self.plot_roc(ml_method, classifier, n_pca)

    def plot_all(self):
        if self.config['pca_features'] is True:
            for n in self.config['n-pca']:
                self.load_results(n)
                self._plot_all(n)
                self._plot_roc(n)
                self._plot_pi(n)
        else:
            self.load_results('no-pca')
            self._plot_all('no-pca')
            self._plot_roc('no-pca')
            self._plot_pi('no-pca')

    def _plot_pi(self, n_pca):
        for ml_method, classifiers in self.results.items():
            for classifier, result in classifiers.items():
                if len(result['permutation_importance']) > 0:
                    self.plot_permutation_importance(ml_method, classifier, n_pca)

    def plot_permutation_importance(self, ml_method, classifier, n_pca):
        title = compose_configuration(f'Permutation Importances of {ml_method} {classifier}',
                                      self.config['filter_latent'], self.config['standardization'], n_pca, self.name)
        print(f"Plotting {title}")

        features = self.results[ml_method][classifier]['features']
        perm_imps = self.results[ml_method][classifier]['permutation_importance']

        permutation_importances = {}
        for i, f in enumerate(features):
            feature_imps = [perm_imp[i] for perm_imp in perm_imps]
            permutation_importances[f] = (np.mean(feature_imps), np.std(feature_imps))

        permutation_importances = {
            k: v for k, v in sorted(permutation_importances.items(), key=lambda item: item[1][0], reverse=True)[:30]}

        x_size = len(permutation_importances) / 2.5
        y_size = x_size / 1.375
        plt.figure(figsize=(x_size, y_size))
        plt.bar(permutation_importances.keys(), [val[0] for val in permutation_importances.values()],
                yerr=[val[1] for val in permutation_importances.values()], width=1, capsize=5)
        plt.xticks(rotation=45, rotation_mode='anchor', ha='right')
        plt.ylabel('Permutation Importance')
        plt.xlabel('Feature')
        plt.title(title, wrap=True)
        plt.tight_layout()
        filename = compose_filename(self.config['output_pi_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca,
                                    f'permutation_importance_{ml_method}_{classifier}', self.name, '')
        plt.savefig(filename)
        plt.clf()
