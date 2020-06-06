import json
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, average_precision_score

from Util import compose_filename, compose_configuration, output_filename

color_dict = {'green': '#5cb85c', 'blue': '#5bc0de', 'orange': '#f0ad4e', 'red': '#d9534f', 'silver': '#c0c0c0',
              'dimgray': '#696969'}
color_phase_dict = {'immediate-early': color_dict['blue'], 'early': color_dict['green'], 'late': color_dict['orange'],
                    'latent': color_dict['red'], 'micro': color_dict['silver'], 'macro': color_dict['dimgray']}
color_ml_dict = {'ML': color_dict['green'], '1vsA': color_dict['blue'], 'RR': color_dict['orange']}


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
            "ba": f"BA",
            "a_ba": f"ABA",
            "roc_auc_ovo": f"ROC_AUC_ovo",
            "roc_auc_ovr": f"ROC_AUC_ovr"
        }
        self.YLABEL = {
            "ba": 'Balanced Accuracy',
            "a_ba": 'Adjusted Balanced Accuracy',
            "roc_auc_ovo": 'AUC',
            "roc_auc_ovr": 'AUC'
        }
        self.YMAX = {
            "ba": 0.65,
            "a_ba": 0.45,
            "roc_auc_ovo": 0.85,
            "roc_auc_ovr": 0.85
        }

        self.MULTI_CLASS_NAME = {
            "1vsA": "OvA",
            "ML": "Built-in",
            "RR": "OvO"
        }

        self.CLASSIFIER_NAME = {
            "KNN": "k-NN",
            "RF": "RF",
            "XGBoost": "XGBoost"
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

        for i, ml_method in enumerate(self.config["ML-method-options"]):
            classifier = self.results[ml_method]
            classifier_tuples = list(classifier.items())
            score_name = self.SCORE_NAME[score_metric]
            x = []
            y = []
            error = []
            for cl, res in classifier_tuples:
                if score_name in res:
                    x.append(self.CLASSIFIER_NAME[cl])
                    score = np.mean(res.get(score_name, None))
                    y.append(score)
                    error.append(np.std(res.get(score_name), None))
                    if score >= max_score:
                        max_score = score
                        max_configuration = (self.MULTI_CLASS_NAME[ml_method], self.CLASSIFIER_NAME[cl])

            if len(x) > 0:
                plt.bar(index + (i * bar_width), y, yerr=error, width=bar_width, label=self.MULTI_CLASS_NAME[ml_method],
                        capsize=5, color=color_ml_dict[ml_method])

                print(f"\t{self.MULTI_CLASS_NAME[ml_method]}")
                for j, mean in enumerate(y):
                    print(f"\t\t{x[j]}: {100 * mean:.2f}% +-{100 * error[j]:.2f}%")
        print(f"Maximum score: {max_configuration[0]}, {max_configuration[1]}: {100 * max_score:.2f}%")

        # plt.title(title, wrap=True)
        plt.ylim(top=self.YMAX[score_metric])
        plt.xticks(index + bar_width, [self.CLASSIFIER_NAME[cl] for cl in list(self.results.values())[0].keys()])
        plt.xlabel('Classifier')
        plt.ylabel(self.YLABEL[score_metric])
        plt.legend(title='Multiclass strategy')
        plt.tight_layout()
        filename = output_filename(self.config['output_bar_plot_directory'], self.config['filter_latent'],
                                   self.config['standardization'], n_pca, self.SAVE_TITLE[score_metric], self.name, '')
        plt.savefig(filename)
        plt.close()
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

        plt.plot(mean_fpr, mean_tpr, ls, color=color_phase_dict[class_], lw=3,
                 label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc))

    def plot_roc(self, ml_method, classifier, n_pca):
        title = compose_configuration(f'ROC curves of {ml_method} {classifier}', self.config['filter_latent'],
                                      self.config['standardization'], n_pca, self.name)
        print(f"Plotting {title}")

        plt.figure(figsize=(6, 4.5))

        for class_ in ['immediate-early', 'early', 'late']:
            self.plot_roc_class(ml_method, classifier, class_, f'{class_}', '-')
        self.plot_roc_class(ml_method, classifier, 'micro', f'micro-average', ':')
        self.plot_roc_class(ml_method, classifier, 'macro', f'macro-average', ':')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # plt.title(title)
        plt.tight_layout()
        filename = compose_filename(self.config['output_roc_curves_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca,
                                    f'ROC_{self.MULTI_CLASS_NAME[ml_method]}_{self.CLASSIFIER_NAME[classifier]}',
                                    self.name, '')
        plt.savefig(filename)
        plt.close()

    def plot_pr_class(self, ml_method, classifier, class_, label, ls):
        y_real = np.concatenate(self.results[ml_method][classifier]['y_real'][class_])
        y_proba = np.concatenate(self.results[ml_method][classifier]['y_proba'][class_])

        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        ap = average_precision_score(y_real, y_proba, average='weighted')
        plt.plot(recall, precision, ls, lw=2.5, color=color_phase_dict[class_], label=r'%s (AP = %0.2f)' % (label, ap))

    def plot_pr(self, ml_method, classifier, n_pca):
        title = compose_configuration(f'PR curves of {ml_method} {classifier}', self.config['filter_latent'],
                                      self.config['standardization'], n_pca, self.name)
        print(f"Plotting {title}")
        plt.figure(figsize=(6, 4.5))

        self.plot_pr_class(ml_method, classifier, 'micro', f'micro-average', ':')

        for class_ in ['immediate-early', 'early', 'late']:
            self.plot_pr_class(ml_method, classifier, class_, f'{class_}', '-')

        plt.legend()
        # plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.tight_layout()
        filename = compose_filename(self.config['output_pr_curves_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca,
                                    f'PR_{self.MULTI_CLASS_NAME[ml_method]}_{self.CLASSIFIER_NAME[classifier]}',
                                    self.name, '')
        plt.savefig(filename)
        plt.close()

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

    def _plot_pr(self, n_pca):
        for ml_method, classifiers in self.results.items():
            for classifier, result in classifiers.items():
                if 'y_real' in result and len(result['y_real']['early']) > 0:
                    self.plot_pr(ml_method, classifier, n_pca)

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
            self._plot_pr('no-pca')
            self.pi_corr_matrix()
            self._plot_pi('no-pca')
            self.plot_wrong_classifications()

    def _plot_pi(self, n_pca):
        # for ml_method, classifiers in self.results.items():
        #     for classifier, result in classifiers.items():
        #         if len(result['permutation_importance']) > 0:
        #             self.plot_permutation_importance(ml_method, classifier, n_pca)
        for classifier in ['RF', 'KNN', 'XGBoost']:
            self.plot_permutation_importance_summary(classifier, 50, n_pca)

    def plot_permutation_importance_summary(self, classifier, max_n, n_pca):
        title = compose_configuration(f'Summarized Permutation Importances of {classifier}',
                                      self.config['filter_latent'], self.config['standardization'], n_pca, self.name)
        print(f"Plotting {title}")

        features = self.results['RR'][classifier]['features']

        perm_imps = []
        for mc_technique in ['ML', '1vsA', 'RR']:
            perm_imps.extend(self.results[mc_technique][classifier]['permutation_importance'])

        permutation_importances = {}
        for i, f in enumerate(features):
            feature_imps = [perm_imp[i] for perm_imp in perm_imps]
            permutation_importances[f] = (np.mean(feature_imps), np.std(feature_imps))

        permutation_importances = {
            k: v for k, v in sorted(permutation_importances.items(), key=lambda item: item[1][0], reverse=True)[:max_n]}

        x_size = len(permutation_importances) / 3.2
        y_size = x_size / 1.375
        plt.figure(figsize=(x_size, y_size))
        plt.bar(permutation_importances.keys(), [val[0] for val in permutation_importances.values()],
                yerr=[val[1] for val in permutation_importances.values()], width=0.9,
                error_kw=dict(lw=1, capsize=3, capthick=1))
        plt.xticks(rotation=45, rotation_mode='anchor', ha='right')
        plt.ylabel('Permutation Importance')
        plt.xlabel('Feature')
        # plt.title(title, wrap=True)
        plt.tight_layout()
        filename = compose_filename(self.config['output_pi_plot_directory'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca,
                                    f'pi_{classifier}', self.name, '')
        plt.savefig(filename, dpi=150)
        plt.close()

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
        plt.close()

    def pi_corr_matrix(self):
        features = list(list(self.results.items())[0][1].items())[0][1]['features']

        pi = {f: [] for f in features}
        for ml_method, classifiers in self.results.items():
            for classifier, result in classifiers.items():
                for feature_results in result['permutation_importance']:
                    for i, feature_result in enumerate(feature_results):
                        pi[features[i]].append(feature_result)

        df = pd.DataFrame(pi)
        corr = df.corr()
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()
        im = ax.matshow(corr)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)

        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, fontsize=10, rotation=-45, rotation_mode='anchor', ha='right')
        ax.set_yticks(range(df.shape[1]))
        ax.set_yticklabels(df.columns, fontsize=10)

        plt.title('Correlation Matrix of permutation importances of all features', y=-0.07, x=-11)
        fig.tight_layout(pad=2)

        plt.savefig(f"{self.config['output_pi_corr_matrix_plot_directory']}pi_correlation_matrix", dpi=150)

    def plot_wrong_classifications(self):
        fig = plt.figure(figsize=(14, 5))
        ax = plt.gca()
        # plt.title('Number of wrong classifications for each classifier\n(p1 -> p2 == a p1 sequence was wrongly classified as a p2 sequence)', wrap=True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        x = []
        y = []
        z = []
        ml_methods = list(self.results.keys())
        classifiers = list(self.results[ml_methods[0]].keys())
        gap = False
        for classifier in classifiers:
            for ml_method in ml_methods:
                results = self.results[ml_method][classifier]['wrong_classifications']
                for correct, wrongs in results.items():
                    if correct == 'immediate-early':
                        correct = 'IE'
                    else:
                        correct = correct.capitalize()
                    for wrong, number in wrongs.items():
                        if wrong == 'immediate-early':
                            wrong = 'IE'
                        else:
                            wrong = wrong.capitalize()
                        if number > 0:
                            x.append(f'{self.MULTI_CLASS_NAME[ml_method]}\n{self.CLASSIFIER_NAME[classifier]}')
                            y.append(f'{correct} ' + r'$\rightarrow$' + f' {wrong}')
                            z.append(number / 200.)
            if not gap:
                x.append('')
                y.append('Late ' + r'$\rightarrow$' + ' Early')
                z.append(0)
                gap = True
            else:
                x.append('  ')
                y.append('Late ' + r'$\rightarrow$' + ' Early')
                z.append(0)

        viridis_big = cm.get_cmap('Blues', 512)
        new_cmp = ListedColormap(viridis_big(np.linspace(0.25, 1, 256)))

        im = ax.scatter(x, y, s=[_z * 200 for _z in z], c=z, cmap=new_cmp, alpha=1)
        d = ax.collections[0]
        offsets = d.get_offsets()
        for i, (off_x, off_y) in enumerate(offsets):
            if z[i] > 0:
                if z[i] > 7:
                    ax.text(off_x, off_y, round(z[i]), fontsize=np.sqrt(z[i]) * 7, ha='center', va='center',
                            color='white')
                else:
                    ax.text(off_x, off_y, round(z[i]), fontsize=np.sqrt(z[i]) * 7, ha='center', va='center')

        cb = fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig('Classification/Output/plots/wrong_classifications/wrong_classifications', dpi=150)
        plt.close()
