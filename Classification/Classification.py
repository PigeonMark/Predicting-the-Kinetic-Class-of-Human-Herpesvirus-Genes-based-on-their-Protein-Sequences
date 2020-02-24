from sklearn.metrics import plot_precision_recall_curve, balanced_accuracy_score

import helper

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from DataCollection.input_data import get_viruses_data
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import interp
from PCA import pca

PHASES = ['immediate-early', 'early', 'late']


def one_vs_rest(data):
    one_vs_rest_data = {}
    for phase in PHASES:
        new_data = data.copy()
        new_data['label'] = new_data['label'] == phase

        x = new_data[new_data.columns[2:-1]]
        y = new_data[new_data.columns[-1]]

        one_vs_rest_data[phase] = (x, y)

    return one_vs_rest_data


def multilabel(data):
    return data[data.columns[2:-1]], data[data.columns[-1]]


def plot_finish(title, filename):
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"Output/{filename}", dpi=900)
    print(f"{filename} done")


def plot_finish_pr(title, filename):
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(f"Output/{filename}", dpi=900)
    print(f"{filename} done")


class Classification:
    def __init__(self, name, classifier, data):
        self.classifier = deepcopy(classifier)
        self.classifiers = {phase: deepcopy(classifier) for phase in PHASES}
        self.name = name
        self.multilabel_data = multilabel(data)
        self.data = one_vs_rest(data)

    def balanced_accuracy(self, adjusted):
        print(self.name)
        res = {}
        for phase in PHASES:
            print(f'Balanced Accuracy for {phase}')
            k_fold = ShuffleSplit(n_splits=100, test_size=1 / 3.)
            x, y = self.data[phase]

            ba_s = []
            for train, test in k_fold.split(x):
                self.classifiers[phase].fit(x.iloc[train], y.iloc[train])
                y_pred = self.classifiers[phase].predict(x.iloc[test])
                ba = balanced_accuracy_score(y.iloc[test], y_pred, adjusted=adjusted)
                ba_s.append(ba)
            res[phase] = (np.mean(ba_s), np.std(ba_s))
            print(f'\tMean: {np.mean(ba_s)}\n\tStd Dev: {np.std(ba_s)}\n')

        print(f'Balanced Accuracy for multilabel')
        k_fold = ShuffleSplit(n_splits=100, test_size=1 / 3.)
        x, y = self.multilabel_data

        ba_s = []
        for train, test in k_fold.split(x):
            self.classifier.fit(x.iloc[train], y.iloc[train])
            y_pred = self.classifier.predict(x.iloc[test])
            ba = balanced_accuracy_score(y.iloc[test], y_pred, adjusted=adjusted)
            ba_s.append(ba)

        res['multilabel'] = (np.mean(ba_s), np.std(ba_s))
        print(f'\tMean: {np.mean(ba_s)}\n\tStd Dev: {np.std(ba_s)}')

        print('----------\n')
        return res

    def plot_curve(self, filename, curve):
        plt.figure()
        for phase, color in zip(PHASES, ['limegreen', 'tomato', 'cornflowerblue']):
            y_vals = []
            base_x_val = np.linspace(0, 1, 101)
            k_fold = ShuffleSplit(n_splits=30, test_size=1 / 3.)
            x, y = self.data[phase]

            for train, test in k_fold.split(x):
                self.classifiers[phase].fit(x.iloc[train], y.iloc[train])
                y_score = self.classifiers[phase].predict_proba(x.iloc[test])[:, 1]
                if curve == "ROC":
                    x_val, y_val, thresholds = metrics.roc_curve(y.iloc[test], y_score)
                    y_val = interp(base_x_val, x_val, y_val)
                    y_val[0] = 0.0
                elif curve == "PR":
                    y_val, x_val, thresholds = metrics.precision_recall_curve(y.iloc[test], y_score)
                    y_val = interp(base_x_val, y_val, x_val)
                    y_val[0] = 1.
                else:
                    return

                y_vals.append(y_val)

            y_vals = np.array(y_vals)
            mean_y_vals = y_vals.mean(axis=0)
            std = y_vals.std(axis=0)

            y_vals_upper = np.minimum(mean_y_vals + std, 1)
            y_vals_lower = mean_y_vals - std

            plt.plot(base_x_val, mean_y_vals, color=color, lw=2,
                     label=f'{curve} curve {phase} (AUC={metrics.auc(base_x_val, mean_y_vals):.2f})')
            plt.fill_between(base_x_val, y_vals_lower, y_vals_upper, color=color, alpha=0.1)

        if curve == "ROC":
            plot_finish(f"{curve} curves for {self.name}", filename)
        elif curve == "PR":
            plot_finish_pr(f"{curve} curves for {self.name}", filename)


def plot_ROC_PR(combined_features):
    # for n_components in range(3, 15):
    #     pca_results, pca_obj = pca(combined_features, n_components)
    #
    #     classifier = Classification(f"Random forest with PCA ({n_components} components)",
    #                                 RandomForestClassifier(), pca_results)
    #     classifier.plot_curve(f"ROC/RandomForest_PCA_{n_components}_components.png", "ROC")
    #     classifier.plot_curve(f"PR/RandomForest_PCA_{n_components}_components.png", "PR")

    classifier = Classification("Random forest", RandomForestClassifier(), combined_features)
    classifier.plot_curve("ROC/RandomForest.png", "ROC")
    classifier.plot_curve("PR/RandomForest.png", "PR")


def balanced_acc(combined_features, adjusted):
    res = {}

    classifier = Classification(f"no PCA", RandomForestClassifier(), combined_features)
    res[classifier.name] = classifier.balanced_accuracy(adjusted)

    for n_components in range(3, 15):
        pca_results, pca_obj = pca(combined_features, n_components)

        classifier = Classification(f"PCA {n_components}",
                                    RandomForestClassifier(), pca_results)
        res[classifier.name] = classifier.balanced_accuracy(adjusted)

    print(res)

    plt.figure()
    x_s = []
    ie = []
    e = []
    l = []
    ml = []
    for config, ress in res.items():
        x_s.append(config)
        ie.append(ress['immediate-early'][0])
        e.append(ress['early'][0])
        l.append(ress['late'][0])
        ml.append(ress['multilabel'][0])
    plt.plot(ie, '.-', label="IE")
    plt.plot(e, '.-', label="Early")
    plt.plot(l, '.-', label="Late")
    plt.plot(ml, '.-', label="Multilabel")

    plt.xticks(np.arange(13), x_s, rotation=20)
    plt.legend()
    if adjusted:
        plt.title(f'Balanced Adjusted Accuracy with different input data')
        plt.savefig(f'Output/Balanced_accuracy/BAcc_adjusted.png', dpi=900)

    else:
        plt.title(f'Balanced Accuracy with different input data')
        plt.savefig(f'Output/Balanced_accuracy/BAcc.png', dpi=900)


def main():
    combined_features = pd.DataFrame()

    for virus in get_viruses_data():
        features = helper.read_csv_data(f"Output/features/{virus['name']}_features.csv")
        combined_features = pd.concat([combined_features, features], ignore_index=True)

    # plot_ROC_PR(combined_features)
    balanced_acc(combined_features, False)
    balanced_acc(combined_features, True)


if __name__ == "__main__":
    main()
