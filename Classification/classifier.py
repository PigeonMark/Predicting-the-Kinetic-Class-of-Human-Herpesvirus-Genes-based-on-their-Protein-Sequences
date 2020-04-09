import json
import pandas as pd
import numpy as np
import progressbar
import xgboost as xgb

from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

from Util import compose_filename


def create_scorers(RR=False):
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
    adjusted_balanced_accuracy_scorer = make_scorer(balanced_accuracy_score, adjusted=True)
    roc_auc_ovo_scorer = make_scorer(roc_auc_score, average='weighted', multi_class='ovo', needs_proba=True)
    roc_auc_ovr_scorer = make_scorer(roc_auc_score, average='weighted', multi_class='ovr', needs_proba=True)

    scoring_dict = {
        "balanced_accuracy": balanced_accuracy_scorer,
        "adjusted_balanced_accuracy": adjusted_balanced_accuracy_scorer
    }

    if not RR:
        scoring_dict.update({"roc_auc_ovo_score": roc_auc_ovo_scorer,
                             "roc_auc_ovr_score": roc_auc_ovr_scorer})
    return scoring_dict


def add_class_roc_scores(cv_results, estimator, y_test_bin, y_score):
    for i, class_ in enumerate(estimator.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        cv_results['fpr'][class_].append(fpr)
        cv_results['tpr'][class_].append(tpr)
        cv_results['roc_auc'][class_].append(auc(fpr, tpr))


def add_micro_roc_scores(cv_results, y_test_bin, y_score):
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    cv_results['fpr']['micro'].append(fpr)
    cv_results['tpr']['micro'].append(tpr)
    cv_results['roc_auc']['micro'].append(auc(fpr, tpr))


def add_macro_roc_scores(cv_results, estimator):
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([cv_results['fpr'][class_][-1] for class_ in estimator.classes_]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for class_ in estimator.classes_:
        mean_tpr += np.interp(all_fpr, cv_results['fpr'][class_][-1], cv_results['tpr'][class_][-1])

    # Finally average it and compute AUC
    mean_tpr /= len(estimator.classes_)
    cv_results['fpr']['macro'].append(all_fpr)
    cv_results['tpr']['macro'].append(mean_tpr)
    cv_results['roc_auc']['macro'].append(auc(all_fpr, mean_tpr))


def add_roc_pr_scores(cv_results, estimator, x_test, y_test):
    if hasattr(estimator, 'predict_proba'):
        y_score = estimator.predict_proba(x_test)
        y_test_bin = label_binarize(y_test, classes=estimator.classes_)

        for i, class_ in enumerate(estimator.classes_):
            cv_results['y_real'][class_].append(y_test_bin[:, i])
            cv_results['y_proba'][class_].append(y_score[:, i])

        cv_results['y_real']['micro'].append(y_test_bin.ravel())
        cv_results['y_proba']['micro'].append(y_score.ravel())

        add_class_roc_scores(cv_results, estimator, y_test_bin, y_score)
        add_micro_roc_scores(cv_results, y_test_bin, y_score)
        add_macro_roc_scores(cv_results, estimator)


def add_scorer_scores(cv_results, estimator, scoring, x_test, y_test):
    for name, scorer in scoring.items():
        score = scorer(estimator, x_test, y_test)
        cv_results[name].append(score)


def add_feature_permutation_importance(cv_results, estimator, x_train, y_train):
    result = permutation_importance(estimator, x_train, y_train, 'balanced_accuracy', 5, 42)
    cv_results['permutation_importance'].append(result['importances_mean'])


def add_wrong_classifications(cv_results, estimator, x_test, y_test):
    predicted = estimator.predict(x_test)
    for i, y_t in enumerate(y_test):
        if predicted[i] != y_t:
            cv_results['wrong_classifications'][y_t][predicted[i]] += 1


def cross_validate(estimator, x, y, scoring, cv: ShuffleSplit):
    cv_results = {name: [] for name, _ in scoring.items()}
    cv_results['permutation_importance'] = []
    for metric in ['fpr', 'tpr', 'roc_auc', 'y_real', 'y_proba']:
        cv_results[metric] = {class_: [] for class_ in ['early', 'immediate-early', 'late', 'micro', 'macro']}
    cv_results['wrong_classifications'] = {class_: {class_2: 0 for class_2 in ['early', 'immediate-early', 'late']} for
                                           class_ in ['early', 'immediate-early', 'late']}

    widgets = [progressbar.Percentage(), ' done']
    bar = progressbar.ProgressBar(widgets=widgets, max_value=cv.n_splits).start()
    bar.update(0)
    i = 0
    for train, test in cv.split(x, y):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        estimator.fit(x_train, y_train)

        add_wrong_classifications(cv_results, estimator, x_test, y_test)
        add_roc_pr_scores(cv_results, estimator, x_test, y_test)
        add_scorer_scores(cv_results, estimator, scoring, x_test, y_test)
        add_feature_permutation_importance(cv_results, estimator, x_train, y_train)

        i += 1
        bar.update(i)
    bar.finish()

    return cv_results


class Classifier:
    def __init__(self, config_filepath, ml_method, classifier):
        self.ml_method = ml_method
        self.classifier = classifier
        self.config = None
        self.data = None  # type: pd.DataFrame
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            ml_method_options = config['ML-method-options']
            classifier_options = config['Classifier-options']

            if self.ml_method not in ml_method_options:
                raise Exception(f'Multilabel method "{self.ml_method}" is not one of the options')
            if self.classifier not in classifier_options:
                raise Exception(f'Classifier "{self.classifier}" is not one of the options')

            self.config = config

    def __create_classifier(self, grid_search=False):
        if self.classifier == "KNN":
            base_classifier = KNeighborsClassifier()
        elif self.classifier == "RF":
            base_classifier = RandomForestClassifier()
        elif self.classifier == "XGBoost":
            if grid_search:
                base_classifier = xgb.XGBClassifier()
            else:
                base_classifier = xgb.XGBClassifier(**self.config['XGBoost-options'][self.ml_method])
        else:
            raise Exception(f'Classifier "{self.classifier}" is not one of the options')

        if self.ml_method == "1vsA":
            classifier = OneVsRestClassifier(base_classifier)
        elif self.ml_method == "RR":
            classifier = OneVsOneClassifier(base_classifier)
        elif self.ml_method == "ML":
            classifier = base_classifier
        else:
            raise Exception(f'Multilabel method "{self.ml_method}" is not one of the options')

        return classifier

    def grid_search(self, name, grid, splits, n_pca=None):
        filename = compose_filename(self.config['input_data_folder'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca, 'features', name, 'csv')
        all_data = pd.read_csv(filename, index_col=0)
        x = all_data[[col_name for col_name in all_data.columns if col_name not in self.config['skip-features']]]
        y = all_data['label']
        classifier = self.__create_classifier(grid_search=True)
        gs_cv = GridSearchCV(classifier, grid, scoring=make_scorer(balanced_accuracy_score, adjusted=True),
                             cv=ShuffleSplit(n_splits=splits, train_size=0.67))
        gs_cv.fit(x.values, y.values)
        print(gs_cv.best_params_)
        print(gs_cv.best_score_)
        print()
        for i, param in enumerate(gs_cv.cv_results_['params']):
            print(param, gs_cv.cv_results_['mean_test_score'][i], 'time:', gs_cv.cv_results_['mean_fit_time'][i])
            print()

    def fit(self, name, n_pca=None):
        filename = compose_filename(self.config['input_data_folder'], self.config['filter_latent'],
                                    self.config['standardization'], n_pca, 'features', name, 'csv')

        all_data = pd.read_csv(filename, index_col=0)
        features = [col_name for col_name in all_data.columns if col_name not in self.config['skip-features']]
        x = all_data[features]
        y = all_data['label']
        k_fold = ShuffleSplit(n_splits=self.config['k-fold-splits'], train_size=self.config['k-fold-train-size'])

        print(f"Classifying with {self.ml_method} and {self.classifier}")

        if self.ml_method == 'RR':
            cv_results = cross_validate(self.__create_classifier(), x.values, y.values, create_scorers(True),
                                        k_fold)
        else:
            cv_results = cross_validate(self.__create_classifier(), x.values, y.values, create_scorers(), k_fold)

        cv_results['features'] = features

        return cv_results
