import json
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import ShuffleSplit, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

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

        if self.ml_method != 'RR':
            cv_results = cross_validate(self.__create_classifier(), x.values, y.values, scoring=create_scorers(),
                                        cv=k_fold, return_estimator=True)
        else:
            cv_results = cross_validate(self.__create_classifier(), x.values, y.values, scoring=create_scorers(True),
                                        cv=k_fold, return_estimator=True)

        if hasattr(cv_results['estimator'][0], 'feature_importances_'):
            cv_results['feature_importance'] = [estimator.feature_importances_ for estimator in cv_results['estimator']]
        del cv_results['estimator']
        cv_results['features'] = features

        return cv_results
