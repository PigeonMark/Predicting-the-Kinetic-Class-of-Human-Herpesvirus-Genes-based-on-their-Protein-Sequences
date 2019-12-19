from numpy import mean, std
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from DataCollection.input_data import get_viruses_data
import helper


def select_one_label(data, label):
    new_data = data.copy()
    new_data['label'] = new_data['label'] == label
    return new_data


def classification(data, classifier):
    scores = []
    for i in range(100):
        train, test = train_test_split(data, test_size=1 / 3.)
        x_train = train[train.columns[2:-1]]
        y_train = train[train.columns[-1]]
        x_test = test[test.columns[2:-1]]
        y_test = test[test.columns[-1]]

        classifier.fit(x_train, y_train)
        scores.append(classifier.score(x_test, y_test))
    return mean(scores), std(scores)


def rf_classification(data):
    return classification(data, RandomForestClassifier())


def nb_classification(data):
    return classification(data, GaussianNB())


if __name__ == "__main__":

    combined_features = pd.DataFrame()

    for virus in get_viruses_data():
        features = helper.read_csv_data(f"Classification/Output/features/{virus['name']}_features.csv")
        combined_features = pd.concat([combined_features, features], ignore_index=True)

    for phase in ['immediate-early', 'early', 'late']:
        selected = select_one_label(combined_features, phase)
        scores_rf = rf_classification(selected)
        scores_nb = nb_classification(selected)
        print(f'Classifying only {phase}:')
        print(f'\tRandom forest:\n\t\tmean score: {scores_rf[0]}\n\t\tSTD: {scores_rf[1]}')
        print(f'\tNaive Bayes:\n\t\tmean score: {scores_nb[0]}\n\t\tSTD: {scores_nb[1]}')
        print()

    scores_rf = rf_classification(combined_features)
    scores_nb = nb_classification(combined_features)
    print(f'Multilabel classification:')
    print(f'\tRandom forest:\n\t\tmean score: {scores_rf[0]}\n\t\tSTD: {scores_rf[1]}')
    print(f'\tNaive Bayes:\n\t\tmean score: {scores_nb[0]}\n\t\tSTD: {scores_nb[1]}')
