from numpy import mean, std
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
    print(classifier.feature_importances_)
    return mean(scores), std(scores)


if __name__ == "__main__":

    combined_features = pd.DataFrame()

    for virus in get_viruses_data():
        features = helper.read_csv_data(f"Classification/Output/features/{virus['name']}_features.csv")
        combined_features = pd.concat([combined_features, features], ignore_index=True)

    classifiers = {
        "Random forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    for phase in ['immediate-early', 'early', 'late']:
        print(f'Classifying only {phase}:')

        selected = select_one_label(combined_features, phase)

        for name, classifier in classifiers.items():
            scores = classification(selected, classifier)
            print(f'\t{name}:\n\t\tmean score: {scores[0]}\n\t\tSTD: {scores[1]}')

        print()

    print(f'Multilabel classification:')
    for name, classifier in classifiers.items():
        scores = classification(combined_features, classifier)
        print(f'\t{name}:\n\t\tmean score: {scores[0]}\n\t\tSTD: {scores[1]}')
