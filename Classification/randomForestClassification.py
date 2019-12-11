from numpy import mean, std
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from DataCollection.input_data import get_viruses_data
import helper


def select_one_label(data, label):
    new_data = data.copy()
    new_data['label'] = new_data['label'] == label
    return new_data


def rf_classification(data):
    scores = []
    for i in range(100):
        train, test = train_test_split(data, test_size=1 / 3.)
        x_train = train[train.columns[2:-1]]
        y_train = train[train.columns[-1]]
        x_test = test[test.columns[2:-1]]
        y_test = test[test.columns[-1]]

        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(x_train, y_train)
        scores.append(rf_classifier.score(x_test, y_test))
    print(std(scores))
    return mean(scores)


if __name__ == "__main__":

    combined_features = pd.DataFrame()

    for virus in get_viruses_data():
        features = helper.read_csv_data(f"Classification/Output/features/{virus['name']}_features.csv")
        combined_features = pd.concat([combined_features, features], ignore_index=True)

    for phase in ['immediate-early', 'early', 'late']:
        selected = select_one_label(combined_features, phase)
        score = rf_classification(selected)
        print(f'Mean score for classifying only {phase} is: {score}')

    score = rf_classification(combined_features)
    print(f'Mean score for multilabel classification is: {score}')
