import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import helper
from DataCollection.input_data import get_viruses_data


def t_sne(features):
    X = features[features.columns[2:-1]]
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=10)
    results = tsne.fit_transform(X)
    return results


def pca(features):
    X = features[features.columns[2:-1]]
    pca = PCA()

    pca.fit(X)
    results = pca.transform(X)
    return results, pca


def plot(features, pca_results, filename, method):
    ie_x, ie_y, e_x, e_y, l_x, l_y = ([], [], [], [], [], [])
    for i, feat in features.iterrows():
        if feat['label'] == 'immediate-early':
            ie_x.append(pca_results[i][0])
            ie_y.append(pca_results[i][1])
        elif feat['label'] == 'early':
            e_x.append(pca_results[i][0])
            e_y.append(pca_results[i][1])
        elif feat['label'] == 'late':
            l_x.append(pca_results[i][0])
            l_y.append(pca_results[i][1])

    alpha = 0.4
    size = 30
    fig, ax = plt.subplots()
    ax.scatter(ie_x, ie_y, c='b', s=size, alpha=alpha, label='immediate_early', edgecolors='none')
    ax.scatter(e_x, e_y, c='r', s=size, alpha=alpha, label='early', edgecolors='none')
    ax.scatter(l_x, l_y, c='g', s=size, alpha=alpha, label='late', edgecolors='none')

    ax.legend()
    plt.savefig(f"Classification/Output/{method}_plots/{filename}.png", dpi=600)


def main():
    combined_features = pd.DataFrame()

    for virus in get_viruses_data():
        features = helper.read_csv_data(f"Classification/Output/features/{virus['name']}_features.csv")
        combined_features = pd.concat([combined_features, features], ignore_index=True)

        tsne_results = t_sne(features)
        plot(features, tsne_results, virus["name"], 'TSNE')

        pca_results, pca_obj = pca(features)

        print(f'Singular values for {virus["name"]}:\n{pca_obj.singular_values_}\n')

        plot(features, pca_results, virus['name'], 'PCA')

    pca_results, pca_obj = pca(combined_features)
    print(f'Singular values for combination of all herpes viruses:\n{pca_obj.singular_values_}\n')
    tsne_results = t_sne(combined_features)
    plot(combined_features, pca_results, 'all_herpes_viruses', 'PCA')
    plot(combined_features, tsne_results, 'all_herpes_viruses', 'TSNE')


if __name__ == "__main__":
    main()
