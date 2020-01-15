import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import helper
from DataCollection.input_data import get_viruses_data
from sklearn.preprocessing import StandardScaler


def t_sne(features):
    X = features[features.columns[2:-1]]

    std_scaler = StandardScaler()
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=10)

    scaled_X = std_scaler.fit_transform(X)

    results = tsne.fit_transform(scaled_X)
    return results


def pca(features, n_components):
    X = features[features.columns[2:-1]]

    std_scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    scaled_X = std_scaler.fit_transform(X)

    results = pca.fit_transform(scaled_X)
    results = pd.concat([features[features.columns[0:2]], pd.DataFrame(results), features[features.columns[-1]]], axis=1)

    return results, pca


def plot(features, pca_results, filename, method):
    # Wont probably work because of changes in pca()
    ie_x, ie_y, e_x, e_y, l_x, l_y = ([], [], [], [], [], [])
    for i, feat in features.iterrows():
        if feat['label'] == 'immediate-early':
            ie_x.append(pca_results[i][0])
            if len(pca_results[i]) > 1:
                ie_y.append(pca_results[i][1])
            else:
                ie_y.append(0)
        elif feat['label'] == 'early':
            e_x.append(pca_results[i][0])
            if len(pca_results[i]) > 1:
                e_y.append(pca_results[i][1])
            else:
                e_y.append(0)
        elif feat['label'] == 'late':
            l_x.append(pca_results[i][0])
            if len(pca_results[i]) > 1:
                l_y.append(pca_results[i][1])
            else:
                l_y.append(0)
    alpha = 0.4
    size = 30
    fig, ax = plt.subplots()
    ax.scatter(ie_x, ie_y, c='b', s=size, alpha=alpha, label='immediate_early', edgecolors='none')
    ax.scatter(e_x, e_y, c='r', s=size, alpha=alpha, label='early', edgecolors='none')
    ax.scatter(l_x, l_y, c='g', s=size, alpha=alpha, label='late', edgecolors='none')

    ax.legend()
    plt.savefig(f"Classification/Output/{method}_plots/scaled_{filename}.png", dpi=600)


def main():

    # Wont probably work because of changes in pca()

    combined_features = pd.DataFrame()

    for virus in get_viruses_data():
        features = helper.read_csv_data(f"Classification/Output/features/{virus['name']}_features.csv")
        combined_features = pd.concat([combined_features, features], ignore_index=True)

        tsne_results = t_sne(features)
        plot(features, tsne_results, virus["name"], 'TSNE')

        pca_results, pca_obj = pca(features)

        print(
            f'Singular values and explained variance for {virus["name"]}:\n{pca_obj.singular_values_}\n{pca_obj.explained_variance_ratio_}\n')
        print(f'{pca_obj.components_}')

        plot(features, pca_results, virus['name'], 'PCA')

    pca_results, pca_obj = pca(combined_features)
    print(
        f'Singular values and explained variace for combination of all herpes viruses:\n{pca_obj.singular_values_}\n{pca_obj.explained_variance_ratio_}\n')
    print(f'{pca_obj.components_}')

    tsne_results = t_sne(combined_features)
    plot(combined_features, pca_results, 'all_herpes_viruses', 'PCA')
    plot(combined_features, tsne_results, 'all_herpes_viruses', 'TSNE')


if __name__ == "__main__":
    main()
