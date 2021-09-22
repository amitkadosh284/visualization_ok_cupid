import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_data():
    df = pd.read_csv('selfdata.csv')

    names = df[['name', 'age']]
    # For each row, combine all the columns into one column
    names1 = names.apply(lambda x: ','.join(x.astype(str)), axis=1)
    return df[['summary']], names1


def bert_data(data):
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    all_persons = []
    for index, row in data.iterrows():
        person = np.array([])
        for col in row:
            person = np.concatenate([person, embedder.encode(col)])
        all_persons.append(person)
    return all_persons


def create_pca_and_plt(data, names1):
    # Create a PCA instance: pca
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    # Plot the explained variances
    features = range(pca.n_components_)

    # Save components to a DataFrame
    PCA_components = pd.DataFrame(principalComponents)

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(principalComponents)
    y_kmeans = kmeans.predict(principalComponents)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    for i, name in enumerate(names1):
        ax.annotate(name, (principalComponents[:, 0][i], principalComponents[:, 1][i]))
    plt.show()

if __name__ == "__main__":
    data, names = load_data()
    bert_data = bert_data(data)
    create_pca_and_plt(bert_data, names)