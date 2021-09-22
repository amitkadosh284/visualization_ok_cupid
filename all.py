import np as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def load_data():
    df = pd.read_csv('selfdata.csv')
    names = df[['name', 'age']]
    # For each row, combine all the columns into one column
    names1 = names.apply(lambda x: ','.join(x.astype(str)), axis=1)

    columns = df[['age', 'smoke', 'drink', 'short', 'long', 'hookup']]
    col_to_bert = df[['summary', 'family', 'background', 'location']]
    return columns, col_to_bert, names1


# Select features from original dataset to form a new dataframe

def bert_data(columns_to_bert, columns):
    for col in columns_to_bert:
        data_col = col_to_bert[[col]]
        all_persons = []
        for index, row in data_col.iterrows():
            for data in row:
                all_persons.append(embedder.encode(data))
        all_persons = np.asarray(all_persons)
        if col == 'summary':
            person = cosine_similarity([all_persons[0]], all_persons)
        else:
            person = np.concatenate([person, cosine_similarity([all_persons[0]], all_persons)])
    data = {'summary': person[0],
            'family': person[1],
            'background': person[2],
            'location': person[3]}

    df_finish = pd.DataFrame(data)

    for col in columns:
        df_finish[col] = columns[col].values

    return df_finish


def pca_and_plt(data, names):
    # # Create a PCA instance: pca
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(principalComponents)
    y_kmeans = kmeans.predict(principalComponents)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    for i, name in enumerate(names):
        ax.annotate(name, (principalComponents[:, 0][i], principalComponents[:, 1][i]))
    plt.show()


if __name__ == "__main__":
    columns, col_to_bert, names = load_data()
    bert_data = bert_data(col_to_bert, columns)
    pca_and_plt(bert_data, names)
