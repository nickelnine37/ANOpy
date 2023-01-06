import numpy as np
from sklearn.cluster import KMeans



def select_Q_active(n: int, seed: int = 0, n_clusters: int = 10):
    """

    """

    n_clusters = 10

    np.random.seed(seed)

    predictions = kmeans[seed].predict(X)

    groups = [np.argwhere(predictions == i).reshape(-1).tolist() for i in range(n_clusters)]

    np.random.shuffle(groups)

    j = 0

    nqs = []

    for i in range(n):

        group = groups[j % n_clusters]

        while len(group) == 0:
            j += 1
            group = groups[j % n_clusters]

        j += 1

        k = np.random.randint(len(group))
        nqs.append(group[k])
        del group[k]

    out = np.zeros(N * T)
    out[nqs] = 1

    return mat(out, like=J).astype(bool)
