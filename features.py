import numpy as np
from utils import vec, mat
from numpy import ndarray
from sklearn.cluster import KMeans


class Features:

    def __init__(self,
                 G: ndarray,
                 UT: ndarray,
                 UN: ndarray,
                 S_: ndarray,
                 A_: ndarray,
                 n_clusters: int = 10):

        self.N, self.T = G.shape

        self.X = np.array([np.ones(self.N * self.T),
                           vec(S_),
                           vec(UN @ (G * (UN.T @ S_ @ UT)) @ UT.T),
                           vec((UN ** 2) @ G @ (UT.T ** 2)),
                           vec((UN ** 2) @ (G ** 2) @ (UT.T ** 2)),
                           vec(A_),
                           vec(UN @ (G * (UN.T @ A_ @ UT)) @ UT.T)
                           ]).T

        self.X[:, 1:] = self.X[:, 1:] / self.X[:, 1:].std(axis=0)
        self.X[:, 1:] = self.X[:, 1:] - self.X[:, 1:].mean(axis=0)

        self.n_clusters = n_clusters
        self.clusters = KMeans(n_clusters=n_clusters).fit_predict(self.X)

    def select_Q_active(self, n: int, seed: int = 0):

        np.random.seed(seed)

        groups = [np.argwhere(self.clusters == i).reshape(-1).tolist() for i in range(self.n_clusters)]

        np.random.shuffle(groups)
        for group in groups:
            np.random.shuffle(group)

        j = 0

        nqs = np.zeros(n, dtype=int)

        for i in range(n):

            group = groups[j % self.n_clusters]

            while len(group) == 0:
                j += 1
                group = groups[j % self.n_clusters]

            nqs[i] = group.pop()
            j += 1


        Q = np.zeros(self.N * self.T)
        Q[nqs] = 1

        return mat(Q, shape=(self.N, self.T)).astype(bool)


    def select_Q_passive(self, n: int, seed=0):

        np.random.seed(seed)
        Q = np.zeros(self.N * self.T)
        Q[np.random.choice(self.N * self.T , size=n, replace=False)] = 1

        return mat(Q, shape=(self.N, self.T)).astype(bool)

