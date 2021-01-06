import numpy as np

from random import choice

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
#from sklearn.metrics.cluster import pair_confusion_matrix


class Cluster(object):

    def __init__(self, num_sims, X=None, metric="bottleneck", dmat=None):
        if X is None:
            X = np.array(list(range(num_sims)))[:, np.newaxis]

        self.X = X
        self.num_points = self.X.shape[0]
        self.dmat = dmat

        if metric == "bottleneck":
            self.metric = lambda i, j: self.dmat[int(i), int(j)]
        elif metric == "euclidean":
            self.metric = metric

        self.km = None
        self.class_members = None

    def fit(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.km = KMedoids(n_clusters, random_state=0, metric=self.metric)
        self.km.fit(self.X)
        self.class_members = [list() for _ in range(n_clusters)]
        for idx in range(self.num_points):
            self.class_members[self.km.labels_[idx]].append(idx)

    def maximize_silhouette(self, n_min=1, n_max=10):
        max_silhouette = -1.0
        max_n = 0

        for n in range(n_min, n_max + 1):
            self.fit(n_clusters=n)
            s = self.score()
            if s > max_silhouette:
                max_silhouette = s
                max_n = n

        self.fit(n_clusters=max_n)
            
    def cluster_grid(self, params1, params2, paramlist):
        grid = np.zeros((len(params2), len(params1)))
        P1, P2 = np.meshgrid(params1, params2)

        for i in range(P1.shape[0]):
            for j in range(P1.shape[1]):
                p1, p2 = P1[i, j], P2[i, j]
                idx = np.argmin([np.linalg.norm([p1 - x1, p2 - x2]) for (x1, x2) in paramlist])
                grid[i, j] = self.km.labels_[idx]

        return grid.T

    def score(self):
        return silhouette_score(self.X, self.km.labels_, metric=self.metric)

    #def confusion(self, newcluster):
    #    return pair_confusion_matrix(self.km.labels_, newcluster.km.labels_)

    def class_representative(self, k):
        return choice(self.class_members[k])