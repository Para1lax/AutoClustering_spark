import sys
import math
import numpy as np

from utils import *
from measure import Measure


class GD53Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, delta=None, centroid_dists=None,
                 sums=None, diameter=0):
        self.centroids = none_check(centroids)
        self.cluster_sizes = none_check(cluster_sizes)
        self.delta = none_check(delta)
        self.centroid_dists = none_check(centroid_dists)
        self.sums = none_check(sums)
        self.diameter = diameter

    def find(self, data, spark_context):
        rows, columns = spark_shape(data)
        n_clusters = get_n_clusters(data, data.columns[-1])
        self.diameter = find_diameter(data, spark_context, 2)
        self.centroids = cluster_centroid(data, spark_context, n_clusters, 2)
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)
        self.centroid_dists = [spark_context.accumulator(0) for _ in range(rows)]
        self.delta = [[0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        self.sums = [spark_context.accumulator(0) for _ in range(n_clusters)]
        data = add_iter(data)

        def f(x, centroids, sums, centroid_dists):
            dist = euclidian_dist(x[:-3], centroids[x[-2]])
            centroid_dists[x[-1] - 1] += dist
            sums[x[-2]] += dist

        data.foreach(lambda x: f(x, self.centroids, self.sums, self.centroid_dists))

        self.centroid_dists = list(map(lambda x: x.value, self.centroid_dists))
        self.sums = list(map(lambda x: x.value, self.sums))

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    self.delta[i][j] = (self.sums[i] + self.sums[j]) / float(self.cluster_sizes[i] + self.cluster_sizes[j])
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
        denominator = list(self.sums)
        for i in range(n_clusters):
            denominator[i] *= (2 / self.cluster_sizes[i])
        return -(minimum_dif_c / max(denominator))

    def update(self, data, spark_context, n_clusters, k, l, ids):
        rows, columns = spark_shape(data)
        columns -= 2
        df = add_iter(data)
        delta = 10 ** (-math.log(rows, 10) - 1)
        points = df.filter(df.row_idx.isin(ids))
        prev_centroids = np.copy(self.centroids)
        prev_cluster_sizes = np.copy(self.cluster_sizes)
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)
        self.centroids = update_centroids(self.centroids, np.copy(self.cluster_sizes), points, k, l, 3)
        minimum_dif_c = sys.float_info.max  # min dist in different clusters

        def f(x, k, l, delta, diameter, centroids, new_centroid_dists):
            if (x[-2] == k and euclidian_dist(prev_centroids[k], centroids[k]) > delta * diameter
               or x[-2] == l and euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * diameter):
                new_centroid_dists[x[-1] - 1] += euclidian_dist(x[:-3], centroids[x[-2]])

        new_centroid_dists = [spark_context.accumulator(0) for _ in range(rows)]
        df.foreach(lambda x: f(x, k, l, delta, self.diameter, self.centroids, new_centroid_dists))
        new_centroid_dists = [self.centroid_dists[i]
                              if new_centroid_dists[i].value == 0 else
                              new_centroid_dists[i].value
                              for i in range(rows)]

        for i in range(n_clusters):
            for j in range(n_clusters):
                self.delta[i][j] *= (prev_cluster_sizes[i] + prev_cluster_sizes[j])

        new_sums = [spark_context.accumulator(0.0) for _ in range(n_clusters)]
        for i in range(n_clusters):
            if i != k and i != l:
                new_sums[i] = spark_context.accumulator(self.sums[i])

        def g(x, k, l, new_sums, new_centroid_dists):
            if x[-2] == k or x[-2] == l:
                new_sums[x[-2]] += new_centroid_dists[x[-1]]

        df.foreach(lambda x: g(x, k, l, new_sums, new_centroid_dists))
        new_sums = list(map(lambda x: x.value, new_sums))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    if self.cluster_sizes[i] + self.cluster_sizes[j] == 0:
                        self.delta[i][j] = float('inf')
                    else:
                        self.delta[i][j] = (new_sums[i] + new_sums[j]) / float(self.cluster_sizes[i] + self.cluster_sizes[j])
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])


        #update denominator
        denominator = list(new_sums)
        #print(denominator)
        for i in range(n_clusters):
            if self.cluster_sizes[i] == 0:
                denominator[i] = float('inf')
            else:
                denominator[i] *= (2 / self.cluster_sizes[i])

        return -(minimum_dif_c / max(denominator))

