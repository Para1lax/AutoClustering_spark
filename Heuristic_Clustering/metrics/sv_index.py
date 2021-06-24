import heapq
import sys
import math

import numpy as np

from utils import *
from measure import Measure


class SVIndex(Measure):

    def __init__(self, centroids=None, clusters_sizes=None, centroid_dists=None,
                 dists=None, diameter=0):
        self.centroids = none_check(centroids)
        self.cluster_sizes = none_check(clusters_sizes)
        self.centroid_dists = none_check(centroid_dists)
        self.dists = none_check(dists)
        self.diameter = diameter

    def count_index(self, numerator, n_clusters, rows):
        denominator = 0.0
        self.dists = [[self.dists[i][j].value for j in range(rows)] for i in range(n_clusters)]
        for k in range(n_clusters):
            # get sum of 0.1*|Ck| largest elements
            acc = 0.0
            max_n = heapq.nlargest(int(math.ceil(0.1 * self.cluster_sizes[k])), self.dists[k])
            for i in range(0, len(max_n)):
                acc += max_n[i]
            denominator += acc * 10.0 / self.cluster_sizes[k]
        return -(numerator / denominator)

    def find(self, data, spark_context):
        rows, columns = spark_shape(data)
        n_clusters = get_n_clusters(data, data.columns[-1])
        self.diameter = find_diameter(data, spark_context, 2)
        self.centroids = cluster_centroid(data, spark_context, n_clusters, 2)
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)

        self.centroid_dists = [[sys.float_info.max for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.dists = [[spark_context.accumulator(0.0) for _ in range(rows)] for _ in range(n_clusters)]
        numerator = 0.0
        for k in range(0, n_clusters - 1):
            for l in range(k + 1, n_clusters):
                self.centroid_dists[k][l] = euclidian_dist(self.centroids[k], self.centroids[l])
                self.centroid_dists[l][k] = self.centroid_dists[k][l]
        for i in range(n_clusters):
            min_dist = np.amin(self.centroid_dists[i])
            numerator += min_dist
        data = add_iter(data)

        def f(x, dists, centroids):
            dists[x[-2]][x[-1] - 1] += euclidian_dist(x[:-3], self.centroids[x[-2]])

        data.foreach(lambda x: f(x, self.dists, self.centroids))
        return self.count_index(numerator, n_clusters, rows)

    def update(self, data, spark_context, n_clusters, k, l, ids):
        rows, columns = spark_shape(data)
        columns -= 2
        df = add_iter(data)
        prev_centroids = np.copy(self.centroids)
        points = df.filter(df.row_idx.isin(ids))
        delta = 10 ** (-math.log(rows, 10) - 1)
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)
        self.centroids = update_centroids(self.centroids, np.copy(self.cluster_sizes), points, k, l, 3)
        for i in range(n_clusters):
            if i > k:
                self.centroid_dists[k][i] = euclidian_dist(self.centroids[i], self.centroids[k])
                self.centroid_dists[i][k] = self.centroid_dists[k][i]
            if i > l:
                self.centroid_dists[l][i] = euclidian_dist(self.centroids[i], self.centroids[l])
                self.centroid_dists[i][l] = self.centroid_dists[l][i]
        numerator = 0.0
        for i in range(n_clusters):
            min_dist = np.amin(self.centroid_dists[i])
            numerator += min_dist
        for i in ids:
            self.dists[k][i] = 0.
        self.dists = [[spark_context.accumulator(self.dists[i][j]) for j in range(rows)] for i in range(n_clusters)]

        def f(x, k, l, diameter, centroids, dists, prev_centroids):
            label = x[-2]
            if (label == k and euclidian_dist(prev_centroids[k], centroids[k]) > delta * diameter
               or label == l and euclidian_dist(prev_centroids[l], centroids[l]) > delta * diameter):
                dists[label][x[-1] - 1] = euclidian_dist(x[:-3], self.centroids[label])

        df.foreach(lambda x: f(x, k, l, self.diameter, self.centroids, self.dists, prev_centroids))
        return self.count_index(numerator, n_clusters, rows)


