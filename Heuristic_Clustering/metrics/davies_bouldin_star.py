from utils import *
from measure import Measure
import sys
import math


class DBSIndex(Measure):

    def __init__(self):
        self.s_clusters = []
        self.cluster_sizes = []
        self.centroids = []
        self.max_s_sum = []
        self.min_centroids_dist = []
        self.diameter = 0

    def s(self, data, spark_context, cluster_k_index):
        accum = spark_context.accumulator(0)

        def f(x, accum, cluster_k_index, cetroids):
            if x[-1] == cluster_k_index:
                accum += euclidian_dist(x[:-2], cetroids[cluster_k_index])

        data.foreach(lambda x: f(x, accum, cluster_k_index, self.centroids))
        if self.cluster_sizes[cluster_k_index] == 0:
            return float('inf')
        return accum.value / self.cluster_sizes[cluster_k_index]

    # db_star, DB*-index, min is better
    def find(self, data, spark_context):
        n_clusters = get_n_clusters(data, data.columns[-1])
        self.centroids = cluster_centroid(data, spark_context, n_clusters, 2)
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)
        self.max_s_sum = [[sys.float_info.min for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.min_centroids_dist = [[sys.float_info.max for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.s_clusters = [0 for _ in range(n_clusters)]
        self.diameter = find_diameter(data, spark_context, 2)
        for i in range(n_clusters):
            self.s_clusters[i] = self.s(data, spark_context, i)
        numerator = 0.0
        for k in range(0, n_clusters):
            for l in range(k + 1, n_clusters):
                self.max_s_sum[k][l] = self.s_clusters[k] + self.s_clusters[l]
                self.min_centroids_dist[k][l] = euclidian_dist(self.centroids[k], self.centroids[l])
            numerator += np.max(self.max_s_sum[k]) / np.min(self.min_centroids_dist[k])
        return numerator / n_clusters

    def update(self, data, spark_context, n_clusters, k, l, ids):
        rows, columns = spark_shape(data)
        columns -= 2
        df = add_iter(data)
        delta = 10 ** (-math.log(rows, 10) - 1)
        points = df.filter(df.row_idx.isin(ids))
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)
        self.centroids = update_centroids(self.centroids, np.copy(self.cluster_sizes), points, k, l, 3)
        if euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter:
            self.s_clusters[k] = self.s(data, spark_context, k)
        if euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter:
            self.s_clusters[l] = self.s(data, spark_context, l)
        for i in range(n_clusters):
            if i > k:
                self.max_s_sum[k][i] = self.s_clusters[i] + self.s_clusters[k]
                self.min_centroids_dist[k][i] = euclidian_dist(self.centroids[i], self.centroids[k])
            if i > l:
                self.max_s_sum[l][i] = self.s_clusters[i] + self.s_clusters[l]
                self.min_centroids_dist[l][i] = euclidian_dist(self.centroids[i], self.centroids[l])
        numerator = 0.0
        for i in range(n_clusters):
            numerator += np.max(self.max_s_sum[i]) / np.min(self.min_centroids_dist[i])
        return numerator / n_clusters