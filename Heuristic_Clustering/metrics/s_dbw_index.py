import numpy as np

from utils import *
from measure import Measure
import math


class SDbwIndex(Measure):

    def __init__(self):
        self.centroids = []
        self.cluster_sizes = []
        self.sigmas = []
        self.normed_sigma_x = 0
        self.diameter = 0
        self.dens = 0

    @staticmethod
    def solve_function(x_i, centroid_k, std):
        if std < euclidian_dist(x_i, centroid_k):
            return 0
        else:
            return 1

    @staticmethod
    def mean(x_i, x_j):
        return (x_i + x_j) / 2


    def den2(self, data, spark_context, k, l, std):
        acc = spark_context.accumulator(0)

        def f(x, acc, k, l, std, solve_function, mean, centroids):
            if x[-1] == k or x[-1] == l:
                acc += solve_function(x[:-2], mean(centroids[k], centroids[l]), std)

        data.foreach(lambda x: f(x, acc, k, l, std, self.solve_function, self.mean, self.centroids))
        return acc.value


    def den1(self, data, spark_context, k, std):
        acc = spark_context.accumulator(0)

        def f(x, acc, k, std, solve_function, centroids):
            if x[-1] == k:
                acc += solve_function(x[:-2], centroids[k], std)

        data.foreach(lambda x: f(x, acc, k, std, self.solve_function, self.centroids))
        return acc.value


    def normed_sigma(self, data, spark_context):
        rows, columns = spark_shape(data)
        acc = spark_context.accumulator(columns - 2, NumpyAccumulatorParam())

        def f(x, acc):
                acc += x[:-2]

        data.foreach(lambda x: f(x, acc))

        avg = acc.value / rows
        sigma = spark_context.accumulator(columns - 2, NumpyAccumulatorParam())

        def g(x, avg, sigma):
            sigma += np.square(x[:-2] - avg)

        data.foreach(lambda x: g(x, avg, sigma))
        sigma = sigma.value / rows
        return math.sqrt(np.dot(sigma, np.transpose(sigma)))


    def normed_cluster_sigma(self, data, spark_context, k):
        rows, columns = spark_shape(data)
        acc = spark_context.accumulator(columns - 2, NumpyAccumulatorParam())
        counter = spark_context.accumulator(0)

        def f(x, acc, counter, k):
            if x[-1] == k:
                acc += x[:-2]
                counter += 1

        data.foreach(lambda x: f(x, acc, counter, k))
        avg = acc.value / counter.value
        sigma = spark_context.accumulator(columns - 2, NumpyAccumulatorParam())

        def g(x, sigma, k, avg):
            if x[-1] == k:
                sigma += np.square(x[:-2] - avg)

        data.foreach(lambda x: g(x, sigma, k, avg))
        sigma = sigma.value / counter.value
        return math.sqrt(np.dot(sigma, np.transpose(sigma)))


    def stdev(self, n_clusters):
        sum = 0.0
        for k in range(n_clusters):
            sum += self.sigmas[k]
        sum = math.sqrt(sum)
        sum /= n_clusters
        return sum


    # s_dbw, S_Dbw index, min is better
    def find(self, data, spark_context):
        n_clusters = get_n_clusters(data, data.columns[-1])
        self.diameter = find_diameter(data, spark_context, 2)
        self.sigmas = [0 for _ in range(n_clusters)]
        self.centroids = cluster_centroid(data, spark_context, n_clusters, 2)
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)

        for k in range(n_clusters):
            self.sigmas[k] = self.normed_cluster_sigma(data, spark_context, k)
        self.normed_sigma_x = self.normed_sigma(data, spark_context)
        term1 = sum(self.sigmas) / (n_clusters * self.normed_sigma_x)
        stdev_val = self.stdev(n_clusters)

        self.dens = 0.0
        for k in range(0, n_clusters):
            for l in range(0, n_clusters):
                div = max(self.den1(data, spark_context, k, stdev_val),
                          self.den1(data, spark_context, l, stdev_val))
                if div != 0:
                    self.dens += self.den2(data, spark_context, k, l, stdev_val) / div

        self.dens /= n_clusters * (n_clusters - 1)
        return term1 + self.dens

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
            self.sigmas[k] = self.normed_cluster_sigma(data, spark_context, k)
        if euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter:
            self.sigmas[l] = self.normed_cluster_sigma(data, spark_context, l)
        term1 = sum(self.sigmas) / (n_clusters * self.normed_sigma_x)
        stdev_val = self.stdev(n_clusters)

        if (euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter
           or euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter):
            self.dens = 0.0
            for k in range(0, n_clusters):
                for l in range(0, n_clusters):
                    div = max(self.den1(data, spark_context, k, stdev_val),
                              self.den1(data, spark_context, l, stdev_val))
                    if div != 0:
                        self.dens += self.den2(data, spark_context, k, l, stdev_val) / div

        self.dens /= n_clusters * (n_clusters - 1)
        return term1 + self.dens
