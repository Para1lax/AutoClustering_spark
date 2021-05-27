import numpy as np
import math

from pyspark.sql.functions import mean as _mean, col

from .spark_custom import spark_join
from .utils import *
from .measure import Measure
from pyspark import SparkContext, SQLContext


class ChIndex(Measure):
    '''
    Note: doesn't found info about ch-index
    '''

    def __init__(self, centroids=None, cluster_sizes=None, x_center=0, numerator=None,
                 denominator=None, diameter=0):
        self.centroids = none_check(centroids)
        self.cluster_sizes = none_check(cluster_sizes)
        self.x_center = x_center
        self.numerator = none_check(numerator)
        self.denominator = none_check(denominator)
        self.diameter = diameter

    def find(self, data, spark_context, labels, n_clusters):
        sql = SQLContext(spark_context)
        rows, columns = spark_shape(data)
        mean_columns = map(lambda x: _mean(col(x)).alias('mean'), data.columns)
        df_stats = data.select(
            *mean_columns
        ).collect()
        df = spark_join(data, labels, 'labels', sql)
        self.x_center = np.array(df_stats[0])
        self.centroids = cluster_centroid(df, spark_context, n_clusters)
        self.diameter = find_diameter(df, spark_context)
        ch = float(rows - n_clusters) / float(n_clusters - 1)

        self.cluster_sizes = count_cluster_sizes(labels, n_clusters)
        self.numerator = [0 for _ in range(n_clusters)]
        for i in range(0, n_clusters):
             self.numerator[i] = self.cluster_sizes[i] * euclidian_dist(self.centroids[i], self.x_center)
        denominator_sum = spark_context.accumulator(0)

        def f(row, acc, centroid):
            acc += euclidian_dist(row[:-2], centroid[row[-2]])

        df.foreach(lambda row: f(row, denominator_sum, self.centroids))
        self.denominator = denominator_sum.value
        ch *= np.sum(self.numerator)
        ch /= self.denominator
        return -ch

    def update(self, data, spark_context, n_clusters, labels, k, l, id): # doesn't work
        rows, columns = spark_shape(data)
        sql = SQLContext(spark_context)
        df = spark_join(data, labels, 'labels', sql)
        delta = 10 ** (-math.log(rows, 10) - 1)
        point = df.filter(df.row_idx.isin(id))
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = count_cluster_sizes(labels, n_clusters)
        self.centroids = update_centroids(self.centroids, np.copy(self.cluster_sizes), point, k, l)
        ch = float(len(labels) - n_clusters) / float(n_clusters - 1)
        self.numerator[k] = self.cluster_sizes[k] * euclidian_dist(self.centroids[k], self.x_center)
        self.numerator[l] = self.cluster_sizes[l] * euclidian_dist(self.centroids[l], self.x_center)
        denom = spark_context.accumulator(0)

        def f(k, l, prev_centroids, centroids, delta, diameter, row, denom):
            if (row[-2] == k and euclidian_dist(prev_centroids[k], centroids[k]) > delta * diameter
               or row[-2] == l and euclidian_dist(prev_centroids[l], centroids[l]) > delta * diameter):
                denom += (euclidian_dist(row[:-2], centroids[row[-2]])
                                     - euclidian_dist(row[:-2], prev_centroids[row[-2]]))

        df.foreach(lambda row: f(k, l, prev_centroids, self.centroids, delta, self.diameter, row, denom))
        self.denominator += denom.value
        ch *= sum(self.numerator)
        ch /= self.denominator
        return -ch
