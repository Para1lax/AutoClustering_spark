import itertools

import numpy as np
import operator
import pyspark

from pyspark import RDD
from numba import njit
from itertools import starmap

from pyspark.sql.functions import round, rand
from HeuristicDataset import HeuristicDataset as HD


class Distance:
    functions = frozenset(['euclidean', 'sqr_euclidean', 'manhattan', 'chebyshev', 'cosine'])
    # numba requires numpy array, so should extract ndarray from pyspark.DenseVector

    @staticmethod
    def euclidean(x, y):
        return Distance._numba_euclidean(x.values, y.values)

    @staticmethod
    @njit
    def _numba_euclidean(x, y):
        diff = x - y
        return np.sqrt(np.sum(diff * diff))

    @staticmethod
    def sqr_euclidean(x, y):
        return Distance._numba_sqr_euclidean(x.values, y.values)

    @staticmethod
    @njit
    def _numba_sqr_euclidean(x, y):
        diff = x - y
        return np.sum(diff * diff)

    @staticmethod
    def manhattan(x, y):
        return Distance._numba_manhattan(x.values, y.values)

    @staticmethod
    @njit
    def _numba_manhattan(x, y):
        diff = x - y
        return np.sum(np.abs(diff))

    @staticmethod
    def chebyshev(x, y):
        return Distance._numba_chebyshev(x.values, y.values)

    @staticmethod
    @njit
    def _numba_chebyshev(x, y):
        diff = x - y
        return np.amax(np.abs(diff))

    @staticmethod
    def cosine(x, y):
        return Distance._numba_cosine(x.values, y.values)

    @staticmethod
    @njit
    def _numba_cosine(x, y):
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        return np.dot(x, y) / (x_norm * y_norm)


class Measure:
    SIL, CH, SCORE, DB = 'silhouette', 'calinski_harabasz', 'score', 'davies_bouldin'
    DUNN, G31, G33, G41, G43, G51, G53 = 'dunn', 'g31', 'g33', 'g41', 'g43', 'g51', 'g53'
    S_DBW, SV = 's_dbw', 'sv'  # needs improvement

    functions = frozenset([SIL, CH, SCORE, DB, S_DBW, DUNN, G31, G33, G41, G43, G51, G53])
    increasing = frozenset([SIL, CH, SCORE, DUNN, G31, G33, G41, G43, G51, G53])
    decreasing = frozenset([DB, S_DBW])

    def __init__(self, algorithm, distance, **kw):
        if algorithm not in Measure.functions:
            raise ValueError('No such measure algorithm: {}'.format(algorithm))
        self.algorithm, self.kw, self.measure_func = algorithm, kw, self.__getattribute__(algorithm)
        self.distance = distance if callable(distance) else getattr(Distance, distance)

    def __call__(self, df, minimise=False):
        labels = HD.get_unique_labels(df)
        if len(labels) < 2:
            return float('inf') if minimise else float('-inf')
        clusters = [df.rdd.filter(lambda x: x.labels == label).cache() for label in labels]
        centroids, amounts = list(map(self.__get_centroid, clusters)), list(map(lambda c: c.count(), clusters))
        return self.measure_func(df, list(zip(clusters, centroids, amounts, labels)))

    def __get_centroid(self, spark_rdd):
        return spark_rdd.map(lambda x: x.features).reduce(operator.add) / spark_rdd.count()

    def __dists_to_centroid(self, cluster, reducer):
        centroid = cluster[1]  # extract centroid from tuple, so Spark can serialize vector
        return reducer(cluster[0].map(lambda x: self.distance(x.features, centroid)))

    def dunn(self, full_df, clusters):
        return self.__dunn_index(clusters, self.__dunn_dist_1, self.__dunn_diam_1)

    def g31(self, full_df, clusters):
        return self.__dunn_index(clusters, self.__dunn_dist_3, self.__dunn_diam_1)

    def g41(self, full_df, clusters):
        return self.__dunn_index(clusters, self.__dunn_dist_4, self.__dunn_diam_1)

    def g51(self, full_df, clusters):
        return self.__dunn_index(clusters, self.__dunn_dist_5, self.__dunn_diam_1)

    def g33(self, full_df, clusters):
        return self.__dunn_index(clusters, self.__dunn_dist_3, self.__dunn_diam_3)

    def g43(self, full_df, clusters):
        return self.__dunn_index(clusters, self.__dunn_dist_4, self.__dunn_diam_3)

    def g53(self, full_df, clusters):
        return self.__dunn_index(clusters, self.__dunn_dist_5, self.__dunn_diam_3)

    def __dunn_index(self, clusters, dist, diam):
        dists = [dist(clusters[x], clusters[y]) for x in range(1, len(clusters)) for y in range(x)]
        return min(dists) / max(map(diam, clusters))

    def __dunn_diam_1(self, cluster):
        return cluster[0].cartesian(cluster[0]).map(lambda pp: self.distance(pp[0].features, pp[1].features)).max()

    def __dunn_diam_3(self, cluster):
        return 2 * self.__dists_to_centroid(cluster, RDD.sum) * cluster[2]

    def __dunn_dist_1(self, x, y):
        return x[0].cartesian(y[0]).map(lambda pp: self.distance(pp[0].features, pp[1].features)).min()

    def __dunn_dist_3(self, x, y):
        mul = 0.5 / x[2] * y[2]  # overall sum doubles, so 0.5 instead of 1
        return x[0].cartesian(y[0]).map(lambda pp: self.distance(pp[0].features, pp[1].features)).sum() * mul

    def __dunn_dist_4(self, x, y):
        return self.distance(x[1], y[1])

    def __dunn_dist_5(self, x, y):
        x_sum, y_sum = self.__dists_to_centroid(x, RDD.sum), self.__dists_to_centroid(y, RDD.sum)
        return (x_sum + y_sum) / (x[2] + y[2])

    def calinski_harabasz(self, full_df, clusters):
        global_centroid = self.__get_centroid(full_df.rdd)
        global_sum = sum(map(lambda c: c[2] * self.distance(c[1], global_centroid), clusters))
        clusters_sum = sum(map(lambda c: self.__dists_to_centroid(c, RDD.sum), clusters))
        mul = (full_df.count() - len(clusters)) / (len(clusters) - 1)
        return mul * global_sum / clusters_sum

    def silhouette(self, full_df, clusters):
        amount_by_label = {c[3]: c[2] for c in clusters}
        sil_values = [(self.__sil_b(c, full_df, amount_by_label), self.__sil_a(c)) for c in clusters]
        return sum(map(self.__sil_cluster, sil_values)) / full_df.count()

    def __sil_a(self, cluster):
        """
        Average distances between points in similar cluster
        point -> (point, point) -> (point.id, dists) -> (point.id, sum_dist) -> (point.id, avg_dist)
        """
        # cluster_size = cluster[2] - 1  will match with Spark ClusteringEvaluator
        cluster_size = cluster[2]  # Serialize size
        return cluster[0].cartesian(cluster[0]).map(
            lambda pp: (pp[0].id, self.distance(pp[0].features, pp[1].features))
        ).reduceByKey(operator.add).mapValues(lambda sum_dist: sum_dist / cluster_size)

    def __sil_b(self, cluster, full_df, amounts):
        """
        Minimal distance from each point of cluster to another clusters
        x_point -> (x_point, y_point) -> ((x_point.id, y_label), point_dists) ->
        ((x_point.id, y_label), sum_dist) -> (x_point.id, avg_dists) -> (x_point.id, min_dist)
        """
        cluster_label = cluster[3]  # Serialize label
        another_clusters_points = full_df.rdd.filter(lambda x: x.labels != cluster_label)
        return cluster[0].cartesian(another_clusters_points).map(
            lambda pp: ((pp[0].id, pp[1].labels), self.distance(pp[0].features, pp[1].features))
        ).reduceByKey(operator.add).map(
            lambda id_label_dist: (id_label_dist[0][0], (id_label_dist[1] / amounts[id_label_dist[0][1]]))
        ).reduceByKey(min)

    def __sil_cluster(self, b_a_values):
        """
        Sum of sil_metrics for one cluster
        (point.id, b) -> (point.id, (b, a)) -> (b - a) / max(b, a) -> cluster_sum
        """
        return b_a_values[0].join(b_a_values[1]).map(
            lambda id_b_a: (id_b_a[1][0] - id_b_a[1][1]) / max(id_b_a[1][0], id_b_a[1][1])
        ).sum()

    def score(self, full_df, clusters):
        n, global_centroid = full_df.count(), self.__get_centroid(full_df.rdd)
        return 1 - 1 / np.exp(np.exp(self.__bcd(clusters, global_centroid, n) - self.__wcd(clusters)))

    def __bcd(self, clusters, global_centroid, n):
        return sum(map(lambda c: c[2] * self.distance(c[1], global_centroid), clusters)) / (n * len(clusters))

    def __wcd(self, clusters):
        return sum(map(lambda c: self.__dists_to_centroid(c, RDD.mean), clusters))

    def davies_bouldin(self, full_df, clusters):
        s = list(map(lambda c: self.__dists_to_centroid(c, RDD.mean), clusters))
        v_matrix = [[(s[x] + s[y]) / self.distance(clusters[x][1], clusters[y][1]) if x != y
                     else float('-inf') for y in range(len(clusters))] for x in range(len(clusters))]
        return sum(starmap(max, v_matrix)) / len(clusters)

    # need 10% improvement, for now consider only the farthest from centroid point
    def sv(self, full_df, clusters):
        v = sum(map(lambda c: self.__dists_to_centroid(c, RDD.max), clusters))
        s_matrix = [[self.distance(clusters[x][1], clusters[y][1]) if x != y else float('inf')
                     for y in range(len(clusters))] for x in range(len(clusters))]
        return sum(starmap(min, s_matrix)) / v

    def s_dbw(self, full_df, clusters):
        clusters_norm, k = [np.linalg.norm(self.__sqr_sum(c[0], c[1])) for c in clusters], len(clusters)
        stddev, global_x = np.sqrt(sum(clusters_norm)) / k, self.__sqr_sum(full_df.rdd) / full_df.count()
        dens = [self.__den_single(c, stddev) for c in clusters]
        den_matrix = [self.__den_pair(clusters[x], clusters[y], stddev) / max(dens[x], dens[y])
                      if x != y and dens[x] != 0 and dens[y] != 0 else 0.0 for y in range(k) for x in range(k)]
        return sum(clusters_norm) / (k * np.linalg.norm(global_x)) + sum(den_matrix) / (k * (k - 1))

    def __sqr_sum(self, vectors, centroid=None):
        c = centroid if centroid is not None else self.__get_centroid(vectors)
        return vectors.map(lambda x: (x.features - c).values ** 2).reduce(operator.add)

    def __den_single(self, cluster, stddev):
        return self.__s_dbw_f(cluster[0], cluster[1], stddev)

    def __den_pair(self, x, y, stddev):
        pair_centroid = (x[1] + y[1]) / 2
        return self.__s_dbw_f(x[0], pair_centroid, stddev) + self.__s_dbw_f(y[0], pair_centroid, stddev)

    def __s_dbw_f(self, points, centroid, stddev):
        return points.map(lambda p: self.distance(p.features, centroid)).filter(lambda d: d < stddev).count()

    @staticmethod
    def test_run():
        """
        Calculates all available measures on iris dataset twice:
        once for original labels, another for randomly generated.
        The same unit test can be found in MeasureTest.py
        """
        columns = (['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'species')
        csv_path, k, distance = './datasets/normalized/labled_iris.csv', 3, Distance.euclidean

        sc = pyspark.SparkContext.getOrCreate(conf=pyspark.SparkConf().setMaster('local[2]').setAppName('measures'))
        df = pyspark.SQLContext(sc).read.csv(csv_path, header=True, inferSchema=True)
        df = HD.make_id(HD.assemble(df, columns[0])).withColumnRenamed(columns[1], 'labels')
        true_df, random_df = df, df.withColumn('labels', round(rand() * (k - 1)).cast('int'))

        for metric in Measure.functions:
            measure = Measure(metric, distance)
            original, random = measure(true_df), measure(random_df)
            print("--> '%s': original: %f, random: %f" % (metric, original, random))
