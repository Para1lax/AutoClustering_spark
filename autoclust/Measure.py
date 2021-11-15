import importlib
from abc import abstractmethod

import numpy as np
import operator

from pyspark import RDD
from itertools import starmap

from Distance import Distance
from AutoDataset import AutoDataset as HD


class Measure:
    """
    Represents base class for clustering evaluators.
    """
    SIL, CH, SCORE, DB, S_DBW, OS = 'Silhouette', 'CH', 'Score', 'DB', 'SDbw', 'OSIndex'
    DUNN, G31, G33, G41, G43, G51, G53 = 'Dunn', 'G31', 'G33', 'G41', 'G43', 'G51', 'G53'

    functions = frozenset([SIL, CH, SCORE, DB, S_DBW, OS, DUNN, G31, G33, G41, G43, G51, G53])
    increasing = frozenset([SIL, CH, SCORE, DUNN, G31, G33, G41, G43, G51, G53])
    decreasing = frozenset([DB, S_DBW, OS])

    def __init__(self, distance):
        self.algorithm = type(self).__name__
        if self.algorithm not in self.functions:
            raise ValueError('Unknown Measure: {}'.format(self.algorithm))
        self.distance = distance if callable(distance) else getattr(Distance, distance)
        self.should_decrease = self.algorithm in self.decreasing

    def __call__(self, df, minimise=None):
        """
        The method is entry point to estimate dataframe clustering.
        Parameters
        ----------
        df: spark dataframe
        minimise: bool, specifies expected monotonicity of result.
        If measure's monotonicity does not match desired, changes the sign of a result
        Default is None, which means return result as is
        Returns
        -------
        Float value of dataframe clustering according to instantiated Measure subclass
        """
        labels = HD.get_unique_labels(df)
        if len(labels) < 2:
            return float('inf') if minimise is not None or minimise else float('-inf')
        clusters = [df.rdd.filter(lambda x: x.labels == label).cache() for label in labels]
        centroids, amounts = list(map(self.get_centroid, clusters)), list(map(lambda c: c.count(), clusters))
        result = self.exec(df, list(zip(clusters, centroids, amounts, labels)))
        if minimise is None:
            return result
        sign = -1 if minimise ^ self.should_decrease else 1
        return sign * result

    @staticmethod
    def make(algorithm, **kwargs):
        """
        Fabric method to instantiate measure by string name.
        Should use Measure fields, e.g. Measure.SIL, Measure.OS, ...
        Parameters
        ----------
        algorithm: One of the Measure static fields, string name of desired algorithm
        kwargs: additional arguments, forwards them into constructor of subclass

        Returns
        -------
        Instance of Measure subclass
        """
        module = importlib.import_module('Measure')
        return getattr(module, algorithm)(**kwargs)

    @staticmethod
    def get_centroid(spark_rdd):
        return spark_rdd.map(lambda x: x.features).reduce(operator.add) / spark_rdd.count()

    @staticmethod
    def dists_to_centroid(cluster, p_dist, reducer):
        centroid = cluster[1]  # extract centroid from tuple, so Spark can serialize vector
        return reducer(cluster[0].map(lambda x: p_dist(x.features, centroid)))

    @abstractmethod
    def exec(self, full_df, clusters):
        pass


class Silhouette(Measure):
    def __init__(self, distance):
        super(Silhouette, self).__init__(distance)

    def exec(self, full_df, clusters):
        amount_by_label = {c[3]: c[2] for c in clusters}
        sil_values = [(self._b(c, full_df, amount_by_label),
                       self._a(c)) for c in clusters]
        return sum(map(self._cluster, sil_values)) / full_df.count()

    def _a(self, cluster):
        """
        Average distances between points in similar cluster
        point -> (point, point) -> (point.id, dists) -> (point.id, sum_dist) -> (point.id, avg_dist)
        """
        # cluster_size = cluster[2] - 1  will match with Spark ClusteringEvaluator
        cluster_size = cluster[2]  # Serialize size
        return cluster[0].cartesian(cluster[0]).map(
            lambda pp: (pp[0].id, self.distance(pp[0].features, pp[1].features))
        ).reduceByKey(operator.add).mapValues(lambda sum_dist: sum_dist / cluster_size)

    def _b(self, cluster, full_df, amounts):
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

    @staticmethod
    def _cluster(b_a_values):
        """
        Sum of sil_metrics for one cluster
        (point.id, b) -> (point.id, (b, a)) -> (b - a) / max(b, a) -> cluster_sum
        """
        return b_a_values[0].join(b_a_values[1]).map(
            lambda id_b_a: (id_b_a[1][0] - id_b_a[1][1]) / max(id_b_a[1][0], id_b_a[1][1])
        ).sum()


class DunnMetrics(Measure):
    """
    Base class for different variations of Dunn metrics.
    Contains functions, which will be combined by subclasses
    """
    def __init__(self, distance, cl_dist, cl_diam):
        super(DunnMetrics, self).__init__(distance)
        self.cl_dist, self.cl_diam = cl_dist, cl_diam

    def exec(self, full_df, clusters):
        dists = [self.cl_dist(self.distance, clusters[x], clusters[y])
                 for x in range(1, len(clusters)) for y in range(x)]
        return min(dists) / max(map(lambda c: self.cl_diam(self.distance, c), clusters))

    @staticmethod
    def cl_diam_1(p_dist, cluster):
        return cluster[0].cartesian(cluster[0]).map(lambda pp: p_dist(pp[0].features, pp[1].features)).max()

    @staticmethod
    def cl_diam_3(p_dist, cluster):
        return 2 * Measure.dists_to_centroid(cluster, p_dist, RDD.sum) * cluster[2]

    @staticmethod
    def cl_dist_1(p_dist, x, y):
        return x[0].cartesian(y[0]).map(lambda pp: p_dist(pp[0].features, pp[1].features)).min()

    @staticmethod
    def cl_dist_3(p_dist, x, y):
        mul = 0.5 / x[2] * y[2]  # overall sum doubles, so 0.5 instead of 1
        return x[0].cartesian(y[0]).map(lambda pp: p_dist(pp[0].features, pp[1].features)).sum() * mul

    @staticmethod
    def cl_dist_4(p_dist, x, y):
        return p_dist(x[1], y[1])

    @staticmethod
    def cl_dist_5(p_dist, x, y):
        x_sum, y_sum = Measure.dists_to_centroid(x, p_dist, RDD.sum), Measure.dists_to_centroid(y, p_dist, RDD.sum)
        return (x_sum + y_sum) / (x[2] + y[2])


class Dunn(DunnMetrics):
    def __init__(self, distance):
        super(Dunn, self).__init__(distance, DunnMetrics.cl_dist_1, DunnMetrics.cl_diam_1)


class G31(DunnMetrics):
    def __init__(self, distance):
        super(G31, self).__init__(distance, DunnMetrics.cl_dist_3, DunnMetrics.cl_diam_1)


class G41(DunnMetrics):
    def __init__(self, distance):
        super(G41, self).__init__(distance, DunnMetrics.cl_dist_4, DunnMetrics.cl_diam_1)


class G51(DunnMetrics):
    def __init__(self, distance):
        super(G51, self).__init__(distance, DunnMetrics.cl_dist_5, DunnMetrics.cl_diam_1)


class G33(DunnMetrics):
    def __init__(self, distance):
        super(G33, self).__init__(distance, DunnMetrics.cl_dist_3, DunnMetrics.cl_diam_3)


class G43(DunnMetrics):
    def __init__(self, distance):
        super(G43, self).__init__(distance, DunnMetrics.cl_dist_4, DunnMetrics.cl_diam_3)


class G53(DunnMetrics):
    def __init__(self, distance):
        super(G53, self).__init__(distance, DunnMetrics.cl_dist_5, DunnMetrics.cl_diam_3)


class CH(Measure):
    def __init__(self, distance=Distance.euclidean):
        super(CH, self).__init__(distance)

    def exec(self, full_df, clusters):
        global_centroid = self.get_centroid(full_df.rdd)
        global_sum = sum(map(lambda c: c[2] * self.distance(c[1], global_centroid), clusters))
        clusters_sum = sum(map(lambda c: self.dists_to_centroid(c, self.distance, RDD.sum), clusters))
        mul = (full_df.count() - len(clusters)) / (len(clusters) - 1)
        return mul * global_sum / clusters_sum


class Score(Measure):
    def __init__(self, distance=Distance.euclidean):
        super(Score, self).__init__(distance)

    def exec(self, full_df, clusters):
        n, global_centroid = full_df.count(), self.get_centroid(full_df.rdd)
        wcd = sum(map(lambda c: self.dists_to_centroid(c, self.distance, RDD.mean), clusters))
        bcd = sum(map(lambda c: c[2] * self.distance(c[1], global_centroid), clusters)) / (n * len(clusters))
        return 1 - 1 / np.exp(np.exp(bcd - wcd))


class DB(Measure):
    def __init__(self, distance=Distance.euclidean):
        super(DB, self).__init__(distance)

    def exec(self, full_df, clusters):
        s = list(map(lambda c: self.dists_to_centroid(c, self.distance, RDD.mean), clusters))
        v_matrix = [[(s[x] + s[y]) / self.distance(clusters[x][1], clusters[y][1]) if x != y
                     else float('-inf') for y in range(len(clusters))] for x in range(len(clusters))]
        return sum(starmap(max, v_matrix)) / len(clusters)


class SDbw(Measure):
    def __init__(self, distance=Distance.euclidean):
        super(SDbw, self).__init__(distance)

    def exec(self, full_df, clusters):
        clusters_norm, k = [np.linalg.norm(self._sqr_sum(c[0], c[1])) for c in clusters], len(clusters)
        stddev, global_x = np.sqrt(sum(clusters_norm)) / k, self._sqr_sum(full_df.rdd) / full_df.count()
        dens = [self._den_single(c, stddev) for c in clusters]
        den_matrix = [self._den_pair(clusters[x], clusters[y], stddev) / max(dens[x], dens[y])
                      if x != y and dens[x] != 0 and dens[y] != 0 else 0.0 for y in range(k) for x in range(k)]
        return sum(clusters_norm) / (k * np.linalg.norm(global_x)) + sum(den_matrix) / (k * (k - 1))

    def _sqr_sum(self, vectors, centroid=None):
        c = centroid if centroid is not None else self.get_centroid(vectors)
        return vectors.map(lambda x: (x.features - c).values ** 2).reduce(operator.add)

    def _den_single(self, cluster, stddev):
        return self._s_dbw_f(cluster[0], cluster[1], stddev)

    def _den_pair(self, x, y, stddev):
        pair_centroid = (x[1] + y[1]) / 2
        return self._s_dbw_f(x[0], pair_centroid, stddev) + self._s_dbw_f(y[0], pair_centroid, stddev)

    def _s_dbw_f(self, points, centroid, stddev):
        return points.map(lambda p: self.distance(p.features, centroid)).filter(lambda d: d < stddev).count()


class OSIndex(Measure):
    def __init__(self, distance=Distance.euclidean, t=0.4):
        super(OSIndex, self).__init__(distance)
        self.t = t

    def exec(self, full_df, clusters):
        centroid_label = [(c[1], c[3]) for c in clusters]
        o_val = full_df.rdd.map(lambda p: self._ox_val(p, centroid_label)).sum()
        s_matrix = [[self.distance(clusters[x][1], clusters[y][1]) if x != y else float('inf')
                     for y in range(len(clusters))] for x in range(len(clusters))]
        return o_val / sum(starmap(min, s_matrix))

    def _ox_val(self, p, cent_label):
        p_cluster = next(filter(lambda c: c[1] == p.labels, cent_label))
        a, p_result = self.distance(p_cluster[0], p.features), []
        for c in cent_label:
            if c[1] == p.labels:
                continue
            b = self.distance(c[0], p.features)
            ox_cluster = a / b if (b - a) / (b + a) < self.t else 0.0
            if ox_cluster > 0.1:
                p_result.append(ox_cluster)
        return sum(p_result) / len(p_result) if len(p_result) != 0 else 0.0
