import math
import numpy as np

from abc import abstractmethod

from graphframes import *
from itertools import combinations
from collections import defaultdict

from pyspark.ml.linalg import DenseVector
from pyspark.sql import Row

from Measure import Distance


class Clusteriser:
    checkpoint_dir = './checkpoint'

    @abstractmethod
    def __call__(self, spark_ds):
        pass


class SparkCluster(Clusteriser):
    def __init__(self, model, **config):
        self.config = config
        self.model = model(**config)

    def __call__(self, spark_ds):
        estimator = self.model.fit(spark_ds.df)
        predictions = estimator.transform(spark_ds.df)
        return predictions.withColumnRenamed('prediction', 'labels')


class KMeansSpark(SparkCluster):
    def __init__(self, **config):
        from pyspark.ml.clustering import KMeans
        SparkCluster.__init__(self, KMeans, **config)


class GaussianMixtureSpark(SparkCluster):
    def __init__(self, **config):
        from pyspark.ml.clustering import GaussianMixture
        SparkCluster.__init__(self, GaussianMixture, **config)


class BisectingKMeansSpark(SparkCluster):
    def __init__(self, **config):
        from pyspark.ml.clustering import BisectingKMeans
        SparkCluster.__init__(self, BisectingKMeans, **config)


class DBCSAN(Clusteriser):
    def __init__(self, **configuration):
        self.eps = configuration['eps']
        self.min_pts = configuration['min_pts']
        self.pivot, dist = None, configuration['distance']
        self.distance = dist if callable(dist) else getattr(Distance, dist)

    def __distance_from_pivot(self, x):
        pivot_dist = self.distance(x.features, self.pivot)
        partition_index = math.floor(pivot_dist / self.eps)
        rows = [Row(id=x.id, features=x.features, pivot_dist=pivot_dist)]
        return [(partition_index, rows), (partition_index + 1, rows)]

    def __scan(self, x):
        # x = (partition_index, data_points)
        # out = {point.id: set(eps_neighbours)}
        out, partition_data = defaultdict(set), x[1]
        for i in range(len(partition_data)):
            for j in range(i + 1, len(partition_data)):
                if self.distance(partition_data[i].features, partition_data[j].features) < self.eps:
                    # both i and j are within epsilon distance to each other
                    out[partition_data[i].id].add(partition_data[j].id)
                    out[partition_data[j].id].add(partition_data[i].id)
        # returns point and its neighbours as tuple
        return [Row(item[0], item[1]) for item in out.items()]

    def __label(self, x):
        if len(x[1]) + 1 < self.min_pts:
            return []
        # use id as cluster label
        cluster_label = x[0]
        # return True for a core point, False for a base point
        out = [(cluster_label, [(cluster_label, True)])]
        for idx in x[1]:
            out.append((idx, [(cluster_label, False)]))
        return out

    @staticmethod
    def __combine_labels(x):
        # x = (point.id, list((cluster, core_point.label))
        point, cluster_labels = x[0], x[1]
        core_point, clusters = False, []
        for (label, point_type) in cluster_labels:
            if point_type is True:
                core_point = True
            clusters.append(label)
        # if core point keep all cluster otherwise only one
        return point, clusters if core_point is True else [clusters[0]], core_point

    def __call__(self, spark_ds):
        self.pivot = DenseVector(np.zeros(spark_ds.dims))
        combine_cluster_rdd = spark_ds.df.rdd. \
            flatMap(self.__distance_from_pivot).reduceByKey(lambda x, y: x + y). \
            flatMap(self.__scan).reduceByKey(lambda x, y: x.union(y)). \
            flatMap(self.__label).reduceByKey(lambda x, y: x + y). \
            map(self.__combine_labels).cache()
        id_cluster_rdd = combine_cluster_rdd. \
            map(lambda x: Row(point=x[0], labels=x[1][0], core_point=x[2]))
        id_cluster_df = id_cluster_rdd.toDF()
        vertices = combine_cluster_rdd.flatMap(
            lambda x: [Row(id=item) for item in x[1]]
        ).toDF().distinct()
        edges = combine_cluster_rdd.flatMap(
            lambda x: [Row(src=item[0], dst=item[1]) for item in combinations(x[1], 2)]
        ).toDF().distinct()
        id_cluster_df.rdd.context.setCheckpointDir(Clusteriser.checkpoint_dir)
        g = GraphFrame(vertices, edges)
        connected_df = g.connectedComponents()
        id_cluster_df = id_cluster_df.join(
            connected_df, connected_df.id == id_cluster_df.labels
        ).select("point", "component").withColumnRenamed('component', 'labels')
        return id_cluster_df.join(spark_ds.df, spark_ds.df.id == id_cluster_df.point, 'right').drop('point')


get_clusteriser = {
    'kmeans': KMeansSpark, 'gaussian_mixture': GaussianMixtureSpark,
    'bisecting_kmeans': BisectingKMeansSpark, 'dbscan': DBCSAN
}

available = frozenset(get_clusteriser.keys())
native = frozenset(['kmeans', 'gaussian_mixture', 'bisecting_kmeans'])
