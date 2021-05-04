import threading
import traceback

import numpy as np
import sys
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
# from sklearn.cluster import KMeans
from pyspark.ml.clustering import KMeans as KMeans_spark
from pyspark.ml.clustering import GaussianMixture as GaussianMixture_spark
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from smac.configspace import ConfigurationSpace
from pyspark.ml.feature import VectorAssembler

import Constants
import Metric


class ClusteringArmThread:

    def __init__(self, data, algorithm_name, metric, seed):
        self.algorithm_name = algorithm_name
        self.metric = metric
        if algorithm_name in Constants.rewrited:
            self.data = data
            vector_assembler = VectorAssembler(inputCols=self.data.columns,
                                               outputCol="features")
            self.data = vector_assembler.transform(self.data)
        else:
            self.data = data.toPandas().values
        self.value = Constants.bad_cluster
        self.parameters = dict()
        self.seed = seed
        self.configuration_space = ConfigurationSpace()

        if algorithm_name == Constants.kmeans_algo:
            self.configuration_space.add_hyperparameters(self.get_kmeans_configspace())

        # elif algorithm_name == Constants.affinity_algo:
        #     damping = UniformFloatHyperparameter("damping", 0.5, 1.0)
        #     max_iter = UniformIntegerHyperparameter("max_iter", 100, 1000)
        #     convergence_iter = UniformIntegerHyperparameter("convergence_iter", 5, 20)
        #     self.configuration_space.add_hyperparameters([damping, max_iter, convergence_iter])
        #
        # elif algorithm_name == Constants.mean_shift_algo:
        #     quantile = UniformFloatHyperparameter("quantile", 0.0, 1.0)
        #     bin_seeding = UniformIntegerHyperparameter("bin_seeding", 0, 1)
        #     min_bin_freq = UniformIntegerHyperparameter("min_bin_freq", 1, 100)
        #     cluster_all = UniformIntegerHyperparameter("cluster_all", 0, 1)
        #     self.configuration_space.add_hyperparameters([quantile, bin_seeding, min_bin_freq, cluster_all])
        #
        # elif algorithm_name == Constants.ward_algo:
        #     linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average"])
        #     affinity_all = CategoricalHyperparameter("affinity_a",
        #                                              ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
        #     affinity_ward = CategoricalHyperparameter("affinity_w", ["euclidean"])
        #     n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
        #     self.configuration_space.add_hyperparameters([n_clusters, affinity_all, affinity_ward, linkage])
        #     self.configuration_space.add_condition(InCondition(child=affinity_ward, parent=linkage, values=["ward"]))
        #     self.configuration_space.add_condition(
        #         InCondition(child=affinity_all, parent=linkage, values=["ward", "complete", "average"]))
        #
        # elif algorithm_name == Constants.dbscan_algo:
        #     algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        #     eps = UniformFloatHyperparameter("eps", 0.1, 0.9)
        #     min_samples = UniformIntegerHyperparameter("min_samples", 2, 10)
        #     leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 100)
        #     self.configuration_space.add_hyperparameters([eps, min_samples, algorithm, leaf_size])

        elif algorithm_name == Constants.gm_algo:
            # cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])
            # tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
            # reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)
            # n_com = UniformIntegerHyperparameter("n_components", 2, 15)
            # max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
            self.configuration_space.add_hyperparameters(self.get_gaussian_mixture_configspace())

        elif algorithm_name == Constants.bisecting_kmeans:
            self.configuration_space.add_hyperparameters(self.get_bisecting_kmeans_configspace())

        # elif algorithm_name == Constants.bgm_algo:
        #     cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])
        #     tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        #     reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)
        #     wcp = UniformFloatHyperparameter("weight_concentration_prior", 1e-10, 0.1)
        #     mpp = UniformFloatHyperparameter("mean_precision_prior", 1e-10, 0.1)
        #     n_com = UniformIntegerHyperparameter("n_components", 2, 15)
        #     max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
        #     self.configuration_space.add_hyperparameters([wcp, mpp, cov_t, tol, reg_c, n_com, max_iter])

    def cluster(self, configuration):
        model = None
        if self.algorithm_name == Constants.kmeans_algo:
            model = KMeans_spark(**configuration)
        # elif self.algorithm_name == Constants.affinity_algo:
        #     model = AffinityPropagation(**configuration)
        # elif self.algorithm_name == Constants.mean_shift_algo:
        #     bandwidth = estimate_bandwidth(self.data, quantile=configuration['quantile'])
        #     model = MeanShift(bandwidth=bandwidth, bin_seeding=bool(configuration['bin_seeding']),
        #                       min_bin_freq=configuration['min_bin_freq'],
        #                       cluster_all=bool(configuration['cluster_all']))
        # elif self.algorithm_name == Constants.ward_algo:
        #     linkage = configuration["linkage"]
        #     aff = ""
        #     if "ward" in linkage:
        #         aff = configuration["affinity_w"]
        #     else:
        #         aff = configuration["affinity_a"]
        #     n_c = configuration["n_clusters"]
        #     model = AgglomerativeClustering(n_clusters=n_c, linkage=linkage, affinity=aff)
        # elif self.algorithm_name == Constants.dbscan_algo:
        #     model = DBSCAN(**configuration)
        elif self.algorithm_name == Constants.gm_algo:
            model = GaussianMixture(predictionCol='labels', **configuration)
        # elif self.algorithm_name == Constants.bgm_algo:
        #     model = BayesianGaussianMixture(**configuration)

        # Some problems with smac, old realization, don't change
        try:
            model.fit(self.data)
        except:
            try:
                exc_info = sys.exc_info()
                try:
                    model.fit(self.data)  # try again
                except:
                    pass
            finally:
                print("Error occured while fitting " + self.algorithm_name)
                print("Error occured while fitting " + self.algorithm_name, file=sys.stderr)
                traceback.print_exception(*exc_info)
                del exc_info
                return Constants.bad_cluster

        if self.algorithm_name in Constants.rewrited:
            predictions = model.transform(self.data)
            # TODO: change to spark
            labels = predictions.select('labels').toPandas()['labels']
        # elif (self.algorithm_name == Constants.gm_algo) or (self.algorithm_name == Constants.bgm_algo):
        #     labels = model.predict(self.data)
        else:
            labels = model.labels_

        return labels

    def clu_run(self, cfg):
        labels = self.cluster(cfg)
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        value = Metric.metric(self.data, n_clusters, labels, self.metric)

        return value

    @staticmethod
    def get_kmeans_configspace():
        """
        k : number of clusters
        initMode : The initialization algorithm. This can be either "random" to choose random points as initial cluster
                   centers, or "k-means||" to use a parallel variant of k-means++
        initSteps : The number of steps for k-means|| initialization mode. Must be > 0
        maxIter : max number of iterations (>= 0)
        seed : random seed
        distanceMeasure : Supported options: 'euclidean' and 'cosine'.

        Returns
        -----------------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, Constants.n_clusters_upper_bound)
        initMode = CategoricalHyperparameter("initMode", ['random', 'k-means||'])
        initSteps = UniformIntegerHyperparameter("initSteps", 1, 5)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distanceMeasure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        return k, initMode, initSteps, maxIter, distanceMeasure

    @staticmethod
    def get_gaussian_mixture_configspace():
        """
        k : number of clusters
        aggregationDepth : suggested depth for treeAggregate (>= 2)
        maxIter : max number of iterations (>= 0)
        tol : the convergence tolerance for iterative algorithms (>= 0)

        Returns
        -------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, Constants.n_clusters_upper_bound)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        return k, maxIter, tol

    @staticmethod
    def get_bisecting_kmeans_configspace():
        """
        k : number of clusters
        initSteps : The number of steps for k-means|| initialization mode. Must be > 0
        maxIter : max number of iterations (>= 0)
        seed : random seed
        distanceMeasure : Supported options: 'euclidean' and 'cosine'.
        minDivisibleClusterSize : The minimum number of points (if >= 1.0) or the minimum proportion of points
                                                               (if < 1.0) of a divisible cluster.
                                                               we use only proportion
        Returns
        -----------------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, Constants.n_clusters_upper_bound)
        initSteps = UniformIntegerHyperparameter("initSteps", 1, 5)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distanceMeasure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        minDivisibleClusterSize = UniformFloatHyperparameter("minDivisibleClusterSize", 0.01, 1.0)
        return k, initSteps, maxIter, distanceMeasure, minDivisibleClusterSize