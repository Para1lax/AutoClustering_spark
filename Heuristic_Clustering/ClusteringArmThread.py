import threading
import traceback

import numpy as np
import sys
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from pyspark.ml.clustering import KMeans as KMeans_spark
from pyspark.ml.clustering import GaussianMixture as GaussianMixture_spark
from pyspark.ml.clustering import BisectingKMeans as BisectingKMeans_spark

from smac.configspace import ConfigurationSpace
from pyspark.ml.feature import VectorAssembler

from Constants import Constants
import Metric
from utils import debugging_printer


class ClusteringArmThread:

    def __init__(self, data, algorithm_name, metric, seed):
        self.algorithm_name = algorithm_name
        self.metric = metric
        self.data = data
        self.current_labels = None
        self.value = Constants.bad_cluster
        self.parameters = dict()
        self.seed = seed
        self.configuration_space = ConfigurationSpace()

        if algorithm_name == Constants.kmeans_algo:
            self.configuration_space.add_hyperparameters(self.get_kmeans_configspace())

        elif algorithm_name == Constants.gm_algo:
            self.configuration_space.add_hyperparameters(self.get_gaussian_mixture_configspace())

        elif algorithm_name == Constants.bisecting_kmeans:
            self.configuration_space.add_hyperparameters(self.get_bisecting_kmeans_configspace())

    def update_labels(self, configuration):
        if self.algorithm_name == Constants.kmeans_algo:
            algorithm = KMeans_spark(predictionCol='labels', **configuration)
        elif self.algorithm_name == Constants.gm_algo:
            algorithm = GaussianMixture_spark(predictionCol='labels', **configuration)
        elif self.algorithm_name == Constants.bisecting_kmeans:
            algorithm = BisectingKMeans_spark(predictionCol='labels', **configuration)

        model = algorithm.fit(self.data)

        # if Constants.DEBUG:
        #     model.fit(self.data)
        # else:
        #     # Some problems with smac, old realization, don't change
        #     try:
        #         model.fit(self.data)
        #     except:
        #         try:
        #             exc_info = sys.exc_info()
        #             try:
        #                 model.fit(self.data)  # try again
        #             except:
        #                 pass
        #         finally:
        #             print("Error occured while fitting " + self.algorithm_name)
        #             print("Error occured while fitting " + self.algorithm_name, file=sys.stderr)
        #             traceback.print_exception(*exc_info)
        #             del exc_info
        #             return Constants.bad_cluster

        if self.algorithm_name in Constants.rewrited:
            predictions = model.transform(self.data)
            self.current_labels = predictions
        else:
            self.current_labels = model.labels_

    def clu_run(self, cfg):
        self.update_labels(cfg)
        return Metric.metric(self.current_labels)

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
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distanceMeasure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        minDivisibleClusterSize = UniformFloatHyperparameter("minDivisibleClusterSize", 0.01, 1.0)
        return k, maxIter, distanceMeasure, minDivisibleClusterSize
