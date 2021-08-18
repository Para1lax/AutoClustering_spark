from ConfigSpace.hyperparameters import *
from smac.configspace import ConfigurationSpace

from clustering.DBSCAN import DBCSAN
from clustering.SparkClusters import SparkCluster

from Constants import Constants
from utils import get_df_dimensions
from Metric import Measure, Distance


class ClusteringArmThread:
    def __init__(self, spark_df, algorithm_name, metric, params):
        self.algorithm_name = algorithm_name
        self.metric = metric
        self.spark_df = spark_df
        self.predictions = None
        self.n_clusters_upper_bound = params.n_clusters_upper_bound
        self.value = Constants.bad_cluster
        self.parameters = dict()
        self.configuration_space = ConfigurationSpace()

        if algorithm_name == Constants.kmeans_algo:
            algo_config = self.get_kmeans_configspace()
        elif algorithm_name == Constants.gm_algo:
            algo_config = self.get_gaussian_mixture_configspace()
        elif algorithm_name == Constants.bisecting_kmeans:
            algo_config = self.get_bisecting_kmeans_configspace()
        elif algorithm_name == Constants.dbscan_algo:
            algo_config = self.get_dbscan_configspace()
        else:
            raise ValueError('No such clustering algorithm: %s' % algorithm_name)

        self.configuration_space.add_hyperparameters(algo_config)

    def update_labels(self, configuration):
        if self.algorithm_name in SparkCluster.models:
            self.predictions = SparkCluster(self.algorithm_name, **configuration)(self.spark_df)
        elif self.algorithm_name == Constants.dbscan_algo:
            self.predictions = DBCSAN(**configuration)(self.spark_df)
        else:
            raise ValueError("Unknown clustering algorithm: %s" % self.algorithm_name)

    def clu_run(self, cfg):
        self.update_labels(cfg)
        return Measure(Measure.CH, Distance.manhattan)(self.predictions)

    def get_kmeans_configspace(self):
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
        k = UniformIntegerHyperparameter("k", 2, self.n_clusters_upper_bound)
        init_mode = CategoricalHyperparameter("initMode", ['random', 'k-means||'])
        init_steps = UniformIntegerHyperparameter("initSteps", 1, 5)
        max_iter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distance_measure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        return k, init_mode, init_steps, max_iter, distance_measure

    def get_gaussian_mixture_configspace(self):
        """
        k : number of clusters
        aggregationDepth : suggested depth for treeAggregate (>= 2)
        maxIter : max number of iterations (>= 0)
        tol : the convergence tolerance for iterative algorithms (>= 0)

        Returns
        -------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, self.n_clusters_upper_bound)
        max_iter = UniformIntegerHyperparameter("maxIter", 5, 50)
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        aggregation_depth = UniformIntegerHyperparameter("aggregationDepth", 2, 15)
        return k, max_iter, tol, aggregation_depth

    def get_bisecting_kmeans_configspace(self):
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
        k = UniformIntegerHyperparameter("k", 2, self.n_clusters_upper_bound)
        max_iter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distance_measure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        min_divisible_cluster_size = UniformFloatHyperparameter("minDivisibleClusterSize", 0.01, 1.0)
        return k, max_iter, distance_measure, min_divisible_cluster_size

    def get_dbscan_configspace(self):
        eps = UniformFloatHyperparameter("eps", lower=0.1, upper=0.7)
        min_pts = UniformIntegerHyperparameter("min_pts", lower=2, upper=6)
        distance = CategoricalHyperparameter("distance", ["euclidean", "manhattan"])
        dims = Constant('dims', get_df_dimensions(self.spark_df))
        return eps, min_pts, distance, dims

