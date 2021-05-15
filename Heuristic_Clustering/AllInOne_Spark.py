from sklearn import datasets
import numpy as np
import sys
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import Metric
import Constants

n_samples = 500

# 3 easily observed clusters
# X, y = datasets.make_blobs(n_samples=n_samples, n_features=4, centers=3, cluster_std=1, center_box=(-10.0, 10.0),
#                            shuffle=True)

max_eval = 10
metric = Constants.silhouette_metric


def run(data, model):
    model.fit(data)
    predictions = model.transform(data)
    return -ClusteringEvaluator(predictionCol='prediction', distanceMeasure='squaredEuclidean').evaluate(data)


def km_run(cfg):
    cl = KMeans(**cfg)
    return run(cl)


# Build Configuration Space which defines all parameters and their ranges
def get_km_scenario():
    km_cs = ConfigurationSpace()
    algorithm = CategoricalHyperparameter("algorithm", ["auto", "full", "elkan"])
    tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2)
    n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
    n_init = UniformIntegerHyperparameter("n_init", 2, 15)
    max_iter = UniformIntegerHyperparameter("max_iter", 50, 1500)
    verbose = UniformIntegerHyperparameter("verbose", 0, 10)
    km_cs.add_hyperparameters([n_clusters, n_init, max_iter, tol, verbose, algorithm])

    km_scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                            "runcount-limit": max_eval,  # maximum function evaluations
                            "cs": km_cs,  # configuration space
                            "deterministic": "true"
                            })
    return km_scenario


def get_ms_scenario():
    ms_cs = ConfigurationSpace()
    quantile = UniformFloatHyperparameter("quantile", 0.0, 1.0)
    bin_seeding = UniformIntegerHyperparameter("bin_seeding", 0, 1)
    min_bin_freq = UniformIntegerHyperparameter("min_bin_freq", 1, 100)
    cluster_all = UniformIntegerHyperparameter("cluster_all", 0, 1)
    ms_cs.add_hyperparameters([quantile, bin_seeding, min_bin_freq, cluster_all])

    ms_scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                            "runcount-limit": max_eval,  # maximum function evaluations
                            "cs": ms_cs,  # configuration space
                            "deterministic": "true"
                            })
    return ms_scenario


def get_aff_scenario():
    aff_cs = ConfigurationSpace()
    damping = UniformFloatHyperparameter("damping", 0.5, 1.0)
    max_iter = UniformIntegerHyperparameter("max_iter", 100, 1000)
    convergence_iter = UniformIntegerHyperparameter("convergence_iter", 5, 20)
    aff_cs.add_hyperparameters([damping, max_iter, convergence_iter])

    aff_scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": max_eval,  # maximum function evaluations
                             "cs": aff_cs,  # configuration space
                             "deterministic": "true"
                             })
    return aff_scenario


def get_w_scenario():
    w_cs = ConfigurationSpace()
    affinity = CategoricalHyperparameter("affinity", ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
    linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average"])
    n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 15)
    w_cs.add_hyperparameters([n_clusters, affinity, linkage])

    w_scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                           "runcount-limit": max_eval,  # maximum function evaluations
                           "cs": w_cs,  # configuration space
                           "deterministic": "true"
                           })
    return w_scenario


def get_db_scenario():
    db_cs = ConfigurationSpace()
    algorithm = CategoricalHyperparameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    eps = UniformFloatHyperparameter("eps", 0.1, 0.9)
    min_samples = UniformIntegerHyperparameter("min_samples", 2, 10)
    leaf_size = UniformIntegerHyperparameter("leaf_size", 5, 100)
    db_cs.add_hyperparameters([eps, min_samples, algorithm, leaf_size])

    db_scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                            "runcount-limit": max_eval,  # maximum function evaluations
                            "cs": db_cs,  # configuration space
                            "deterministic": "true"
                            })
    return db_scenario


def launch(X):
    best_val = sys.float_info.max
    best_algo = "-1"
    best_params = dict()
    achieved_values = dict()

    i = 0
    algos = {Constants.dbscan_algo: 0, Constants.kmeans_algo: 0, Constants.affinity_algo: 0,
             Constants.mean_shift_algo: 0, Constants.ward_algo: 0}
    metrics = [Constants.silhouette_metric, Constants.dunn_metric]
    saved_parameters = [""] * len(metrics)
    num_parameters_for_algo = {Constants.kmeans_algo: [], Constants.affinity_algo: [], Constants.mean_shift_algo: [],
                               Constants.ward_algo: [], Constants.dbscan_algo: []}

    for algo in algos.keys():
        value = 1
        parameters = ""
        print('\n\n=============================\n\n{}\n\n=============================\n\n'.format(algo))
        if Constants.kmeans_algo in algo:
            smac = SMAC(scenario=get_km_scenario(), rng=np.random.RandomState(42), tae_runner=km_run)
            parameters = smac.optimize()
            value = km_run(parameters)
            achieved_values[Constants.kmeans_algo] = value
        elif Constants.affinity_algo in algo:
            smac = SMAC(scenario=get_aff_scenario(), rng=np.random.RandomState(42), tae_runner=aff_run)
            parameters = smac.optimize()
            value = aff_run(parameters)
            achieved_values[Constants.affinity_algo] = value
        elif Constants.mean_shift_algo in algo:
            smac = SMAC(scenario=get_ms_scenario(), rng=np.random.RandomState(42), tae_runner=ms_run)
            parameters = smac.optimize()
            value = ms_run(parameters)
            achieved_values[Constants.mean_shift_algo] = value
        elif Constants.ward_algo in algo:
            smac = SMAC(scenario=get_w_scenario(), rng=np.random.RandomState(42), tae_runner=w_run)
            parameters = smac.optimize()
            value = w_run(parameters)
            achieved_values[Constants.ward_algo] = value
        elif Constants.dbscan_algo in algo:
            smac = SMAC(scenario=get_db_scenario(), rng=np.random.RandomState(42), tae_runner=db_run)
            parameters = smac.optimize()
            value = db_run(parameters)
            achieved_values[Constants.dbscan_algo] = value
        print(('For algo ' + algo + ' lowest function value found: %f' % value))
        print(('Parameter setting %s' % parameters))
        if value < best_val:
            best_val = value
            best_algo = algo
            best_params = parameters
    algos[best_algo] += 1
    saved_parameters[i] = best_params
    num_parameters_for_algo[best_algo].append(i)
    i += 1

    chosen_algo = ""
    num_cases = 0
    for algo in algos.keys():
        if algos[algo] > num_cases:
            num_cases = algos[algo]
            chosen_algo = algo

    best_params = saved_parameters[num_parameters_for_algo[chosen_algo][0]]
    cl = ""
    if Constants.kmeans_algo in chosen_algo:
        cl = KMeans(**best_params)
    elif Constants.affinity_algo in chosen_algo:
        cl = AffinityPropagation(**best_params)
    elif Constants.mean_shift_algo in chosen_algo:
        bandwidth = estimate_bandwidth(X, quantile=best_params["quantile"])
        cl = MeanShift(bandwidth=bandwidth, bin_seeding=bool(best_params["bin_seeding"]),
                       min_bin_freq=best_params["min_bin_freq"],
                       cluster_all=bool(best_params["cluster_all"]))
    elif Constants.ward_algo in chosen_algo:
        cl = AgglomerativeClustering(**best_params)
    elif Constants.dbscan_algo in chosen_algo:
        cl = DBSCAN(**best_params)

    cl.fit(X)

    print('Best algorithm = ' + chosen_algo)
    print('Best parameters = ' + str(best_params))
    print('All achieved results = ' + str(achieved_values))
    print(str(algos))
    return cl
