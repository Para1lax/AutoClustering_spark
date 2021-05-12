import time
from os import walk
from sys import argv

import numpy as np
import pandas as pd

import AlgorithmExecutor as ae
import Constants
import Metric
from mab_solvers import Softmax as sf

script, argfile, argseed = argv


def run(data, logfile_name='clustering_log.txt', metric=Constants.silhouette_metric, seed=42):
    run_start = time.time()

    # May be delete?
    Metric.global_trace = {}  # clear all trace from previous dataset

    # Logging everything here
    f = open(logfile_name, 'w', 1)

    # MOST IMPORTANT PART
    algo_e = ae.AlgorithmExecutor(Constants.num_algos, metric, X, seed)
    soft_max = sf.Softmax(algo_e, Constants.num_algos, Constants.tau)

    start = time.time()
    soft_max.initialize(f)
    print("#PROFILE: time spent in initialize()" + str(time.time() - start))

    soft_max.iterate(Constants.bandit_iterations, f)

    f.write("Metric: " + metric + ' : ' + str(algo_e.best_val) + '\n')
    f.write("Algorithm: " + str(algo_e.best_algo) + '\n')
    f.write(str(algo_e.best_param) + '\n\n')

    f.close()

    print("#PROFILE: TOTAL time consumed by run: " + str(time.time() - run_start))

    if len(Metric.global_trace) != 0:
        s = 0.0
        for i in range(0, len(Metric.global_trace[metric])):
            s = s + Metric.global_trace[metric][i]
        print("#PROFILE: time spent in calculating metrics (" + str(len(Metric.global_trace[metric]))
              + " calls) " + str(metric) + ": " + str(s))

        print("#PROFILE: average metrics call consumes " + str(s / len(Metric.global_trace[metric])))
        print("Metrics " + str(metric) + " calls: " + Metric.global_trace[metric])
    else:
        print("#PROFILE: no metrics calculation outside SMAC runs found")



if argfile == "all":
    for (dirpath, dirnames, files) in walk(Constants.experiment_path):
        for ii in range(0, len(files)):
            file = files[ii]
            run(file)
else:
    run(argfile)
