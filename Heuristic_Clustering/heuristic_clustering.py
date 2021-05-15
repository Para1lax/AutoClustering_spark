# UP TO SPARK

import time
import numpy as np

from Constants import Constants
from RLrfAlgoEx import RLrfAlgoEx
from mab_solvers.UCB_SRSU import UCBsrsu
from utils import debugging_printer, preprocess


# arglabel = None
# if len(argv) == 7:
#     script, argfile, argseed, argmetric, argiter, argbatch, argtl = argv
#     algorithm = Constants.algorithm
# elif len(argv) == 8:
#     script, argfile, argseed, argmetric, argiter, argbatch, argtl, algorithm = argv
# elif len(argv) == 9:
#     script, argfile, argseed, argmetric, arglabel, argiter, argbatch, argtl, algorithm = argv
# else:
#     raise "Invalid error"


# def configure_mab_solver(algorithm, metric, X, seed):
#     """
#     Creates and configures the corresponding MAB-solver.
#     :param algorithm: algorithm to be used.
#     """
#
#     if algorithm.startswith("rl-ei"):
#         algo_e = RLsmacEiAlgoEx(metric, X, seed, batch_size)
#         mab_solver = Softmax(action=algo_e, tau=Constants.tau, is_fair=False, time_limit=time_limit)
#
#         # Advanced MAB:
#     elif algorithm.startswith("rfrsls-uni"):
#         algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
#         mab_solver = Uniform(action=algo_e, time_limit=time_limit)
#     elif algorithm.startswith("rfrsls-smx-R"):
#         algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
#         mab_solver = SoftmaxR(action=algo_e, time_limit=time_limit)
#     elif algorithm.startswith("rfrsls-ucb-SRU"):
#         algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
#         mab_solver = UCBsru(action=algo_e, time_limit=time_limit)
#     elif algorithm.startswith("rfrsls-ucb-SRSU"):
#         algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
#         mab_solver = UCBsrsu(action=algo_e, time_limit=time_limit)
#     elif algorithm.startswith("rfrsls-smx-SRSU"):
#         algo_e = RLrfrsAlgoEx(metric, X, seed, batch_size, expansion=100)
#         mab_solver = SoftmaxSRSU(action=algo_e, time_limit=time_limit)
#
#         # Old MABs still supported
#     else:
#         algo_e = ae.RLsmacAlgoEx(metric, X, seed, batch_size)
#         # choose algo:
#         if algorithm == "rl-ucb-f-smac":
#             mab_solver = UCB(action=algo_e, is_fair=True, time_limit=time_limit)
#         elif algorithm == "rl-ucb1-smac":
#             mab_solver = UCB(action=algo_e, is_fair=False, time_limit=time_limit)
#         elif algorithm.startswith("rl-smx-smac"):
#             mab_solver = Softmax(action=algo_e, tau=Constants.tau, is_fair=False, time_limit=time_limit)
#         elif algorithm == "rl-smx-f-smac":
#             mab_solver = Softmax(action=algo_e, tau=Constants.tau, is_fair=True, time_limit=time_limit)
#         # elif argalgo == "rl-ucb1-rs":
#         elif algorithm.startswith("rl-max-ei"):
#             mab_solver = MaxEi(action=algo_e, optimizers=algo_e.smacs, time_limit=time_limit)
#         else:
#             raise "X3 algo: " + algorithm
#
#     return mab_solver

# checking rfrsls-ucb-SRSU only
def configure_mab_solver(data, seed, metric):
    """
    Creates and configures the corresponding MAB-solver.
    :param algorithm: algorithm to be used.
    """
    # algorithm.startswith("rfrsls-ucb-SRSU"):
    algorithm_executor = RLrfAlgoEx(data=data, metric=metric, seed=seed, batch_size=batch_size, expansion=100)
    mab_solver = UCBsrsu(action=algorithm_executor, time_limit=time_limit)

    return mab_solver


# algorithm = Constants.algorithm
# # algorithm = 'rl-ei'
# batch_size = Constants.batch_size
# # batch_size = 1200
# time_limit = Constants.tuner_timeout
# # time_limit = 1000
# iterations = Constants.bandit_iterations
# # iterations = 5000
def run(spark_df, seed=42, metric='sil', output_file='AutoClustering_output.txt', algorithm=Constants.algorithm,
        batch_size=Constants.batch_size, timeout=Constants.bandit_timeout, iterations=Constants.bandit_iterations,
        max_clusters=Constants.n_clusters_upper_bound, algorithms=Constants.algos):
    true_labels = None

    Constants.algorithm = algorithm
    Constants.num_algos = len(algorithm)
    Constants.batch_size = Constants.batch_size
    Constants.bandit_timeout = timeout
    Constants.bandit_iterations = iterations
    Constants.n_clusters_upper_bound = max_clusters
    Constants.tuner_timeout = timeout * (iterations + 1) / Constants.num_algos

    f = open(file=output_file, mode='a')

    spark_df = preprocess(spark_df)

    # core part:
    # initializing multi-arm bandit solver:
    mab_solver = configure_mab_solver(spark_df, algorithm=algorithm, metric=metric, seed=seed)

    start = time.time()

    # Random initialization:
    mab_solver.initialize(f, true_labels)
    time_init = time.time() - start

    start = time.time()

    # RUN actual Multi-Arm:
    its = mab_solver.iterate(iterations, f)
    time_iterations = time.time() - start

    print("#PROFILE: time spent in initialize: " + str(time_init))
    print("#PROFILE: time spent in iterations:" + str(time_iterations))

    # algorithm_executor
    algorithm_executor = mab_solver.action

    f.write("Metric: " + metric + ' : ' + str(algorithm_executor.best_val) + '\n')
    f.write("Algorithm: " + str(algorithm_executor.best_algo) + '\n')
    f.write("# Target func calls: " + str(its * batch_size) + '\n')
    f.write("# Time init: " + str(time_init) + '\n')
    f.write("# Time spent: " + str(time_iterations) + '\n')
    f.write("# Arms played: " + str(mab_solver.n) + '\n')
    f.write("# Arms algos: " + str(Constants.algos) + '\n')

    try:
        f.write("# Arms avg time: " + str([np.average(plays) for plays in mab_solver.spendings]) + '\n')
    except:
        f.write("# Arms avg time: []")
        pass

    f.write(str(algorithm_executor.best_param) + "\n\n")

    f.write("SMACS: \n")
    if hasattr(algorithm_executor, "smacs"):
        for s in algorithm_executor.smacs:
            try:
                stats = s.get_tae_runner().stats
                t_out = stats._logger.info
                stats._logger.info = lambda x: f.write(x + "\n")
                stats.print_stats()
                stats._logger.info = t_out
            except:
                pass

        f.write("\n")
        for i in range(0, Constants.num_algos):
            s = algorithm_executor.smacs[i]
            _, Y = s.solver.rh2EPM.transform(s.solver.runhistory)
            f.write(Constants.algos[i] + ":\n")
            f.write("Ys:\n")
            for x in Y:
                f.write(str(x[0]))
                f.write("\n")
            f.write("-----\n")

    f.write("###\n")
    f.write("\n\n")

    if algorithm.startswith("rl-max-ei"):
        log = mab_solver.tops_log
    elif algorithm.startswith("rl-ei"):
        log = algorithm_executor.tops_log
    else:
        log = []

    for i in range(0, len(log)):
        f.write(str(i + 1) + ": " + str(log[i]))
        f.write("\n")

    f.flush()

    return algorithm_executor
