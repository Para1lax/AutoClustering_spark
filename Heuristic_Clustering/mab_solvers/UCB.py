import math

import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

import Constants
import Metric
from mab_solvers.MabSolver import MabSolver
from utils import debugging_printer


class UCB(MabSolver):
    def __init__(self, action, is_fair=False, time_limit=None):
        MabSolver.__init__(self, action, time_limit)
        self.num_algos = Constants.num_algos
        self.rewards = np.zeros(Constants.num_algos)
        # self.spendings = [[] for i in range(0, self.num)]
        # self.avg_spendings = [1] * Constants.num_algos
        self.n = np.array([1] * self.num_algos)
        self.name = "ucb"
        self.iter = 1
        self.is_fair = is_fair

    def initialize(self, log_file, true_labels=None):
        """
        Initialize rewards. We use here the same value,
        gained by calculating metrics on randomly assigned labels.
        """
        print("\nInit UCB1")
        # labels = np.random.randint(0, n_clusters, self.action.data.count())
        # for c in range(0, n_clusters):
        #     labels[c] = c
        # np.random.shuffle(labels)

        # Random initialization of cluster labels
        self.action.data = self.action.data.withColumn('labels', round(rand()*Constants.n_clusters_upper_bound).cast(IntegerType()))

        debugging_printer(place='UCB.py -> initialize', info_name='Data after random initialization',
                          info=self.action.data.head())
        # TODO: rewrite Metric to Spark
        res = Metric.metric(self.action.data)
        debugging_printer(place='UCB.py -> initialize', info_name='Type of result of metric',
                          info=type(res))

        # start = time.time()
        for i in range(0, Constants.num_algos):
            self.rewards[i] = -res  # the smallest value is, the better.
        # self.consume_limit(time.time() - start)
        log_file.write("Init rewards: " + str(self.rewards) + '\n')

    def draw(self):
        values = self.rewards
        if self.is_fair:
            values = values / (self.sum_spendings / self.n)

        values = values + math.sqrt(2 * math.log(self.iter)) / self.n
        return np.argmax(values)

    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.rewards[arm] += reward
        self.n[arm] += 1
