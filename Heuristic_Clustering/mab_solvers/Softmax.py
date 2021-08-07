import numpy as np
from numpy.random import choice

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

from Metric import Measure
from mab_solvers.MabSolver import MabSolver
from Constants import Constants


class Softmax(MabSolver):
    def __init__(self, action, params, is_fair=True):
        MabSolver.__init__(self, action, params)
        self.num_algos = params.num_algos
        self.rewards = np.zeros(params.num_algos)
        self.n = np.ones(self.num_algos, dtype=np.int)
        self.tau = params.tau
        self.is_fair = is_fair

    def initialize(self, log_file):
        """
        Initialize rewards. We use here the same value,
        gained by calculating metrics on randomly assigned labels.
        """
        print("\nInit Softmax with tau = %d" % self.tau)
        init_measure = float('-inf')
        while init_measure == float('-inf'):
            self.action.spark_df = self.action.spark_df.withColumn(
                'labels', round(rand() * self.params.n_clusters_upper_bound).cast(IntegerType())
            )
            # spark_measure = metric(self.action.spark_df)
            init_measure = Measure(Measure.CH, 'manhattan')(self.action.spark_df)
        # start = time.time()
        for i in range(self.params.num_algos):
            self.rewards[i] = -init_measure  # the smallest value is, the better.
        # self.consume_limit(time.time() - start)
        log_file.write("Init rewards: " + str(self.rewards) + '\n')

    def draw(self):
        if not self.is_fair:
            s_max = self.softmax_normalize(self.rewards)
        else:
            x = np.array(self.rewards)
            x = x / (self.sum_spendings / self.n)
            s_max = self.softmax_normalize(x)
        try:
            d = choice(np.arange(self.num_algos), p=s_max)
        except ValueError:
            d = choice(np.arange(self.num_algos))
        return d

    def register_action(self, arm, time_consumed, reward):
        self.rewards[arm] += reward
        self.n[arm] += 1

    @staticmethod
    def softmax_normalize(rewards):
        x, norm = rewards, np.linalg.norm(rewards)
        x = x / norm if norm == 0.0 else x
        e_x = np.exp(x - np.amax(x))
        s_max = e_x / e_x.sum(axis=0)
        return s_max
