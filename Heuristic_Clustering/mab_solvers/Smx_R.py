import sys

import numpy as np
from numpy.random import choice

import Constants
import Metric
from mab_solvers.MabSolver import MabSolver
from RLthreadBase import ClusteringArmThread
from mab_solvers.Softmax import Softmax


class SoftmaxR(Softmax):
    def __init__(self, action, tau=Constants.tau, is_fair=False, time_limit=None):
        super().__init__(action, tau, is_fair, time_limit)
        self.raw_rewards = []

    def initialize(self, f, true_labels=None):
        super().initialize(f)
        self.raw_rewards = np.array(self.rewards)

    def register_action(self, arm, time_consumed, reward):
        self.n[arm] += 1
        self.raw_rewards[arm] = reward
        self.rewards[arm] = reward
