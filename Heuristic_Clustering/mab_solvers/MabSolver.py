import abc
import math
import time

import numpy as np


class TL:
    def __init__(self, time_limit):
        self.time_remaining = time_limit

    def consume_limit(self, t):
        self.time_remaining -= t

    def is_limit_exceeded(self):
        if self.time_remaining is None:
            return False
        return self.time_remaining <= 0


class MabSolver(TL):
    def __init__(self, action, params=None):
        TL.__init__(self, params.tuner_timeout)
        self.params = params
        self.sum_spendings = np.ones(self.params.num_algos, dtype=np.int)
        self.spendings = [[] for _ in range(self.params.num_algos)]
        self.action = action
        self.time_limit, self.its = params.tuner_timeout, 0

    @abc.abstractmethod
    def initialize(self, log_file):
        return 0

    @abc.abstractmethod
    def draw(self):
        return 0

    @abc.abstractmethod
    def register_action(self, arm, time_consumed, reward):
        """

        This method is for calculating reward.
        :param arm: the arm, which was called
        :param time_consum
        ed: time consumed by that arm to run
        :param reward: reward gained by this call
        """
        return 0

    def iteration(self, f, current_time=0):
        # choosing arm
        cur_arm = self.draw()
        start = time.time()
        # CALL ARM here:
        # the last arm call will be cut off if time limit exceeded.
        reward = self.action.apply(cur_arm, f, self.its, self.time_remaining, current_time)
        consumed = time.time() - start
        self.consume_limit(consumed)
        # Time spent on each algo
        self.sum_spendings[cur_arm] += consumed
        # all spendings
        self.spendings[cur_arm].append(consumed)
        self.register_action(cur_arm, consumed, reward)
        self.its += 1
        return reward

    def iterate(self, log_file):
        start = time.time()
        self.its = 0
        while not self.is_limit_exceeded():
            self.iteration(log_file, int(time.time() - start))

    @staticmethod
    def u_correction(sum_spendings, num_algos):
        sp = np.add(sum_spendings, 1)
        T = np.sum(np.log(sp))
        numerator = math.sqrt(2 * math.log(num_algos + T))
        denom = np.sqrt(1 + np.log(sp))
        return numerator / denom
