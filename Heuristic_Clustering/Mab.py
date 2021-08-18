import time
import numpy as np

from abc import abstractmethod
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType as Int


class MabSolver:
    def __init__(self, ds, is_fair, optimisers):
        self.optimisers, self.arms = optimisers, len(optimisers)
        self.is_fair, self.arms_usage = is_fair, np.ones(self.arms, np.int)
        self.consumed = np.ones(self.arms, dtype=np.int)

        init_measure = float('-inf')
        while init_measure == float('-inf'):
            random_init = ds.df.withColumn('labels', round(rand() * ds.max_clusters).cast(Int()))
            init_measure = ds.measure(random_init)
        self.rewards = np.full(self.arms, init_measure)

    def __call__(self, batch_size, time_limit, *args, **kwargs):
        self.its, remain = 0, time_limit
        while remain > 0:
            cur_arm, start = self.draw(), time.time()
            arm_reward = self.optimisers[cur_arm](remain, batch_size)
            arm_time = time.time() - start
            self.consumed[cur_arm] += arm_time
            self.arms_usage[cur_arm] += 1
            self.its, remain = self.its + 1, remain - arm_time
            self.update(cur_arm, arm_reward)

    @abstractmethod
    def draw(self):
        pass

    def update(self, arm, reward):
        self.rewards[arm] += reward


class SoftmaxMab(MabSolver):
    def __init__(self, df, is_fair, optimisers):
        MabSolver.__init__(self, df, is_fair, optimisers)

    @staticmethod
    def soft_norm(x):
        e_x = np.exp(x - np.amax(x))
        return e_x / e_x.sum()

    def draw(self):
        x = self.rewards / (self.consumed / self.arms_usage) if self.is_fair else self.rewards
        return np.random.choice(np.arange(self.arms), p=self.soft_norm(x))


class UcbMab(MabSolver):
    def __init__(self, df, is_fair, optimisers):
        MabSolver.__init__(self, df, is_fair, optimisers)

    def draw(self):
        x = self.rewards / (self.consumed / self.arms_usage) if self.is_fair else self.rewards
        return np.argmax(x + np.sqrt(2 * np.log(self.its) / self.arms_usage))


get_solver = {'softmax': SoftmaxMab, 'fair_softmax': SoftmaxMab, 'ucb': UcbMab, 'fair_ucb': UcbMab}
available = frozenset(get_solver.keys())
