import logging
import time
import numpy as np

from abc import abstractmethod
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType as Int


class MabSolver:
    """
    Base class for switching clustering algorithms
    """
    def __init__(self, ds, is_fair, arms):
        """
        Parameters
        ----------
        ds: initialised HeuristicDataset
        is_fair: define solver's policy if he should consider time consuming
        arms: amount of switching algorithms
        """
        self.arms, self.is_fair = arms, is_fair
        self.arms_usage = np.ones(self.arms, np.int)
        self.consumed = np.ones(self.arms, dtype=np.int)

        logging.info('Initialising mab solver')
        self.best_result, self.best_config = float('-inf'), {}
        while self.best_result == float('-inf'):
            random_init = ds.df.withColumn('labels', round(rand() * ds.max_clusters).cast(Int()))
            self.best_result = ds.measure(random_init, minimise=False)
        self.rewards, self.its = np.full(self.arms, self.best_result), 0

    def __call__(self, optimisers, batch_size, time_limit):
        """
        Launches procedure of switching and optimising clustering algorithms configurations
        Parameters
        ----------
        optimisers: list of HyperOptimiser (de-facto, arms)
        batch_size: defines amount of algorithm calculations, when drawing a particular arm
        time_limit: time budget (in seconds) for current run
        Returns
        -------
        Pair of (the best reached measure value, the best configuration)
        """
        while time_limit > 0:
            cur_arm, start = self.draw(), time.time()
            algo = optimisers[cur_arm].algorithm
            logging.info('Calling ' + algo + ' optimiser')
            # optimisers are minimising reward, so need to inverse monotonicity
            arm_reward = -optimisers[cur_arm](time_limit, batch_size)
            arm_time = time.time() - start
            logging.info(str(int(arm_time)) + 's spent for ' + algo + ' optimisation')
            if arm_reward > self.best_result:
                self.best_result, self.best_arm = arm_reward, cur_arm
                self.best_config = dict(**optimisers[cur_arm].get_best_config())
            self.consumed[cur_arm] += arm_time
            self.arms_usage[cur_arm] += 1
            self.its, time_limit = self.its + 1, time_limit - arm_time
            self.update(cur_arm, arm_reward)

    @abstractmethod
    def draw(self):
        pass

    def update(self, arm, reward):
        self.rewards[arm] += reward


class SoftmaxMab(MabSolver):
    def __init__(self, df, is_fair, arms):
        MabSolver.__init__(self, df, is_fair, arms)

    @staticmethod
    def soft_norm(x):
        e_x = np.exp(x - np.amax(x))
        return e_x / e_x.sum()

    def draw(self):
        x = self.rewards / (self.consumed / self.arms_usage) if self.is_fair else self.rewards
        return np.random.choice(np.arange(self.arms), p=self.soft_norm(x))


class UcbMab(MabSolver):
    def __init__(self, df, is_fair, arms):
        MabSolver.__init__(self, df, is_fair, arms)

    def draw(self):
        x = self.rewards / (self.consumed / self.arms_usage) if self.is_fair else self.rewards
        return np.argmax(x + np.sqrt(2 * np.log(self.its) / self.arms_usage))


get_solver = {'softmax': SoftmaxMab, 'fair_softmax': SoftmaxMab, 'ucb': UcbMab, 'fair_ucb': UcbMab}
available = frozenset(get_solver.keys())
