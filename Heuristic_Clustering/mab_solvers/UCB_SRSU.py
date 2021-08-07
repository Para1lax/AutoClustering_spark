import numpy as np

from mab_solvers.Softmax import Softmax
from mab_solvers.UCB import UCB

s_norm = Softmax.softmax_normalize


class UCBsrsu(UCB):
    def __init__(self, action, is_fair=False, params=None):
        super().__init__(action, is_fair, params)
        self.raw_rewards = []

    def initialize(self, f, true_labels=None):
        super().initialize(f)
        self.raw_rewards = np.array(self.rewards)

    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.n[arm] += 1
        self.raw_rewards[arm] = reward
        self.rewards = s_norm(self.raw_rewards) + s_norm(self.u_correction(self.sum_spendings, self.params.num_algos))

