import numpy as np

from mab_solvers.Softmax import Softmax
from mab_solvers.UCB import UCB

s_norm = Softmax.softmax_normalize


class UCBsrsu(UCB):
    def __init__(self, action, is_fair=False, time_limit=None):
        super().__init__(action, is_fair, time_limit)
        self.raw_rewards = []

    def initialize(self, f, true_labels=None):
        super().initialize(f)
        self.raw_rewards = np.array(self.rewards)

    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.n[arm] += 1
        self.raw_rewards[arm] = reward
        # TODO : DELETE PRINT
        print("==========================\n \
               ==========================> UCB_SRSU -> register_action <==========================\n \
               ==========================\n \
               \n \
               reward:   {}\n \
               raw_rewards:   {}\n \
               \n \
               ==========================".format(reward, self.raw_rewards))
        self.rewards = s_norm(self.raw_rewards) + s_norm(self.u_correction(self.sum_spendings))

