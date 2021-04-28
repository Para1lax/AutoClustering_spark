import time

import numpy as np

#TODO: Use spark RF regressor
from sklearn.ensemble import RandomForestRegressor

import Constants
from RLthreadRFRS import RLthreadRFRS


class RLrfrsAlgoEx:
    clu_algos = Constants.algos

    def __init__(self, data, metric='sil', seed=42, batch_size=Constants.batch_size, expansion=5000):
        self.metric = metric
        self.data = data
        self.run_num = np.array([0] * Constants.num_algos)
        self.best_val = Constants.best_init
        self.best_param = dict()
        self.best_algo = ""
        self.seed = seed
        self.batch_size = batch_size
        self.optimizers = []
        self.clustering_threads = []

        # create all clustering threads in advance:
        for i in range(0, Constants.num_algos):
            self.clustering_threads.append(
                RLthreadRFRS(self.clu_algos[i], self.metric, self.data, self.seed, self.batch_size, expansion=expansion))
            self.optimizers.append(self.clustering_threads[i].optimizer)

        self.rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    def apply(self, arm, log_file, iteration_number, remaining_time=None, current_time=0):
        clustering_thread = self.clustering_threads[arm]

        # initially, run_num for each arm == 0, thus we allocate 1 batch of target f calls:
        clustering_thread.new_scenario(self.run_num[arm] + 1, remaining_time)  # add budget

        run_start = time.time()
        clustering_thread.run()
        run_spent = int(time.time()-run_start)

        self.run_num[arm] += 1
        reward = clustering_thread.value

        if reward < self.best_val:
            self.best_val = reward
            self.best_param = clustering_thread.parameters
            self.best_algo = clustering_thread.thread_name
        log_string = str(iteration_number) \
                     + ', ' + self.metric \
                     + ', ' + str(self.best_val) \
                     + ', ' + str(self.best_algo) \
                     + ', ' + str(self.clu_algos[arm]) \
                     + ', ' + str(reward) \
                     + ', ' + str(current_time + run_spent)

        file.write(log_string + '\n')
        file.flush()

        # best value in random forest if the smallest one. Algo Executor provides the REWARD.
        # The smaller value is, the better reward should be.
        return -1.0 * clustering_thread.optimizer.get_best_from_forest()
