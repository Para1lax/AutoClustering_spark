from Constants import Constants


class Parameters:
    def __init__(self, algorithms=Constants.algos, n_clusters_upper_bound=15,
                 bandit_timeout=30, bandit_iterations=40, batch_size=40):
        self.algorithms = algorithms
        self.num_algos = len(self.algorithms)
        self.n_clusters_upper_bound = n_clusters_upper_bound
        # seconds for each bandit iteration
        self.bandit_timeout = bandit_timeout
        # iterations number
        self.bandit_iterations = bandit_iterations
        self.batch_size = batch_size
        self.tuner_timeout = bandit_timeout * (bandit_iterations + 1) / self.num_algos
