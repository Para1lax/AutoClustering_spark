import logging

import HyperOptimiser
import Mab
import Clusteriser

from HeuristicDataset import HeuristicDataset


class HeuristicClustering:
    """
    The entry point of the framework. Should be configured only once, then can be launched multiple times
    """
    def __init__(self, spark_df, algorithms=Clusteriser.native,
                 mab_solver='fair_softmax', hpo='optuna'):
        """
        Framework configuration
        Parameters
        ----------
        spark_df: source spark or pandas dataframe
        algorithms: list<string> of clustering algorithms to use.
         Options are available by Clusteriser.available.
         Default is ['kmeans', 'gaussian_mixture', 'bisecting_kmeans']
        mab_solver: <string> algorithm used to switch clustering algorithms.
         Options are available by Mab.available
         Default is 'fair_softmax'
        hpo: <string> hyper parameter optimiser,
         searches for best solution for each clustering algorithm individually.
         Options are available by HyperOptimeser.available
         Default is 'optuna'
        """
        assert hpo in HyperOptimiser.available, \
            'Unknown HPO: {}. Available: {}'.format(hpo, HyperOptimiser.available)
        for algorithm in algorithms:
            assert algorithm in Clusteriser.available,\
                'Unknown algorithm: {}. Available: {}'.format(algorithm, Clusteriser.available)
        assert mab_solver in Mab.available,\
            'Unknown mab_solver: {}. Available: {}'.format(mab_solver, Mab.available)

        logging.basicConfig(level=logging.INFO)
        spark_dataset = HeuristicDataset(spark_df)
        self.hpo = HyperOptimiser.get_optimiser[hpo]
        self.optimisers = [self.hpo(algorithm, spark_dataset) for algorithm in algorithms]

        fair_mab = mab_solver.startswith('fair_')
        self.mab_solver = Mab.get_solver[mab_solver](spark_dataset, fair_mab, len(self.optimisers))

    def __call__(self, batch_size=20, time_limit=1000):
        """
        Actual launch of execution for a given seconds
        Parameters
        ----------
        batch_size: amount of different configurations, which will be applied to clustering algorithm,
         selected by mab solver. Default is 20
        time_limit: time budget (in seconds) for current run. Default is 1000

        Returns
        -------
        Pair of (the best reached measure value, the best configuration)
        """
        self.mab_solver(self.optimisers, batch_size, time_limit)
        best_opt = self.optimisers[self.mab_solver.best_arm]
        logging.info('Collecting results')
        return dict(
            algorithm=best_opt.algorithm,
            metric=best_opt.ds.measure.algorithm,
            result=self.mab_solver.best_result,
            config=self.mab_solver.best_config,
            labels=best_opt.get_labels_by_config(self.mab_solver.best_config)
        )
