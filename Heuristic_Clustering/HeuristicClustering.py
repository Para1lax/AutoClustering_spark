import logging

import HyperOptimiser
import Mab
import Classifiers

from HeuristicDataset import HeuristicDataset


class HeuristicClustering:
    def __init__(self, spark_df, algorithms=Classifiers.native, mab_solver='fair_softmax', hpo='smac'):
        assert hpo in HyperOptimiser.available, \
            'Unknown HPO: {}. Available: {}'.format(hpo, HyperOptimiser.available)
        for algorithm in algorithms:
            assert algorithm in Classifiers.available,\
                'Unknown algorithm: {}. Available: {}'.format(algorithm, Classifiers.available)
        assert mab_solver in Mab.available,\
            'Unknown mab_solver: {}. Available: {}'.format(mab_solver, Mab.available)

        logging.basicConfig(level=logging.INFO)
        spark_dataset = HeuristicDataset(spark_df)
        self.hpo = HyperOptimiser.get_optimiser[hpo]
        self.optimisers = [self.hpo(algorithm, spark_dataset) for algorithm in algorithms]

        fair_mab = mab_solver.startswith('fair_')
        self.mab_solver = Mab.get_solver[mab_solver](spark_dataset, fair_mab, len(self.optimisers))

    def __call__(self, batch_size=20, time_limit=1000, **kwargs):
        return self.mab_solver(self.optimisers, batch_size, time_limit)
