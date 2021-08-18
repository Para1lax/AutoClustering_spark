from abc import abstractmethod
from smac.facade.func_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory

from ConfigSpace import ConfigurationSpace
import ConfigSpace
import optuna
import Classifiers


class HyperOptimiser:
    INT, FLOAT, CATEGORY = 'int', 'float', 'categorical'

    def __init__(self, algorithm, ds):
        self.algorithm, self.ds = algorithm, ds
        self.classifier = Classifiers.get_classifier[algorithm]
        self.get_config = self.__getattribute__(algorithm + '_config')

    def dbscan_config(self):
        return dict(
            eps=(HyperOptimiser.FLOAT, (0.1, 0.7)),
            min_pts=(HyperOptimiser.INT, (2, 6)),
            distance=(HyperOptimiser.CATEGORY, (['euclidean', 'manhattan'],)),
        )

    def kmeans_config(self):
        return dict(
            k=(HyperOptimiser.INT, (2, self.ds.max_clusters)),
            initMode=(HyperOptimiser.CATEGORY, (['random', 'k-means||'],)),
            initSteps=(HyperOptimiser.INT, (1, 5)),
            maxIter=(HyperOptimiser.INT, (5, 50)),
            distanceMeasure=(HyperOptimiser.CATEGORY, (['euclidean', 'cosine'],))
        )

    def gaussian_mixture_config(self):
        return dict(
            k=(HyperOptimiser.INT, (2, self.ds.max_clusters)),
            maxIter=(HyperOptimiser.INT, (5, 50)),
            tol=(HyperOptimiser.FLOAT, (1e-6, 0.1)),
            aggregationDepth=(HyperOptimiser.INT, (2, 15))
        )

    def bisecting_kmeans_config(self):
        return dict(
            k=(HyperOptimiser.INT, (2, self.ds.max_clusters)),
            maxIter=(HyperOptimiser.INT, (5, 50)),
            distanceMeasure=(HyperOptimiser.CATEGORY, (['euclidean', 'cosine'],)),
            minDivisibleClusterSize=(HyperOptimiser.FLOAT, (0.01, 1.0))
        )

    @abstractmethod
    def __call__(self, time_limit, batch_size):
        pass


class SmacOptimiser(HyperOptimiser):
    hyper_map = {
        HyperOptimiser.INT: ConfigSpace.UniformIntegerHyperparameter,
        HyperOptimiser.FLOAT: ConfigSpace.UniformFloatHyperparameter,
        HyperOptimiser.CATEGORY: ConfigSpace.CategoricalHyperparameter,
    }

    def __init__(self, algorithm, ds):
        HyperOptimiser.__init__(self, algorithm, ds)
        self.cs, self.best_val = ConfigurationSpace(), float('inf')
        for name, space in self.get_config().items():
            param = SmacOptimiser.hyper_map[space[0]]
            self.cs.add_hyperparameter(param(name, *space[1]))
        self.run_history, self.best_config = RunHistory(), self.cs.get_default_configuration()

    def opt_function(self, config):
        # !!!!!!!!!!!!!!!!!!  MIN-MAX
        predictions = self.classifier(**config)(self.ds)
        return self.ds.measure(predictions)

    def __call__(self, time_limit, batch_size):
        scenario = Scenario({
            'run_obj': 'quality', 'cs': self.cs, 'deterministic': 'true', 'runcount-limit': batch_size,
            'algo_runs_timelimit': time_limit, 'initial_incumbent': self.best_config
        })
        opt = SMAC4HPO(scenario=scenario, runhistory=self.run_history, tae_runner=self.opt_function)
        incumbent = opt.optimize()
        opt_val = opt.runhistory.get_cost(incumbent)
        if opt_val <= self.best_val:
            self.best_config = incumbent
            self.best_val = opt_val
        self.run_history = opt.runhistory
        return self.best_val


class OptunaOptimiser(HyperOptimiser):
    __hyper_map__ = {
        HyperOptimiser.INT: optuna.Trial.suggest_int,
        HyperOptimiser.FLOAT: optuna.Trial.suggest_float,
        HyperOptimiser.CATEGORY: optuna.Trial.suggest_categorical
    }

    def __init__(self, algorithm, ds):
        HyperOptimiser.__init__(self, algorithm, ds)
        self.session = optuna.create_study()
        self.cs = self.get_config()

    def objective(self, trial):
        config = dict()
        for name, space in self.cs.items():
            suggest = self.__hyper_map__[space[0]]
            config[name] = suggest(trial, name, *space[1])
        predictions = self.classifier(**config)(self.ds)
        return self.ds.measure(predictions)

    def __call__(self, time_limit, batch_size):
        self.session.optimize(self.objective, n_trials=batch_size, timeout=time_limit)
        return self.session.best_value


get_optimiser = {'smac': SmacOptimiser, 'optuna': OptunaOptimiser}
available = frozenset(get_optimiser.keys())
