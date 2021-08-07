from pyspark.sql.dataframe import DataFrame

from ClusteringArmThread import ClusteringArmThread
from RFRS import RFRS
from smac.scenario.scenario import Scenario


class RLthreadRFRS(ClusteringArmThread):
    def __init__(self, spark_df, algorithm_name, metric, batch_size, expansion=5000, params=None):
        self.run_count = batch_size
        ClusteringArmThread.__init__(self, spark_df, algorithm_name, metric, params=params)
        self.new_scenario(1)  # initial scenario
        self.optimizer = RFRS(self.clu_scenario, self.clu_run, algorithm_name,
                              expansion_number=expansion, batch_size=batch_size)

    def new_scenario(self, c, remaining_time=None):
        # remaining_time is usually expected to be way more than needed for one call,
        # but sometimes it's guarding from hanging arbitraty long in single iteration
        self.clu_scenario = Scenario({
            "run_obj": "quality", "cs": self.configuration_space,
            "deterministic": "true", "runcount-limit": self.run_count * c
        }) if remaining_time is None else Scenario({
            "run_obj": "quality", "cs": self.configuration_space,
            "deterministic": "true", "tuner-timeout": remaining_time,
            "wallclock_limit": remaining_time, "cutoff_time": remaining_time,
            "runcount-limit": self.run_count * c}
        )

    def run(self):
        self.parameters = self.optimizer.optimize()
        self.value = self.optimizer.get_runhistory().get_cost(self.parameters)
