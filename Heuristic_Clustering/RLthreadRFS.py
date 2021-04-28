from RFRS import RFRS
from smac.scenario.scenario import Scenario

from RLthreadBase import ClusteringArmThread


class RLthreadRFRS(ClusteringArmThread):

    def __init__(self, name, metric, X, seed, batch_size, expansion=5000):
        self.run_count = batch_size
        ClusteringArmThread.__init__(self, name, metric, X, seed)  # populates config space
        self.new_scenario(1)  # initial scenario

        self.optimizer = RFRS(scenario=self.clu_scenario,
                         tae_runner=self.clu_run,
                         expansion_number=expansion,
                         batch_size=batch_size)

    def new_scenario(self, c, remaining_time=None):
        # remaining_time is usually expected to be way more than needed for one call,
        # but sometimes it's guarding from hanging arbitraty long in single iteration
        if remaining_time is None:
            self.clu_scenario = Scenario({"run_obj": "quality",
                                          "cs": self.clu_cs,
                                          "deterministic": "true",
                                          "runcount-limit": self.run_count * c
                                          })
        else:
            self.clu_scenario = Scenario({"run_obj": "quality",
                                          "cs": self.clu_cs,
                                          "deterministic": "true",
                                          "tuner-timeout": remaining_time,
                                          "wallclock_limit": remaining_time,
                                          "cutoff_time": remaining_time,
                                          "runcount-limit": self.run_count * c
                                          })

    def run(self):
        print('Run RFRS ' + self.thread_name)

        self.parameters = self.optimizer.optimize()
        self.value = self.optimizer.get_runhistory().get_cost(self.parameters)
