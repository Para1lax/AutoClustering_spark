import json
import logging

import pyspark
import Heuristic_Clustering
from Heuristic_Clustering import HeuristicClustering, Mab, Measure, Clusteriser, HyperOptimiser
import pickle


# read dataset
sc = pyspark.SparkContext(appName='heuristic', master='local[*]')
sql = pyspark.SQLContext(sc)
df = sql.read.csv('Heuristic_Clustering/datasets/normalized/balance.csv', inferSchema=True)

# show string names of available components
print(Mab.available, Clusteriser.available, HyperOptimiser.available, sep='\n')

# create instance of executor with appropriate components
heuristic_executor = HeuristicClustering(df, algorithms=['kmeans', 'bisecting_kmeans', 'gaussian_mixture'],
                                         mab_solver='fair_softmax', hpo='optuna')

# can run multiple times with different budget, return best result among all runs
result1 = heuristic_executor(batch_size=5, time_limit=100)
result2 = heuristic_executor(batch_size=3, time_limit=70)

with open('best_dump.json', 'w') as best:
    json.dump(result2, best, indent=2)
with open('full_dump.json', 'w') as full:
    # accesses full logs since the moment executor instance in created
    json.dump(heuristic_executor.logs, full, indent=2)
