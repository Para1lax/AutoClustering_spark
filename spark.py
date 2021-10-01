import logging

import pyspark
import Heuristic_Clustering
from Heuristic_Clustering import HeuristicClustering, Mab, Measure, Clusteriser, HyperOptimiser
import pickle


sc = pyspark.SparkContext(appName='sparkTest', master='local[*]')
sql = pyspark.SQLContext(sc)
df = sql.read.csv('Heuristic_Clustering/datasets/normalized/balance.csv', inferSchema=True)
print(Mab.available, Clusteriser.available, HyperOptimiser.available, sep='\n')
heuristic_executor = HeuristicClustering(df, mab_solver='fair_softmax', hpo='optuna')
result = heuristic_executor(batch_size=3, time_limit=70)
print(result)

