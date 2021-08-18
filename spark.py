import pyspark
import Heuristic_Clustering
from Heuristic_Clustering import HeuristicClustering, HeuristicDataset


sc = pyspark.SparkContext(appName='sparkTest', master='local[*]')
sql = pyspark.SQLContext(sc)
df = sql.read.csv('Heuristic_Clustering/datasets/normalized/iris.csv', inferSchema=True)
HeuristicClustering(HeuristicDataset(df), hpo='optuna')(batch_size=3)
