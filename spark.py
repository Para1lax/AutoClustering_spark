import pyspark
import Heuristic_Clustering


sc = pyspark.SparkContext(appName='sparkTest', master='local[2]')
sql = pyspark.SQLContext(sc)
df = sql.read.csv('Heuristic_Clustering/datasets/normalized/iris.csv', inferSchema=True)
df = Heuristic_Clustering.make_id(Heuristic_Clustering.assemble(df))
res = Heuristic_Clustering.run(df, output_file='log.txt', time_limit=200, batch_size=10)
print(res)
