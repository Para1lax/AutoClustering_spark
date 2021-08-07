from pyspark.ml.clustering import KMeans, GaussianMixture, BisectingKMeans
from Constants import Constants


class SparkCluster:
    models = {
        Constants.kmeans_algo: KMeans,
        Constants.gm_algo: GaussianMixture,
        Constants.bisecting_kmeans: BisectingKMeans
    }

    def __init__(self, clustering_algo, **configuration):
        self.model = SparkCluster.models[clustering_algo](**configuration)

    def __call__(self, *args, **kwargs):
        spark_df = args[0]
        estimator = self.model.fit(spark_df)
        predictions = estimator.transform(spark_df)
        return predictions.withColumnRenamed('prediction', 'labels')
