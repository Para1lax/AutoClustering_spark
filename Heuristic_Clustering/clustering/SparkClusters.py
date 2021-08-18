from pyspark.ml.clustering import KMeans, GaussianMixture, BisectingKMeans


class SparkCluster:
    def __init__(self, model, **config):
        self.config = config
        self.model = model(**config)

    def __call__(self, *args, **kwargs):
        spark_df = args[0]
        estimator = self.model.fit(spark_df)
        predictions = estimator.transform(spark_df)
        return predictions.withColumnRenamed('prediction', 'labels')


class KMeansSpark(SparkCluster):
    name = 'kmeans'
    def __init__(self, **config):
        SparkCluster.__init__(self, KMeans, **config)


class GaussianMixtureSpark(SparkCluster):
    def __init__(self, **config):
        SparkCluster.__init__(self, GaussianMixture, **config)


class BisectingKMeansSpark(SparkCluster):
    def __init__(self, **config):
        SparkCluster.__init__(self, BisectingKMeans, **config)
