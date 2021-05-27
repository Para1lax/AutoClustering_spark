
from pyspark.ml.evaluation import ClusteringEvaluator


# TODO: change when more metrics arrived
# TODO: delete prints when found where use metrics
def metric(data, **kwargs):
    try:
        res = -ClusteringEvaluator(predictionCol='labels', distanceMeasure='squaredEuclidean').evaluate(data)
        return res
    except TypeError:
        print("\n\nTYPE ERROR OCCURED IN Metric.py:\n\nDATA: {}\n\n".format(data))
        return 0
