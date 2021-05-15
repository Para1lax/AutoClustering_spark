# AutoClustering_spark
This is new repository devoted to the development of AutoClustering framework in Apache Spark

**Realized algorithms:** KMeans, BisectingKMeans, Gaussian mixture

**Realized metrics:** sil _(waiting for more)_

#### To run:

```
from heuristic_clustering import run

# data - your data in pyspark dataframe
# metric - one of realized metrics
# log_file - path to file where you want to see logs
result = run(data, seed, metric, log_file)
```

