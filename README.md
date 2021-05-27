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

#### To run spark in Colab:

```
!apt-get update
!apt-get upgrade
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://www-us.apache.org/dist/spark/spark-2.4.1/spark-2.4.1-bin-hadoop2.7.tgz
!tar xf spark-2.4.1-bin-hadoop2.7.tgz
!pip install -q findspark
```
