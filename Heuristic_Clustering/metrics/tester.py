import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext, SQLContext
import pyspark
from spark_custom import *
import pandas as pd
import datetime
import ch_index as ch_index_new


def check_measures(old_m, new_m, pandas_X, pandas_y,
                   pyspark_X, pyspark_y, name):
    f_new = new_m.find(pyspark_X, pyspark_y, 9)
    print(f_new)
    f_old = old_m.find(pandas_X, pandas_y, 9)
    print(f_old)
    if abs(f_old - f_new) < 1e-6:
        print(f"|{name}| find work correct!\n")
    else:
        print(f"Error on |{name}| find!!\n")


def d(acc, row):
    acc.add(np.sum(np.array(row).astype(float)))


if __name__ == '__main__':
    data_path = "../datasets/exp/ok/wholesale_customers.csv"
    number_cores = 4
    memory_gb = 6
    conf = (
        pyspark.SparkConf()
            .setMaster('local[{}]'.format(number_cores))
            .set('spark.driver.memory', '{}g'.format(memory_gb))
    )
    # spark = SparkSession \
    #     .builder \
    #     .appName("DBSCAN") \
    #     .config('spark.driver.host', '192.168.0.9', conf=conf) \
    #     .getOrCreate()
    sc = SparkContext(conf=conf)
    sql_sc = SQLContext(sc)
    data_pandas_cv = pd.read_csv(data_path).head(999)
    print(data_pandas_cv.head(5))
    data_pandas_cv["use"] = [[1, 2, 3]] * len(data_pandas_cv)
    data_pandas_cv['predict'] = (data_pandas_cv['Delicassen'] // 1000).apply(lambda x: min(8, x))
    print(set(data_pandas_cv['predict']))
    data_pandas_cv = data_pandas_cv.drop(['Delicassen'], axis=1)
    df = sql_sc.createDataFrame(data_pandas_cv)
    acc = sc.accumulator(0)
    labels = data_pandas_cv['predict'].to_numpy()

    new_labels = np.copy(labels)
    id = [0, 2, 8, 25, 26, 28]
    for i in id:
        new_labels[i] = 3


    data_pandas_cv['predict'] = new_labels
    new_df = sql_sc.createDataFrame(data_pandas_cv)

    start_t = datetime.datetime.now()
    b = ch_index_new.ChIndex()
    res = b.find(df, sc)
    res2 = b.update(new_df, sc, 9, 2, 3, id)
    print(f"Spark custom time {datetime.datetime.now() - start_t}")
    print(res)
    print(res2)
