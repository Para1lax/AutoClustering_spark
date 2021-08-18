import pandas as pd

from pyspark.sql.functions import monotonically_increasing_id as unique_id
import pyspark.sql.functions as F
from scipy.stats import tmean, tvar, skew, kurtosis
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.types import *

from Metric import *


class HeuristicDataset:
    default_spark = pyspark.SparkConf().setMaster('[*]')
    eq_types = {'datetime64[ns]': TimestampType, 'int64': LongType, 'int32': IntegerType, 'float64': FloatType}

    def __init__(self, df, features=None, measure=None, max_clusters=15):
        if isinstance(df, pd.DataFrame):
            sc = pyspark.SparkContext.getOrCreate(self.default_spark)
            df = self.pandas_to_spark(sc, df)
        elif not isinstance(df, pyspark.sql.DataFrame):
            raise ValueError('Expected pandas_df or spark_df')

        self.df, self.max_clusters = self.make_id(self.vector_norm(self.assemble(df, features))), max_clusters
        self.dims, self.n = self.get_df_dimensions(self.df), self.df.count()
        self.measure = measure if measure is not None else self.get_measure(df, columns=features)

    @staticmethod
    def pandas_to_spark(spark_context, pandas_df):
        sql = pyspark.sql.SQLContext(spark_context)
        columns = list(pandas_df.columns)
        types = list(pandas_df.dtypes)
        struct_list = []
        for column, t in zip(columns, types):
            typo = StringType() if t not in HeuristicDataset.eq_types else HeuristicDataset.eq_types[t]()
            struct_list.append(StructField(column, typo))
        p_schema = StructType(struct_list)
        return sql.createDataFrame(pandas_df, p_schema)

    @staticmethod
    def get_measure(df, columns=None):
        # !!! cvi_predictor should be here
        # columns = columns if columns is not None else df.columns
        # norm = HeuristicDataset.sparse_norm(df, columns)
        # meta_dims = [[norm.select(sql_func(col)).collect()[0][0] for col in columns]
        #              for sql_func in [F.mean, F.variance, F.skewness, F.kurtosis]]
        # meta_combined = [[meta_func(meta_dim) for meta_dim in meta_dims]
        #                  for meta_func in [tmean, tvar, skew, kurtosis]]
        return Measure(Measure.SIL, Distance.euclidean)

    @staticmethod
    def assemble(spark_df, columns=None):
        drops = spark_df.columns if columns is None else columns
        return VectorAssembler(inputCols=drops, outputCol='features').transform(spark_df).drop(*drops)

    @staticmethod
    def make_id(spark_df):
        return spark_df if 'id' in spark_df.columns else spark_df.withColumn('id', unique_id())

    @staticmethod
    def get_df_dimensions(spark_df):
        return len(spark_df.first().features) if 'features' in spark_df.columns \
            else len(spark_df.drop('id', 'features', 'labels').columns)

    @staticmethod
    def get_unique_labels(labeled_df, exclude_none=True):
        if 'labels' not in labeled_df.columns:
            raise ValueError("Dataframe does not contain column 'labels'\n. Columns are: {}".format(labeled_df.columns))
        labels = list(map(lambda sample: sample.labels, labeled_df.drop_duplicates(['labels']).collect()))
        if None in labels and exclude_none:
            labels.remove(None)
        return labels

    @staticmethod
    def vector_norm(vectorised_df):
        normalised = MinMaxScaler(inputCol='features', outputCol='__f__').fit(vectorised_df).transform(vectorised_df)
        return normalised.drop('features').withColumnRenamed('__f__', 'features')

    @staticmethod
    def sparse_norm(df, columns=None):
        columns, normalised = columns if columns is not None else df.columns, df
        vector = HeuristicDataset.vector_norm(HeuristicDataset.assemble(df, columns))
        return vector.rdd.map(lambda row: [*row.features.values.tolist()]).toDF(columns)
