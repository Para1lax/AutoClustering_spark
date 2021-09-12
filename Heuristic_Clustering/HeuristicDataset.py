import os
import pickle

import numpy as np
import pandas as pd
import logging

import pyspark
from pyspark.sql.functions import monotonically_increasing_id as unique_id, udf
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector, VectorUDT

from Distance import Distance


class HeuristicDataset:
    default_spark = pyspark.SparkConf().setMaster('[*]')
    cvi_fitness, cvi_path = 'meta_fitness.csv', 'cvi_predictor.pkl'
    eq_types = {'datetime64[ns]': TimestampType, 'int64': LongType, 'int32': IntegerType, 'float64': FloatType}

    def __init__(self, df, features=None, measure=None, max_clusters=15):
        if isinstance(df, pd.DataFrame):
            sc = pyspark.SparkContext.getOrCreate(self.default_spark)
            df = self.pandas_to_spark(sc, df)
        elif not isinstance(df, pyspark.sql.DataFrame):
            raise ValueError('Expected pandas_df or spark_df')

        logging.info('Preprocessing dataframe')
        self.df, self.max_clusters = self.make_id(self.vector_norm(self.assemble(df, features))), max_clusters
        self.df = self.df.withColumn('features', self.to_dense(self.df.features))
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
        logging.info('Measure recommendation in process')
        columns = columns if columns is not None else df.columns
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + '/' + HeuristicDataset.cvi_path, 'rb') as f:
            cvi_classifier = pickle.load(f)
            meta_features = HeuristicDataset.get_meta_features(df, columns)
            measure_id = cvi_classifier.predict(np.array([meta_features]))[0]
            from Measure import Measure
            measure_name = Measure.functions[measure_id]
            logging.info('Measure ' + measure_name + ' is selected')
            return Measure(measure_name, Distance.euclidean)

    @staticmethod
    def __columns_stat(norm, columns):
        import pyspark.sql.functions as F
        from scipy.stats import tmean, tvar, skew, kurtosis
        meta_dims = [[norm.select(sql_func(col)).collect()[0][0] for col in columns]
                     for sql_func in [F.mean, F.variance, F.skewness, F.kurtosis]]
        meta_combined = [[meta_func(meta_dim) for meta_dim in meta_dims]
                         for meta_func in [tmean, tvar, skew, kurtosis]]
        return np.array([*meta_combined])

    @staticmethod
    def get_meta_features(df, features):
        vectorised = HeuristicDataset.make_id(HeuristicDataset.assemble(df, features)). \
            rdd.map(lambda p: (p.id, p.features))
        distances = vectorised.cartesian(vectorised).map(
            lambda pp: Distance.euclidean(pp[0][1], pp[1][1]) if pp[0][0] != pp[1][0] else float('-inf')
        ).filter(lambda d: d != float('-inf'))
        max_dist, n = distances.max(), distances.count()
        distances = distances.map(lambda d: d / max_dist)
        d_mean, d_var, d_std, d_skew, d_kurt = HeuristicDataset.get_rdd_stat(distances)
        md_percentage = np.array(distances.histogram(10)[1]) / n
        z_values = distances.map(lambda d: (d - d_mean) / d_std)
        z_percentage = np.array(z_values.histogram([0, 1, 2, 3, float('inf')])[1]) / n
        return np.array([d_mean, d_var, d_std, d_skew, d_kurt, *md_percentage, *z_percentage])

    @staticmethod
    def get_rdd_stat(rdd):
        rdd_mean, rdd_var = rdd.mean(), rdd.variance()
        rdd_std, n = np.sqrt(rdd_var), rdd.count()
        moment_3rd = rdd.map(lambda d: (d - rdd_mean) ** 3).sum() / n
        moment_4th = rdd.map(lambda d: (d - rdd_mean) ** 4).sum() / n
        rdd_skew, rdd_kurt = moment_3rd / rdd_std ** 3, moment_4th / rdd_std ** 4 - 3
        return rdd_mean, rdd_var, rdd_std, rdd_skew, rdd_kurt

    @staticmethod
    def assemble(spark_df, columns=None):
        drops = spark_df.columns if columns is None else columns
        return VectorAssembler(inputCols=drops, outputCol='features').transform(spark_df).drop(*drops)

    @staticmethod
    @udf(returnType=VectorUDT())
    def to_dense(features):
        return DenseVector(features.toArray())

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

    @staticmethod
    def _build_cvi_predictor():
        """
        1 -> CH
        2 -> SIL
        3 -> OS
        4 -> G41
        """
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBClassifier
        meta_data = pd.read_csv(HeuristicDataset.cvi_fitness, header=None, sep=',')
        x, y = meta_data.values[:, :-1], meta_data.values[:, -1].astype(int) - 1
        predictor = GridSearchCV(estimator=XGBClassifier(
            objective='multi:softmax', num_class=4, seed=5, use_label_encoder=False
        ), param_grid={}, scoring='balanced_accuracy', n_jobs=4, cv=4)
        predictor.fit(x, y)
        with open(HeuristicDataset.cvi_path, 'wb') as f:
            pickle.dump(predictor, f)


# To build and serialize classifier, use
# HeuristicDataset._build_cvi_predictor()
