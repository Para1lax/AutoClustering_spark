import os
import pickle

import numpy as np
import pandas as pd

import pyspark
from pyspark.sql.functions import monotonically_increasing_id as unique_id, udf
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector, VectorUDT

from Distance import Distance


class AutoDataset:
    """
    Wrapper for spark dataset. Provides preprocessing functions and measure predictor
    """

    # if spark session has not been created, default configuration for launch
    default_spark = pyspark.SparkConf().setMaster('[*]')

    # local path of pickled CVI Predictor and its fit dataset
    cvi_fitness, cvi_path = 'meta_fitness.csv', 'cvi_predictor.pkl'

    # conversions from pandas to spark data types
    eq_types = {'datetime64[ns]': TimestampType, 'int64': LongType, 'int32': IntegerType, 'float64': FloatType}

    def __init__(self, df, features=None, measure=None, max_clusters=15):
        """
        Parameters
        ----------
        df: either spark of pandas dataframe
        features: list of column names, which specifies features of samples
        measure: if measure is given, it will be applied, else recommendation by CVI Predictor will bi given
        max_clusters: upper bound on clusters amount. Search will be in range [2, max_clusters]
        """
        if isinstance(df, pd.DataFrame):
            sc = pyspark.SparkContext.getOrCreate(self.default_spark)
            df = self.pandas_to_spark(sc, df)
        elif not isinstance(df, pyspark.sql.DataFrame):
            raise ValueError('Expected pandas_df or spark_df')

        self.df, self.max_clusters = self.make_id(self.vector_norm(self.assemble(df, features))), max_clusters
        self.df = self.df.withColumn('features', self.to_dense(self.df.features))
        self.dims, self.n = self.get_df_dimensions(self.df), self.df.count()
        from Measure import Measure
        if issubclass(type(measure), Measure):
            self.measure = measure
        elif isinstance(measure, str) and measure in Measure.functions:
            self.measure = Measure.make(measure, distance=Distance.euclidean)
        elif measure is None:
            self.measure = self.get_measure(df, columns=features)
        else:
            raise ValueError('Unknown measure argument: {}'.format(measure))

    @staticmethod
    def pandas_to_spark(spark_context, pandas_df):
        sql = pyspark.sql.SQLContext(spark_context)
        columns = list(pandas_df.columns)
        types = list(pandas_df.dtypes)
        struct_list = []
        for column, t in zip(columns, types):
            typo = StringType() if t not in AutoDataset.eq_types else AutoDataset.eq_types[t]()
            struct_list.append(StructField(column, typo))
        p_schema = StructType(struct_list)
        return sql.createDataFrame(pandas_df, p_schema)

    @staticmethod
    def get_measure(df, columns=None):
        """
        Measure recommendation. Using predefined pickled CVI Predictor.
        Parameters
        ----------
        df: spark dataframe
        columns: list of column names, which represent sample's features

        Returns
        -------
        Instance of Measure, which will be used to estimate clustering
        """
        columns = columns if columns is not None else df.columns
        from Measure import Measure
        # Define integer labels for measure to match with CVI Predictor output
        cvi_measure_by_id = [Measure.CH, Measure.SIL, Measure.OS, Measure.G41]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + '/' + AutoDataset.cvi_path, 'rb') as f:
            cvi_classifier = pickle.load(f)
            meta_features = AutoDataset.get_meta_features(df, columns)
            measure_id = cvi_classifier.predict(np.array([meta_features]))[0]
            measure_name = cvi_measure_by_id[measure_id]
            return Measure.make(measure_name, distance=Distance.euclidean)

    @staticmethod
    def __columns_stat(norm, columns):
        """
        Not in use at the moment. Another approach to calculate meta-features of dataset
        Parameters
        ----------
        norm: normalised to [0, 1] range spark dataframe (not vectorised!)
        columns: list of column names, which represent sample's features

        Returns
        -------
        Numpy array of 16 combinations of mean, variance, skewness and kurtosis,
        applied for each column separately, then combined
        """
        import pyspark.sql.functions as F
        from scipy.stats import tmean, tvar, skew, kurtosis
        meta_dims = [[norm.select(sql_func(col)).collect()[0][0] for col in columns]
                     for sql_func in [F.mean, F.variance, F.skewness, F.kurtosis]]
        meta_combined = [[meta_func(meta_dim) for meta_dim in meta_dims]
                         for meta_func in [tmean, tvar, skew, kurtosis]]
        return np.array([*meta_combined])

    @staticmethod
    def get_meta_features(df, features):
        """
        Extract meta-features of given dataframe in order to suggest recommendation by CVI Predictor.
         Calculates all pairwise distances between samples. Normalises them to upper bound 1.0.
         Extract statistical values: [mean, variance, standard deviation, skewness, kurtosis].
         Next 10 values represents percentage of distances in ranges [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0];
         To calculate the last 4 meta-features, distances values forced to obtain zero mean and unit variance;
         after that count percentage of values in ranges [0, 1), [1, 2), [2, 3) and other positives
        Parameters
        ----------
        df: source spark dataframe
        features: list of column names, which represent sample's features

        Returns
        -------
        Numpy array of 19 meta features in specified above order
        """
        vectorised = AutoDataset.make_id(AutoDataset.assemble(df, features)). \
            rdd.map(lambda p: (p.id, p.features))
        distances = vectorised.cartesian(vectorised).map(
            lambda pp: Distance.euclidean(pp[0][1], pp[1][1]) if pp[0][0] != pp[1][0] else float('-inf')
        ).filter(lambda d: d != float('-inf'))
        max_dist, n = distances.max(), distances.count()
        distances = distances.map(lambda d: d / max_dist)
        d_mean, d_var, d_std, d_skew, d_kurt = AutoDataset.get_rdd_stat(distances)
        md_percentage = np.array(distances.histogram(10)[1]) / n
        z_values = distances.map(lambda d: (d - d_mean) / d_std)
        z_percentage = np.array(z_values.histogram([0, 1, 2, 3, float('inf')])[1]) / n
        return np.array([d_mean, d_var, d_std, d_skew, d_kurt, *md_percentage, *z_percentage])

    @staticmethod
    def get_rdd_stat(rdd):
        """
        Get statistical moments of given rdd
        Parameters
        ----------
        rdd: pyspark.RDD of floats

        Returns
        -------
        Numpy array of [mean, variance, standard deviation, skewness, kurtosis]
        """
        rdd_mean, rdd_var = rdd.mean(), rdd.variance()
        rdd_std, n = np.sqrt(rdd_var), rdd.count()
        moment_3rd = rdd.map(lambda d: (d - rdd_mean) ** 3).sum() / n
        moment_4th = rdd.map(lambda d: (d - rdd_mean) ** 4).sum() / n
        rdd_skew, rdd_kurt = moment_3rd / rdd_std ** 3, moment_4th / rdd_std ** 4 - 3
        return rdd_mean, rdd_var, rdd_std, rdd_skew, rdd_kurt

    @staticmethod
    def assemble(spark_df, columns=None):
        """
        Collects separate columns of spark dataframe into a single vector
        Parameters
        ----------
        spark_df: source dataframe
        columns: columns to assemble. If None, gather all columns of spark_df

        Returns
        -------
        Spark dataframe with one vector column instead of specified list of columns
        """
        drops = spark_df.columns if columns is None else columns
        return VectorAssembler(inputCols=drops, outputCol='features').transform(spark_df).drop(*drops)

    @staticmethod
    @udf(returnType=VectorUDT())
    def to_dense(features):
        """
        Converts spark vector (dense or sparse) to dense vector
        Parameters
        ----------
        features: sparse or dense vector
        Returns
        -------
        pyspark.ml.linalg.DenseVector
        """
        return DenseVector(features.toArray())

    @staticmethod
    def make_id(spark_df):
        """
        Add column of unique identifiers, if column 'id' does not exist.
        If column is provided, it is up to user to make sure of uniqueness
        Parameters
        ----------
        spark_df: source dataframe

        Returns
        -------
        Spark dataframe with column 'id'
        """
        return spark_df if 'id' in spark_df.columns else spark_df.withColumn('id', unique_id())

    @staticmethod
    def get_df_dimensions(spark_df):
        """
        Counts the number of sample's dimensions
        Parameters
        ----------
        spark_df: spark df with 'features' column

        Returns
        -------
        Integer number of dims
        """
        return len(spark_df.first().features) if 'features' in spark_df.columns \
            else len(spark_df.drop('id', 'features', 'labels').columns)

    @staticmethod
    def get_unique_labels(labeled_df, exclude_none=True):
        """
        Get list of unique labels of labeled dataframe
        Parameters
        ----------
        labeled_df: spark dataframe with column 'labels'
        exclude_none: if contains None, exclude it from result

        Returns
        -------
        List of unique label values
        """
        if 'labels' not in labeled_df.columns:
            raise ValueError("Dataframe does not contain column 'labels'\n. Columns are: {}".format(labeled_df.columns))
        labels = list(map(lambda sample: sample.labels, labeled_df.drop_duplicates(['labels']).collect()))
        if None in labels and exclude_none:
            labels.remove(None)
        return labels

    @staticmethod
    def vector_norm(vectorised_df):
        """
        Normalises vector column 'features'. Each vector component value compresses into range [0 .. 1]
        Parameters
        ----------
        vectorised_df: spark dataframe with vectorised column 'features'

        Returns
        -------
        Spark dataframe with vectorised column 'features', each component of vector in range [0 .. 1]
        """
        normalised = MinMaxScaler(inputCol='features', outputCol='__f__').fit(vectorised_df).transform(vectorised_df)
        return normalised.drop('features').withColumnRenamed('__f__', 'features')

    @staticmethod
    def sparse_norm(df, columns=None):
        """
        Normalises input columns into range [0 .. 1]
        Parameters
        ----------
        df: spark dataframe with separate columns of features
        columns: list of column names to normalise. If None, normalise each column

        Returns
        -------
        Spark dataframe with same columns, but normalised content
        """
        columns, normalised = columns if columns is not None else df.columns, df
        vector = AutoDataset.vector_norm(AutoDataset.assemble(df, columns))
        return vector.rdd.map(lambda row: [*row.features.values.tolist()]).toDF(columns)

    @staticmethod
    def _build_cvi_predictor():
        """
        Should not be used. Fits and serialises CVI Predictor
          1 -> CH, 2 -> SIL, 3 -> OS, 4 -> G41
        Returns
        -------
        Serialised by pickle model in 'cvi_path' file
        """
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBClassifier
        meta_data = pd.read_csv(AutoDataset.cvi_fitness, header=None, sep=',')
        x, y = meta_data.values[:, :-1], meta_data.values[:, -1].astype(int) - 1
        predictor = GridSearchCV(estimator=XGBClassifier(
            objective='multi:softmax', num_class=4, seed=5, use_label_encoder=False
        ), param_grid={}, scoring='balanced_accuracy', n_jobs=4, cv=4)
        predictor.fit(x, y)
        with open(AutoDataset.cvi_path, 'wb') as f:
            pickle.dump(predictor, f)


# To build and serialize classifier, use
# HeuristicDataset._build_cvi_predictor()
