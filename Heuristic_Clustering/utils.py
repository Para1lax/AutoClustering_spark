import pyspark.sql.functions

from Constants import DEBUG, DEBUG_PREFIX

# try:
#     from pyspark import SparkContext, SparkConf
#     from pyspark.sql import SparkSession
# except ImportError as e:
#     print('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')
# sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
#
# spark = SparkSession \
#     .builder \
#     .getOrCreate()

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer
from pyspark.sql.functions import monotonically_increasing_id


# sqlCtx = SQLContext(sc)

def print_log(f, s):
    f.write if f is not None else print(s)


def assemble(spark_df, columns=None):
    drops = spark_df.columns if columns is None else columns
    return VectorAssembler(inputCols=drops, outputCol='features').transform(spark_df).drop(*drops)


def make_id(spark_df):
    return spark_df.withColumn('id', monotonically_increasing_id())


def get_df_dimensions(spark_df):
    return len(spark_df.first().features) if 'features' in spark_df.columns \
        else len(spark_df.drop('id', 'features', 'labels').columns)


def get_unique_labels(labeled_df, exclude_none=True):
    if 'labels' not in labeled_df.columns:
        raise ValueError("Dataframe does not contain column 'labels'\n. Columns are: {}".format(labeled_df.columns))
    labels = list(map(lambda sample: sample.labels, labeled_df.drop_duplicates(['labels']).collect()))
    if None in labels and exclude_none:
        labels.remove(None)
    return labels

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]':
        return TimestampType()
    elif f == 'int64':
        return LongType()
    elif f == 'int32':
        return IntegerType()
    elif f == 'float64':
        return FloatType()
    else:
        return StringType()


def define_structure(string, format_type):
    try:
        typo = equivalent_type(format_type)
    except:
        typo = StringType()
    return StructField(string, typo)


# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(spark_context, pandas_df):
    sql = SQLContext(spark_context)
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types):
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sql.createDataFrame(pandas_df, p_schema)
