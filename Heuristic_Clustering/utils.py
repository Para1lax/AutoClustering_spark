from Constants import DEBUG, DEBUG_PREFIX

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer

sqlCtx = SQLContext(sc)


#preprocessing data for spark
# TODO: OneHotEncoder, StringIndexer (maybe)
def preprocess(data):
    vectorAssembler = VectorAssembler(inputCols=data.columns,
                                      outputCol="features")
    return vectorAssembler.transform(data)


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
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types):
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlCtx.createDataFrame(pandas_df, p_schema)


def debugging_printer(place, info_name=None, info=None):
    if not DEBUG:
        return

    if info is None:
        print("{}{}".format(DEBUG_PREFIX, place))
    else:
        print("\n \
               ==========================> {} <==========================\n \
               \n \
               {}:  \n{}\n \
               \n \
               ==========================".format(place, info_name, info))

