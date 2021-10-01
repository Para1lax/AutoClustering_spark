import unittest
import pyspark
from pyspark.sql.functions import rand, round

from Measure import Measure
from Distance import Distance
from HeuristicDataset import HeuristicDataset as HD


class MeasureTest(unittest.TestCase):
    """
    Calculates all available measures on iris dataset twice:
    once for original labels, another for randomly generated.
    """

    columns = (['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'species')
    csv_path, k, distance = './datasets/normalized/labled_iris.csv', 3, Distance.euclidean

    @classmethod
    def setUpClass(cls):
        sc = pyspark.SparkContext.getOrCreate(conf=pyspark.SparkConf().setMaster('local[2]').setAppName('measures'))
        df = pyspark.SQLContext(sc).read.csv(MeasureTest.csv_path, header=True, inferSchema=True)

        df = HD.make_id(HD.assemble(df, MeasureTest.columns[0])).withColumnRenamed(MeasureTest.columns[1], 'labels')
        cls.true_df, cls.random_df = df, df.withColumn('labels', round(rand() * (MeasureTest.k - 1)).cast('int'))

    def __invoke_measure(self, algorithm):
        measure = Measure.make(algorithm, distance=MeasureTest.distance)
        original, random = measure(self.true_df), measure(self.random_df)
        print("$> '%s': original: %f, random: %f" % (algorithm, original, random))
        self.assertLessEqual(original, random) if measure.should_decrease else self.assertGreaterEqual(original, random)

    def test_silhouette(self):
        self.__invoke_measure(Measure.SIL)

    def test_os(self):
        self.__invoke_measure(Measure.OS)

    def test_ch(self):
        self.__invoke_measure(Measure.CH)

    def test_score(self):
        self.__invoke_measure(Measure.SCORE)

    def test_db(self):
        self.__invoke_measure(Measure.DB)

    def test_sdbw(self):
        self.__invoke_measure(Measure.S_DBW)

    def test_dunn(self):
        self.__invoke_measure(Measure.DUNN)

    def test_g31(self):
        self.__invoke_measure(Measure.G31)

    def test_g41(self):
        self.__invoke_measure(Measure.G41)

    def test_g51(self):
        self.__invoke_measure(Measure.G51)

    def test_g33(self):
        self.__invoke_measure(Measure.G33)

    def test_g43(self):
        self.__invoke_measure(Measure.G43)

    def test_g53(self):
        self.__invoke_measure(Measure.G53)
