import numpy as np
from numba import njit


class Distance:
    functions = frozenset(['euclidean', 'sqr_euclidean', 'manhattan', 'chebyshev', 'cosine'])

    # numba requires numpy array, so should extract ndarray from pyspark.DenseVector

    @staticmethod
    def euclidean(x, y):
        return Distance._numba_euclidean(x.values, y.values)

    @staticmethod
    @njit
    def _numba_euclidean(x, y):
        diff = x - y
        return np.sqrt(np.sum(diff * diff))

    @staticmethod
    def sqr_euclidean(x, y):
        return Distance._numba_sqr_euclidean(x.values, y.values)

    @staticmethod
    @njit
    def _numba_sqr_euclidean(x, y):
        diff = x - y
        return np.sum(diff * diff)

    @staticmethod
    def manhattan(x, y):
        return Distance._numba_manhattan(x.values, y.values)

    @staticmethod
    @njit
    def _numba_manhattan(x, y):
        diff = x - y
        return np.sum(np.abs(diff))

    @staticmethod
    def chebyshev(x, y):
        return Distance._numba_chebyshev(x.values, y.values)

    @staticmethod
    @njit
    def _numba_chebyshev(x, y):
        diff = x - y
        return np.amax(np.abs(diff))

    @staticmethod
    def cosine(x, y):
        return Distance._numba_cosine(x.values, y.values)

    @staticmethod
    @njit
    def _numba_cosine(x, y):
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        return np.dot(x, y) / (x_norm * y_norm)
