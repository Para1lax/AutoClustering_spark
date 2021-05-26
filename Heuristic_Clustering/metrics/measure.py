from abc import abstractmethod, ABC


class Measure(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def find(self, data, spark_context, labels, n_clusters):
        pass

    @abstractmethod
    def update(self, data, spark_context, n_clusters, labels, k, l, id):
        """

        :param k: old_labels for id
        :param l: new_labels for id
        :param id: ids for rows with new labels
        """
        pass
