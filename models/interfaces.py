""" Generic retrieval interface (to extend for all retrieval models) """

import abc


class RetrievalInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def top_n_documents(self, document, n):
        return
