""" Definitions of each particular retrieval model """

import abc


class RetrievalInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def top_n_documents(self, n):
        return
