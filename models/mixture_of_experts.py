""" Mixture of experts (experts here being gensim models) """
from models.gensim_models import TfidfRetrieval, LsiRetrieval, LdaRetrieval, Word2VecRetrieval, Doc2VecRetrieval
from models.interfaces import RetrievalInterface
from serialization.dictionary import CorpusDictionary


class MixtureOfExperts(RetrievalInterface):

    def __init__(self, experts, dic_prefix=''):
        self.dictionary = CorpusDictionary(prefix=dic_prefix)
        self.experts = [expert(self.dictionary) for expert in experts]

        self.model = None #### TODO

    def top_n_documents(self, document, n):
        return self.model[document]


if __name__ == '__main__':
    experts = [
        TfidfRetrieval,
        LsiRetrieval,
        LdaRetrieval,
        Word2VecRetrieval,
        Doc2VecRetrieval,
    ]

    moe = MixtureOfExperts(experts, dic_prefix='v20000')

