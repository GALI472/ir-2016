""" Mixture of experts (experts here being gensim models) """
from models.gensim_models import TfidfRetrieval, LsiRetrieval, LdaRetrieval, Word2VecRetrieval, Doc2VecRetrieval
from models.interfaces import RetrievalInterface
from serialization.dictionary import CorpusDictionary


class MixtureOfExperts(RetrievalInterface):

    def __init__(self, dictionary, experts):
        self.dictionary = dictionary
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
    ]

    dic = CorpusDictionary(prefix='v20000')
    model = LdaRetrieval(dic, num_best=10)

    docs = dic.get_docs(num=100)
    q, a = docs[0], docs[1]

    print(model.index[q])


    # moe = MixtureOfExperts(dic, experts)

