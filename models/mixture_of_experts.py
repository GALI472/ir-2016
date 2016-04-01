""" Mixture of experts (experts here being gensim models) """
import operator

from models.gensim_models import TfidfRetrieval, LsiRetrieval, LdaRetrieval, Word2VecRetrieval
from models.interfaces import RetrievalInterface
from serialization.dictionary import CorpusDictionary


class MixtureOfExperts(RetrievalInterface):

    def __init__(self, dictionary, experts):
        self.num_best = 10

        self.dictionary = dictionary
        self.experts = [expert(self.dictionary, num_best=self.num_best) for expert in experts]

    def heuristic(self, i, score):
        return (self.num_best - i) ** 1.5

    def top_n_documents(self, document, n):
        scores = {}

        for expert in self.experts:
            print('A')
            docs = expert.top_n_documents(self.dictionary.vocab.doc2bow(CorpusDictionary.tokenize(document)), self.num_best)
            print('B')
            for i, doc in enumerate(docs):
                doc_id, score = doc[0], doc[1]
                if doc_id in scores:
                    scores[doc_id] += self.heuristic(i, score)
                else:
                    scores[doc_id] = self.heuristic(i, score)
            print(expert)
            print(scores)

        return sorted(scores, key=scores.get)[:n]


if __name__ == '__main__':
    experts = [
        TfidfRetrieval,
        LdaRetrieval,
        Word2VecRetrieval,
        LsiRetrieval,
    ]
    dic = CorpusDictionary(prefix='v20000')

    moe = MixtureOfExperts(dic, experts)

    while True:
        d = raw_input('Enter a question: ')
        docs = moe.top_n_documents(d, 5)
        print(docs)

