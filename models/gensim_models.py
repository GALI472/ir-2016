""" Gensim-based retrieval models """

import abc
import os

import gensim

import config
from models.interfaces import RetrievalInterface
from serialization.dictionary import CorpusDictionary


class GensimInterface(RetrievalInterface):
    def __init__(self, dictionary, name, num_features):
        assert name in config.MODELS, '"%s" not found in models, please specify in config.py' % name
        self.name = name
        self.index_name = config.MODELS[name] + '.index'
        self.num_features = num_features

        if os.path.exists(self.index_name):
            print('Loading matrix similarities for <%s>' % name)
            self.index = gensim.similarities.Similarity.load(self.index_name)
        else:
            print('Generating matrix similarities for <%s>' % name)
            self.index = self.generate_index(dictionary, name)
            self.index.save(self.index_name)

    def generate_index(self, dictionary, name):

        # make sure the model exists, otherwise generate it
        if not os.path.exists(config.MODELS[name]):
            print('Generating <%s> model at "%s"' % (name, config.MODELS[name]))
            model = self.generate_model(dictionary.mm_corpus)
            model.save(config.MODELS[name])
        else:
            print('Loading <%s> model from "%s"' % (name, config.MODELS[name]))
            model = self.load_model(config.MODELS[name])

        index = gensim.similarities.Similarity(self.index_name,
                                               model[dictionary.mm_corpus],
                                               self.num_features)

        return index

    def top_n_documents(self, document, n):
        sims = sorted(enumerate(self.index[document]), key=lambda item: -item[1])
        return sims[:n]

    @abc.abstractmethod
    def generate_model(self, corpus):
        return

    @abc.abstractmethod
    def load_model(self, fname):
        return


class TfidfRetrieval(GensimInterface):
    def generate_model(self, corpus):
        return gensim.models.TfidfModel(corpus)

    def load_model(self, fname):
        return gensim.models.TfidfModel.load(fname)

if __name__ == '__main__':
    dic = CorpusDictionary(prefix='v20000') # get the shorter dictionary
    print(dic.vocab)

    model = TfidfRetrieval(dictionary=dic, name='tfidf', num_features=dic.vocab.num_docs)
    model =
