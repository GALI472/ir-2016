""" Gensim-based vector space representation retrieval models """

import abc
import os

import gensim
import itertools

import config
from models.interfaces import RetrievalInterface
from serialization.dictionary import CorpusDictionary

import logging
logger = logging.getLogger(__name__)


class GensimInterface(RetrievalInterface):
    def __init__(self, dictionary, name, num_features, num_best=None):
        assert name in config.MODELS, '"%s" not found in models, please specify in config.py' % name
        self.name = name
        self.index_name = config.MODELS[name] + '.index'
        self.num_features = num_features
        self.num_best = num_best

        if os.path.exists(self.index_name):
            logger.info('Loading matrix similarities for <%s>' % name)
            self.index = gensim.similarities.Similarity.load(self.index_name)
        else:
            logger.info('Generating matrix similarities for <%s>' % name)
            self.index = self.generate_index(dictionary, name)
            self.index.save(self.index_name)

    def generate_index(self, dictionary, name):

        # make sure the model exists, otherwise generate it
        if not os.path.exists(config.MODELS[name]):
            logger.info('Generating <%s> model at "%s"' % (name, config.MODELS[name]))
            model = self.generate_model(dictionary)
            model.save(config.MODELS[name])
        else:
            logger.info('Loading <%s> model from "%s"' % (name, config.MODELS[name]))
            model = self.load_model(config.MODELS[name])

        index = gensim.similarities.Similarity(self.index_name,
                                               model[dictionary.mm_answer_corpus],
                                               self.num_features,
                                               num_best=self.num_best)

        return index

    def top_n_documents(self, document, n):
        assert self.num_best is None or n >= self.num_best, 'num_best must be at least number of requested docs'

        if self.num_best is None:
            sims = sorted(enumerate(self.index[document]), key=lambda item: -item[1])
        else:
            sims = sorted(self.index[document], key=lambda item: -item[1])

        return sims[:n]

    @abc.abstractmethod
    def generate_model(self, dictionary):
        return

    @abc.abstractmethod
    def load_model(self, fname):
        return


class TfidfRetrieval(GensimInterface):
    def __init__(self, dictionary, num_best=None):
        GensimInterface.__init__(self, dictionary=dictionary, name='tfidf', num_features=dictionary.vocab.num_docs, num_best=num_best)

    def generate_model(self, dictionary):
        return gensim.models.TfidfModel(dictionary.mm_answer_corpus)

    def load_model(self, fname):
        return gensim.models.TfidfModel.load(fname, mmap='r')


class LdaRetrieval(GensimInterface):
    def __init__(self, dictionary, n_topics=1000, num_best=None):
        GensimInterface.__init__(self, dictionary=dictionary, name='lda', num_features=n_topics, num_best=num_best)

    def generate_model(self, dictionary):
        return gensim.models.LdaModel(dictionary.mm_answer_corpus, num_topics=self.num_features)

    def load_model(self, fname):
        return gensim.models.LdaModel.load(fname, mmap='r')


class LsiRetrieval(GensimInterface):
    def __init__(self, dictionary, n_topics=100, num_best=None):
        GensimInterface.__init__(self, dictionary=dictionary, name='lsi', num_features=n_topics, num_best=num_best)

    def generate_model(self, dictionary):
        return gensim.models.LdaModel(dictionary.mm_answer_corpus, num_topics=self.num_features)

    def load_model(self, fname):
        return gensim.models.LdaModel.load(fname, mmap='r')


class Word2VecRetrieval(GensimInterface):
    def __init__(self, dictionary, n_topics=100, num_best=None):
        GensimInterface.__init__(self, dictionary=dictionary, name='word2vec', num_features=n_topics, num_best=num_best)

    def generate_model(self, dictionary):
        return gensim.models.Word2Vec(dictionary, size=self.num_features)

    def load_model(self, fname):
        return gensim.models.Word2Vec.load(fname, mmap='r')


class Doc2VecRetrieval(GensimInterface):
    def __init__(self, dictionary, n_topics=100, num_best=None):
        GensimInterface.__init__(self, dictionary=dictionary, name='doc2vec', num_features=n_topics, num_best=num_best)

    def generate_model(self, dictionary):
        return gensim.models.Doc2Vec(dictionary, size=self.num_features)

    def load_model(self, fname):
        return gensim.models.Doc2Vec.load(fname, mmap='r')

if __name__ == '__main__':

    # build the models (if they aren't already built)
    dic = CorpusDictionary(prefix='v20000') # get the shorter dictionary
    print(dic.vocab)
    print(dic.mm_answer_corpus)

    model = TfidfRetrieval(dic)
    print(model)

    model = LsiRetrieval(dic)
    print(model)

    model = LdaRetrieval(dic)
    print(model)

    model = Word2VecRetrieval(dic)
    print(model)

    # model = Doc2VecRetrieval(dic)
    # print(model)


