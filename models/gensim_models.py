""" Gensim-based retrieval models """

import abc
import os
import config
from models.interfaces import RetrievalInterface


class GensimInterface(RetrievalInterface):
    def __init__(self, dictionary, name):
        assert name in config.MODELS, '"%s" not found in models, please specify in config.py' % name

        self.dictionary = dictionary
        self.name = name

        # for different gensim versions, get a particular model
        self.model_type = self.get_gensim_model()

        # make sure the model exists, otherwise generate it
        if not os.path.exists(config.MODELS[name]):
            self.model = self.model_type(self.dictionary.mm_corpus)
            self.model.save(config.MODELS[name])
        else:
            self.model = self.model_type.load(config.MODELS[name])

    def top_n_documents(self, document, n):
        scores = sorted([self.model[d] for d in self.dictionary.get_docs()])
        return scores[:n]

    @abc.abstractmethod
    def get_gensim_model(self):
        return
