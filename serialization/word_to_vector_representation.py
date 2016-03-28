""" Create document representations """

from __future__ import print_function

import os

import gensim
from gensim.corpora import Dictionary

import config

from serialization.sqlalchemy_db import DBSession, Category, Question, Answer, init_db


class YahooDictionary:

    def __init__(self, vocab_size=20000, dict_file_name='vocab.dict'):
        self.vocab = Dictionary()
        self.vocab_file = os.path.join(config.BASE_DATA_PATH, dict_file_name)
        self.vocab_size = vocab_size

        # start a db session
        self.session = DBSession()

        # load the vocabulary if it exists
        if os.path.exists(self.vocab_file):
            self.vocab.load_from_text(self.vocab_file)
        else:
            self._generate_vocabulary()

        # get the categories as a set
        self.categories = [c[0] for c in self.session.query(Category.text).distinct().all()]
        self.cat_to_idx = dict((c, i+1) for i, c in enumerate(self.categories))
        self.idx_to_cat = dict((i+1, c) for i, c in enumerate(self.categories))

        # commit and close the session
        self.session.commit(); self.session.close()

    @staticmethod
    def tokenize(text):
        return gensim.utils.tokenize(text, to_lower=True)

    def _generate_vocabulary(self, yield_per=100):
        n_questions = self.session.query(Question).count()
        n_answers = self.session.query(Answer).count()

        i = 0
        for question in self.session.query(Question).yield_per(yield_per):
            i += 1
            if i % 10000 == 0:
                print('Processed %d / %d questions :: %d unique tokens' % (i, n_questions, self.vocab.num_docs))

            strings = [question.title, question.content] if question.content is not None else [question.title]
            self.vocab.add_documents([YahooDictionary.tokenize(s) for s in strings])

        i = 0
        for answer in self.session.query(Answer).yield_per(yield_per):
            i += 1
            if i % 10000 == 0:
                print('Processed %d / %d answers :: %d unique tokens' % (i, n_answers, self.vocab.num_docs))

            self.vocab.add_documents([YahooDictionary.tokenize(answer.content)])

        # save the vocabulary
        self.vocab.save_as_text(self.vocab_file)

if __name__ == '__main__':
    dictionary = YahooDictionary()
