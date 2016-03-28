""" Create document representations """

from __future__ import print_function

import os

import gensim
import itertools
from gensim.corpora import Dictionary
from keras.preprocessing.sequence import pad_sequences
import numpy as np

import config

from serialization.sqldb import DBSession, Category, Question, Answer


class YahooDictionary:
    def __init__(self, dict_file_name='vocab.dict', yield_per=100, print_per=10000):
        """
        Dictionary for storing and accessing questions and answers from the Yahoo QA database

        :param dict_file_name: Dictionary location on disk is { config.BASE_DATA_PATH }/{ dict_file_name }
        :param yield_per: For iterating the database, number of documents to retrieve at once. Depends on system memory.
        :param print_per: For long operations, the number of times to print out the status
        """

        self.vocab = Dictionary()
        self.vocab_file = os.path.join(config.BASE_DATA_PATH, dict_file_name)
        self.yield_per = yield_per
        self.print_per = print_per

        # start a db session
        session = DBSession()

        # some information about the dictionary
        self.n_questions = session.query(Question).count()
        self.n_answers = session.query(Answer).count()

        os.remove(self.vocab_file)

        # load the vocabulary if it exists
        if os.path.exists(self.vocab_file):
            print('Loading vocabulary from "%s"' % self.vocab_file)
            self.vocab.load_from_text(self.vocab_file)
        else:
            print('Generating vocabulary')
            self._generate_vocabulary()

        # get the categories as a set
        self.categories = [c[0] for c in session.query(Category.text).distinct().all()]
        self.cat_to_idx = dict((c, i + 1) for i, c in enumerate(self.categories))
        self.idx_to_cat = dict((i + 1, c) for i, c in enumerate(self.categories))

        # commit and close the session
        session.close()

    @staticmethod
    def tokenize(text):
        """
        Defines how to tokenize a string.

        :param text: The string to tokenize.
        :return: A generator for tokenized text
        """

        return gensim.utils.tokenize(text, to_lower=True)

    def _generate_vocabulary(self):
        session = DBSession()

        i = 0
        for question in session.query(Question).yield_per(self.yield_per):
            i += 1
            if i % self.print_per == 0:
                print('Processed %d / %d questions :: %d unique tokens' % (i, self.n_questions, self.vocab.num_docs))

            strings = [question.title, question.content] if question.content is not None else [question.title]
            self.vocab.add_documents([YahooDictionary.tokenize(s) for s in strings])

        i = 0
        for answer in session.query(Answer).yield_per(self.yield_per):
            i += 1
            if i % self.print_per == 0:
                print('Processed %d / %d answers :: %d unique tokens' % (i, self.n_answers, self.vocab.num_docs))

            self.vocab.add_documents([YahooDictionary.tokenize(answer.content)])

        # save the vocabulary
        self.vocab.save_as_text(self.vocab_file)

        # commit and close the session
        session.commit(); session.close()

    def token2id(self, token):
        return self.vocab.token2id.get(token)

    def id2token(self, id):
        return self.vocab.id2token.get(id)

    def get_docs(self, num=-1):
        """
        Get encoded documents.

        :param num: Number of documents to return (defaults to all documents)
        :return: The first `num` documents in the dictionary
        """

        session = DBSession()

        answers = []
        questions = []
        categories = []

        if num < 0:
            num = self.n_answers

        i = 0
        for answer in itertools.islice(session.query(Answer).yield_per(self.yield_per), num):
            i += 1
            if i % self.print_per == 0:
                print('Processed %d / %d answers' % (i, self.n_answers))

            question = answer.question
            question_title_tokens = YahooDictionary.tokenize(question.title)
            question_content_tokens = [] if question.content is None else YahooDictionary.tokenize(question.content)

            # encode using the dictionary
            question_enc = [self.token2id(x) for x in itertools.chain(question_title_tokens, question_content_tokens)]

            # category indices
            category_enc = self.cat_to_idx[question.category.text]

            # answer indices
            answer_tokens = YahooDictionary.tokenize(answer.content)
            answer_enc = [self.vocab.token2id[x] for x in answer_tokens]

            # append encoded versions to the list to keep track of them
            answers.append(answer_enc)
            questions.append(question_enc)
            categories.append(category_enc)

        question_length = config.STRING_LENGTHS['question_title'] + config.STRING_LENGTHS['question_content']
        return pad_sequences(answers, config.STRING_LENGTHS['answer_content']), \
               pad_sequences(questions, question_length), \
               np.array(categories)

if __name__ == '__main__':
    dictionary = YahooDictionary()
    print(dictionary.get_docs())
