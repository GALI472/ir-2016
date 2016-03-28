""" Create document representations """

from __future__ import print_function

import os

import gensim
import itertools

from gensim.corpora import Dictionary
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import cPickle as pickle

import config

from serialization.sqldb import DBSession, Category, Question, Answer


class YahooDictionary:
    def __init__(self,
                 prefix='',
                 yield_per=100,
                 print_per=10000):
        """
        Dictionary for storing and accessing questions and answers from the Yahoo QA database

        :param prefix: The prefix for files associated with this dictionary (default = None)
        :param yield_per: For iterating the database, number of documents to retrieve at once. Depends on system memory.
        :param print_per: For long operations, the number of times to print out the status
        """

        self.yield_per = yield_per
        self.print_per = print_per
        self.prefix = prefix

        vocab_file = os.path.join(config.BASE_DATA_PATH, self.prefix + 'vocab.dict')
        id2token_file = os.path.join(config.BASE_DATA_PATH, self.prefix + 'id2token_mapping.pkl')
        cat_file = os.path.join(config.BASE_DATA_PATH, self.prefix + 'categories.pkl')

        # start a db session
        session = DBSession()

        # some information about the dictionary
        self.n_questions = session.query(Question).count()
        self.n_answers = session.query(Answer).count()

        # load the vocabulary if it exists
        if os.path.exists(vocab_file):
            print('Loading vocabulary from "%s"' % vocab_file)
            self.vocab = Dictionary.load_from_text(vocab_file)
        else:
            print('Generating vocabulary')
            self.vocab = self._generate_vocabulary()
            self.vocab.save_as_text(vocab_file)

        # load or generate the reverse matchings
        if os.path.exists(id2token_file):
            print('Loading id2token from "%s"' % id2token_file)
            self.vocab.id2token = pickle.load(open(id2token_file, 'rb'))
        else:
            print('Generating id2token')
            self.vocab.id2token = dict((e, i) for i, e in self.vocab.token2id.iteritems())
            pickle.dump(self.vocab.id2token, open(id2token_file, 'wb'))

        # for tokens that aren't in the vocabulary, return a misc id
        self.empty_token_id = len(self.vocab.token2id)
        self.empty_id_token = 'UNKNOWN_TOKEN'

        # get the categories as a set
        if os.path.exists(cat_file):
            print('Loading category mappings from "%s"' % cat_file)
            self.cat_to_idx_dict, self.idx_to_cat_dict = pickle.load(open(cat_file, 'rb'))
        else:
            print('Generating category mappings')
            categories = [c[0] for c in session.query(Category.text).distinct().all()]
            self.cat_to_idx_dict = dict((c, i + 1) for i, c in enumerate(categories))
            self.idx_to_cat_dict = dict((i + 1, c) for i, c in enumerate(categories))
            pickle.dump((self.cat_to_idx_dict, self.idx_to_cat_dict), open(cat_file, 'wb'))

        self.empty_category_idx = session.query(Category).distinct().count()
        self.empty_idx_category = 'UNKNOWN_CATEGORY'

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
        vocab = Dictionary()
        session = DBSession()

        i = 0
        for question in session.query(Question).yield_per(self.yield_per):
            i += 1
            if i % self.print_per == 0:
                print('Processed %d / %d questions :: %d unique tokens' % (i, self.n_questions, vocab.num_docs))

            strings = [question.title, question.content] if question.content is not None else [question.title]
            vocab.add_documents([YahooDictionary.tokenize(s) for s in strings])

        i = 0
        for answer in session.query(Answer).yield_per(self.yield_per):
            i += 1
            if i % self.print_per == 0:
                print('Processed %d / %d answers :: %d unique tokens' % (i, self.n_answers, vocab.num_docs))

            vocab.add_documents([YahooDictionary.tokenize(answer.content)])

        # commit and close the session
        session.commit(); session.close()

        return vocab

    def token2id(self, token):
        return self.vocab.token2id.get(token, self.empty_token_id)

    def id2token(self, id):
        return self.vocab.id2token.get(id, self.empty_id_token)

    def cat_to_idx(self, category):
        return self.cat_to_idx_dict.get(category.text, self.empty_category_idx) if category is not None \
               else self.empty_category_idx

    def idx_to_cat(self, idx):
        return self.idx_to_cat_dict.get(idx, self.empty_idx_category)

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
            category_enc = self.cat_to_idx(question.category)

            # answer indices
            answer_tokens = YahooDictionary.tokenize(answer.content)
            answer_enc = [self.token2id(x) for x in answer_tokens]

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
    answers, questions, categories = dictionary.get_docs(100)
    print('Answers:', answers)
    print('Questions:', questions)
    print('Categories:', categories)
