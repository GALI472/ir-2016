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

import logging
logger = logging.getLogger(__name__)


class CorpusDictionary:
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
        self.prefix = prefix + '_' if len(prefix) > 0 else ''

        # locations of files in the system
        files = {
            'vocab': os.path.join(config.BASE_DATA_PATH, 'dicts', self.prefix + 'vocab.dict'),
            'id2token': os.path.join(config.BASE_DATA_PATH, 'dicts', self.prefix + 'id2token_mapping.pkl'),
            'cat': os.path.join(config.BASE_DATA_PATH, 'dicts', self.prefix + 'categories.pkl'),
            'mm_question_corpus': os.path.join(config.BASE_DATA_PATH, 'dicts', self.prefix + 'question_corpus.mm'),
            'mm_answer_corpus': os.path.join(config.BASE_DATA_PATH, 'dicts', self.prefix + 'answer_corpus.mm'),
        }

        # start a db session
        session = DBSession()

        # some information about the dictionary
        self.n_questions = session.query(Question).count()
        self.n_answers = session.query(Answer).count()

        # load the vocabulary if it exists
        if os.path.exists(files['vocab']):
            logger.info('Loading vocabulary from "%s"' % files['vocab'])
            self.vocab = Dictionary.load(files['vocab'])
        else:
            logger.info('Generating vocabulary')
            self.vocab = self._generate_vocabulary()
            self.vocab.save(files['vocab'])

        # load or generate the reverse matchings
        if os.path.exists(files['id2token']):
            logger.info('Loading id2token from "%s"' % files['id2token'])
            self.vocab.id2token = pickle.load(open(files['id2token'], 'rb'))
        else:
            logger.info('Generating id2token')
            self.vocab.id2token = dict((e, i) for i, e in self.vocab.token2id.iteritems())
            pickle.dump(self.vocab.id2token, open(files['id2token'], 'wb'))

        # for tokens that aren't in the vocabulary, return a misc id
        self.empty_token_id = len(self.vocab.token2id)
        self.empty_id_token = 'UNKNOWN_TOKEN'

        # get the categories as a set
        if os.path.exists(files['cat']):
            logger.info('Loading category mappings from "%s"' % files['cat'])
            self.cat_to_idx_dict, self.idx_to_cat_dict = pickle.load(open(files['cat'], 'rb'))
        else:
            logger.info('Generating category mappings')
            categories = [c[0] for c in session.query(Category.text).distinct().all()]
            self.cat_to_idx_dict = dict((c, i + 1) for i, c in enumerate(categories))
            self.idx_to_cat_dict = dict((i + 1, c) for i, c in enumerate(categories))
            pickle.dump((self.cat_to_idx_dict, self.idx_to_cat_dict), open(files['cat'], 'wb'))

        self.empty_category_idx = session.query(Category).distinct().count()
        self.empty_idx_category = 'UNKNOWN_CATEGORY'

        # create the corpus if it doesn't exist
        def corpus(what):
            i = 0
            for a in session.query(what).yield_per(self.yield_per):
                if a.content is None:
                    continue

                doc = CorpusDictionary.tokenize(a.content)

                i += 1
                if i % self.print_per == 0:
                    logger.info('Added %d documents to corpus' % i)

                yield self.vocab.doc2bow(doc)

        if not os.path.exists(files['mm_question_corpus']):
            gensim.corpora.MmCorpus.serialize(files['mm_question_corpus'], corpus(Question))

        if not os.path.exists(files['mm_answer_corpus']):
            gensim.corpora.MmCorpus.serialize(files['mm_answer_corpus'], corpus(Answer))

        # load the corpus
        logger.info('Loading corpus from "%s"' % files['mm_question_corpus'])
        self.mm_corpus = gensim.corpora.MmCorpus(files['mm_question_corpus'])

        logger.info('Loading corpus from "%s"' % files['mm_answer_corpus'])
        self.mm_corpus = gensim.corpora.MmCorpus(files['mm_answer_corpus'])

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
                logger.info('Processed %d / %d questions :: %d unique tokens' % (i, self.n_questions, vocab.num_docs))

            strings = [question.title, question.content] if question.content is not None else [question.title]
            vocab.add_documents([CorpusDictionary.tokenize(s) for s in strings])

        i = 0
        for answer in session.query(Answer).yield_per(self.yield_per):
            i += 1
            if i % self.print_per == 0:
                logger.info('Processed %d / %d answers :: %d unique tokens' % (i, self.n_answers, vocab.num_docs))

            vocab.add_documents([CorpusDictionary.tokenize(answer.content)])

        # commit and close the session
        session.commit()
        session.close()

        return vocab

    def token2id(self, token):
        return self.vocab.token2id.get(token, self.empty_token_id)

    def id2token(self, tid):
        return self.vocab.id2token.get(tid, self.empty_id_token)

    def cat_to_idx(self, category):
        return self.cat_to_idx_dict.get(category.text, self.empty_category_idx) if category is not None \
               else self.empty_category_idx

    def idx_to_cat(self, idx):
        return self.idx_to_cat_dict.get(idx, self.empty_idx_category)

    def get(self, what, num=-1):
        session = DBSession()

        if num < 0:
            num = self.n_answers

        for answer in itertools.islice(session.query(what).yield_per(self.yield_per), num):
            yield answer

        session.close()

    def doc2vec(self, doc):
        return self.vocab.doc2bow(CorpusDictionary.tokenize(doc))

    def __iter__(self):
        session = DBSession()

        for answer in session.query(Answer).yield_per(self.yield_per):
            for line in answer.content.splitlines():
                yield list(line.split())

        session.close()

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
                logger.info('Processed %d / %d answers' % (i, self.n_answers))

            question = answer.question
            question_title_tokens = CorpusDictionary.tokenize(question.title)
            question_content_tokens = [] if question.content is None else CorpusDictionary.tokenize(question.content)

            # encode using the dictionary
            question_enc = [self.token2id(x) for x in itertools.chain(question_title_tokens, question_content_tokens)]

            # category indices
            category_enc = self.cat_to_idx(question.category)

            # answer indices
            answer_tokens = CorpusDictionary.tokenize(answer.content)
            answer_enc = [self.token2id(x) for x in answer_tokens]

            # append encoded versions to the list to keep track of them
            answers.append(answer_enc)
            questions.append(question_enc)
            categories.append(category_enc)

        question_length = config.STRING_LENGTHS['question_title'] + config.STRING_LENGTHS['question_content']
        return (pad_sequences(answers, config.STRING_LENGTHS['answer_content']),
                pad_sequences(questions, question_length),
                np.array(categories))

if __name__ == '__main__':
    dic = CorpusDictionary()

    if False: # a filtered version of the vocabulary (for performance). can change if time permits?
        dic.vocab.filter_extremes(no_above=0.5, keep_n=20000)
        dic.vocab.save(os.path.join(config.BASE_DATA_PATH, 'dicts', 'v20000_vocab.dict'))

    # answers, questions, categories = dic.get_docs(100)
    # logger.info('Answers:', answers)
    # logger.info('Questions:', questions)
    # logger.info('Categories:', categories)
