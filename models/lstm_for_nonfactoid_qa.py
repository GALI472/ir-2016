"""
LSTM model for non-factoid question answering (sort of my interpretation on the theme)

Notes:
    - Character-level modeling did not converge (might be improved somehow, not sure how though)

"""

import random

import gensim
import numpy as np
from gensim.corpora import Dictionary
from keras.layers import Embedding, Convolution1D, MaxPooling1D, LSTM, Merge, Dense, Dropout, AveragePooling1D, Flatten, \
    RepeatVector
from keras.models import Graph
from keras.preprocessing.sequence import pad_sequences
from sqlalchemy import func

import itertools
import os

import config
from serialization.sqldb import DBSession, Answer

import logging
logger = logging.getLogger(__name__)

# lengths of strings
ac_len = 100 # config.STRING_LENGTHS['answer_content']
qt_len = 100 # config.STRING_LENGTHS['question_title']
qc_len = 10 # config.STRING_LENGTHS['question_content']

# generate document encodings
vocab = Dictionary.load(os.path.join(config.BASE_DATA_PATH, 'dicts', 'v20000_vocab.dict'))

max_char = len(vocab.token2id) + 1
default_id = max_char - 1


def encode_doc(doc, max_len):
    if doc is None:
        return np.asarray([])

    enc = np.asarray([vocab.token2id.get(c, default_id) for c in itertools.islice(gensim.utils.tokenize(doc, to_lower=True), max_len)])
    return enc


###################
# build the model #
###################

embedding_dims = 256
lstm_dims = 128
pool_length = 5
encode_dims = 1000

logger.info('Building model')

model = Graph()

model.add_input('question_title', input_shape=(qt_len,), dtype=int)
model.add_node(Embedding(max_char, embedding_dims, input_length=qt_len), name='qt_emb', input='question_title')
model.add_input('question_content', input_shape=(qc_len,), dtype=int)
model.add_node(Embedding(max_char, embedding_dims, input_length=qc_len), name='qc_emb', input='question_content')
model.add_node(LSTM(lstm_dims, return_sequences=True, dropout_U=0.1, dropout_W=0.1), name='q_flstm1',
               inputs=['qt_emb', 'qc_emb'], merge_mode='concat', concat_axis=1)
model.add_node(LSTM(lstm_dims, go_backwards=True, return_sequences=True, dropout_U=0.1, dropout_W=0.1), name='q_blstm1',
               inputs=['qt_emb', 'qc_emb'], merge_mode='concat', concat_axis=1)
model.add_node(MaxPooling1D(pool_length=pool_length), name='q_mp', inputs=['q_flstm1', 'q_blstm1'], merge_mode='ave')
model.add_node(AveragePooling1D(pool_length=pool_length), name='q_ap', inputs=['q_flstm1', 'q_blstm1'],
               merge_mode='ave')
model.add_node(Flatten(), name='q_flat', inputs=['q_mp', 'q_ap'], merge_mode='concat')
model.add_node(Dense(encode_dims, activation='tanh'), name='q_out', input='q_flat')

model.add_node(Dense(embedding_dims, activation='tanh'), name='q_dense', input='q_flat')
model.add_node(RepeatVector(ac_len), name='q_rep', input='q_dense')

model.add_input(name='answer', input_shape=(ac_len,), dtype=int)
model.add_node(Embedding(max_char, embedding_dims, input_length=ac_len), name='a_emb', input='answer')
model.add_node(LSTM(lstm_dims, return_sequences=True, dropout_U=0.1, dropout_W=0.1), name='a_flstm1', inputs=['a_emb', 'q_rep'], merge_mode='sum')
model.add_node(LSTM(lstm_dims, go_backwards=True, return_sequences=True, dropout_U=0.1, dropout_W=0.1), name='a_blstm1', inputs=['a_emb', 'q_rep'], merge_mode='sum')
model.add_node(MaxPooling1D(pool_length=2), name='a_mp', inputs=['a_flstm1', 'a_blstm1'], merge_mode='ave')
model.add_node(AveragePooling1D(pool_length=2), name='a_ap', inputs=['a_flstm1', 'a_blstm1'], merge_mode='ave')
model.add_node(Flatten(), name='a_flat', inputs=['a_mp', 'a_ap'], merge_mode='concat')
model.add_node(Dense(encode_dims, activation='tanh'), name='a_out', input='a_flat')

model.add_output(name='output', inputs=['q_out', 'a_out'], merge_mode='dot')

logger.info('Compiling...')
model.compile('adam', {'output': 'binary_crossentropy'})

logger.info('Done!')

#################################
# generate training set from db #
#################################


def generate_data(generate_every=50):
    assert generate_every % 2 == 0, 'Must provide an even number of points to generate'

    logger.info('Generating QA sessions')

    answers = []
    question_titles = []
    question_contents = []

    session = DBSession()

    while True:
        for i, answer in enumerate(session.query(Answer).order_by(func.random()).yield_per(generate_every)):
            if i > 0 and i % (generate_every / 2) == 0:
                targets = [1] * len(answers) + [-1] * len(answers)
                question_titles = question_titles + question_titles[1:] + [question_titles[0]]
                question_contents = question_contents + question_contents[1:] + [question_contents[0]]
                answers = answers + answers

                combined = zip(targets, question_titles, question_contents, answers)
                random.shuffle(combined)
                targets[:], question_titles[:], question_contents[:], answers[:] = zip(*combined)

                yield {'question_title': pad_sequences(question_titles, maxlen=qt_len),
                       'question_content': pad_sequences(question_contents, maxlen=qc_len),
                       'answer': pad_sequences(answers, maxlen=ac_len),
                       'output': np.asarray(targets)}

                question_titles = []
                question_contents = []
                answers = []

            # positive training sample
            answers.append(encode_doc(answer.content, ac_len))
            question_titles.append(encode_doc(answer.question.title, qt_len))
            question_contents.append(encode_doc(answer.question.content, qc_len))

    session.close()

samples_per_epoch = 1000
nb_epoch = 100

train = generate_data()
test = generate_data()

# 10 * 100 * 1000 = 1,000,000
# looks at every data set at least once

for i in range(10):
    model.fit_generator(train, samples_per_epoch, nb_epoch, validation_data=test, nb_val_samples=10)
    model.save_weights(config.MODELS['lstm_cnn'], overwrite=True)

logger.info('Done training')
