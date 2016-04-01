from __future__ import print_function

import os

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation, Convolution1D, MaxPooling1D, Dense
from nltk.corpus import reuters
import numpy as np
import pickle

import config
import logging

logger = logging.getLogger(__name__)

reuters_enc_path = os.path.join(config.BASE_DATA_PATH, 'reuters_enc.npz')
char_idx_path = os.path.join(config.BASE_DATA_PATH, 'char_idx.pkl')
max_len = 1000
max_char = 256

default_char = 'X'
default_idx = 0


def encode_doc(doc):
    d = np.zeros((1, max_len-1, n_chars), dtype=np.bool)
    for p, j in enumerate(doc.lower()[:max_len]):
        d[0, p, char_to_idx[j]] = 1
    return d

if os.path.exists(char_idx_path):
    with open(char_idx_path, 'rb') as f:
        logger.info('Loading character encodings from "%s"' % char_idx_path)
        idx_to_char = pickle.load(f)
        char_to_idx = pickle.load(f)
        cat_enc = pickle.load(f)
else:
    n_docs = len(reuters.fileids())
    cat_enc = dict((x, i + 1) for i, x in enumerate(set(reuters.categories())))

    chars = set()
    for fid in reuters.fileids():
        chars = chars.union(set(reuters.raw(fid).lower()))

    idx_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_idx = dict((c, i) for i, c in enumerate(chars))

    cat_enc = dict((x, i + 1) for i, x in enumerate(set(reuters.categories())))

    with open(char_idx_path, 'wb') as f:
        logger.info('Saving character encodings to "%s"' % char_idx_path)
        pickle.dump(idx_to_char, f)
        pickle.dump(char_to_idx, f)
        pickle.dump(cat_enc, f)

if os.path.exists(reuters_enc_path):
    logging.info('Loading reuters encodings from "%s"' % reuters_enc_path)
    np_file = np.load(reuters_enc_path)
    cats = np_file['arr_0']
    docs = np_file['arr_1']
else:
    n_docs = len(reuters.fileids())

    chars = set()
    for fid in reuters.fileids():
        chars = chars.union(set(reuters.raw(fid).lower()))

    n_chars = len(chars)

    # encode the categories
    cats_enc = [[cat_enc[i] for i in reuters.categories(fid)] for fid in reuters.fileids()]
    cats = np.zeros((len(cats_enc), max([max(i) for i in cats_enc])), dtype=np.bool)
    for i, cat in enumerate(cats_enc):
        for j in cat:
            cats[i,j-1] = 1

    # encode the documents
    docs = np.zeros((n_docs, max_len, n_chars), dtype=np.bool)
    for i, fid in enumerate(reuters.fileids()):
        for p, j in enumerate(reuters.raw(fid).lower()[:max_len]):
            docs[i, p, char_to_idx[j]] = 1

    # save the output
    logging.info('Saving reuters encodings to "%s"' % reuters_enc_path)
    np.savez(reuters_enc_path, cats, docs)

n_chars = len(char_to_idx)

X = docs[:, :max_len-1, :]
y = docs[:, 1:, :]

logging.info('Building model...')
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(max_len-1, n_chars)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(LSTM(n_chars, return_sequences=True))
model.add(Activation('sigmoid'))

logging.info('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model_save_location = os.path.join(config.BASE_DATA_PATH, 'language_model.h5')

if os.path.exists(model_save_location):
    model.load_weights(model_save_location)

def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def evaluate():
    p = model.predict(encode_doc('what is my name? '))
    print(p.shape)
    print(sample(p))
    print(p)

logging.info('Fitting model...')
for i in range(10):
    evaluate()
    model.fit(X, y, batch_size=128, nb_epoch=10, verbose=True)

model.save_weights(model_save_location, overwrite=True)
