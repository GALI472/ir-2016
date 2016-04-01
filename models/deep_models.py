from __future__ import print_function

import os

import pickle

import theano
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import reuters
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import config

reuters_enc_path = os.path.join(config.BASE_DATA_PATH, 'reuters_enc.npz')
max_len = 1000

if os.path.exists(reuters_enc_path):
    npfile = np.load(reuters_enc_path)
    cats = npfile['arr_0']
    docs = npfile['arr_1']
else:
    cat_enc = dict((x, i) for i, x in enumerate(set(reuters.categories())))

    def encode(x):
        r = [int(i) for i in x[:max_len].encode('ascii','ignore')]
        return r

    encs = [([cat_enc[i] for i in reuters.categories(fid)], encode(reuters.raw(fid))) for fid in reuters.fileids()]

    cats_enc = [i[0] for i in encs]
    cats = np.zeros((len(cats_enc), max([max(i) for i in cats_enc])), dtype=theano.config.floatX)

    for i, cat in enumerate(cats_enc):
        for j in cat:
            cats[i,j-1] = 1

    docs = pad_sequences([i[1] for i in encs], maxlen=max_len, dtype=theano.config.floatX)

    np.savez(reuters_enc_path, cats, docs)

print(cats.shape)
print(docs.shape)
