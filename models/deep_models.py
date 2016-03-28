from __future__ import print_function

from keras.layers import Embedding, Convolution1D, MaxPooling1D, LSTM, Dropout, Dense, Merge
from keras.models import Sequential

import config

WEIGHTS_FILE = config.RESOURCES['weights']['cnn_lstm']
LOAD_WEIGHTS = False
VOCAB_SIZE = 20000
BATCH_SIZE = 32
EPOCHS = 200

# TODO: Trim dictionary to top `VOCAB_SIZE` elements

# build the network
HIDDEN_NEURONS = 50
EMBEDDING_SIZE = 50
LSTM_DROPOUT_U = 0.15
LSTM_DROPOUT_W = 0.25

# TODO: Might work better as a graphical model?
# It might be ideal to train the same language model on both the questions and answers

# predict question given answer
print('Building network...')

# answer part
a_model = Sequential()
a_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=config.STRING_LENGTHS['answer_content']))
a_model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='valid', activation='relu', subsample_length=1))
a_model.add(MaxPooling1D(pool_length=2))
a_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W))
a_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W, go_backwards=True))
a_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W))
a_model.add(Dropout(0.25))
a_model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='valid', activation='relu',
                           subsample_length=1))
a_model.add(MaxPooling1D(pool_length=2))
a_model.add(LSTM(50, return_sequences=False))

# question part
q_model = Sequential()
q_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=config.STRING_LENGTHS['answer_content']))
q_model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='valid', activation='relu', subsample_length=1))
q_model.add(MaxPooling1D(pool_length=2))
q_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W))
q_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W, go_backwards=True))
q_model.add(LSTM(HIDDEN_NEURONS, return_sequences=True, dropout_U=LSTM_DROPOUT_U, dropout_W=LSTM_DROPOUT_W))
q_model.add(Dropout(0.25))
q_model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='valid', activation='relu',
                           subsample_length=1))
q_model.add(MaxPooling1D(pool_length=2))
q_model.add(LSTM(50, return_sequences=False))

aq_model = Sequential()
aq_model.add(Merge([a_model, q_model], mode='cos'))

aq_model.compile(optimizer='adam', loss='categorical_crossentropy')

# TODO: Train on real data