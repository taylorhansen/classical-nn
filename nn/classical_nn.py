import keras
from keras.activations import relu, sigmoid, tanh
from keras.layers import Dense, Dropout, Input, LSTM, TimeDistributed
from keras.losses import mean_squared_error
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, SGD
import numpy as np

noise_length = 100
note_data_length = 3

g_input = Input(shape=(None, noise_length))
lstm1 = LSTM(units=100, dropout=0.4, return_sequences=True)(g_input)
lstm2 = LSTM(units=50, dropout=0.3, return_sequences=True)(lstm1)
g_output = Dense(units=note_data_length, activation="sigmoid",
        name="g_output")(lstm1)
gm = Model(inputs=g_input, outputs=g_output)
#gm.summary(line_length=80)
gm.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
        loss=mean_squared_error)

d_input = Input(shape=(None, note_data_length))
lstm1 = LSTM(units=100, dropout=0.3)(d_input)
d_output = Dense(units=1, activation="sigmoid", name="d_output")(lstm1)
dm = Model(inputs=d_input, outputs=d_output)
#dm.summary(line_length=80)
dm.compile(optimizer=RMSprop(lr=0.1, clipvalue=1.0, decay=6e-8),
        loss=mean_squared_error)

am = Sequential()
am.add(gm)
am.add(dm)
#am.summary(line_length=80)
am.compile(optimizer=RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8),
        loss=keras.losses.mean_squared_error, metrics=["accuracy"])
