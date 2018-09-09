from keras.activations import relu, sigmoid
from keras.layers import Dense, Dropout, LSTM
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
import numpy as np

# leaky rectified linear unit activation function
leaky_relu = lambda x: relu(x, alpha=0.01)

class Network(object):
    def __init__(self, callbacks=[], noise_length=100):
        self.callbacks = callbacks
        self.noise_length = noise_length
        self.output_length = 1

        self.gm = Network.generator(noise_length, self.output_length)
        self.dm = Network.discriminator(self.output_length)
        self.am = Network.adversarial(self.gm, self.dm)

    # network that creates synthetic data using noise as input
    @staticmethod
    def generator(noise_length, output_length):
        gm = Sequential()

        gm.add(LSTM(units=noise_length input_dim=100, dropout=0.3))
        gm.add(LSTM(units=output_length)

        gm.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                loss=binary_crossentropy)

        return gm

    # network that tells between real and synthetic data
    @staticmethod
    def discriminator(output_length):
        dm = Sequential()
        dm.add(LSTM(units=output_length, input_dim=output_length, dropout=0.3))
        # at the end of the sequence, the discriminator should return the last
        #  output, which is the final "fakeness" value
        dm.add(LSTM(units=1, dropout=0.3, return_sequences=True))
        dm.compile(optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8),
                loss=binary_crossentropy)

        return dm

    # full adversarial network used in training
    @staticmethod
    def adversarial(gm, dm):
        am = Sequential()
        am.add(gm)
        am.add(dm)
        am.compile(optimizer=RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8),
                loss=binary_crossentropy, metrics=["accuracy"])
        
        return am
    
    # trains the neural networks using real audio data
    # real_data should be a list of numpy arrays that contain all the wav data
    def train(self, real_data):
        # get a mixture of fake (generated) data and real data
        noise = self.get_noise_input()
        fake_data = self.gm.predict(noise)
        x = np.concatenate((real_data, fake_data))

        # expected output of the discriminator network
        # 1 is fake, 0 is real
        y = np.ones([2 * self.num_samples, 1])
        # real_data is the first section of x, so set that to 0 (real)
        y[self.num_samples:, :] = 0

        # train the discriminator network to recognize fake data
        self.dm.fit(x, y, epochs=1, verbose=1)

        # train the entire adversarial network using random noise
        y = np.ones([self.num_samples, 1])
        noise = self.get_noise_input()
        self.am.fit(noise, y, epochs=1, verbose=1, callbacks=self.callbacks)
    
    # generates a matrix of noise data for the generator
    def get_noise_input(self, num_samples):

        np.random.uniform(-1.0, 1.0, size=[])


        return np.random.uniform(0.0, 1.0,
            size=[num_samples, self.noise_length])
