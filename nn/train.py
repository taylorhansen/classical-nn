from classical_nn import am, dm, gm, noise_length
from generate import generate, get_noise_vector
from mid import real_d_in
import numpy as np

# trains the discriminator network for one full epoch
def train_discriminator():

    # expected output for each type song
    real = np.zeros(shape=(1, 1))
    fake = np.ones(shape=(1, 1))

    # generate training data
    d_in = []
    d_out = []
    for i in range(len(real_d_in)):
        d_in.append(real_d_in[i])
        d_out.append(np.array(real))
        d_in.append(generate().reshape((1, -1, 3)))
        d_out.append(np.array(fake))

    # train one full epoch with all the training data
    for x, y in zip(d_in, d_out):
        dm.train_on_batch(x, y)

# trains the adversarial (generator+discriminator) network for one full batch
def train_adversarial():
    g_notes = np.random.randint(100, 1000)
    noise_vector = get_noise_vector()
    a_in = np.full(shape=(1, g_notes, noise_length), fill_value=noise_vector)
    a_out = np.ones(shape=(1, 1))
    am.train_on_batch(a_in, a_out)
