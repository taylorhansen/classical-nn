from keras.callbacks import TensorBoard
from nn.nn import Network
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from time import time

filename="988-v04.lehman1.wav"

with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    audio_data = sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()
    
    tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
    network = Network(num_samples=len(audio_data), callbacks=[tensorboard])
    network.train(np.array(audio_data).transpose())
