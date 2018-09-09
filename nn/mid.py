import glob
import keras
from music21 import converter, instrument, note, chord
import numpy as np
import os

# notes of every song
songs = []

# the set of all possible notes
unique_notes = set()

# creates an array of encoded notes data, or an empty list if unnecessary
# each element contains an array consisting of a note's pitch, offset, and
#  duration
# chords are expanded into multiple notes
def parse_element(element):
    if isinstance(element, chord.Chord):
        return [tuple(pitch.nameWithOctave for pitch in element.pitches),
                    float(element.offset),
                    float(element.duration.quarterLength)]
    if isinstance(element, note.Note):
        return [element.nameWithOctave, float(element.offset),
                    float(element.duration.quarterLength)]
    return []

for file in glob.glob("./mid/*"):
    midi = converter.parse(file)
    notes_to_parse = None
    notes = []

    parts = instrument.partitionByInstrument(midi)

    if parts:
        # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:
        # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        parsed_note = parse_element(element)
        if len(parsed_note) > 0:
            notes.append(parsed_note)
            unique_notes.add(parsed_note[0])
            x = parsed_note[0]

    songs.append(notes)

num_songs = len(songs)

# map note/chord names to a unique identifier
int_to_note = list(unique_notes)
note_to_int = dict((note, number) for number, note in enumerate(int_to_note))

# number of different notes in dataset
n_vocab = len(unique_notes)

real_d_in = []

for i in range(0, len(songs)):
    current_input = []
    # wrap note id in array brackets
    for j in range(0, len(songs[i])):
        note_data = songs[i][j]
        current_input.append([note_to_int[note_data[0]], note_data[1],
                    note_data[2]])
    real_d_in.append(current_input)

n_patterns = len(real_d_in)

# reshape input so it's compatible with lstm layers
# basically we need to have each note be its own column vector to be inputted
#  into the discriminator
for i in range(len(real_d_in)):
    real_d_in[i] = np.array(real_d_in[i]).reshape(
        (1, len(real_d_in[i]), 3)) / float(n_vocab)

# TODO: move to separate file

from classical_nn import am, dm, gm, noise_length

def get_noise_vector():
    return np.random.uniform(size=(1, noise_length))

# generate an amount of notes using the generator
def generate(g_notes=500):
    noise_vector = get_noise_vector()
    g_in = np.full(shape=(1, g_notes, noise_length), fill_value=noise_vector)
    return gm.predict(g_in, batch_size=1)

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
        d_in.append(generate(np.random.randint(100, 1000)).reshape((1, -1, 3)))
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

train_discriminator()
train_adversarial()

# test stuff
generated_song = generate()[0]
processed_song = []
for note_data in generated_song:
    processed_song.append([int_to_note[int(note_data[0] * n_vocab)],
                note_data[1], note_data[2]])

from create_midi import create_midi
create_midi(processed_song)
