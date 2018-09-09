import glob
import keras
from music21 import converter, instrument, note, chord
import numpy as np
import os

# notes of every song
songs = []

# the set of all possible notes
unique_notes = set()

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
        if isinstance(element, note.Note):
            notes.append(element.pitch)
            unique_notes.add(element.pitch)
        elif isinstance(element, chord.Chord):
            c = '.'.join(str(n) for n in element.normalOrder)
            notes.append(c)
            unique_notes.add(c)

    songs.append(notes)

num_songs = len(songs)

# map note/chord names to a unique identifier
note_to_int = dict((note, number) for number, note in enumerate(unique_notes))

# number of different notes in dataset
n_vocab = len(unique_notes)

d_in = []

for i in range(0, len(songs)):
    current_input = []
    # wrap note id in array brackets
    for j in range(0, len(songs[i])):
        current_input.append(note_to_int[songs[i][j]])
    d_in.append(current_input)

n_patterns = len(d_in)

# reshape input so it's compatible with lstm layers
# basically we need to have each note be its own column vector to be inputted
#  into the discriminator
for i in range(len(d_in)):
    d_in[i] = np.array(d_in[i]).reshape((1, len(d_in[i]), 1)) / float(n_vocab)

from classical_nn import am, dm, gm

# expected output for each song
d_out = np.zeros(shape=(1, 1))

#dm.fit(d_in, d_out, batch_size=1, epochs=1, verbose=1)
for di in d_in:
    dm.train_on_batch(di, d_out)
    print(dm.predict(di, batch_size=1))
