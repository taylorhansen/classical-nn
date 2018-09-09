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

# real midi data
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
