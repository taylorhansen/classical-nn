from music21 import converter, instrument, note, chord, stream, duration

# takes in chords and notes, outputs a midi or plays it if no file name is given
def create_midi(note_data, filename=None):
    s = stream.Stream()
    for sublist in note_data:
        if type(sublist[0]) is tuple:
            c = chord.Chord(sublist[0], quarterLength=sublist[2])
            c.offset = sublist[1]
            s.append(c)
        else:
            n = note.Note(sublist[0],
                type=duration.convertQuarterLengthToType(sublist[2]))
            n.offset = sublist[1]
            s.append(n)

    if filename is None:
        s.show('midi')
    else:
        s.write("midi", filename)
