from music21 import converter, instrument, note, chord, stream, duration

# takes in chords and notes, outputs a midi or plays it if no file name is given
def create_midi(note_data, filename=None):
    s = stream.Stream()
    for sublist in note_data:
        if type(sublist[0]) is tuple:
            c = chord.Chord(sublist[0],
                    quarterLength=int(sublist[2] * 32) / 32.0)
            c.offset = int(sublist[1] * 32) / 32.0
            s.append(c)
        else:
            n = note.Note(sublist[0],
                type=duration.quarterLengthToClosestType(
                        int(sublist[2] * 32) / 32.0)[0])
            n.offset = int(sublist[1] * 32) / 32.0
            s.append(n)

    if filename is None:
        s.show('midi')
    else:
        s.write("midi", filename)
