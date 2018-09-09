from music21 import converter, instrument, note, chord, stream, duration

def create_midi(note_data, filename=None):
    s = stream.Stream()
    for sublist in note_data:
        n = note.Note(sublist[0],
            type=duration.convertQuarterLengthToType(sublist[2]))
        n.offset = sublist[1]
        s.append(n)
        print(n)
    print(s)

    if filename is None:
        s.show('midi')
    else:
        s.write("midi", filename)
