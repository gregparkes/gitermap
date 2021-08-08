"""Processes potential audio sounds if 'simpleaudio' is installed"""

import itertools as it

from ._utils import is_simpleaudio_installed


def _note_hz(n, A_4=440):
    import numpy as np
    return A_4 * np.power(np.power(2, 1. / 12.), n)


def _notes_flat():
    return "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"


def _notes_sharp():
    return "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"


def _major_arp():
    return {
        "C": ["E", "G"],
        "D": ["F#", "A"],
        "E": ["G#", "B"],
        "F": ["A", "C"],
        "G": ["B", "D"],
        "A": ["C#", "E"],
        "B": ["D#", "F#"],
        "Db": ["F", "Ab"],
        "Eb": ["G", "Bb"],
        "F#": ["A#", "Db"],
        "Ab": ["C", "Eb"],
        "Bb": ["D", "F"],
        # also map equivalents e.g F#=Gb
        "C#": ["F", "Ab"],
        "D#": ["G", "Bb"],
        "Gb": ["A#", "Db"],
        "G#": ["C", "Eb"],
        "A#": ["D", "F"],
    }


def _minor_arp():
    return {
        "C": ["Eb", "G"],
        "D": ["F", "A"],
        "E": ["G", "B"],
        "F": ["Ab", "C"],
        "G": ["Bb", "D"],
        "A": ["C", "E"],
        "B": ["D", "F#"],
        "C#": ["E", "G#"],
        "Eb": ["Gb", "Bb"],
        "F#": ["A", "C#"],
        "G#": ["B", "D#"],
        "Bb": ["Db", "F"],
        # also map equivalents e.g F#=Gb
        "Db": ["E", "G#"],
        "D#": ["Gb", "Bb"],
        "Gb": ["A", "C#"],
        "Ab": ["B", "D#"],
        "A#": ["Db", "F"],
    }


def _get_arp(note='C', key='major', level=4):
    arpeg = _major_arp() if key == 'major' else _minor_arp()
    prog = arpeg[note]
    li, lj = ["_%s" % str(i) for i in (level, level + 1)]
    return [note + li, prog[0] + li, prog[1] + li, note + lj]


def _notepack():
    # map together 'notes' with hertz frequency
    import numpy as np
    # lambda func
    append_n = lambda x: x[1] + "_" + str(x[0])
    hz = _note_hz(n=np.arange(-57, 51, 1))
    note_range_flat = list(map(append_n, it.product(range(9), _notes_flat())))
    note_range_sharp = list(map(append_n, it.product(range(9), _notes_sharp())))

    return {**dict(zip(note_range_flat, hz)), **dict(zip(note_range_sharp, hz))}


def _produce_audio(notes, seconds=2, fs=44100):
    import numpy as np
    notepack = _notepack()
    fz = [notepack[a] for a in notes]
    # create a timeloop
    sec_scaled = seconds / len(notes)
    t = np.linspace(0, sec_scaled, int(sec_scaled * fs), False)
    # create sine waves
    sine = np.hstack([np.sin(n * t * 2 * np.pi) for n in fz])
    # join together and move note into 16-bit range
    audio = sine * (np.power(2, 15) - 1) / np.max(np.abs(sine))
    # cast as 16-bit and return
    return audio.astype(np.int16)


def _play_audio(audio, fs=44100):
    # plays the audio
    if is_simpleaudio_installed(False):
        import simpleaudio as sa
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        # wait for playback to finish
        play_obj.wait_done()


def play_arpeggio(note="C", key="major", level=4):
    """plays the arpeggio given a note and key"""
    arp = _get_arp(note=note, key=key, level=level)
    # get audio
    aud = _produce_audio(arp)
    # play
    _play_audio(aud)
