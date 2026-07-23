# Build the SYMBOLIC tab-fingering dataset from GuitarSet annotations
# (Zenodo 3371780, CC BY 4.0). For each note the human picked a (string, fret);
# we learn to score those placements so arrangeTab's Viterbi fingers like a human.
#
# Output mirrors TabCNN's decoder contract EXACTLY so the shipped
# tab_emission_decoder + DP are reused verbatim — only the INPUT differs (symbolic
# note-column pitch-presence, not audio CQT):
#   X : float32[N, PITCH_BINS, WINDOW, 1]  multi-hot pitch presence per column,
#                                          a WINDOW of columns centred on target
#   Y : int64[N, 6]                        per-string class: 0 = silent (string
#                                          not played), class k = fret k-1
#
# String order = the CometBeat decoder's (Tuning.standardGuitar): index 0 = high
# e (E4=64) … index 5 = low E (E2=40). GuitarSet annotates the reverse (idx 0 =
# low E), so decoder_str = 5 - gs_str.
import glob
import os
import sys

import jams
import numpy as np

ANNO = sys.argv[1] if len(sys.argv) > 1 else "anno"
OUT = sys.argv[2] if len(sys.argv) > 2 else "labeler_data.npz"
# Max |semitone| transposition augmentation on the TRAIN split (0 = off). A whole
# passage shifted by k keeps the SAME strings with frets+k — a valid, idiomatic
# fingering — so it teaches position-relativity and fills the fretboard the 6
# GuitarSet players don't cover. Val stays un-augmented (honest held-out player).
AUG = int(os.environ.get("AUG", "0"))
# Which GuitarSet guitarist (00..05) is the held-out val split. Vary it for
# leave-one-guitarist-out CV — player 05 alone can under/over-state generalization.
HOLDOUT = os.environ.get("HOLDOUT", "05")

# Decoder-order open-string MIDIs (index 0 = high e … 5 = low E).
OPEN = [64, 59, 55, 50, 45, 40]
GS_OPEN = [40, 45, 50, 55, 59, 64]  # GuitarSet annotation order (idx 0 = low E)
MAXFRET = 19  # class 20 = fret 19; drop anything higher
WINDOW = 9  # columns of context (centred), like TabCNN's 9-frame window
HALF = WINDOW // 2
PITCH_LO, PITCH_HI = 40, 88  # E2 … E6-ish; PITCH_BINS bins
PITCH_BINS = PITCH_HI - PITCH_LO + 1
ONSET_TOL = 0.045  # notes within 45 ms share a column (a strummed chord)


def columns_for(jam):
    """A time-ordered list of columns; each column is {decoder_str: fret} plus
    the set of MIDI pitches sounded at that attack."""
    events = []  # (onset, decoder_str, fret, midi)
    for gs_str, anno in enumerate(jam.search(namespace="note_midi")):
        ds = 5 - gs_str
        for obs in anno.data:
            midi = int(round(obs.value))
            fret = midi - OPEN[ds]
            if 0 <= fret <= MAXFRET:
                events.append((float(obs.time), ds, fret, midi))
    events.sort(key=lambda e: e[0])
    cols = []
    i = 0
    while i < len(events):
        t0 = events[i][0]
        frets, pitches = {}, set()
        j = i
        while j < len(events) and events[j][0] - t0 <= ONSET_TOL:
            _, ds, fret, midi = events[j]
            # If two attacks land on the same string in the window, keep the
            # first (a real chord seats one note per string anyway).
            frets.setdefault(ds, fret)
            pitches.add(midi)
            j += 1
        cols.append((frets, pitches))
        i = j
    return cols


def encode_column(pitches):
    # uint8 multi-hot (binary) — 4× smaller on disk / in RAM than float32; the
    # trainer casts per batch. Matters on a 16 GB box with augmentation on.
    v = np.zeros(PITCH_BINS, np.uint8)
    for m in pitches:
        if PITCH_LO <= m <= PITCH_HI:
            v[m - PITCH_LO] = 1
    return v


def shifted(cols, k):
    """Transpose every column by k semitones (same strings, frets+k). Returns
    None if any note would leave [0, MAXFRET] — a uniform shift keeps the
    fingering valid and idiomatic."""
    out = []
    for frets, pitches in cols:
        nf = {}
        for ds, fret in frets.items():
            f2 = fret + k
            if not (0 <= f2 <= MAXFRET):
                return None
            nf[ds] = f2
        out.append((nf, {m + k for m in pitches}))
    return out


def emit(cols, Xs, Ys):
    enc = [encode_column(p) for (_, p) in cols]
    for c in range(len(cols)):
        win = np.zeros((PITCH_BINS, WINDOW), np.uint8)
        for w in range(WINDOW):
            k = c - HALF + w
            if 0 <= k < len(cols):
                win[:, w] = enc[k]
        y = np.zeros(6, np.int64)  # 0 = silent
        for ds, fret in cols[c][0].items():
            y[ds] = fret + 1  # class k = fret k-1
        Xs.append(win[:, :, None])
        Ys.append(y)


def main():
    files = sorted(glob.glob(os.path.join(ANNO, "*.jams")))
    Xtr, Ytr, Xva, Yva = [], [], [], []
    ncols = 0
    for f in files:
        val = os.path.basename(f)[:2] == HOLDOUT
        cols = columns_for(jams.load(f))
        ncols += len(cols)
        if val:
            emit(cols, Xva, Yva)
        else:
            emit(cols, Xtr, Ytr)
            for k in range(-AUG, AUG + 1):
                if k == 0:
                    continue
                sh = shifted(cols, k)
                if sh is not None:
                    emit(sh, Xtr, Ytr)
    Xtr = np.asarray(Xtr, np.uint8)
    Ytr = np.asarray(Ytr, np.int64)
    Xva = np.asarray(Xva, np.uint8)
    Yva = np.asarray(Yva, np.int64)
    np.savez_compressed(OUT, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
    print(f"{len(files)} files, {ncols} columns")
    print(f"train X {Xtr.shape} Y {Ytr.shape} | val X {Xva.shape} Y {Yva.shape}")
    # Sanity: class balance (how often each string is played vs silent).
    played = (Ytr > 0).mean(axis=0)
    print("train per-string played-rate (0=high e … 5=low E):",
          np.round(played, 3).tolist())
    print("mean notes/column (train):", round(float((Ytr > 0).sum(1).mean()), 2))


if __name__ == "__main__":
    main()
