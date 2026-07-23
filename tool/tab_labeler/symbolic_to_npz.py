# Fold a symbolic corpus (bin/tab_corpus.dart JSON — GP/MusicXML fingering) into
# the labeler training set: window/encode each song's note-columns exactly like
# extract.py, and APPEND to a base GuitarSet .npz's TRAIN split (val is left
# untouched — it stays the honest held-out GuitarSet player). Output is a merged
# .npz ready for train.py.
#
#   python symbolic_to_npz.py corpus.json base.npz out.npz
import json
import sys

import numpy as np

CORPUS = sys.argv[1]
BASE = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "labeler_merged.npz"

# Must match extract.py.
OPEN = [64, 59, 55, 50, 45, 40]  # decoder order (idx 0 = high e)
MAXFRET, WINDOW, HALF = 19, 9, 4
PLO, PHI = 40, 88
PBINS = PHI - PLO + 1


def encode(pitches):
    v = np.zeros(PBINS, np.uint8)
    for m in pitches:
        if PLO <= m <= PHI:
            v[m - PLO] = 1
    return v


def emit(columns, human, Xs, Ys):
    enc = [encode(c) for c in columns]
    for c in range(len(columns)):
        win = np.zeros((PBINS, WINDOW), np.uint8)
        for w in range(WINDOW):
            k = c - HALF + w
            if 0 <= k < len(columns):
                win[:, w] = enc[k]
        y = np.zeros(6, np.int64)
        for s, fret in human[c]:
            if 0 <= s < 6 and 0 <= fret <= MAXFRET:
                y[s] = fret + 1
        Xs.append(win[:, :, None])
        Ys.append(y)


def main():
    songs = json.load(open(CORPUS))
    Xs, Ys = [], []
    for song in songs:
        emit(song["columns"], song["human"], Xs, Ys)
    Xsym = np.asarray(Xs, np.uint8)
    Ysym = np.asarray(Ys, np.int64)
    print(f"symbolic: {len(songs)} parts → {Xsym.shape[0]} columns")

    base = np.load(BASE)
    Xtr = np.concatenate([base["Xtr"], Xsym]) if len(Xsym) else base["Xtr"]
    Ytr = np.concatenate([base["Ytr"], Ysym]) if len(Ysym) else base["Ytr"]
    np.savez_compressed(OUT, Xtr=Xtr, Ytr=Ytr, Xva=base["Xva"], Yva=base["Yva"])
    print(f"merged train {base['Xtr'].shape[0]} + {Xsym.shape[0]} = {Xtr.shape[0]} "
          f"| val {base['Xva'].shape[0]} (unchanged) → {OUT}")


if __name__ == "__main__":
    main()
