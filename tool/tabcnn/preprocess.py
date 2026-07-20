# Preprocess GuitarSet (flat dir) into per-clip CQT features + per-string fret
# labels, EXACTLY matching andywiggins/tab-cnn's TabDataReprGen (the frozen CQT
# spec): mic audio -> peak-normalize waveform -> resample 22050 -> |cqt| (hop
# 512, n_bins 192, bins_per_octave 24, fmin default C1) -> [frames,192].
import glob
import os
import sys

import jams
import librosa
import numpy as np
from scipy.io import wavfile

SR = 22050
HOP = 512
N_BINS = 192
BPO = 24
STRING_MIDI = [40, 45, 50, 55, 59, 64]
HIGHEST_FRET = 19
NUM_CLASSES = HIGHEST_FRET + 2  # 21


def cqt_repr(wav_path):
    sr0, data = wavfile.read(wav_path)
    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(data, orig_sr=sr0, target_sr=SR)
    c = np.abs(librosa.cqt(data, hop_length=HOP, sr=SR, n_bins=N_BINS,
                           bins_per_octave=BPO))
    return np.swapaxes(c, 0, 1)  # [frames, 192]


def labels_for(jams_path, n_frames):
    jam = jams.load(jams_path)
    times = librosa.frames_to_time(range(n_frames), sr=SR, hop_length=HOP)
    lab = []
    for s in range(6):
        anno = jam.annotations["note_midi"][s]
        samp = anno.to_samples(times)
        col = []
        for i in range(n_frames):
            if samp[i] == []:
                col.append(-1)
            else:
                col.append(int(round(samp[i][0]) - STRING_MIDI[s]))
        lab.append(col)
    lab = np.swapaxes(np.array(lab), 0, 1)  # [frames, 6]
    # correct_numbering: n+1; clamp <0 or >19 -> 0. one-hot 21.
    out = np.zeros((n_frames, 6, NUM_CLASSES), np.float32)
    for i in range(n_frames):
        for s in range(6):
            n = lab[i, s] + 1
            if n < 0 or n > HIGHEST_FRET:
                n = 0
            out[i, s, n] = 1.0
    return out


def main():
    GS, OUT = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(GS, "*_mic.wav")))
    print("clips:", len(wavs))
    for i, w in enumerate(wavs):
        cid = os.path.basename(w)[:-len("_mic.wav")]
        outp = os.path.join(OUT, cid + ".npz")
        if os.path.exists(outp):
            continue
        jpath = os.path.join(GS, cid + ".jams")
        if not os.path.exists(jpath):
            print("no jams for", cid)
            continue
        repr_ = cqt_repr(w)
        y = labels_for(jpath, len(repr_))
        np.savez_compressed(outp, repr=repr_.astype(np.float32), labels=y)
        if i % 30 == 0:
            print(f"[{i}/{len(wavs)}] {cid}: {len(repr_)} frames")
    print("DONE preprocess")


if __name__ == "__main__":
    main()
