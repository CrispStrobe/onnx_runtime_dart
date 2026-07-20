# Generate the 192-bin CQT filterbank blob in CometBeat's CqtFilterBank binary
# format, matching TabCNN's librosa.cqt (sr 22050, hop 512, n_bins 192,
# bins_per_octave 24, fmin C1) via the direct fft_basis form (all bins at base
# sr on a boxcar STFT — the same trick BTC uses; matches librosa.cqt closely).
# Verifies §4: median per-bin magnitude ratio ≈ 1 vs librosa.cqt (NOT just cos).
import struct
import sys

import librosa
import librosa.core.constantq as cq
import numpy as np

SR, HOP, N_BINS, BPO = 22050, 512, 192, 24
FMIN = librosa.note_to_hz("C1")
OUT = sys.argv[1] if len(sys.argv) > 1 else "tabcnn-cqt.bin"

# librosa's exact CQT filterbank (VQT gamma=0). Access the module-dunder builder.
vqt_filter_fft = getattr(cq, "__vqt_filter_fft")
freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BPO)
fft_basis, n_fft, lengths = vqt_filter_fft(
    SR, freqs, 1, 1, 0.01, window="hann", gamma=0
)
fft_basis = np.asarray(fft_basis.todense())  # [n_bins, n_fft] complex (full)
n_freq = n_fft // 2 + 1
fb = fft_basis[:, :n_freq]  # positive-freq half
print("n_fft", n_fft, "n_freq", n_freq, "fb", fb.shape)


def boxcar_stft(y):
    # librosa __cqt_response uses a 'ones' (boxcar) window STFT.
    D = librosa.stft(y, n_fft=n_fft, hop_length=HOP, window="ones",
                     pad_mode="constant")
    return D  # [n_freq, frames]


def cqt_from_blob(y):
    D = boxcar_stft(y)
    C = fb.conj() @ D if False else fb @ D  # response = fft_basis . STFT
    # librosa scales the response by sqrt(lengths) then... match empirically:
    return np.abs(C) / np.sqrt(lengths)[:, None]


def main():
    # --- band the filterbank + write the blob ---
    lo = np.zeros(N_BINS, np.int32)
    hi = np.zeros(N_BINS, np.int32)
    re_bands, im_bands = [], []
    for k in range(N_BINS):
        row = fb[k]
        nz = np.where(np.abs(row) > 0)[0]
        a, b = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)
        lo[k], hi[k] = a, b
        re_bands.append(row[a:b].real.astype(np.float32))
        im_bands.append(row[a:b].imag.astype(np.float32))
    re = np.concatenate(re_bands) if re_bands else np.zeros(0, np.float32)
    im = np.concatenate(im_bands) if im_bands else np.zeros(0, np.float32)

    with open(OUT, "wb") as f:
        f.write(struct.pack("<4i", N_BINS, n_fft, n_freq, HOP))
        f.write(struct.pack("<2f", 0.0, 1.0))  # mean/std unused (raw magnitude)
        f.write(lengths.astype(np.float32).tobytes())
        f.write(lo.tobytes())
        f.write(hi.tobytes())
        f.write(re.tobytes())
        f.write(im.tobytes())
    print("wrote", OUT, "bands total", len(re))

    # --- §4 parity vs librosa.cqt on a real GuitarSet clip ---
    import glob
    from scipy.io import wavfile
    w = sorted(glob.glob("GuitarSet/*_mic.wav"))[0]
    sr0, data = wavfile.read(w)
    data = librosa.util.normalize(data.astype(float))
    data = librosa.resample(data, orig_sr=sr0, target_sr=SR)
    ref = np.abs(librosa.cqt(data, hop_length=HOP, sr=SR, n_bins=N_BINS,
                             bins_per_octave=BPO))  # [192, frames]
    mine = cqt_from_blob(data)  # [192, frames]
    T = min(ref.shape[1], mine.shape[1])
    ref, mine = ref[:, :T], mine[:, :T]
    cos = float((ref * mine).sum() /
                (np.linalg.norm(ref) * np.linalg.norm(mine) + 1e-12))
    mask = ref > 1e-4
    ratio = float(np.median(mine[mask] / ref[mask]))
    print(f"PARITY  cosine={cos:.6f}  median_magnitude_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
