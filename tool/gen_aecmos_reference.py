#!/usr/bin/env python
"""Generates test/aecmos_reference.json for test/aecmos_pipeline_test.dart.

Synthesizes a deterministic 4-second echo-cancellation scene at 16 kHz
(far-end sum of sines, mic = delayed+attenuated echo + near-end tone bursts +
noise, enh = mic with most of the echo removed), then replicates the AECMOS
reference pipeline (microsoft/AEC-Challenge AECMOS_local/aecmos.py) end to end
with librosa + onnxruntime on the 16 kHz scenario-marker model, and records:

  - the three float32 signals,
  - the normalized mel features of the mic signal (no scenario markers),
  - the (echo_mos, other_mos) scores for talk types 'st', 'nst', 'dt'.

Run from the repo root:  .venv/bin/python tool/gen_aecmos_reference.py
Requires the model at ~/.cache/onnx_runtime_dart_models/aecmos_1663915512.onnx.
"""
import json
import os
import sys

import librosa
import numpy as np
import onnxruntime as ort

MODEL_NAME = "aecmos_1663915512.onnx"
MODEL_PATH = os.path.expanduser(
    os.path.join("~", ".cache", "onnx_runtime_dart_models", MODEL_NAME))

SR = 16000
DFT_SIZE = 512
HIDDEN_SIZE = (4, 1, 64)
HOP_FRACTION = 0.5
MAX_LEN = 20  # seconds
DURATION = 4  # seconds


def synthesize():
    rng = np.random.default_rng(20260717)
    n = SR * DURATION
    t = np.arange(n) / SR

    # Far-end reference (loopback): sum of sines with slow amplitude modulation.
    lpb = (0.30 * np.sin(2 * np.pi * 320 * t)
           + 0.20 * np.sin(2 * np.pi * 725 * t + 0.7)
           + 0.10 * np.sin(2 * np.pi * 1310 * t + 1.9))
    lpb *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t)

    # Echo path: 50 ms delay, attenuated.
    delay = 800
    echo = np.zeros(n)
    echo[delay:] = 0.35 * lpb[:-delay]

    # Near-end "speech": 440/660 Hz tone bursts gated by a 2 Hz envelope.
    env = np.clip(np.sin(2 * np.pi * 2.0 * t), 0.0, None) ** 2
    near = env * (0.25 * np.sin(2 * np.pi * 440 * t)
                  + 0.12 * np.sin(2 * np.pi * 660 * t + 0.3))

    noise = 0.004 * rng.standard_normal(n)

    mic = echo + near + noise
    enh = mic - 0.93 * echo
    return (lpb.astype(np.float32), mic.astype(np.float32),
            enh.astype(np.float32))


def mel_transform(sample):
    mel_spec = librosa.feature.melspectrogram(
        y=sample, sr=SR, n_fft=DFT_SIZE + 1,
        hop_length=int(HOP_FRACTION * DFT_SIZE), n_mels=160)
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def run(session, talk_type, lpb_sig, mic_sig, enh_sig):
    assert len(lpb_sig) == len(mic_sig) == len(enh_sig)
    seg = MAX_LEN * SR
    lpb_sig, mic_sig, enh_sig = lpb_sig[:seg], mic_sig[:seg], enh_sig[:seg]

    lpb_sig = mel_transform(lpb_sig)
    mic_sig = mel_transform(mic_sig)
    enh_sig = mel_transform(enh_sig)

    assert talk_type in ("nst", "st", "dt")
    ne_st = 1 if talk_type == "nst" else 0
    fe_st = 1 if talk_type == "st" else 0

    cols = mic_sig.shape[1]
    mic_sig = np.concatenate(
        (mic_sig, np.ones((20, cols)) * (1 - fe_st), np.zeros((20, cols))))
    lpb_sig = np.concatenate(
        (lpb_sig, np.ones((20, cols)) * (1 - ne_st), np.zeros((20, cols))))
    enh_sig = np.concatenate(
        (enh_sig, np.ones((20, cols)), np.zeros((20, cols))))

    feats = np.stack((lpb_sig, mic_sig, enh_sig)).astype(np.float32)
    feats = np.expand_dims(feats, axis=0)

    h0 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    result = session.run([], {"input": feats, "h0": h0})[0]
    return float(result[0]), float(result[1])


def intermediates(lpb, mic):
    """Stage-by-stage dumps of the mic-signal mel pipeline plus the 'st'
    marker/stacking stages, so the Dart test can pinpoint the first diverging
    stage. Derived with the same librosa calls the pipeline uses; a manual
    replica is asserted against librosa to pin down padding/window semantics
    (librosa >= 0.10 pads with mode='constant', i.e. zeros)."""
    n_fft = DFT_SIZE + 1
    hop = int(HOP_FRACTION * DFT_SIZE)

    win = librosa.filters.get_window("hann", n_fft, fftbins=True)
    pad = n_fft // 2
    padded = np.pad(mic.astype(np.float64), pad, mode="constant")

    stft = librosa.stft(y=mic, n_fft=n_fft, hop_length=hop)
    power = np.abs(stft) ** 2  # [257, frames]

    # Cross-check the manual frame-0 replica against librosa.
    manual0 = np.abs(np.fft.rfft(padded[:n_fft] * win, n=n_fft)) ** 2
    assert np.allclose(manual0, power[:, 0], rtol=1e-4, atol=1e-10), \
        "librosa stft semantics drifted (padding/window?)"

    mel_fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=160)
    mel_spec = librosa.feature.melspectrogram(
        y=mic, sr=SR, n_fft=n_fft, hop_length=hop, n_mels=160)
    assert np.allclose(mel_fb @ power, mel_spec, rtol=1e-4, atol=1e-12), \
        "melspectrogram no longer equals mel_fb @ |stft|^2"

    db_ref = float(mel_spec.max())
    db = librosa.power_to_db(mel_spec, ref=np.max)
    feats = ((db + 40) / 40).T  # [frames, 160]

    # 'st' marker stage: aecmos.py APPENDS 20 one-frames then 20 zero-frames
    # in mel-feature space (there is no signal-space marker). For 'st',
    # ne_st=0 so the lpb marker block is ones then zeros.
    lpb_feats = mel_transform(lpb)
    frames = lpb_feats.shape[0]
    lpb_marked = np.concatenate(
        (lpb_feats, np.ones((20, 160)), np.zeros((20, 160))))
    marker_flat = lpb_marked[frames:].flatten()

    return {
        "hann_window": win.tolist(),
        "padded_head": padded[:600].astype(np.float32).tolist(),
        "frame0_power": power[:, 0].astype(np.float64).tolist(),
        "mel_fb_rowsums": mel_fb.sum(axis=1).astype(np.float64).tolist(),
        "mel_fb_row0": mel_fb[0].astype(np.float64).tolist(),
        "mel_frame0": mel_spec[:, 0].astype(np.float64).tolist(),
        "db_ref": db_ref,
        "db_frame0": db[:, 0].astype(np.float64).tolist(),
        "features_frame0_norm": feats[0].astype(np.float32).tolist(),
        "marker_rows_start": frames,
        "marker_head_st": marker_flat[:400].tolist(),
    }


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"model not found: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH)

    lpb, mic, enh = synthesize()
    mic_feats = mel_transform(mic).astype(np.float32)

    inter = intermediates(lpb, mic)
    # First 160 values of the [1, 3, frames, 160] tensor for 'st' == the lpb
    # channel's normalized frame 0.
    inter["model_input_slice"] = \
        mel_transform(lpb).astype(np.float32)[0].tolist()

    scores = {}
    for talk_type in ("st", "nst", "dt"):
        echo_mos, other_mos = run(session, talk_type, lpb, mic, enh)
        scores[talk_type] = {"echo_mos": echo_mos, "other_mos": other_mos}
        print(f"{talk_type}: echo_mos={echo_mos:.6f} other_mos={other_mos:.6f}")

    out = {
        "model": MODEL_NAME,
        "sampling_rate": SR,
        "dft_size": DFT_SIZE,
        "signals": {
            "lpb": lpb.tolist(),
            "mic": mic.tolist(),
            "enh": enh.tolist(),
        },
        "mic_features": {
            "shape": list(mic_feats.shape),
            "data": mic_feats.flatten().tolist(),
        },
        "intermediates": inter,
        "scores": scores,
    }
    path = os.path.join(os.path.dirname(__file__), "..", "test",
                        "aecmos_reference.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {os.path.normpath(path)} "
          f"({os.path.getsize(path) / 1e6:.1f} MB), "
          f"features shape {mic_feats.shape}")


if __name__ == "__main__":
    main()
