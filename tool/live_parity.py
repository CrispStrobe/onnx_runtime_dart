#!/usr/bin/env python3
"""Runs a real model through native onnxruntime on deterministic inputs and
writes <out.json> with {"inputs": ..., "expected": ...} in the same tensor
JSON format as the fixtures. Compare with:  dart run tool/live_parity.dart

Usage: .venv/bin/python tool/live_parity.py <model.onnx> <out.json> [seq]
"""
import json
import sys

import numpy as np
import onnxruntime as ort


def tensor_json(a: np.ndarray) -> dict:
    if a.dtype in (np.float32, np.float64):
        return {"dtype": "float32", "shape": list(a.shape),
                "data": [float(v) for v in a.astype(np.float32).ravel()]}
    return {"dtype": "int64", "shape": list(a.shape),
            "data": [int(v) for v in a.astype(np.int64).ravel()]}


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    feed = {}
    for inp in sess.get_inputs():
        # Dynamic dims: batch-like -> 1; time/sequence-like (by name, or the
        # trailing axis) -> seq.
        nd = len(inp.shape)
        def resolve(i, d):
            if isinstance(d, int) and d > 0:
                return d
            if i == 0:
                return 1  # leading dynamic dim is batch-like, always
            if isinstance(d, str):
                dl = d.lower()
                if "channel" in dl:
                    # RGB by default; diffusion latents are 4-channel.
                    return 4 if "latent" in inp.name.lower() else 3
                if "feature" in dl or "mel" in dl:
                    return 80  # mel-bin count (whisper-style encoders)
                if any(t in dl for t in ("height", "width", "frame", "seq",
                                         "time", "len")):
                    return seq
            # Rank-4 tensors are image-like: interior dynamic dims are
            # spatial, not batch.
            return seq if (i == nd - 1 or nd == 4) else 1
        dims = [resolve(i, d) for i, d in enumerate(inp.shape)]
        if inp.name == "sr":  # sample-rate scalar (VAD-style audio models)
            feed[inp.name] = np.array(16000, dtype=np.int64)
        elif "size" in inp.name.lower() and inp.type == "tensor(float)":
            # image-size vectors (SAM's orig_im_size): plausible dimensions
            feed[inp.name] = np.full(dims, 512, dtype=np.float32)
        elif "label" in inp.name.lower() and inp.type == "tensor(float)":
            n = int(np.prod(dims))
            feed[inp.name] = (np.arange(n) % 2).reshape(dims).astype(
                np.float32)
        elif ("state" in inp.name or "hidden" in inp.name
              or inp.name in ("h0", "c0")):
            feed[inp.name] = np.zeros(dims, np.float32)
        elif "len" in inp.name.lower():  # sequence/waveform lengths: full
            it = np.int32 if inp.type == "tensor(int32)" else np.int64
            feed[inp.name] = np.full(dims, seq, dtype=it)
        elif "elo" in inp.name.lower():  # rating conditioning (maia-style)
            it = np.int32 if inp.type == "tensor(int32)" else np.int64
            feed[inp.name] = np.full(dims, 1500, dtype=it)
        elif inp.type == "tensor(bool)":  # branch flags (merged decoders)
            feed[inp.name] = np.zeros(dims, dtype=bool)
        elif inp.type == "tensor(uint8)":  # raw image bytes
            n = int(np.prod(dims))
            feed[inp.name] = ((np.arange(n) * 37) % 256).reshape(dims).astype(
                np.uint8)
        elif inp.type == "tensor(int32)":
            n = int(np.prod(dims))
            feed[inp.name] = ((np.arange(n) * 37) % 97).reshape(dims).astype(
                np.int32)
        elif inp.type == "tensor(int64)":
            n = int(np.prod(dims))
            if "mask" in inp.name:
                v = np.ones(dims, np.int64)
            elif "type" in inp.name:
                v = np.zeros(dims, np.int64)
            elif "position" in inp.name:
                v = (np.arange(n) % seq).reshape(dims).astype(np.int64)
            else:
                v = (1000 + (np.arange(n) * 37) % 999).reshape(dims)
                v.ravel()[0] = 101
                v.ravel()[-1] = 102
            feed[inp.name] = v.astype(np.int64)
        else:
            n = int(np.prod(dims))
            v = (((np.arange(n) * 2654435761) & 0xFFFF) / 32768.0 - 1.0)
            feed[inp.name] = v.reshape(dims).astype(np.float32)

    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feed)
    json.dump({
        "inputs": {k: tensor_json(v) for k, v in feed.items()},
        "expected": {k: tensor_json(v) for k, v in zip(out_names, outs)},
    }, open(out_path, "w"))
    print(f"wrote {out_path}: inputs {[list(v.shape) for v in feed.values()]}"
          f" -> outputs {[list(v.shape) for v in outs]}")


if __name__ == "__main__":
    main()
