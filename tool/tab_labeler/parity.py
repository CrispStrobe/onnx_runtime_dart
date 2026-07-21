# ONNX export sanity: run tab-labeler.onnx through onnxruntime on the real val
# parity fixture and compare to the torch reference logprobs (saved by train.py).
# Confirms the export is faithful before the pure-Dart onnx_runtime_dart check.
import sys

import numpy as np
import onnxruntime as ort

ONNX = sys.argv[1] if len(sys.argv) > 1 else "tab-labeler.onnx"
FIX = sys.argv[2] if len(sys.argv) > 2 else "tab-labeler_parity.npz"

fix = np.load(FIX)
X, ref = fix["X"].astype(np.float32), fix["logprobs"]
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
out = sess.run(None, {name: X})[0]

max_abs = float(np.abs(out - ref).max())
# Per-string argmax agreement (the class = fret+1 the decoder reads).
agree = (out.argmax(-1) == ref.argmax(-1)).mean()
print(f"onnxruntime vs torch: max|Δlogprob| {max_abs:.3e}  "
      f"per-string argmax agreement {agree * 100:.2f}%  "
      f"({X.shape[0]} examples × 6 strings)")
print("PASS" if max_abs < 1e-4 and agree > 0.999 else "CHECK")
