# tab-labeler → ONNX (symbolic guitar-tab fingering) pipeline

Trains the **symbolic tab labeler** — a small CNN that scores `(string, fret)`
placements for note-columns so a tab arranger's Viterbi fingers like a human — and
exports `tab-labeler.onnx`. It's the score→tab (symbolic) counterpart to the
audio→tab TabCNN in `../tabcnn/`, and shares TabCNN's `[6,21]` per-string
LogSoftmax contract so CometBeat's shipped tab decoder / `arrangeTab` consume both.
Published on the `models-v1` release + HF `cstr/tab-labeler-onnx`; consumed by
CometBeat's `tab_labeler.dart` (`TabPositionModel`).

## Provenance / licence
- **Data: GuitarSet, CC BY 4.0** (`zenodo.org/records/3371780`, `annotation.zip`
  only — the symbolic arm needs the note→string/fret labels, not the audio).
  Derived weights redistributable **with attribution**. No DadaGP.

## Steps (needs torch + numpy + jams + onnxruntime)
1. `python extract.py <anno-dir> labeler_data.npz`
   — parse the 6 per-string `note_midi` annotations → onset-quantised columns;
     labels = per-string fret class (0 = silent, k = fret k-1). String index 0 =
     high e (the decoder order; GuitarSet is the reverse, so `dec = 5 - gs`).
     Held out by guitarist (player 05 = val).
2. `python train.py labeler_data.npz tab-labeler.onnx 40`
   — a CNN over a 9-column pitch-presence window (MIDI 40..88); per-string
     softmax-CE; exports the LogSoftmax-head ONNX + a parity fixture.
3. `python parity.py tab-labeler.onnx tab-labeler_parity.npz`
   — onnxruntime vs torch on real val data (asserts max|Δ| < 1e-4, 100% argmax).
4. `python export_acceptance.py <anno-dir> acceptance.json`
   — held-out (player 05) songs as column sequences + human fingering, for the
     Dart-side `test/tab_labeler_accept_test.dart`.

## Verified (this port)
- Runs on pure-Dart `onnx_runtime_dart`; parity vs onnxruntime **cosine
  1.000000000, max|Δ| 5.7e-6** (`tool/live_parity.dart`).
- Val string+fret agreement vs held-out human **0.76**; ~244 k params, ~1 MB.
- **Acceptance** (behind `arrangeTab`, 60 held-out songs / 8,715 positions):
  human-fingering agreement **56.98% → 78.59% (+21.6 pts)** vs the heuristic.

## IO (pin in the provider)
`input: float32[N,49,9,1]` (49 pitch bins × 9-column window × 1) →
`output: float32[N,6,21]` per-string log-probs; class 0 = silent, class k = fret
k-1; string 0 = high e. Emission score for `(string,fret)` = `output[string][fret+1]`.
