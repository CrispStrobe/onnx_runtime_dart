# TabCNN → ONNX (audio→guitar-tablature) reproduction pipeline

Trains the vanilla TabCNN (Wiggins & Kim, ISMIR 2019) on GuitarSet and exports
`tabcnn.onnx` (per-string LogSoftmax head) + `tabcnn-cqt.bin` (192-bin CQT
filterbank) — the audio arm of CometBeat's guitar-tab feature. Published on the
`models-v1` release; consumed by CometBeat's `tab_emission_decoder.dart`.

## Provenance / licence
- Data: **GuitarSet, CC BY 4.0** (`zenodo.org/records/3371780`, `audio_mono-mic`
  + `annotation`). Derived weights redistributable **with attribution**.
- Architecture: andywiggins/tab-cnn (code-only). This is the **vanilla** model;
  the GuitarProFX-augmented weights (DAFx-24) aren't public.

## Steps (needs TF 2.x + tf2onnx + librosa + jams + scipy)
1. `python preprocess.py <GuitarSet-flat-dir> features/`
   — exact CQT: peak-normalize → resample 22050 → `|cqt|` (hop 512, n_bins 192,
     bins_per_octave 24, fmin C1); labels = per-string fret classes (0=closed).
2. `EPOCHS=4 VAL_GUITARIST=5 python train.py features/ tabcnn.onnx`
   — modern-Keras TabCNN, per-string softmax-CE, Adadelta; per-epoch checkpoints
     (`CKPT_DIR`, `RESUME=1`); exports the LogSoftmax-head ONNX.
3. `python gen_cqt_blob.py tabcnn-cqt.bin`
   — 192-bin filterbank in CometBeat's `CqtFilterBank` format via librosa's exact
     `__vqt_filter_fft`; **asserts median magnitude ratio ≈ 1** vs `librosa.cqt`.

## Verified (this port)
- Export runs on pure-Dart `onnx_runtime_dart`; runtime parity vs onnxruntime:
  **240/240 per-string argmax, max|Δlogprob| 2.67e-5**.
- CQT blob vs `librosa.cqt`: **cosine 0.999947, median magnitude ratio 0.9999**.
- Held-out (guitarist-5) **tablature F1 0.745** (paper 0.748).

## IO (pin in the provider)
`input: float32[N,192,9,1]` → `output: float32[N,6,21]` per-string log-probs;
class 0 = closed, class k = fret k−1; frame hop 512/22050 = 0.023220 s.
