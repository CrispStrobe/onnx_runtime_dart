# onnx_runtime_dart

A small, dependency-light **pure-Dart ONNX inference runtime**. No FFI and no
native `onnxruntime` — the graph is interpreted in plain Dart, so the same code
runs on **every Dart/Flutter target, including the web and WebAssembly**.

It implements the operator set used by **transformer / attention** models
(BERT/XLM-R/MPNet/DeBERTa embedders and rerankers, RoPE / ALiBi variants, Qwen3
decoders), the **convolution / pooling** family (CNN classifiers, detectors,
segmentation and diffusion VAEs), **recurrent** ops (`LSTM`/`GRU`/`RNN`),
**control flow** (`If`/`Loop`/`Scan` with subgraph execution) and **quantized
models** (QDQ, QOperator and int4 `MatMulNBits`). In practice this covers text
embedders, rerankers, seq2seq translation with KV-cache, OCR, object detection,
segmentation, diffusion VAEs, three ASR stacks, TTS and audio scoring — all
verified against native ONNX Runtime (see [Verified models](#verified-models)).
It is still not a complete ONNX runtime — see the operator list below for
exactly what is covered.

## Verified models

Every model below is run **against native ONNX Runtime** (`onnxruntime`, CPU
provider) on deterministic inputs and checked op-for-op; the "parity" column
is the metric the architecture actually admits (see
[Parity criteria](#parity-criteria)). On top of that, every operator has
per-op parity fixtures generated from ONNX Runtime (`test/fixtures/`, built by
`tool/gen_fixtures.py`).

Unless noted, **parity = cosine 1.0, max abs diff ≈ 1e-6–1e-5** (float32
rounding). Repos in the `cstr/` namespace are the author's ONNX rehosts;
`†` marks models tested from a local/pre-existing ONNX export (upstream repo
linked).

### Text embeddings & rerankers

| Model | HF repo | Architecture | Parity |
|---|---|---|---|
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | BERT 384d | 1.0 |
| all-MiniLM-L12-v2 | `sentence-transformers/all-MiniLM-L12-v2` | BERT 384d | 1.0 |
| all-mpnet-base-v2 | `sentence-transformers/all-mpnet-base-v2` | MPNet, relative-position buckets | 1.0 |
| bge-small-en-v1.5 | `BAAI/bge-small-en-v1.5` | BERT 384d | 1.0 |
| bge-m3 | `BAAI/bge-m3` | XLM-R 1024d, external-data `Constant` attrs | 1.0 |
| gte-small | `Xenova/gte-small` | BERT 384d | 1.0 |
| gte-base-en-v1.5 | `Alibaba-NLP/gte-base-en-v1.5` | GTE: pre-LN + RoPE + GeGLU | 1.0 |
| gte-modernbert-base | `Alibaba-NLP/gte-modernbert-base` | ModernBERT, global/local sliding-window attn | 1.0 |
| nomic-embed-text-v1.5 | `nomic-ai/nomic-embed-text-v1.5` | NomicBERT, RoPE | 1.0 |
| multilingual-e5-small | `intfloat/multilingual-e5-small` | XLM-R 384d | 1.0 |
| snowflake-arctic-embed-xs | `Snowflake/snowflake-arctic-embed-xs` | BERT CLS | 1.0 |
| granite-embedding-107m | `ibm-granite/granite-embedding-107m-multilingual` | XLM-R | 1.0 |
| jina-embeddings-v2-base-en `†` | `jinaai/jina-embeddings-v2-base-en` | BERT + ALiBi | 1.0 |
| Splade_PP_en_v1 | `Qdrant/Splade_PP_en_v1` | SPLADE sparse-lexical | 1.0 |
| Octen-Embedding-0.6B `†` | `cstr/Octen-Embedding-0.6B-ONNX` | Qwen3 decoder, RoPE, external-data | 1.0 (fp32) |
| ms-marco-MiniLM-L-6-v2 `†` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | BERT cross-encoder | 1.0 |
| mxbai-rerank-xsmall-v1 | `mixedbread-ai/mxbai-rerank-xsmall-v1` | DeBERTa-v2, disentangled attention | 1.0 |
| mxbai-rerank-base-v1 | `mixedbread-ai/mxbai-rerank-base-v1` | DeBERTa-v2 | 1.0 |
| awesome-align | `cstr/awesome-align-onnx` | multilingual BERT word aligner | 1.0 |
| embeddinggemma-300m | `onnx-community/embeddinggemma-300m-ONNX` | **Gemma3**; fused `MultiHeadAttention` + `RotaryEmbedding` + `SimplifiedLayerNormalization` | 1.0 (fp16 export) |
| harrier-270m | `onnx-community/harrier-oss-v1-270m-ONNX` | **Gemma3** decoder; fused **`GroupQueryAttention`** (causal, internal RoPE, GQA grouping) | 1.0 (fp16 export) |
| F2LLM-v2-0.6B `†` | `cstr/F2LLM-v2-0.6B-ONNX` | Qwen3 decoder | int8 dynamic-quant band (0.997) |
| jina-embeddings-v5-small `†` | `jinaai/jina-embeddings-v5-text-small-retrieval` | Qwen3, `IsNaN` attention masking | 1.0 (fp16 export) |
| partitura-jina `†` | jina variant (local export) | BERT/jina | 1.0 |

### Quantized language models (int4 / int8)

| Model | HF repo | Format | Parity |
|---|---|---|---|
| Octen-0.6B int4 `†` | `cstr/Octen-Embedding-0.6B-ONNX` | `MatMulNBits`, 1.8 GB packed | cosine 1.0 (static block quant) |
| llama-nemotron-rerank-1B int4 `†` | `cstr/llama-nemotron-rerank-1b-v2-ONNX` | `MatMulNBits` | logit within 1.7e-5 |
| llama-nemotron-rerank-1B int8 `†` | `cstr/nemotron` int8 (local) | dynamic quant | dynamic-int8 band |
| Octen-0.6B int8 `†` | `cstr/Octen-Embedding-0.6B-ONNX` | dynamic quant, 196 `MatMulInteger` | intrinsic-band (below) |
| zerank-1-small int4 `†` | `cstr/zerank-1-small-ONNX` | int4 + **fp16-compute** regions | fp16-caveat (below) |
| PIXIE-Rune-v1.0 int4 | `cstr/PIXIE-Rune-v1.0-ONNX` | XLM-R, **ONNX-native INT4** QDQ (packed 2/byte) | cosine 1.0 |

### Generative LLMs (autoregressive decoding, KV cache)

| Model | HF repo | Notes | Parity |
|---|---|---|---|
| SmolLM2-135M-Instruct | `HuggingFaceTB/SmolLM2-135M-Instruct` (ONNX export) | Llama-style decoder; 30× fused **`GroupQueryAttention`** with real **`past`/`present` KV cache** (9 query / 3 KV heads), external `RotaryEmbedding` | prefill + decode 1.0 on `logits` **and** `present.*`; 24-step greedy generation token-for-token identical to ORT |
| Qwen2.5-0.5B-Instruct | `onnx-community/Qwen2.5-0.5B-Instruct` (fp16) | **fully decomposed** decoder (no fused ops — RoPE / attention as primitives, 24 layers, 2 KV heads); **graph-level KV cache** via `Concat`/`Slice` | prefill logits cosine 0.99999885, decode 0.99999976 (fp16 band); coherent text generation |

Full autoregressive text generation runs in pure Dart — both the fused-`GroupQueryAttention`
export style and the fully-decomposed one — with `present_key`/`present_value`
feeding straight back as the next step's `past_key`/`past_value`.
`OnnxModel.inputSpecs` reports the decoder's cache shape (layers / KV heads /
head size) so the empty first-step past can be sized generically. `tool/llm_chat.dart`
is a complete **text-in / text-out** demo (byte-level BPE tokenizer loaded
from `tokenizer.json`, ChatML prompt, greedy + temperature/top-k sampling);
`tool/smollm2_generate.dart` is a greedy loop checked token-for-token against ORT.

Two decode-oriented speedups: the GEMM kernel has a single-row (`m==1`) GEMV
path with aligned vector loads (~5× faster decode vs a naive pack-and-tile
kernel, bitwise identical), and `loadOnnxModel(path, lastTokenLogits: true)`
slices the vocab projection to the last position so prefill skips the prompt's
wasted `seq-1` logit rows (1.2×+ less prefill MatMul, last row bit-exact).

### Sequence-to-sequence & OCR

| Model | HF repo | Notes | Parity |
|---|---|---|---|
| NLLB-200-600M (enc + dec + `decoder_with_past`) `†` | `facebook/nllb-200-distilled-600M` (Optimum ONNX export) | full translation loop, KV cache, 256k-vocab logits + all present-KV; `Trilu`, `ScatterND`, external-data `Constant` | 1.0 |
| TrOCR (ViT encoder + text decoder) `†` | `microsoft/trocr-*` (local ONNX export) | image → text | 1.0 |
| CrispTranslator (enc + dec) `†` | NLLB-family, int8 dynamic-quant (local export) | `MatMulInteger`+`DynamicQuantizeLinear`; 256k-vocab logits, KV outputs exact | dynamic-int8 band (enc 0.9999, dec 0.998) |

### Vision

| Model | Source | Task | Parity |
|---|---|---|---|
| MobileNetV2 | ONNX Model Zoo (`mobilenetv2-7`) | classification | 1.0 |
| ResNet18 | ONNX Model Zoo (`resnet18-v1-7`) | classification | 1.0 |
| MobileNetV2 QDQ | ONNX Model Zoo (`mobilenetv2-12-qdq`) | quantized classification | same top-5 (below) |
| SSD-MobileNetV1 | ONNX Model Zoo (`ssd_mobilenet_v1_10`) | detection: uint8 in, preprocessing `Loop`, per-class NMS (`TopK`/`NonZero`/`NonMaxSuppression`) | **all 4 outputs bit-identical** |
| UltraFace RFB-320 | ONNX Model Zoo (`version-RFB-320`) | face detection | 1.0 |
| fast-neural-style candy | ONNX Model Zoo (`candy-9`) | style transfer, `InstanceNorm` + `Upsample` | 1e-4 relative |
| sub-pixel CNN super-resolution | ONNX Model Zoo (`super-resolution-10`) | super-resolution | 1.0 |
| emotion-ferplus | ONNX Model Zoo (`emotion-ferplus-8`) | classification | 1.0 |
| SAM (ViT-H) mask decoder | `Annotation-AI/sam-vit-h-decoder-onnx` | segmentation prompting, 3 outputs | 1.0 |
| TAESD (SD VAE decoder) | `julienkay/taesd` | diffusion VAE, latent `[1,4,64,64]` → 512² | 1.0 (max\|Δ\| 9e-6) |
| Depth-Anything-v2-small | `onnx-community/depth-anything-v2-small` | monocular depth (DINOv2 ViT + **DPT** head: `Resize`×6, `ConvTranspose`); dense `[1,H,W]` depth map | 1.0 (max\|Δ\| 1.5e-5) |

### Speech — ASR, TTS, VAD, scoring, tokenization

| Model | HF / source | Stack | Parity |
|---|---|---|---|
| Whisper-tiny (enc + dec + `decoder_with_past`) | `onnx-community/whisper-tiny` | Transformer ASR, KV cache | dec 1.0; enc 2e-4 rel (below) |
| Moonshine-tiny (enc + merged decoder) | `UsefulSensors/moonshine` | ASR; single top-level `If` decoder, 24-tensor KV cache | 1.0 (encoder-KV exact) |
| Parakeet-TDT 0.6B `†` | NVIDIA Parakeet-TDT 0.6B v3 (ONNX export) | NeMo mel featurizer (`STFT` + float64 weights); RNN-T decoder/joint (`LSTM`+`Split`); int8 conformer encoder | featurizer/decoder 1.0; int8 enc intrinsic-band |
| Kokoro-82M TTS | `onnx-community/Kokoro-82M-v1.0-ONNX` | StyleTTS2 / iSTFT-Net: LSTMs, harmonic sine source, mid-graph `STFT`, 1-D conv/transposed-conv | **log-mel cosine 0.995** (below) |
| Silero VAD | `snakers4/silero-vad` | Conv1D + LSTM + `If` + reflect-`Pad` | 1.0 |
| AECMOS (2 echo-MOS models) | `microsoft/AEC-Challenge` (`AECMOS_local`) | Conv + MaxPool + bidirectional GRU + `ReduceMax`; full scorer in [`example/aecmos/`](example/aecmos/) | 1.0 (max\|Δ\| 1.2e-7) |
| CAM++ speaker embedding `†` | CosyVoice3 (`campplus.onnx`) | x-vector: 225 convs, `ReduceProd` | 1.0 |
| CosyVoice3 speech tokenizer `†` | CosyVoice3 (`speech_tokenizer_v3.onnx`) | discrete speech tokens | **25/25 tokens exactly equal** |
| FastConformer CTC `†` | NeMo FastConformer (local ONNX export) | Conformer + CTC, `LogSoftmax` | 1.0 |
| ECAPA-TDNN language-ID `†` | SpeechBrain ECAPA-TDNN (local ONNX export) | SE-Res2 TDNN, attentive stat pooling | 1.0 |
| Piper (VITS) TTS `†` | rhasspy/piper voices (e.g. `en_US-libritts_r-medium`) | end-to-end VITS, `RandomNormalLike` | 1.0 (noise-scale 0 → deterministic) |

### Games

| Model | HF repo | Notes | Parity |
|---|---|---|---|
| Maia3-5M | `cstr/maia3-onnx-int32` | chess transformer, policy (4352) + WDL value heads, `Einsum` attention | 1.0 |

### Parity criteria

Not every architecture admits bitwise (or even high-cosine) whole-output
parity; the oracle defines what's achievable, so each model is judged by the
right metric:

- **Float models** — cosine 1.0, max abs diff ~1e-5 (or ~2e-4 *relative* for
  very deep encoders like Whisper's, where summation-order alone drifts that
  far; confirmed by running with load-time fusion **on vs off**).
- **QDQ classification (MobileNetV2)** — no bitwise logit parity exists for
  *any* runtime: tiny float-ordering differences flip quantization buckets,
  and ORT's own optimized-vs-unoptimized execution self-agrees only to
  cosine ≈0.993. Criterion: same top-k, cosine inside that band (~0.991). Our
  classification is identical.
- **Static int4 (`MatMulNBits`)** — reproduces *exactly* (cosine 1.0): block
  quantization has no runtime quantization boundaries.
- **Dynamic int8 (`DynamicQuantizeLinear`+`MatMulInteger`)** — transcendental
  ulps amplify across layers, so no independent runtime matches ORT-int8
  bitwise. Criterion: our deviation from ORT-int8 ≪ that export's own
  int8-vs-fp32 quantization error (e.g. Octen 0.96 vs 0.73 pooled; Parakeet
  conformer 0.997 vs 0.63).
- **fp16-compute exports** (e.g. `zerank-1-small` int4, 115 `Cast`-to-fp16
  pairs) — we execute float32 between cast points (rounding *through* fp16 at
  each cast); ORT computes those regions in true half precision, so results
  agree only to the model's fp16 sensitivity (~2%). Ours is the more precise
  side, not bit-matching.
- **TTS vocoders (Kokoro)** — whole-waveform cosine is meaningless: LSTM
  recurrence and sine-phase integration chaos-amplify float ulps for any
  implementation. Verified instead by (a) components in isolation
  (feed-forward path bitwise, harmonic source cosine 0.99996, first 2000
  samples 0.9998) and (b) **log-mel spectrogram cosine 0.995** on the audio.

Execution uses a packed, register-tiled **`Float32x4` SIMD GEMM kernel** on
native targets (scalar fallback on web), im2col convolution, load-time
constant folding and weight prepacking; see `BENCHMARKS.md` for current
numbers vs native ONNX Runtime.

On native targets you can additionally spread large matmuls across an
**isolate worker pool** — each worker permanently owns a column slice of the
big weights, so per-run messages carry only activations:

```dart
await model.parallelize(workers: 4);        // once, after loading
final out = await model.runAsync(inputs, ['output']); // bitwise == run()
model.dispose();                            // shuts the workers down
```

## Install

```yaml
dependencies:
  onnx_runtime_dart: ^0.3.4
```

## Usage

```dart
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart'; // native only (dart:io)

void main() {
  // Resolves companion external-data files (large models) automatically.
  final model = loadOnnxModel('model.onnx');

  final out = model.run(
    {'input': Tensor.float(Float32List.fromList([1, 2, 3, 4]), [1, 4])},
    ['output'],
  );

  print(out['output']!.asFloatList());
}
```

On the web (no `dart:io`), use `OnnxModel.fromBytes(bytes)` directly; for
external-data models pass an `externalData` resolver.

Weights load from float32, float16, float64, int32, int64, bool, and int8 /
uint8 (kept in compact 1-byte storage) tensors, plus 4-bit block-quantized
`MatMulNBits` weights (kept packed) — inline or from a companion `.onnx.data`
file (read on demand, so multi-GB models don't load into memory all at once).

See [`example/onnx_runtime_dart_example.dart`](example/onnx_runtime_dart_example.dart) for a
self-contained, runnable graph built with the protobuf types.

## Tokenizers (end-to-end text in / text out)

Two pure-Dart tokenizers load a HuggingFace `tokenizer.json` directly, so text
models are usable with no external tokenizer — the same code runs on web/WASM:

- **`WordPieceTokenizer`** — the BERT `WordPiece` pipeline (`BertNormalizer`
  incl. accent-stripping via a precomputed NFD table, `BertPreTokenizer`, greedy
  `##` continuation, `[CLS]…[SEP]`). Covers the whole embedder/reranker family
  (BERT / MiniLM / MPNet / GTE / E5 / mxbai …).
- **`UnigramTokenizer`** — SentencePiece Unigram (`Metaspace` + Viterbi over
  the vocab log-probs + `<s>…</s>`), with the `Precompiled` NFKC normalizer
  approximated by a per-codepoint compatibility fold. Covers the multilingual
  XLM-RoBERTa family (multilingual-e5 / bge-m3 / paraphrase-multilingual …).
- **`BpeTokenizer`** — byte-level BPE (GPT-2 / Qwen / Llama-BPE): the GPT-2
  pre-tokenization regex, `bytes_to_unicode`, rank-ordered merges, added/special
  tokens. Drives the generative decoders.

All three are validated for **exact** id-match against the reference
`tokenizers` library (across cased/uncased, accented, CJK, Cyrillic, Greek and
full-width text). Full **text → embedding** pipelines match
`sentence-transformers` at **cosine 1.0** end-to-end (`tool/embed_e2e.dart`):
WordPiece → all-MiniLM-L6, and Unigram → multilingual-e5-small (English,
German, Japanese, Russian). `tool/llm_chat.dart` is the generative
text-in/text-out counterpart.

```dart
final tok = WordPieceTokenizer.fromFile('tokenizer.json');
final ids = tok.encode('Machine learning in pure Dart.'); // [101, 3698, ... , 102]
```

## Supported operators

- **Math:** `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Sqrt`, `Reciprocal`, `Abs`,
  `Neg`, `Exp`, `Log`, `Relu`, `Erf`, `Sigmoid`, `Tanh`, `Cos`, `Sin`, `Clip`,
  `Sign`.
- **Compare / logic:** `Equal`, `Greater`, `Less`, `GreaterOrEqual`,
  `LessOrEqual`, `And`, `Or`, `Not`, `Where`, `Max`, `Min`.
- **Shape / index:** `Shape`, `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`,
  `Concat`, `Gather`, `GatherND`, `GatherElements`, `Expand`, `Slice`, `Range`,
  `Cast`, `Constant`, `ConstantOfShape`.
- **Reduce / linalg:** `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`,
  `ReduceProd`, `CumSum`, `Softmax`, `LayerNormalization`, `MatMul`, `Gemm`,
  `Einsum` (general 1/2-operand equations without ellipsis; the transformer-
  hot patterns keep specialized kernels), `ArgMax`, `ArgMin`, `LogSoftmax`.
- **Fused transformer ops (`com.microsoft`):** `MultiHeadAttention`,
  `GroupQueryAttention` (causal, grouped KV heads, internal RoPE, **real
  `past`/`present` KV cache** for autoregressive decoding — the fused op
  modern LLM exports use: Llama3 / Qwen2-3 / Gemma2-3 / Phi3 / Mistral),
  `RotaryEmbedding` (interleaved + rotate-half, partial-rotation),
  `SimplifiedLayerNormalization` / `SkipSimplifiedLayerNormalization`
  (RMSNorm) — so onnx-community / Optimum-optimized transformer exports
  (which fuse attention rather than emitting decomposed graphs) run directly.
- **Convolution / pooling:** `Conv` (1–3 spatial dims, strides / pads /
  dilations / groups / depthwise / auto_pad, im2col+GEMM fast path),
  `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool`,
  `GlobalMaxPool`, `BatchNormalization`, `InstanceNormalization`, `GroupNormalization`,
  `Resize` (nearest + linear), `GridSample` (bilinear/nearest,
  zeros/border/reflection), `RoiAlign` (avg/max), `Flatten`.
- **Activations:** `LeakyRelu`, `Elu`, `PRelu`, `HardSigmoid`, `HardSwish`,
  `Softplus`, `Gelu` (erf + tanh forms).
- **Recurrent:** `LSTM` (incl. peepholes), `GRU` (incl. linear_before_reset),
  `RNN` — forward / reverse / bidirectional, `sequence_lens`,
  `initial_h`/`initial_c`; default activations only.
- **Control flow / misc:** `If`, `Loop`, `Scan` (subgraphs capture the outer
  scope; scan outputs, non-default axes and reverse directions supported),
  `Identity`, `Pad` (constant / reflect / edge), `Size`, `Tile`, `Trilu`,
  `ScatterND`, `Split`, `NonZero`, `TopK`, `NonMaxSuppression`, `Upsample`
  (deprecated pre-Resize form), `Dropout` (inference identity), `STFT`,
  `Floor`, `Ceil`, `Round`.
- **Quantization:** QDQ format — `QuantizeLinear`, `DequantizeLinear`
  (per-tensor + per-axis, uint8/int8), `DynamicQuantizeLinear`; weight
  `DequantizeLinear` nodes constant-fold at load, so QDQ models run at float
  speed. QOperator format — `MatMulInteger` (scalar / per-row / per-column
  zero points, batched), `ConvInteger`, `QLinearMatMul`, `QLinearConv`
  (per-channel weight scales + int32 bias) with exact int32 accumulation and
  round-half-to-even requantization. int8/uint8 tensors use **compact 1-byte
  storage** end to end — a 1 GB int8 model needs ~1 GB, not the 8 GB an
  int64 widening would cost. **`MatMulNBits`** (`com.microsoft`, 4-bit
  block-quantized weights, packed zero points, partial blocks) runs int4
  exports with weights kept packed (0.5 bytes/weight); dequantization streams
  through the SIMD GEMM per call.
- **Fusion (load-time, automatic):** the erf-GELU chain
  (`0.5·x·(1+Erf(x/√2))`) fuses to a single pass, and the attention epilogue
  `MatMul(Softmax(MatMul(Q,K)·s + mask), V)` fuses so scale + mask + softmax
  happen in one sweep over the attention matrix. Fusions only fire when the
  intermediate values have no other consumers; results stay within float
  rounding of the unfused graph.

Tensors are float32 or int64 (int32 and bool are widened to int64), row-major
(matching ONNX's own layout). An unimplemented op throws `UnsupportedError`
naming the op.

## How it works

ONNX graphs are stored in topological order, so execution is a single linear
pass over the nodes with a `name -> Tensor` value cache — initializers
(weights) are decoded up front, runtime inputs are supplied to `run`, and the
requested outputs are returned.

To build or inspect models programmatically, import
`package:onnx_runtime_dart/onnx_proto.dart` for the ONNX message types (`ModelProto`,
`GraphProto`, `NodeProto`, …).

## Licensing

The runtime code is MIT. The generated protobuf bindings
(`lib/src/onnx.pb*.dart`) derive from the ONNX project's `onnx.proto3` and are
used under Apache-2.0 — see [`NOTICE.md`](NOTICE.md).
