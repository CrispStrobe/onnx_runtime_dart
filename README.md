# onnx_runtime_dart

A small, dependency-light **pure-Dart ONNX inference runtime**. No FFI and no
native `onnxruntime` — the graph is interpreted in plain Dart, so the same code
runs on **every Dart/Flutter target, including the web and WebAssembly**.

It implements the operator set used by **transformer / attention** style models
(BERT-family text embedders, rerankers, and RoPE / ALiBi variants), the
**convolution / pooling** family used by CNN vision models, **recurrent** ops
(`LSTM`/`GRU`/`RNN`), **control flow** (`If`/`Loop`/`Scan` with subgraph
execution) and **quantized models** (both QDQ and QOperator formats). It is
still not a complete ONNX runtime — see the operator list below for exactly
what is covered.

Verified to **cosine-1.0 parity** against ONNX Runtime (via `ort`), max abs diff
~1e-6 (float32 rounding), on: `jina-embeddings-v2-base-en` (BERT + ALiBi),
`bge-small-en-v1.5`, `all-MiniLM-L6-v2`, `ms-marco-MiniLM` (cross-encoder
reranker), the full **`nllb-200-600M`** seq2seq stack — encoder, decoder
(256k-vocab logits + all present-KV outputs) and **`decoder_with_past`**
(KV-cache incremental decoding), so translation loops run end to end —
**TrOCR** (ViT image encoder + text decoder, both cosine-1.0), a 0.6B **RoPE**
embedder (external-data weights), the vision CNNs **MobileNetV2** and
**ResNet18**, **Silero VAD** (Conv1D + LSTM + `If` + reflect-`Pad`),
**AECMOS** (both echo-MOS models: Conv + MaxPool + bidirectional GRU +
ReduceMax — with a complete pure-Dart scoring pipeline in
[`example/aecmos/`](example/aecmos/)), **CAM++** (speaker-embedding x-vector:
225 convs + BatchNorm/AveragePool/Pad/ReduceProd), **Maia3-5M** (chess
transformer, policy + WDL value heads, Einsum attention), and from the
**Parakeet-TDT 0.6B** ASR stack: the NeMo mel featurizer (`STFT` op +
float64 weights), the RNN-T decoder/joint (LSTM + Split), and the int8
conformer encoder (ConvInteger 1-D/2-D + MatMulInteger + Tile; dynamic-quant
model, so judged by the intrinsic-band criterion — our deviation from
ORT-int8, cosine 0.997, is far below that export's own quantization error vs
fp32, cosine 0.63). Also verified: the **CosyVoice3 speech tokenizer**
(all 25 discrete speech tokens exactly equal to ORT's) and the
**llama-nemotron-rerank-1B int4** reranker (logit within 1.7e-5).

One known precision-mode gap: exports that run regions in **fp16 compute**
(115 `Cast`-to-fp16 pairs in `zerank-1-small` int4) execute here in float32
between the cast points (values are rounded through fp16 *at* each cast).
ORT computes those ops in true half precision, so results agree only to the
model's fp16 sensitivity (~2% on zerank's score) — ours is the more precise
of the two, but not bit-matching.
Every op is additionally covered by generated per-op parity fixtures against
native ONNX Runtime (`test/fixtures/`, see `tool/gen_fixtures.py`).

**Quantized MobileNetV2 (QDQ)** classifies identically to ORT (same top-5, in
order). Quantized models have no bitwise logit parity to target: tiny
float-ordering differences flip quantization buckets, and ORT's own optimized
vs unoptimized execution of the same model only agrees with itself to cosine
≈0.993 — our output lands in the same band (≈0.991).

**Octen-0.6B int4** (`MatMulNBits`, 1.8 GB packed weights) runs at
**cosine-1.0** vs ORT — static block quantization has no runtime
quantization boundaries, so int4 models reproduce exactly, unlike dynamic
int8 below.

**Octen-0.6B int8** (QOperator dynamic quantization, 196 `MatMulInteger`,
1 GB of int8 weights running in ~1 GB of RAM via compact storage) runs
end-to-end; the runtime-induced deviation from ORT-int8 (cosine 0.96 on the
pooled embedding) is far smaller than that model's own quantization error vs
its fp32 original (cosine 0.73) — dynamic quantization amplifies
transcendental-function rounding differences across layers, so int8-vs-int8
bitwise parity is unattainable for any independent implementation.

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
  onnx_runtime_dart: ^0.1.0
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

Weights load from float32, float16, int32, int64 and bool tensors, inline or
from a companion `.onnx.data` file (read on demand, so multi-GB models don't
load into memory all at once).

See [`example/onnx_runtime_dart_example.dart`](example/onnx_runtime_dart_example.dart) for a
self-contained, runnable graph built with the protobuf types.

## Supported operators

- **Math:** `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Sqrt`, `Reciprocal`, `Abs`,
  `Neg`, `Exp`, `Log`, `Relu`, `Erf`, `Sigmoid`, `Tanh`, `Cos`, `Sin`, `Clip`.
- **Compare / logic:** `Equal`, `Greater`, `Less`, `GreaterOrEqual`,
  `LessOrEqual`, `And`, `Or`, `Not`, `Where`, `Max`, `Min`.
- **Shape / index:** `Shape`, `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`,
  `Concat`, `Gather`, `GatherND`, `GatherElements`, `Expand`, `Slice`, `Range`,
  `Cast`, `Constant`, `ConstantOfShape`.
- **Reduce / linalg:** `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`,
  `ReduceProd`, `CumSum`, `Softmax`, `LayerNormalization`, `MatMul`, `Gemm`,
  `Einsum` (`bhi,oi->bho`, `bid,bjd->bij`).
- **Convolution / pooling:** `Conv` (1–3 spatial dims, strides / pads /
  dilations / groups / depthwise / auto_pad, im2col+GEMM fast path),
  `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool`,
  `GlobalMaxPool`, `BatchNormalization`, `InstanceNormalization`, `Resize`
  (nearest + linear), `Flatten`.
- **Activations:** `LeakyRelu`, `Elu`, `PRelu`, `HardSigmoid`, `HardSwish`,
  `Softplus`, `Gelu` (erf + tanh forms).
- **Recurrent:** `LSTM` (incl. peepholes), `GRU` (incl. linear_before_reset),
  `RNN` — forward / reverse / bidirectional, `sequence_lens`,
  `initial_h`/`initial_c`; default activations only.
- **Control flow / misc:** `If`, `Loop`, `Scan` (subgraphs capture the outer
  scope; scan outputs, non-default axes and reverse directions supported),
  `Identity`, `Pad` (constant / reflect / edge), `Size`.
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
