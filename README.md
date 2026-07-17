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
reranker), the `nllb-200-600M` encoder (seq2seq / mBART), a 0.6B **RoPE**
embedder (external-data weights), the vision CNNs **MobileNetV2** and
**ResNet18**, **Silero VAD** (Conv1D + LSTM + `If` + reflect-`Pad`),
**AECMOS** (both echo-MOS models: Conv + MaxPool + bidirectional GRU +
ReduceMax), **CAM++** (speaker-embedding x-vector: 225 convs +
BatchNorm/AveragePool/Pad/ReduceProd), and **Maia3-5M** (chess transformer,
policy + WDL value heads, Einsum attention).
Every op is additionally covered by generated per-op parity fixtures against
native ONNX Runtime (`test/fixtures/`, see `tool/gen_fixtures.py`).

**Quantized MobileNetV2 (QDQ)** classifies identically to ORT (same top-5, in
order). Quantized models have no bitwise logit parity to target: tiny
float-ordering differences flip quantization buckets, and ORT's own optimized
vs unoptimized execution of the same model only agrees with itself to cosine
≈0.993 — our output lands in the same band (≈0.991).

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
  scope; scan outputs supported; Scan with default axes/directions),
  `Identity`, `Pad` (constant / reflect / edge), `Size`.
- **Quantization:** QDQ format — `QuantizeLinear`, `DequantizeLinear`
  (per-tensor + per-axis, uint8/int8), `DynamicQuantizeLinear`; weight
  `DequantizeLinear` nodes constant-fold at load, so QDQ models run at float
  speed. QOperator format — `MatMulInteger`, `ConvInteger`, `QLinearMatMul`,
  `QLinearConv` (per-channel weight scales + int32 bias) with exact int32
  accumulation and round-half-to-even requantization.
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
