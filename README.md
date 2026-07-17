# onnx_runtime_dart

A small, dependency-light **pure-Dart ONNX inference runtime**. No FFI and no
native `onnxruntime` — the graph is interpreted in plain Dart, so the same code
runs on **every Dart/Flutter target, including the web and WebAssembly**.

It implements the operator set used by **transformer / attention** style models
(BERT-family text embedders, rerankers, and RoPE / ALiBi variants) plus the
**convolution / pooling** family used by CNN vision models. It is still not a
complete ONNX runtime — notably there are no recurrent (LSTM/GRU), control-flow
(`If`/`Loop`/`Scan`) or quantized ops yet.

Verified to **cosine-1.0 parity** against ONNX Runtime (via `ort`), max abs diff
~1e-6 (float32 rounding), on: `jina-embeddings-v2-base-en` (BERT + ALiBi),
`bge-small-en-v1.5`, `all-MiniLM-L6-v2`, `ms-marco-MiniLM` (cross-encoder
reranker), the `nllb-200-600M` encoder (seq2seq / mBART), a 0.6B **RoPE**
embedder (external-data weights), and the vision CNNs **MobileNetV2** and
**ResNet18**. Every op is additionally covered by generated per-op parity
fixtures against native ONNX Runtime (`test/fixtures/`, see
`tool/gen_fixtures.py`).

Execution uses a packed, register-tiled **`Float32x4` SIMD GEMM kernel** on
native targets (scalar fallback on web), im2col convolution, load-time
constant folding and weight prepacking; see `BENCHMARKS.md` for current
numbers vs native ONNX Runtime.

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
- **Reduce / linalg:** `ReduceMean`, `ReduceSum`, `CumSum`, `Softmax`,
  `LayerNormalization`, `MatMul`, `Gemm`, `Einsum`.
- **Convolution / pooling:** `Conv` (1–3 spatial dims, strides / pads /
  dilations / groups / depthwise / auto_pad, im2col+GEMM fast path),
  `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool`,
  `GlobalMaxPool`, `BatchNormalization`, `InstanceNormalization`, `Resize`
  (nearest + linear), `Flatten`.
- **Activations:** `LeakyRelu`, `Elu`, `PRelu`, `HardSigmoid`, `HardSwish`,
  `Softplus`, `Gelu` (erf + tanh forms).

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
