# onnx_dart

A small, dependency-light **pure-Dart ONNX inference runtime**. No FFI and no
native `onnxruntime` — the graph is interpreted in plain Dart, so the same code
runs on **every Dart/Flutter target, including the web and WebAssembly**.

It implements the operator set used by **transformer / attention** style models
(BERT-family text embedders, rerankers, and RoPE / ALiBi variants). It is
deliberately *not* a complete ONNX runtime — there are no convolution, pooling
or recurrent ops.

Verified to **cosine-1.0 parity** against ONNX Runtime (via `ort`), max abs diff
~1e-6 (float32 rounding), on: `jina-embeddings-v2-base-en` (BERT + ALiBi),
`bge-small-en-v1.5`, `all-MiniLM-L6-v2`, `ms-marco-MiniLM` (cross-encoder
reranker), the `nllb-200-600M` encoder (seq2seq / mBART), and a 0.6B **RoPE**
embedder (external-data weights).

## Install

```yaml
dependencies:
  onnx_dart: ^0.1.0
```

## Usage

```dart
import 'dart:typed_data';
import 'package:onnx_dart/onnx_dart.dart';
import 'package:onnx_dart/onnx_dart_io.dart'; // native only (dart:io)

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

See [`example/onnx_dart_example.dart`](example/onnx_dart_example.dart) for a
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

Tensors are float32 or int64 (int32 and bool are widened to int64), row-major
(matching ONNX's own layout). An unimplemented op throws `UnsupportedError`
naming the op.

## How it works

ONNX graphs are stored in topological order, so execution is a single linear
pass over the nodes with a `name -> Tensor` value cache — initializers
(weights) are decoded up front, runtime inputs are supplied to `run`, and the
requested outputs are returned.

To build or inspect models programmatically, import
`package:onnx_dart/onnx_proto.dart` for the ONNX message types (`ModelProto`,
`GraphProto`, `NodeProto`, …).

## Licensing

The runtime code is MIT. The generated protobuf bindings
(`lib/src/onnx.pb*.dart`) derive from the ONNX project's `onnx.proto3` and are
used under Apache-2.0 — see [`NOTICE.md`](NOTICE.md).
