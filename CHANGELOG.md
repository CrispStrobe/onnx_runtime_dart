# Changelog

## 0.2.0

- Extended the operator set for BERT-family text embedders, rerankers, and
  seq2seq encoders: `Constant`, `ConstantOfShape`, `Range`, `Where`, `Equal`,
  `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual`, `And`, `Or`, `Not`, `Max`,
  `Min`, `Abs`, `Neg`, `Sigmoid`, `Tanh`, `Cos`, `Sin`, `Exp`, `Log`,
  `ReduceSum`, `GatherElements` and `CumSum`.
- **Weights:** load float16 (widened to float32), int32 and bool tensors, and
  **external-data** weights from a companion `.onnx.data` file — read on demand
  via `loadOnnxModel` / `OnnxModel.fromFile` in the new
  `package:onnx_dart/onnx_dart_io.dart` (the web-safe core stays `dart:io`-free;
  pass an `externalData` resolver there).
- `Shape` now honours the `start` / `end` attributes (opset 15+), so models
  that slice the shape to read a single dim (e.g. RoPE position ranges) work.
- **Fixes:** `ReduceMean`, `Unsqueeze` and `Squeeze` now honour an `axes`
  **attribute** (older opsets), not just an `axes` input — previously this
  collapsed reductions to a scalar / failed, breaking BERT LayerNorm and
  older-opset exports.
- Verified **cosine-1.0 parity** vs ONNX Runtime (max abs diff ~1e-6) on
  `jina-embeddings-v2-base-en`, `bge-small-en-v1.5`, `all-MiniLM-L6-v2`,
  `ms-marco-MiniLM` (reranker), the `nllb-200-600M` encoder, and a 0.6B RoPE
  embedder with external-data weights.

## 0.1.0

- Initial release: a pure-Dart ONNX inference runtime (no FFI), extracted from
  the CrispChess app. Interprets an ONNX graph node-by-node over a minimal
  float32/int64 tensor type.
- Supports the transformer / attention operator set: Add, Sub, Mul, Div, Pow,
  Sqrt, Reciprocal, Relu, Erf, Clip, Cast, Shape, Reshape, Transpose, Squeeze,
  Unsqueeze, Concat, Gather, GatherND, Expand, Slice, ReduceMean, Softmax,
  LayerNormalization, MatMul, Gemm and Einsum.
