# Changelog

## 0.2.0

- Extended the operator set for BERT-family text embedders, rerankers, and
  RoPE / ALiBi models: `Constant`, `ConstantOfShape`, `Range`, `Where`,
  `Equal`, `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual`, `And`, `Or`,
  `Not`, `Max`, `Min`, `Abs`, `Neg`, `Sigmoid`, `Tanh`, `Cos`, `Sin`, `Exp`,
  `Log`, `ReduceSum` and `GatherElements`.
- `Cast` now handles bool, and int32 / bool weights load (widened to int64).
- **Fix:** `ReduceMean` now honours an `axes` **attribute** (older opsets, as in
  BERT LayerNorm), not just an `axes` input — previously it collapsed to a
  scalar, breaking normalization.
- Verified **cosine-1.0 parity** against ONNX Runtime on
  `jina-embeddings-v2-base-en` (max abs diff ~5e-6).

## 0.1.0

- Initial release: a pure-Dart ONNX inference runtime (no FFI), extracted from
  the CrispChess app. Interprets an ONNX graph node-by-node over a minimal
  float32/int64 tensor type.
- Supports the transformer / attention operator set: Add, Sub, Mul, Div, Pow,
  Sqrt, Reciprocal, Relu, Erf, Clip, Cast, Shape, Reshape, Transpose, Squeeze,
  Unsqueeze, Concat, Gather, GatherND, Expand, Slice, ReduceMean, Softmax,
  LayerNormalization, MatMul, Gemm and Einsum.
