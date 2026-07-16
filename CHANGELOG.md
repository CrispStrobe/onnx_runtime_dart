# Changelog

## 0.1.0

- Initial release: a pure-Dart ONNX inference runtime (no FFI), extracted from
  the CrispChess app. Interprets an ONNX graph node-by-node over a minimal
  float32/int64 tensor type.
- Supports the transformer / attention operator set: Add, Sub, Mul, Div, Pow,
  Sqrt, Reciprocal, Relu, Erf, Clip, Cast, Shape, Reshape, Transpose, Squeeze,
  Unsqueeze, Concat, Gather, GatherND, Expand, Slice, ReduceMean, Softmax,
  LayerNormalization, MatMul, Gemm and Einsum.
