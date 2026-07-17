# Changelog

## 0.3.2

- **Correctness:** `Clip` with min/max as attributes (opset 6–10, e.g. every
  `Relu6` in TF-converted models) was silently unbounded; `Transpose`
  without `perm` now defaults to reversing axes; `TopK` handles empty axes.
- **Performance:** shared-weight batched `MatMul`s collapse into one GEMM
  (`[64,1,k]@[k,n]` no longer re-packs the weight per batch row — Maia3-5M
  2.5×: 335→135 ms, 102 ms with 4 workers); SIMD row-dot einsum kernels;
  fused RMSNorm (`x·rsqrt(mean(x²)+eps)·γ` chains, 16–113 per Qwen-style
  model).
- **New ops:** general 1/2-operand `Einsum`, `GroupNormalization`,
  `GridSample`, `RoiAlign`, `ArgMax`/`ArgMin`, `NonZero`, `TopK`,
  `NonMaxSuppression`, `Trilu`, `ScatterND`, `Upsample`, `Dropout`;
  `Constant` attributes stored in external data now resolve.
- **Newly verified live** (all vs native ORT): NLLB-600M decoder +
  `decoder_with_past` (KV cache), TrOCR encoder/decoder, SSD-MobileNetV1
  end-to-end **bit-identical**, SAM mask decoder, TAESD SD-VAE decoder,
  Ultraface, style transfer, super-resolution, emotion-ferplus.

## 0.3.1

- **Correctness:** integer `Div` now truncates toward zero per the ONNX spec
  (previously the float quotient was rounded — off-by-one on ceil-div length
  arithmetic); `Cast(to: FLOAT16)` rounds values through half precision
  (ties-to-even) instead of passing them through unchanged.
- Register-blocked 4×8 SIMD GEMM microkernel (accumulator tile in locals):
  ResNet18 422 → 332 ms, MobileNetV2 260 → 230 ms; MiniLM-L6 with the
  4-worker isolate pool: 32.6 ms — 2.0× off single-threaded native ORT.
- `Gemm` weights join the isolate-pool column partitioning (bitwise-identical
  results, transB pre-transposed at `parallelize`).
- New ops: `Tile`, `Floor`, `Ceil`, `Round`; 1-D `ConvInteger`.
- Newly verified live vs native ORT: Parakeet-TDT int8 conformer encoder,
  CosyVoice3 speech tokenizer (discrete tokens exactly equal),
  llama-nemotron-rerank-1B int4; zerank-1-small int4 runs with a documented
  fp16-compute precision caveat (see README).

## 0.3.0 (not published)

Major expansion: CNNs, recurrent models, control flow, quantization (all
formats), audio front-ends, ~7-18x faster execution, and opt-in isolate
parallelism. Every op is covered by generated parity fixtures against native
ONNX Runtime (`test/fixtures/`), and 15 real models are verified live at
cosine-1.0 (see README).

- **Convolution / pooling family:** `Conv` (1-3 spatial dims, groups /
  depthwise / dilations / auto_pad; im2col+GEMM fast path, branchless
  depthwise kernel), `ConvTranspose`, `MaxPool`, `AveragePool`, global pools,
  `BatchNormalization`, `InstanceNormalization`, `Resize`, `Flatten`, and the
  common activations (`LeakyRelu`, `Elu`, `PRelu`, `HardSigmoid`, `HardSwish`,
  `Softplus`, `Gelu`).
- **Recurrent:** `LSTM` (peepholes), `GRU` (linear_before_reset), `RNN` —
  forward / reverse / bidirectional, `sequence_lens`, initial states.
- **Control flow:** `If`, `Loop`, `Scan` (incl. non-default axes/directions)
  with subgraph execution and outer-scope capture; `Identity`, `Pad`
  (constant/reflect/edge), `Size`, `Split`, `STFT`, `ReduceMax/Min/Prod`,
  `ReduceSumSquare`, `QuantizeLinear`/`DequantizeLinear`/
  `DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`, `QLinearMatMul`,
  `QLinearConv`, and `MatMulNBits` (com.microsoft int4).
- **Quantized models:** QDQ and QOperator formats; compact 1-byte int8/uint8
  tensor storage (a 1 GB int8 model runs in ~1 GB); int4 weights stay packed.
  Exact float32-stepwise requantization semantics matching ORT.
- **Performance:** packed register-tiled `Float32x4` SIMD GEMM (scalar web
  fallback), load-time constant folding, weight prepacking, erf-GELU and
  attention (scale+mask+softmax) fusion, broadcast/reduction fast paths.
  MiniLM-L6 seq-32: 775 ms -> 66 ms single-threaded.
- **Isolate pool (native):** `parallelize(workers: N)` + `runAsync` partition
  large MatMul weights by column across worker isolates — bitwise-identical
  results, ~66 -> 42 ms on MiniLM with 4 workers. Conv fan-out exists behind
  `poolConv: true` (off by default; measure first).
- **Profiling:** `run(..., profile: ExecutionProfile())` for per-op timings;
  `run(inputs, ['*'])` returns every intermediate value for debugging.
- **Weights:** int8/uint8 (compact), float64 (narrowed) and int4-packed
  loading added to the existing float32/float16/int32/int64/bool support.
- **Example:** `example/aecmos/` — a complete pure-Dart AECMOS echo-MOS
  scorer (librosa-parity mel front-end + scorer), stage-tested against the
  Python reference.

## 0.2.0

- Extended the operator set for BERT-family text embedders, rerankers, and
  seq2seq encoders: `Constant`, `ConstantOfShape`, `Range`, `Where`, `Equal`,
  `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual`, `And`, `Or`, `Not`, `Max`,
  `Min`, `Abs`, `Neg`, `Sigmoid`, `Tanh`, `Cos`, `Sin`, `Exp`, `Log`,
  `ReduceSum`, `GatherElements` and `CumSum`.
- **Weights:** load float16 (widened to float32), int32 and bool tensors, and
  **external-data** weights from a companion `.onnx.data` file — read on demand
  via `loadOnnxModel` / `OnnxModel.fromFile` in the new
  `package:onnx_runtime_dart/onnx_runtime_dart_io.dart` (the web-safe core stays `dart:io`-free;
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
