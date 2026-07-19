# Changelog

## 0.4.2

- **Another ~1.9× on LLM decode (≈5× total vs 0.3.x).** The single-row GEMV
  path now reads B through a `Float32x4List` view — one vector load per lane
  instead of the four scalar loads the `Float32x4(a,b,c,d)` constructor
  compiles to — whenever `n` is a multiple of 4 and B's base is 16-byte
  aligned (the common case for transformer weight matrices). Bitwise identical
  (the view yields the same four floats; verified max|Δ|=0), with a scalar
  fallback for any odd width or unaligned isolate-worker slice. Qwen2.5-0.5B
  decode 407 → 229 ms/step; SmolLM2 24-step greedy generation 3.7 → 2.3 s,
  still token-for-token identical to ORT. ORT parity unchanged.
- Investigated fp16 weight *storage* (halve memory traffic via in-kernel
  upcast) and measured it a net loss in pure Dart — the non-vectorizable
  `fp16→fp32` conversion costs more than the bandwidth saved — so it was not
  adopted. The vector-load win above is the better lever.

## 0.4.1

- **~2.8× faster LLM decode.** The SIMD GEMM kernel now has a dedicated
  single-row (`m == 1`) GEMV path. The register-tiled kernel packs a column
  panel of B before computing — worthwhile when many A-rows reuse it, but for
  the one row of an autoregressive decode step that packing is pure overhead
  (B is read to pack, then read again to compute). The GEMV path streams B
  exactly once through register-held 16-wide column tiles. Measured on
  Qwen2.5-0.5B: decode 1154 → 407 ms/step (kernel-time −67%); SmolLM2 24-step
  greedy generation 7.3 → 3.7 s, still **token-for-token identical to ORT**.
  Accumulation stays in k-order, so column-partitioned isolate runs remain
  self-consistent and ORT parity is unchanged (Qwen decode cosine 0.99999976).

## 0.4.0

- **`OnnxModel.inputSpecs` / `outputNames` + `TensorSpec`** — introspect a
  graph's required inputs (name, declared shape with symbolic dims as -1, ONNX
  element type) and output names. KV-cache decoders use this to size their
  empty first-step `past_key_values.*` feeds without hard-coded config.
- **End-to-end text generation** verified on **Qwen2.5-0.5B-Instruct** (fp16):
  a *fully decomposed* decoder — no fused ops, RoPE/attention as primitives,
  graph-level KV cache via `Concat`/`Slice` — at logits cosine 0.99999885
  (prefill) / 0.99999976 (decode) vs ORT, plus coherent greedy/sampled text.
  This complements the fused-`GroupQueryAttention` path (SmolLM2), covering
  both modern decoder export styles.
- New tooling: `tool/bpe_tokenizer.dart` (byte-level BPE loaded from a
  HuggingFace `tokenizer.json`; validated exact-match vs `tokenizers`),
  `tool/llm_chat.dart` (text-in/text-out chat with greedy + temperature/top-k
  sampling, decoder shape auto-discovered from `inputSpecs`).

## 0.3.8

- **`GroupQueryAttention` KV cache** — `past_key`/`past_value` are now consumed
  and the real `present_key`/`present_value` are returned (the RoPE'd K and V,
  concatenated `[batch, kv_heads, past+seq, head_size]`), so the present-KV
  feeds directly back as the next decode step's past. This unlocks **real
  autoregressive LLM decoding** in pure Dart. Query token `i` has absolute
  position `past_len+i` and attends causally to keys `0..past_len+i`; internal
  RoPE positions offset by `past_len`.
- Verified end-to-end on **SmolLM2-135M-Instruct** (Llama-style GQA decoder,
  9 query / 3 KV heads): prefill and decode steps at cosine 1.0 vs ORT on
  `logits` **and** `present.*` KV, plus a 24-step greedy generation that is
  **token-for-token identical** to ORT feeding our own cache back as past.
- New `gqa_kvcache` fixture checks the populated-cache path (all three outputs)
  against the ORT oracle without needing the 515MB model.

## 0.3.7

- **`GroupQueryAttention` (`com.microsoft`)** — the fused attention op used
  by nearly every modern LLM export (Llama3 / Qwen2-3 / Gemma2-3 / Phi3 /
  Mistral): grouped KV heads, internal RoPE, and internal **causal masking**
  (GQA is a decoder op — it masks future keys regardless of the additive
  bias). Verified on harrier-oss-v1-270m (Gemma3 decoder, cosine 0.999996)
  and a `gqa_causal` fixture. present-KV outputs are dummies (no KV-cache
  reuse yet).

## 0.3.6

- **Fused transformer ops (`com.microsoft`):** `MultiHeadAttention`,
  `RotaryEmbedding` (interleaved + rotate-half, partial rotation),
  `SimplifiedLayerNormalization` / `SkipSimplifiedLayerNormalization`
  (RMSNorm) — so onnx-community / Optimum-optimized exports that fuse
  attention run directly. New ops: `LogSoftmax`, `IsNaN`, `IsInf`, `Sign`,
  `Atan`, `RandomNormalLike`.
- **Loader:** ONNX-native INT4 (dtype 22) / UINT4 (21) weights (packed
  2/byte); external-data entries with `length: 0` fall back to the
  shape-derived byte count (some Optimum fp16 exports); malformed-protobuf
  and short/oversized-tensor inputs reject with clear errors.
- **Performance:** cache-blocked GEMM kernel — a column-panel outer loop
  keeps the packed B slice in L2 instead of re-streaming it per A-row tile
  (2.3× on large-k matmuls / pointwise convs, e.g. ECAPA-TDNN 20.5 → 9.0 s;
  small-k transformers unaffected; bitwise-identical). `Softmax`/`LayerNorm`
  hot loops use direct float-buffer access; 1-D `Conv` joins the isolate-pool
  conv fan-out.
- **Newly verified live vs native ORT:** embeddinggemma-300m (Gemma3),
  FastConformer CTC, ECAPA-TDNN, Piper VITS TTS, PIXIE-Rune int4
  (ONNX-native INT4), jina-embeddings-v5-small, F2LLM-v2-0.6B,
  CrispTranslator (int8 NLLB), awesome-align, partitura-jina.

## 0.3.5

- **Web build fix:** the float→int Cast saturation added in 0.3.4 used 64-bit
  integer literals (`0x7FFFFFFFFFFFFFFF` / `-0x8000000000000000`), which are a
  compile error under dart2js (on the web `int` is a 53-bit double). The bounds
  are now parsed at runtime — exact on the VM, the nearest double on the web
  (only used when saturating a non-finite Cast, so harmless). Restores web
  (dart2js) compilation.

## 0.3.4

- **Correctness:** `Tensor.reshape` resolved `-1` before substituting
  `0`-copy dims, so a target like `[0, 0, -1]` silently produced a wrong
  shape (present since 0.1; first triggered by Kokoro's LSTM reshape).
  Float→int casts saturate non-finite values like ORT (±inf → int64
  min/max, NaN → 0); `CumSum` float accumulation rounds to float32
  stepwise; `Cast(FLOAT16)` semantics already in 0.3.3.
- **New ops:** `Sign`, `Atan`; `Resize` handles rank-3 NCW (1-D temporal).
- **Performance (Kokoro-82M TTS: 398 s → 7.3 s for 1.5 s of audio):**
  1-D `Conv` rides the im2col+SIMD GEMM path (was the generic per-element
  N-D walk); 1-D `ConvTranspose` reformulated as GEMM + scatter-add; the
  general broadcast path is an incremental odometer (no per-element
  coordinate decomposition); the isolate pool's conv fan-out handles 1-D
  convs (bitwise-identical).
- **Diagnostics:** `OnnxModel.fromBytes(..., fuse: false)` disables pattern
  fusion; executor errors include input shapes.
- **Newly verified live:** Kokoro-82M (log-mel 0.995 — see README for why
  waveform cosine is the wrong metric), Whisper-tiny (enc/dec/dec-with-past),
  Moonshine-tiny (enc + merged If-decoder), BGE-M3, MPNet, GTE-v1.5,
  ModernBERT, NomicBERT, DeBERTa-v2 rerankers, SPLADE, arctic-embed-xs,
  multilingual-e5-small, gte-small, MiniLM-L12, granite-107m.

## 0.3.3

- **Correctness (from an adversarial multi-agent review of 0.3.2):**
  - RMSNorm fusion: gamma now broadcasts along the last axis exactly as the
    unfused chain does (0.3.2 could silently mis-apply it when the reduce
    axis wasn't the last axis), with an explicit length check.
  - The fusion pass protects names captured by `Loop`/`If`/`Scan` body
    subgraphs (previously it could fuse away an intermediate a subgraph
    reads).
  - `RoiAlign` defaults to `output_half_pixel` for opset < 16 models;
    `GroupNormalization` accepts the deprecated per-group scale/bias form;
    `Einsum` rejects repeated output labels and keeps integer dtypes;
    `GridSample` throws on cubic modes / non-4-D inputs and rounds nearest
    ties half-to-even like ORT.
- **Input validation:** `run`/`runAsync` verify provided inputs against the
  graph's declared signatures — missing inputs and fixed-dimension
  mismatches (e.g. feeding a batch to a batch-fixed export) now throw
  `ArgumentError` instead of computing silently wrong results.
- Cleanups: `InstanceNormalization` delegates to `GroupNormalization`;
  `GridSample` hoists per-pixel geometry out of the channel loop.

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
