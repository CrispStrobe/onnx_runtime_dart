# PLAN — closing the gap to ONNX Runtime

> **Status (2026-07-17): all workstreams landed.** B0–B4 ✅ (incl. GELU +
> SDPA pattern fusion; isolate pool) · A1–A5 ✅ (conv/pool, im2col,
> LSTM/GRU/RNN, If/Loop/Scan, QDQ **and** QOperator quantization).
> Compact uint8/int8 tensor storage landed (1 byte/element end to end;
> QuantizeLinear/DynamicQuantizeLinear produce compact outputs).
> Quiet-machine benchmarks are in: MiniLM 66 ms sync / ≈42 ms with 4
> workers (2.6× off 1-thread native ORT, from 48× at baseline);
> MobileNetV2 260 ms, ResNet18 422 ms. Remaining follow-ups: Scan
> non-default axes/directions; pool fan-out for Gemm/conv;
> com.microsoft-domain ops (MatMulNBits for int4 models). Numbers in
> `BENCHMARKS.md`;
> per-op ORT parity fixtures in `test/fixtures/` (`tool/gen_fixtures.py`),
> live-model parity via `tool/live_parity.py|dart`.

Two workstreams, independently useful and largely independent to execute:

- **A. Operator coverage** — conv / pooling / recurrent ops, unlocking CNN vision
  models, audio front-ends, and RNN-era models.
- **B. Performance** — getting from "correct interpreter" toward native-ORT-class
  speed on the models we already run.

Both build on the current architecture (one dispatch `switch` in
`lib/src/onnx_graph.dart`, pure op functions over row-major `Tensor` in
`lib/src/onnx_ops.dart`), which stays as-is. Every step below is verifiable with
the existing methodology: cosine-parity vs native ORT via `tool/run_model.dart`,
node-level diffing via `tool/trace_node.dart`.

---

## Workstream A — operator coverage

### A1. Convolution + pooling (highest practical value)

Unlocks MobileNet / ResNet / SqueezeNet, CLIP-style image towers, and the conv
stems of audio models (wav2vec, Whisper).

Ops, roughly in implementation order:

| Op | Notes |
|---|---|
| `MaxPool`, `AveragePool` | Shared window-iteration helper (also used by Conv). Mind `ceil_mode`, `count_include_pad`, `auto_pad`. |
| `GlobalAveragePool`, `GlobalMaxPool` | Trivial reductions over spatial dims. |
| `BatchNormalization` | Inference mode only = per-channel scale + shift. Trivial. |
| `Conv` | The centerpiece. One op covers 1D/2D/3D via `kernel_shape`, plus `strides`, `pads`, `dilations`, `group` (groups ⇒ depthwise, needed by MobileNet), `auto_pad` (`SAME_UPPER`/`SAME_LOWER`/`VALID`). Start with a naive direct loop (~150 lines, correct); speed comes later via im2col (A2). |
| `Resize` | Nearest + linear modes; present in almost every vision export. |
| `InstanceNormalization`, `GroupNormalization` | Small variations on existing LayerNorm code. |
| Activations | `LeakyRelu`, `HardSwish`, `HardSigmoid`, `Elu`, `PRelu`, `Softplus`, `Gelu` — one-liners on `_elementwiseUnary`. |

Estimated size: ~500–700 lines in `onnx_ops.dart` + dispatch cases.

**Verification targets:** MobileNetV2 and ResNet18 from the ONNX model zoo,
cosine-1.0 vs ORT.

### A2. Conv performance + completeness

- **im2col + MatMul** fast path for `Conv` — reuses `opMatMul`, so all GEMM work
  from Workstream B automatically accelerates convolution too.
- `ConvTranspose`.
- 1D and 3D coverage beyond the 2D common case, if targets demand it.

### A3. Recurrent ops

`LSTM`, `GRU`, `RNN` — mechanically a time-step loop of Gemm + gate
elementwise, but spec-heavy: `direction` (incl. bidirectional), optional
`sequence_lens`, `initial_h`/`initial_c`, per-gate `activations`, `layout`.
~300–400 lines for LSTM+GRU.

Lower priority than conv (modern exports are transformer-dominated), **unless a
target model appears** — Silero VAD is a popular ONNX model that needs LSTM and
would make a great verified target.

### A4. Control flow — the one structural change

`If`, `Loop`, `Scan` carry **subgraphs as GraphProto-typed attributes**; many
seq2seq decoders (KV-cache loops) and RNN-era exports use them. Requires
`OnnxGraphExecutor` to recursively execute a subgraph with a captured outer
scope — an executor change, not just a new op function. Not huge (the executor
is ~290 lines) but the only non-additive item in this workstream.

### A5. Quantized models (later)

`Tensor` is float32/int64 only today. Mobile-sized CNNs usually ship
quantized, which needs int8/uint8 dtype support first, then `QuantizeLinear`,
`DequantizeLinear`, `QLinearConv`, `QLinearMatMul`, `MatMulInteger`. Int8
matmul in pure Dart can even beat f32 on memory traffic, though without SIMD
dot-product instructions the gain is modest.

---

## Workstream B — performance

For the verified transformer models, essentially all time is in `opMatMul` and
the elementwise/broadcast machinery. The plan mirrors ORT's own stack in
miniature: kernels (MLAS) + memory planning + graph transforms + threadpool.

### B0. Measure first (half a day; informs everything)

Per-op-type timing in the dispatch loop of `onnx_graph.dart` — a `Stopwatch`
and a `Map<String, Duration>`, ~10 lines. Confirms whether we're 95%
MatMul-bound (likely) or losing real time to broadcasting/allocation. Add a
wall-clock benchmark script (jina-v2, MiniLM) so every later step has a number.

### B1. Cheap wins in existing code (small diffs, likely 2–5× end-to-end)

1. **Pre-pack weights at load time.** `opGemm` materializes a full transpose of
   B on *every call* when `transB=1` — the standard case for linear layers with
   initializer weights. Transpose initializers **once** in the
   `OnnxGraphExecutor` constructor and mark the node pre-transposed (ORT's
   "weight prepacking"). Free speed.
2. **Trailing-dim broadcast fast path.** A bias add `[B,T,N] + [N]` currently
   falls through `_elementwiseBinary`'s general path: coordinate decomposition
   + two `_flattenBroadcast` calls *per element*. A "b broadcasts over the last
   axis" path is ~10× cheaper on one of the most common ops in any transformer.
3. **Monomorphic inner loops for hot ops.** The `op(x,y)` closure + `getD`'s
   per-element dtype branch block tight AOT codegen. Give `Add`/`Mul`/`Sub`
   dedicated loops indexing `Float32List` directly; keep the generic helper for
   the long tail.
4. **Buffer reuse (memory planner).** Every op allocates a fresh list. A
   size-keyed free-list + one-time liveness pass over the graph ("value's last
   use is node k ⇒ recycle its buffer after") removes most GC pressure. ~60
   lines.

### B2. Real GEMM kernel (single biggest lever)

Current i-k-j loop is cache-friendly but memory-bound (load-add-store on
`out[]` per element; the `aVal == 0` skip costs more on dense data than it
saves).

1. **Register tiling** — compute a 4×4 (or 6×4) output tile per inner loop,
   accumulators in locals, one write per tile. Typically 2–4× on dense GEMM.
2. **SIMD via `dart:typed_data`** — `Float32x4`/`Float32x4List` compile to real
   SSE/NEON on native VM/AOT. Broadcast `aVal` into a lane-splat, process B
   four columns at a time; composes with tiling for another ~2–3×.
   ⚠ Verify what dart2wasm / dart2js do with `Float32x4` on web targets; keep
   the scalar loop as a fallback behind a flag.
3. **Cache blocking + panel packing** for matrices past L2 (768–4096-dim
   transformer layers qualify): pack a block of B contiguously once, reuse
   across the M loop.

### B3. Graph transforms (run once at session load)

1. **Constant folding.** Transformer exports recompute the same
   `Shape → Gather → Concat → Reshape` chains every run. Fold anything whose
   inputs are all initializers/constants.
2. **Pattern fusion**, in value order:
   - MatMul + Add → biased Gemm (one output write);
   - the Erf-based GELU subgraph (≈6 nodes) → one fused pass;
   - **attention fusion**: recognize `Softmax(Q·Kᵀ/√d + mask)·V` and run it as
     one routine with no materialized attention matrix — the largest
     graph-level win ORT gets on BERT-class models, and it slashes peak memory.
3. **Elementwise chain fusion** (add→relu etc.): one pass, one buffer.

### B4. Isolate parallelism (native only; most design effort)

Isolates don't share mutable memory, and copying weights per call is a
non-starter. The design that works:

- Persistent worker pool; **partition each weight matrix by output columns
  across workers at load time** — worker *w* permanently owns columns
  `[w·N/W, (w+1)·N/W)` of every linear layer.
- Per run, send only activations (small: `B·T·K` floats) to each worker;
  concatenate the returned column slices. Memory stays ~1× model size.
- Ship behind the existing conditional-import split
  (`onnx_runtime_dart_io.dart`); web keeps the single-threaded path.

Expect near-linear scaling of the GEMM-bound fraction up to physical cores.

### B5. Ceiling + escape hatch

ORT's MLAS is hand-tuned AVX2/AVX-512/NEON with years of blocking heuristics.
Realistic pure-Dart AOT target after B1–B4: **within ~3–8× of single-threaded
native ORT** on GEMM-bound models (from a starting point likely 30–100× off),
closer end-to-end once fusion lands.

If true parity ever becomes a requirement: dual backend — FFI to XNNPACK/BLAS
for GEMM only on native targets, pure-Dart kernel as the universal fallback.
The `opMatMul` interface doesn't change, so portability survives — but it
dilutes the package's "no FFI" pitch, so this is a product decision, not a
default.

---

## Suggested overall order

1. **B0** — instrument + benchmark baseline.
2. **B1** — prepacking, broadcast fast path, monomorphic loops, buffer reuse.
3. **A1** — pooling + BatchNorm + naive Conv + Resize + activations; verify
   MobileNetV2 / ResNet18 to cosine-1.0.
4. **B2** — tiled + SIMD GEMM (also speeds up A2's im2col conv).
5. **A2** — im2col conv fast path, ConvTranspose.
6. **B3** — constant folding, GELU/bias fusion, then attention fusion.
7. **B4** — isolate pool (multiplies whatever the kernels achieve).
8. **A3/A4** — LSTM/GRU + subgraph execution, when a target model demands it
   (e.g. Silero VAD).
9. **A5** — quantized dtypes + QLinear ops, if mobile-sized models matter.

Every step keeps the invariant that made the package trustworthy so far:
**cosine-1.0 parity vs native ORT on real models before merging.**
