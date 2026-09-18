# Benchmark log

Workload: `all-MiniLM-L6-v2.onnx` (BERT-6L-384d), batch 1, seq 32, deterministic
token ids, `dart run tool/bench.dart <model> --seq 32 --iters 5`, min wall time.
Machine: Apple Silicon (this repo's dev machine), Dart AOT-less `dart run` (JIT).

Native ORT reference on identical inputs (`onnxruntime` 1.27.0 CPU):
**min 16.0 ms single-thread, 6.8 ms multi-thread.**

| Date | Change | min wall | vs ORT 1-thread | Top ops |
|---|---|---|---|---|
| 2026-07-17 | B0 baseline (pre-optimization) | 775.3 ms | 48× | MatMul 81.4%, Add 5.8%, ReduceMean 3.5% |
| 2026-07-17 | B1: monomorphic+suffix-broadcast elementwise, Gemm transB prepack | 518.4 ms | 32× | MatMul 85.5%, ReduceMean 4.7%, Erf 2.7% |
| 2026-07-17 | B2: packed 4-row Float32x4 SIMD GEMM kernel | 142.5 ms | 8.9× | MatMul 46.9%, ReduceMean 16.9%, Erf 9.5% |
| 2026-07-17 | B3: constant folding; Transpose/ReduceMean fast paths; row-scalar broadcast (LayerNorm), Pow/Erf direct loops | 107.4 ms | 6.7× | MatMul 84.4%, Erf 6.0%, Add 2.4% |
| 2026-07-17 | + GELU/SDPA fusion (quiet machine, load ≈ 4) | 66.2 ms | 4.1× | MatMul 83.2%, _FusedGelu 8.0%, _FusedSDPA 3.0% |
| 2026-07-17 | + isolate pool, 4 workers (`runAsync`, single warmed run) | ≈42 ms | 2.6× | — |

| 2026-07-17 | register-blocked 4×8 GEMM microkernel (accumulators in locals) | 67.3 ms | 4.2× | — |
| 2026-07-17 | + isolate pool, 4 workers (min of 15) | **32.6 ms** | **2.0×** | — |

Isolate pool scaling on MiniLM (bitwise-identical outputs): the 4-worker
run is 2.0× off single-threaded native ORT and 4.8× off ORT's own
multi-threaded 6.8 ms. Small-m transformer GEMMs saw little from the
register-blocked kernel (sync ≈ unchanged); large conv GEMMs did —
see the vision table.

Caveat: this machine runs other dev workloads; min-of-15-iters is the robust
number, means can inflate 2–3× under contention. Rows above marked "quiet
machine" were measured at load ≈ 4; the earlier rows at load 10–30.

## Maia3-5M chess transformer (batch 1, min of 25, cosine-1.0 throughout)

| Change | min wall |
|---|---|
| baseline (register-blocked GEMM era) | 335 ms |
| + batch-collapse for shared-weight MatMul (`[64,1,k]@[k,n]` was re-packing the weight 64×, m=1) and SIMD row-dot einsum kernels | 135 ms |
| + isolate pool, 4 workers | 102 ms |

For engine throughput, batch positions: `tokens` accepts `[N, 64, 96]`, and
GEMM efficiency rises with N (m = 64·N rows per matmul).

## Kokoro-82M TTS (16 phonemes -> 1.5 s of 24 kHz audio, quiet machine)

| Change | wall |
|---|---|
| baseline (1-D convs on the generic N-D path) | 398 s |
| Conv1D routed through im2col+SIMD GEMM | 11.5 s |
| odometer general-broadcast path (Mul 2.65 -> 1.15 s) | 9.8 s |
| GEMM-backed 1-D ConvTranspose (overlap-add off the profile) | 9.3 s |
| + isolate pool with 1-D conv fan-out (bitwise-identical) | **7.3 s** |

Remaining profile: Conv 69% (genuine GEMM work, ~15-20 GMAC per run).

## GEMM cache-blocking (large-k matmuls / pointwise convs)

Controlled A/B (old single-panel vs new column-panel kernel, back-to-back on
the same machine so load cancels), ECAPA-TDNN (81 MB, dominated by
`3072×3072` and `1024×1024` pointwise convs):

| kernel | ECAPA min wall |
|---|---|
| single-panel (packs all of B once, re-streams it per A-row tile) | 20.5 s |
| column-panel (keeps a ~192 KB k×panel slice of B in L2) | 9.0 s — **2.3×** |

Accumulation order is unchanged, so results stay bitwise identical (pool
bitwise tests + all live models pass). Small-k matmuls (transformers) are
unaffected — one panel covers all columns, so the path is identical to
before. `Softmax`/`LayerNorm` inner loops also switched from the
dtype-branching `getD()` to direct float-buffer access.

## Vision models (224×224, batch 1, cosine-1.0 parity vs ORT throughout)

| Date | Change | MobileNetV2 | ResNet18 |
|---|---|---|---|
| 2026-07-17 | A1: naive direct conv | 4612 ms | 11520 ms |
| 2026-07-17 | A2: im2col + SIMD GEMM conv (depthwise stays direct) | 655 ms | 625 ms |
| 2026-07-17 | + padded-buffer branchless depthwise (quiet machine) | 260 ms | 422 ms |
| 2026-07-17 | + register-blocked 4×8 GEMM microkernel | 230 ms | 332 ms |

ORT reference: MobileNetV2 15.4 ms (1-thread) / 5.1 ms (default);
ResNet18 47.9 ms / 14.9 ms. Gap ≈ 15× / 6.9× single-threaded; Conv is
~88% of both profiles.

## Spotify Basic Pitch (`nmp.onnx`, one 43844-sample / 2 s window)

Workload: the real Basic Pitch graph on one real audio window, checked each
run against ONNX Runtime — contour sum `4766.68`, frame-0 argmax `92`,
frame-86 argmax `94`. Machine: shared 4-core VPS, load average 10-30
throughout; **min of at least 5 runs**, before/after measured in the same
session, ratios are the usable signal and absolute numbers are not.

Native ORT reference (C++ CPU) on the same window: **174-270 ms**.

| Date | Change | min wall | vs ORT | Top ops |
|---|---|---|---|---|
| 2026-09-18 | baseline (0.10.8, JIT) | 1509 ms | 5.6-8.7× | Conv 81%, Slice 7.1%, ReduceSum 2.5%, Concat 2.2% |
| 2026-09-18 | im2col-free direct conv for mPerGroup ≤ 8 | 900 ms | 3.3-5.2× | Conv 74%, Slice 11%, ReduceSum 3.4% |
| 2026-09-18 | direct-conv crossover raised to 64 (measured) | — | — | — |
| 2026-09-18 | Slice contiguous-run fast path | 736 ms | 2.7-4.2× | Conv 78%, ReduceSum 4.8%, Pad 4.0%, Concat 3.8% |
| 2026-09-18 | ReduceSum/ReduceMin/Pad/Concat fast paths, cached `Tensor.length` | 583 ms | 2.2-3.3× | Conv 86%, Transpose 2.4%, Concat 2.3%, Relu 1.9% |
| 2026-09-18 | register-rotated input window in the few-channel kernel | **498 ms** | **1.8-2.9×** | Conv 86%, Transpose 2.4%, Concat 2.3% |

Same-session A/B of `main` against this branch, alternating three rounds of
`--iters 10` each so the load drift is shared (load 13-21): best min on
`main` **1416 ms**, best min on the branch **498 ms** — **2.8×**. Medians
under that load run 600-730 ms on the branch; 498 ms is the quietest slot
observed, so treat ~500 ms as the floor on this box rather than the typical
figure. Parity held at every step: contour sum 4766.68, argmaxes 92 / 94,
`dart test` 353 passing.

Conv is now 86% of the profile and the single 3×39 conv is 38% of the run:

| node | share | GMAC/s |
|---|---|---|
| `[1,8,172,264] × [8,8,3,39]` | 38.0% | 1.4 |
| `[1,8,172,264] × [32,8,5,5]` | 15.0% | 0.9 |
| `[1,32,172,88] × [1,32,7,3]` | 6.1% | 0.26 |
| `[1,33,172,88] × [1,33,3,3]` | 5.5% | 0.27 |
| CQT `[1,1,1,44098] × [1,1,1,256]` | 4.9% | 0.42 |
| `[1,8,172,264] × [1,8,5,5]` | 4.6% | 0.30 |
| `[1,1,172,264] × [32,1,7,7]` | 3.7% | 2.0 |

**FFT convolution was assessed and not implemented.** For the dominant
conv — 39 taps over a 302-wide padded row — a 512-point real FFT front end
costs roughly 64 MFLOP of transforms plus 68 MFLOP of spectral products
against 680 MFLOP direct: about 5× fewer operations, but they are scalar
complex arithmetic where the direct kernel is `Float32x4` throughout, so the
expected net is well under 2× on that node (≈0.5× of the total run) at the
cost of float reassociation and a parity budget. The CQT front end's genuinely
long kernels (256 taps), where FFT would pay, are only ~6% of the Dart
profile. Measure again before revisiting.

The Dart profile is **not** the ORT profile. ORT's own profiler attributes
66% of node time to `Conv` with the nnAudio CQT front end's very long 1-D
kernels prominent; in Dart the long-kernel CQT convs are ~6% and the cost
sits in the CNN head's *small-output-channel* 2-D convs, above all the
8→8-channel 3×39 (38% of the run on its own). The hypothesis that FFT
convolution was the main lever did not survive the Dart profile.

Why im2col lost: it materializes `cPerGroup·kh·kw × oh·ow` floats, each
reused only `mPerGroup` times. For the 3×39 conv that is a 170 MB column
matrix built (and packed again by the GEMM) to perform 340 MMAC. The
replacement streams a zero-padded input and keeps output accumulators in
`Float32x4` registers — four output channels per vector when `m ≥ 4`, four
output pixels per vector otherwise — accumulating in the same `(c, ky, kx)`
order in float32 lanes, so results stay **bitwise identical**.

Per-shape conv A/B (`tool/conv_bench.dart`, both arms interleaved in one
process so load drift cancels; min of 9; checksums identical):

| conv | im2col+GEMM | direct | ratio |
|---|---|---|---|
| `[1,8,172,264] × [8,8,3,39]` | 2720.0 ms | 245.1 ms | 11.1× |
| `[1,32,172,88] × [1,32,7,3]` | 279.4 ms | 56.6 ms | 4.9× |
| `[1,8,172,264] × [1,8,5,5]` | 188.2 ms | 47.9 ms | 3.9× |
| `[1,1,172,264] × [32,1,7,7]` | 48.7 ms | 16.3 ms | 3.0× |
| `[1,33,172,88] × [1,33,3,3]` | 55.9 ms | 21.7 ms | 2.6× |
| `[1,8,172,264] × [32,8,5,5]` | 193.8 ms | 107.1 ms | 1.8× |
| CQT `[1,1,1,22528] × [36,1,1,256]` | 223.1 ms | 146.2 ms | 1.5× |
| vision `[1,64,56,56] × [64,64,3,3]` | 147.3 ms | 131.3 ms | 1.1× |
| vision `[1,128,28,28] × [128,128,3,3]` | 75.7 ms | 152.6 ms | 0.50× |
| vision `[1,256,14,14] × [256,256,3,3]` | 124.7 ms | 158.2 ms | 0.79× |
| vision `[1,512,7,7] × [512,512,3,3]` | 94.2 ms | 201.0 ms | 0.47× |

Hence `directConvMaxChannels = 64`: above that, each column-matrix element
is reused often enough to pay for materializing it.

Per-op A/B for the generic ops (`tool/op_bench.dart`, min over three
alternating runs of each revision × 12 iters, load ~20-24):

| op | before | after |
|---|---|---|
| `ReduceSum [1,309,172,2]` | 18.20 ms | 2.42 ms |
| `Pad const [1,172,273,1]` | 2.55 ms | 0.27 ms |
| `ReduceMin [1,172,309]` | 2.40 ms | 0.43 ms |
| `Concat ×8 axis -1` | 9.29 ms | 4.80 ms |
| `Concat ×2 axis 1` | 3.97 ms | 2.21 ms |
| `Pad reflect [1,1,43844]` | 1.29 ms | 1.06 ms |
| `Slice [1,172,309,8]` (in-graph) | 112 ms | 8.6 ms |

Tried and rejected, each reverted after measurement:

* **4-wide output blocking with a rotating input window** in the `m ≥ 4`
  kernel (16 MACs per weight-vector load instead of 8): 0.87× — slower, the
  four `Float32x4` accumulators plus splats exceed the register budget.
* **`poolConv` isolate fan-out**, 4 workers: 1625 ms against 1016 ms
  single-threaded on the same box. As its doc-comment already warns, each
  conv message copies the whole input activation to every worker. Note this
  box had no idle cores (load ~23 on 4 cores), so this is not evidence about
  a quiet device.
* **Isolate pool without `poolConv`**: 1016 ms vs 1041 ms — no effect, the
  model has no MatMul large enough to partition.

Per-op shares come from `ExecutionProfile` (`--iters` accumulate). Regenerate
the ORT reference with the snippet in `git log` for this file or ad-hoc via
`.venv/bin/python` + `onnxruntime`.
