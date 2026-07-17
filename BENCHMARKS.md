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

Per-op shares come from `ExecutionProfile` (`--iters` accumulate). Regenerate
the ORT reference with the snippet in `git log` for this file or ad-hoc via
`.venv/bin/python` + `onnxruntime`.
