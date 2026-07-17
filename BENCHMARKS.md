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

Caveat: this machine runs other dev workloads; min-of-15-iters is the robust
number, means can inflate 2–3× under contention.

## Vision models (224×224, batch 1, cosine-1.0 parity vs ORT throughout)

| Date | Change | MobileNetV2 | ResNet18 |
|---|---|---|---|
| 2026-07-17 | A1: naive direct conv | 4612 ms | 11520 ms |
| 2026-07-17 | A2: im2col + SIMD GEMM conv (depthwise stays direct) | 655 ms | 625 ms |

ORT reference: MobileNetV2 15.4 ms (1-thread) / 5.1 ms (default);
ResNet18 47.9 ms / 14.9 ms. Gap ≈ 42× / 13× single-threaded — depthwise conv
(not GEMM-backed) dominates MobileNetV2's remainder.

Per-op shares come from `ExecutionProfile` (`--iters` accumulate). Regenerate
the ORT reference with the snippet in `git log` for this file or ad-hoc via
`.venv/bin/python` + `onnxruntime`.
