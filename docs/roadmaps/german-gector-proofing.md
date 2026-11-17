# German GECToR runtime qualification

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: baseline runtime parity measured; bounded INT4 optimization in progress; application qualification pending.

The [canonical Quds roadmap](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/roadmaps/german-gector-proofing.md) owns the cross-repository scope,
frozen model, acceptance criteria and product integration. Base:
`7bdf47d815a6516da69d29c671240d10e68723ff`.

## This repository's bounded work

1. Add pretokenized WordPiece encoding with original word IDs/first-piece
   mapping and focused tests. Preserve existing text-encoding behavior.
2. Add an exact-input, dual-output AOT benchmark consuming independently
   captured reference fixtures. Load once, request logits_labels and logits_d,
   retain all token positions, and report cold load/first/warm latency and RSS.
3. Check the pinned Proofbird v2.1 graph against the current operator set.
4. Measure a separately identified ONNX MatMulNBits INT4 candidate.
5. Profile before optimizing. Prioritize only measured initializer/intermediate
   lifetime, dequantization, matrix or worker-allocation costs; limit the first
   round to two hypotheses and retain a baseline for every comparison.

## Precision contract

Q4_K is a ggml/GGUF format and is the user's preferred native candidate.
This runtime loads ONNX, not GGUF. ONNX MatMulNBits INT4 must remain separately
named and qualified. The original implementation expanded each packed matrix
to F32 for each call. The first optimization reconstructs at most 32 columns
at a time; lower file size still does not prove faster execution or low peak
memory. Original F32 is a control, not an approved shipping format.

Compare exact IDs/alignment, both head outputs, selected edit/detection
decisions and corrected text. Do not use lastTokenLogits, pooled embeddings,
cosine alone, silent truncation or candidate-regenerated gold.

## Evidence and handoff

Use the Quds-pinned model revision8e3b50beef7d775cedb7b876009e88bcf2e8d6af and
manifest. Report source/model/fixture hashes, SDK and AOT mode, CPU/thread budget,
all shapes and numerical errors, cold/warm timings and peak RSS. Hosted
measurements qualify only their host. Core API changes get CHANGELOG entries
without a release/version bump. Large models and raw arrays stay outside Git.

If the bounded optimization round misses the canonical resource/latency targets,
hand the same fixtures to the conditional CrispEmbed Q4_K or Rust/native ONNX
backend. Source preparation alone does not select the Dart backend.

## First implementation checkpoint

The roadmap was pushed and read back at
`3ac10dd64c655657186ba89e673cb183d2250682` before implementation. The first code
increment adds the pretokenized API, exact-input benchmark and focused tests.
Under Dart 3.14.0, 22 targeted tests pass, changed Dart files pass analysis with
fatal infos, and the benchmark compiles to an AOT executable. These checks use
tiny synthetic graphs; no real-model performance is established by them.

The benchmark records tokenizer and runtime parity separately. A tokenizer
mismatch fails overall qualification while verified saved inputs still permit
measuring the executor. Atomic progress reports retain incomplete runs. The
next gate is hosted execution against both the frozen original and separately
verified ONNX INT4 candidate, each compared with its own native ORT outputs.


## First measured optimization: bounded dequantization tiles

The [retained local baseline](https://github.com/CrispStrobe/quds_office_suite/blob/28cbbe4e780460736bdacccb798be9005aae3d14/docs/validation/german-gector/2026-11-17-local-smoke/README.md)
shows correct saved-input execution but slow MatMulNBits. The full hosted
[baseline run 44053903485](https://github.com/CrispStrobe/quds_office_suite/actions/runs/44053903485)
has now passed all 30 cases / 40 saved passes for both original and INT4
models, including tokenizer/runtime parity and quantized correction
nonregression on that frozen panel. Baseline Dart INT4 aggregate forward
median/p95 were 767/835 ms, with peak RSS 499,945,472 bytes. These are
short 5..12-token inputs, not qualification through 64 tokens or a complete
Dart correction loop.

The first optimization limits reconstructed weight scratch to K*32 floats,
preserving FP32 reconstruction and K-order accumulation. It also fixes
packed zero-point byte padding between columns with an odd number of blocks,
and rejects unsupported floating zero points explicitly. Eighteen focused
operator/reference tests and analysis pass. The synthetic diagnostic keeps
the former full-matrix algorithm for reproducible comparison; noisy synthetic
speed ratios are not a product claim.

The two-case real-model tile probe preserves both heads and exact selected
tags on all four saved passes. Its forward medians were 1.84/1.89 seconds
on the capped local host, versus baseline INT4 1.90/2.15 seconds. This is
insufficient to qualify the checker. The next and final initial hypothesis
is avoiding redundant packing of the already narrow dequantized tile using
the existing aligned GEMM path. Keep each source/binary identity and report
separately; do not replace either baseline or infer hosted speed from the
local comparison.
