# German GECToR runtime qualification

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: full hosted saved-input qualification complete; bounded Dart optimization closed; CrispEmbed/ggml mixed Q4_K selected next; application qualification pending.

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
next gate at this checkpoint was hosted execution against both the frozen
original and separately verified ONNX INT4 candidate, each compared with its
own native ORT outputs. That gate is now complete, as recorded below.


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
insufficient to qualify the checker. These results motivated the second and
final initial hypothesis: avoiding redundant packing of the already narrow
dequantized tile using the existing aligned GEMM path. Keep each source/binary
identity and report separately; do not replace either baseline or infer hosted
speed from the local comparison.


## Second hypothesis and existing experiment contract

The second hypothesis reuses the existing aligned narrow GEMM path to avoid
repacking an already narrow reconstructed tile. It is selected by the existing
`OnnxExperiment.narrowDirectGemm` flag, remains disabled by default, and is
forwarded explicitly through MatMulNBits dispatch. The benchmark accepts
`--narrow-direct true|false` (default false) and records its experiment names
in complete, partial and runtime-failure reports.

Twenty-seven focused operator, graph, benchmark-configuration and existing
ORT-fixture tests pass; analysis and AOT compilation pass. The final opt-in
AOT executable has SHA-256
`648c7928f5a4b19bd09d0d7fd17b3d7c2963da363e0f0aa0bbb5b0d8c447cf9c`.
With the flag enabled, the two-case real-model verification passes all four
saved passes, tokenizer alignment and exact selected tags. Forward medians
were 1.36/1.41 seconds and peak RSS was 463,605,760 bytes. The earlier
unconditional prototype and its faster 1.06/1.25-second observations remain
separate records of the same second hypothesis; local samples are noisy and
must not be cherry-picked.

The initial two runtime hypotheses are complete. They improve the measured
local quantized path but do not qualify the 500 ms complete-check target.
The full hosted repeat below is complete with the experiment explicitly enabled
for the INT4 arm. Preserve the original hosted baseline and the per-model opt-in
boundary. Runtime code remains frozen at
`b10258682d459847b63e012eaaba58a528e93956`; no further initial kernel search is
planned.

## Final hosted qualification and backend handoff

[Run 44062093617](https://github.com/CrispStrobe/quds_office_suite/actions/runs/44062093617)
used Quds source `69d153c0f81596af894e48e21880f0957b963994` and runtime source
`b10258682d459847b63e012eaaba58a528e93956`. Both the original F32 and ONNX INT4
variants passed all **30 cases / 40 saved passes**, including tokenizer
alignment, both output heads, exact selected tags, quantized correction
nonregression and the declared experiment selection. The F32 control explicitly
disabled `narrowDirectGemm`; the INT4 arm explicitly enabled it. The
[complete optimized hosted record](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/validation/german-gector/2026-11-17-hosted-optimized/README.md)
retains source, model, fixture and executable identities and the measured scope.

The following comparison uses the same INT4 candidate bytes within that job:

| Measurement | Dart INT4, `narrowDirectGemm` enabled | Native ORT INT4 control |
| --- | ---: | ---: |
| Warm first-pass forward median | 463.4255 ms | 15.7722095 ms |
| Warm first-pass forward p95 | 545.531 ms | 17.060261 ms |
| Inference/reference process peak RSS | 487,698,432 bytes | 358,817,792 bytes |
| Complete unguarded reference correction p95 | Not implemented/measured | 33.219578 ms |

These aggregate statistics pool 90 warm samples across 30 cases, with one
warmup and three measurements per case. Inputs span only 5..12 wordpieces.
Forward timing excludes tokenization and edit decoding; native complete
reference correction includes the iterative runner but omits application
guards and editor integration. Process peaks are not weights-only allocation.

This final hosted job used an **Intel Xeon 6975P-C**; the original hosted
baseline used an **AMD EPYC 9V74**. The hosts differ, so the cross-run timing
change is not a causal measurement of the kernel optimization. The separate
local observations and both hosted records remain intact.

The Dart forward p95 alone still exceeds the provisional 500 ms complete-check
budget on short inputs. Close this bounded Dart tuning round and select
**CrispEmbed/ggml with the declared mixed Q4_K recipe** for the next native
implementation and qualification slice. This chooses the next engineering
step; it does not qualify a shipping backend. The full Dart correction/guard
loop, longer inputs, target devices and application integration remain open.

Native ONNX through Rust remains a conditional fallback. Although ORT's
top-level license is MIT, its pinned
[1.23.2 ThirdPartyNotices](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/ThirdPartyNotices.txt)
lists Eigen under MPL 2.0. The strict permissive-only license closure for a
shipping native ORT build has **not been approved**. Its measured control
performance does not resolve that boundary; any fallback needs an exact-build
dependency, linked-library and packaged-license inventory before selection.
