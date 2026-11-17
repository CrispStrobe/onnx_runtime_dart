# German GECToR runtime qualification

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: scoped implementation authorized; no model/performance qualification yet.

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
named and qualified. The current implementation expands each packed matrix
to F32 for each call, so lower file size does not prove faster execution or
low peak memory. Original F32 is a control, not an approved shipping format.

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
