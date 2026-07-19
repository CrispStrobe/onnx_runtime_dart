# Reader-robustness fuzzing

Fuzz harnesses for the parsers that read **untrusted input**, using
[`covfuzz`](https://pub.dev/packages/covfuzz). The contract: on *any* input a
parser must parse or reject with a **documented exception** — never leak a
`RangeError` / `StateError` / `TypeError` / protobuf-internal exception, OOM,
hang, or take multiple seconds.

## Surfaces

| Harness | Target | Untrusted input | Clean rejects |
|---|---|---|---|
| `onnx_bytes.dart` | `OnnxModel.fromBytes` | ONNX protobuf bytes | `FormatException`, `UnsupportedError` |
| `onnx_bytes_cov.dart` | same, **coverage-guided** | reaches tensor-loading / folding / fusion behind valid protobuf | same |
| `tokenizer_text.dart` | `*.encode` / `encodePair` | runtime user text | *(none — encode is total)* |
| `tokenizer_json.dart` | `*.fromJson` | a `tokenizer.json` config | `FormatException` |

## Run

```bash
# Blind (fast, ~1M execs/s) — the first pass; catches crashes, hangs, bombs.
dart run tool/fuzz/onnx_bytes.dart
dart run tool/fuzz/tokenizer_text.dart
dart run tool/fuzz/tokenizer_json.dart

# Coverage-guided — for the deep paths behind valid protobuf. Needs the VM
# service. Optional arg = library suffix to score coverage on.
dart run --enable-vm-service=0 --no-pause-isolates-on-exit \
  tool/fuzz/onnx_bytes_cov.dart onnx_proto_loader.dart
dart run --enable-vm-service=0 --no-pause-isolates-on-exit \
  tool/fuzz/onnx_bytes_cov.dart onnx_graph.dart
```

Exit code: `0` clean & fast, `1` escapes found (a bug — with a minimized
reproducer), `2` clean-but-slow. `FUZZ_BUDGET_MS` / `FUZZ_ITERS` shorten a run
(CI uses this). The coverage-guided corpus and any crashes persist under
`.corpus/` and `.crashes/` (git-ignored).

## The hardening loop

1. Blind-fuzz → fix each escape, wrapping the fix in `// GUARD:name >>> … <<<`.
2. Add the minimized reproducer as a regression test
   (`test/parser_robustness_test.dart`, `test/tokenizer_robustness_test.dart`).
3. Prove the guard is load-bearing:
   ```bash
   dart run covfuzz:mutverify --file lib/onnx_runtime_dart.dart \
     --guard protobuf_leak --test 'dart test test/parser_robustness_test.dart'
   ```
4. Coverage-guided for the paths blind can't reach; repeat.

## Guards in place (all mutverified)

- `protobuf_leak` (`lib/onnx_runtime_dart.dart`) — every protobuf decode failure
  → `FormatException` (was leaking `InvalidProtocolBufferException`).
- `wp_config` / `uni_config` / `bpe_config` (tokenizers) — a malformed
  `tokenizer.json` → `FormatException` (was leaking cast/type errors).

The regression tests are the permanent gate and run in `dart test`; the
harnesses above are for periodic deep runs and finding new issues.
