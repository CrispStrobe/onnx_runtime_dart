/// Blind robustness fuzzing of the tokenizer config loaders
/// (`*.fromJson(String)`), which parse a `tokenizer.json`. Contract: a
/// malformed or structurally-wrong config must reject with FormatException —
/// never leak a cast/type error. Seeds are the tiny committed configs.
///
///   dart run tool/fuzz/tokenizer_json.dart
library;

import 'dart:io';

import 'package:covfuzz/covfuzz.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

int _env(String k, int def) =>
    int.tryParse(Platform.environment[k] ?? '') ?? def;

void main() {
  final wpSeed = File('test/data/tiny_wordpiece.json').readAsStringSync();
  final uniSeed = File('test/data/tiny_unigram.json').readAsStringSync();

  var worst = 0;
  for (final spec in [
    ('WordPiece.fromJson', wpSeed, (String s) => WordPieceTokenizer.fromJson(s)),
    ('Unigram.fromJson', uniSeed, (String s) => UnigramTokenizer.fromJson(s)),
    ('BPE.fromJson', uniSeed, (String s) => BpeTokenizer.fromJson(s)),
  ]) {
    final (name, seed, entry) = spec;
    stdout.writeln('--- fuzzing $name ---');
    final report = fuzz<String>(
      seeds: [seed],
      entry: (s) => entry(s),
      mutate: (s, rng) => mutateString(s, rng, maxOps: 16),
      isClean: (e) => e is FormatException, // the only allowed reject
      iterations: _env('FUZZ_ITERS', 150000),
      budgetMs: _env('FUZZ_BUDGET_MS', 40000),
      stressors: ['', '{}', '{"model":{}}', '[]', 'null', '{"model":null}'],
    );
    report.report();
    final code = report.escapes.isNotEmpty
        ? 1
        : (report.maxSingleMs > 3000 ? 2 : 0);
    if (code > worst) worst = code;
  }
  exit(worst);
}
