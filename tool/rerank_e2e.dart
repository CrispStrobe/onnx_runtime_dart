/// End-to-end cross-encoder reranking in pure Dart: pair-tokenize (WordPiece or
/// Unigram, auto-detected) -> ONNX -> relevance logit, checked against a
/// `sentence-transformers` / ORT reference.
///   dart run tool/rerank_e2e.dart MODEL.onnx TOKENIZER.json REF.json
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

(List<int>, List<int>) Function(String, String) _pairEncoder(String path) {
  final type =
      (jsonDecode(File(path).readAsStringSync())['model'] as Map)['type'];
  if (type == 'Unigram') return UnigramTokenizer.fromFile(path).encodePair;
  return WordPieceTokenizer.fromFile(path).encodePair;
}

void main(List<String> a) {
  final model = loadOnnxModel(a[0]);
  final encodePair = _pairEncoder(a[1]);
  final ref = jsonDecode(File(a[2]).readAsStringSync()) as List;
  var worst = 0.0;
  for (final c in ref) {
    final (ids, types) = encodePair(c['a'] as String, c['b'] as String);
    final n = ids.length;
    final out = model.run({
      'input_ids': Tensor.int64(Int64List.fromList(ids), [1, n]),
      'attention_mask': Tensor.int64(Int64List(n)..fillRange(0, n, 1), [1, n]),
      'token_type_ids': Tensor.int64(Int64List.fromList(types), [1, n]),
    }, [
      'logits'
    ])['logits']!;
    final got = (out.f ?? out.asFloatList())[0];
    final want = (c['score'] as num).toDouble();
    final d = (got - want).abs();
    if (d > worst) worst = d;
    print(
        'score=${got.toStringAsFixed(4)} (ref ${want.toStringAsFixed(4)}, Δ${d.toStringAsExponential(1)})  "${(c['b'] as String).substring(0, (c['b'] as String).length.clamp(0, 36))}"');
  }
  model.dispose();
  print('worst |Δ| = ${worst.toStringAsExponential(2)}');
  if (worst > 2e-3) {
    stderr.writeln('FAIL');
    exit(1);
  }
  print('PASS: pure-Dart reranking matches the reference');
}
