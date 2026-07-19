import 'dart:convert';
import 'dart:io';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

(List<int>, List<int>) enc(String path, String a, String b) {
  final type =
      (jsonDecode(File(path).readAsStringSync())['model'] as Map)['type'];
  if (type == 'Unigram') {
    return UnigramTokenizer.fromFile(path).encodePair(a, b);
  }
  return WordPieceTokenizer.fromFile(path).encodePair(a, b);
}

void main(List<String> args) {
  final ref = jsonDecode(File(args[1]).readAsStringSync()) as List;
  var pass = 0, fail = 0;
  for (final c in ref) {
    final (ids, types) = enc(args[0], c['a'] as String, c['b'] as String);
    final wi = (c['ids'] as List).cast<int>(),
        wt = (c['types'] as List).cast<int>();
    final ok = ids.length == wi.length &&
        List.generate(ids.length, (i) => ids[i] == wi[i] && types[i] == wt[i])
            .every((x) => x);
    if (ok) {
      pass++;
    } else {
      fail++;
      stderr.writeln('MISMATCH ${c['a']} | ${c['b']}');
      stderr.writeln('  want ids ${wi}\n  got  ids ${ids}');
      stderr.writeln('  want typ ${wt}\n  got  typ ${types}');
    }
  }
  print('pair: $pass/${pass + fail} exact (ids+types)');
  if (fail > 0) exit(1);
  print('PASS');
}
