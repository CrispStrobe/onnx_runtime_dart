/// Checks the Dart BpeTokenizer against a HuggingFace `tokenizers` reference.
///   dart run tool/tok_check.dart TOKENIZER.json REF.json
library;

import 'dart:convert';
import 'dart:io';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

void main(List<String> args) {
  final tok = BpeTokenizer.fromFile(args[0]);
  final ref = jsonDecode(File(args[1]).readAsStringSync()) as List;
  var pass = 0, fail = 0;
  for (final c in ref) {
    final text = c['text'] as String;
    final want = (c['ids'] as List).cast<int>();
    final got = tok.encode(text);
    final ok = got.length == want.length &&
        List.generate(got.length, (i) => got[i] == want[i]).every((x) => x);
    final round = tok.decode(got, skipSpecial: false);
    final roundOk = round == text;
    if (ok && roundOk) {
      pass++;
    } else {
      fail++;
      stderr.writeln('MISMATCH for ${jsonEncode(text)}');
      if (!ok) stderr.writeln('  ids want ${want.length}: $want\n  ids got  ${got.length}: $got');
      if (!roundOk) stderr.writeln('  decode: ${jsonEncode(round)}');
    }
  }
  print('encode+roundtrip: $pass/${pass + fail} exact');
  if (fail > 0) exit(1);
  print('PASS');
}
