/// Greedy KV-cache generation with the pure-Dart runtime, checked against the
/// ORT reference from tool/smollm2_generate.py. Proves the GroupQueryAttention
/// present_key/value outputs feed correctly back as past_key/value across many
/// autoregressive steps.
///
///   dart run tool/smollm2_generate.dart MODEL.onnx REF.json
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

Future<void> main(List<String> args) async {
  final model =
      loadOnnxModel(args[0], lastTokenLogits: args.contains('--lasttok'));
  final wk = args.indexOf('--workers');
  final workers = wk >= 0 ? int.parse(args[wk + 1]) : 0;
  if (workers > 0) await model.parallelize(workers: workers);
  final ref = jsonDecode(File(args[1]).readAsStringSync()) as Map<String, dynamic>;
  final prompt = (ref['prompt'] as List).cast<int>();
  final nNew = ref['n_new'] as int;
  final nLayers = ref['n_layers'] as int;
  final kvHeads = ref['kv_heads'] as int;
  final headSize = ref['head_size'] as int;
  final refGen = (ref['generated'] as List).cast<int>();

  // past cache: name -> [1, kvHeads, len, headSize], starts empty.
  final past = <String, Tensor>{};
  Tensor emptyPast() =>
      Tensor.float(Float32List(0), [1, kvHeads, 0, headSize]);
  for (var l = 0; l < nLayers; l++) {
    past['past_key_values.$l.key'] = emptyPast();
    past['past_key_values.$l.value'] = emptyPast();
  }

  final outNames = <String>[
    'logits',
    for (var l = 0; l < nLayers; l++) ...['present.$l.key', 'present.$l.value']
  ];

  var cur = List<int>.from(prompt);
  var total = 0;
  final generated = <int>[];
  final sw = Stopwatch()..start();
  for (var step = 0; step < nNew; step++) {
    final seq = cur.length;
    final inputs = <String, Tensor>{
      'input_ids':
          Tensor.int64(Int64List.fromList(cur), [1, seq]),
      'attention_mask':
          Tensor.int64(Int64List(total + seq)..fillRange(0, total + seq, 1),
              [1, total + seq]),
      'position_ids': Tensor.int64(
          Int64List.fromList([for (var i = 0; i < seq; i++) total + i]),
          [1, seq]),
      ...past,
    };
    final out = workers > 0
        ? await model.runAsync(inputs, outNames)
        : model.run(inputs, outNames);
    final logits = out['logits']!;
    final vocab = logits.shape[2];
    final base = (logits.shape[1] - 1) * vocab; // last token's row
    final lf = logits.f ?? logits.asFloatList();
    var best = 0;
    var bestV = lf[base];
    for (var i = 1; i < vocab; i++) {
      if (lf[base + i] > bestV) {
        bestV = lf[base + i];
        best = i;
      }
    }
    generated.add(best);
    total += seq;
    for (var l = 0; l < nLayers; l++) {
      past['past_key_values.$l.key'] = out['present.$l.key']!;
      past['past_key_values.$l.value'] = out['present.$l.value']!;
    }
    cur = [best];
  }
  sw.stop();
  model.dispose();

  var matches = 0;
  for (var i = 0; i < generated.length && i < refGen.length; i++) {
    if (generated[i] == refGen[i]) matches++;
  }
  print('prompt    $prompt');
  print('ORT       $refGen');
  print('dart      $generated');
  print('token match: $matches/${refGen.length}  '
      '(${sw.elapsedMilliseconds}ms for $nNew steps)');
  if (matches != refGen.length) {
    stderr.writeln('MISMATCH: greedy token streams diverge');
    exit(1);
  }
  print('PASS: identical greedy generation');
}
