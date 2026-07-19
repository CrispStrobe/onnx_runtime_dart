/// End-to-end text generation with the pure-Dart runtime: real text in,
/// real text out. Loads a KV-cache decoder (SmolLM2 / Qwen2.5 / any GQA or
/// decomposed decoder exporting `past_key_values.*` / `present.*`) plus a
/// HuggingFace `tokenizer.json`, applies a ChatML prompt, and streams tokens
/// with greedy or temperature/top-k sampling.
///
///   dart run tool/llm_chat.dart MODEL.onnx TOKENIZER.json "your prompt" \
///       [--max N] [--temp T] [--topk K] [--seed S] [--raw]
///
/// The decoder shape (layer count, KV heads, head size) is discovered from
/// the model's inputSpecs, so no per-model config is needed.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

int _argInt(List<String> a, String flag, int def) {
  final i = a.indexOf(flag);
  return i >= 0 ? int.parse(a[i + 1]) : def;
}

double _argDouble(List<String> a, String flag, double def) {
  final i = a.indexOf(flag);
  return i >= 0 ? double.parse(a[i + 1]) : def;
}

void main(List<String> args) {
  // last-token-logits skips the wasted prompt-row vocab projections in prefill;
  // greedy/sampling only reads the last row anyway. --full-logits disables it.
  final model =
      loadOnnxModel(args[0], lastTokenLogits: !args.contains('--full-logits'));
  final tok = BpeTokenizer.fromFile(args[1]);
  final userPrompt = args[2];
  final maxNew = _argInt(args, '--max', 128);
  final temp = _argDouble(args, '--temp', 0.0); // 0 => greedy
  final topK = _argInt(args, '--topk', 40);
  final seed = _argInt(args, '--seed', 1234);
  final raw = args.contains('--raw');
  final rng = math.Random(seed);

  // Discover the decoder's cache shape from the graph inputs.
  final pastKeys = model.inputSpecs
      .where((s) => s.name.startsWith('past_key_values') &&
          s.name.endsWith('.key'))
      .toList();
  final nLayers = pastKeys.length;
  if (nLayers == 0) {
    stderr.writeln('Model has no past_key_values.* inputs — not a KV-cache '
        'decoder.');
    exit(2);
  }
  final kvHeads = pastKeys.first.shape[1];
  final headSize = pastKeys.first.shape[3];

  final outNames = <String>[
    'logits',
    for (var l = 0; l < nLayers; l++) ...['present.$l.key', 'present.$l.value']
  ];

  // ChatML prompt (Qwen / SmolLM2 style). --raw feeds the prompt verbatim.
  final prompt = raw
      ? userPrompt
      : '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
          '<|im_start|>user\n$userPrompt<|im_end|>\n<|im_start|>assistant\n';
  final promptIds = tok.encode(prompt);
  final eosIds = <int>{
    for (final s in const ['<|im_end|>', '<|endoftext|>'])
      if (tok.specials[s] != null) tok.specials[s]!
  };

  final past = <String, Tensor>{
    for (var l = 0; l < nLayers; l++) ...{
      'past_key_values.$l.key':
          Tensor.float(Float32List(0), [1, kvHeads, 0, headSize]),
      'past_key_values.$l.value':
          Tensor.float(Float32List(0), [1, kvHeads, 0, headSize]),
    }
  };

  stdout.write(raw ? userPrompt : userPrompt);
  stdout.write('\n\x1b[36m'); // cyan for the completion
  final generated = <int>[];
  var cur = List<int>.from(promptIds);
  var total = 0;
  final sw = Stopwatch()..start();
  for (var step = 0; step < maxNew; step++) {
    final seq = cur.length;
    final inputs = <String, Tensor>{
      'input_ids': Tensor.int64(Int64List.fromList(cur), [1, seq]),
      'attention_mask': Tensor.int64(
          Int64List(total + seq)..fillRange(0, total + seq, 1),
          [1, total + seq]),
      'position_ids': Tensor.int64(
          Int64List.fromList([for (var i = 0; i < seq; i++) total + i]),
          [1, seq]),
      ...past,
    };
    final out = model.run(inputs, outNames);
    final logits = out['logits']!;
    final vocab = logits.shape[2];
    final lf = logits.f ?? logits.asFloatList();
    // Row of the last position — logits may already be sliced to [1,1,vocab]
    // (lastTokenLogits) or carry every position ([1,seq,vocab]).
    final base = (logits.shape[1] - 1) * vocab;
    final next = _pickToken(lf, base, vocab, temp, topK, rng);

    if (eosIds.contains(next)) break;
    generated.add(next);
    // Stream the newly-decoded text (decode the whole run to keep multi-byte
    // characters intact, print only the delta).
    stdout.write(tok.decode([...generated.sublist(generated.length - 1)]));

    total += seq;
    for (var l = 0; l < nLayers; l++) {
      past['past_key_values.$l.key'] = out['present.$l.key']!;
      past['past_key_values.$l.value'] = out['present.$l.value']!;
    }
    cur = [next];
  }
  sw.stop();
  model.dispose();
  stdout.write('\x1b[0m\n');
  final tps = generated.length / (sw.elapsedMilliseconds / 1000.0);
  stderr.writeln('[${generated.length} tokens, ${sw.elapsedMilliseconds}ms, '
      '${tps.toStringAsFixed(1)} tok/s, '
      '${temp == 0 ? 'greedy' : 'temp=$temp topk=$topK'}]');
}

/// Greedy (temp 0) or temperature + top-k sampling over one logit row.
int _pickToken(Float32List lf, int base, int vocab, double temp, int topK,
    math.Random rng) {
  if (temp <= 0) {
    var best = 0;
    var bestV = lf[base];
    for (var i = 1; i < vocab; i++) {
      if (lf[base + i] > bestV) {
        bestV = lf[base + i];
        best = i;
      }
    }
    return best;
  }
  // Top-k indices by logit (partial selection via a running k-min heap would
  // be faster; a full sort is fine for a CLI demo).
  final idx = List<int>.generate(vocab, (i) => i);
  idx.sort((a, b) => lf[base + b].compareTo(lf[base + a]));
  final k = math.min(topK, vocab);
  var maxLogit = lf[base + idx[0]];
  final probs = List<double>.filled(k, 0);
  var sum = 0.0;
  for (var i = 0; i < k; i++) {
    final p = math.exp((lf[base + idx[i]] - maxLogit) / temp);
    probs[i] = p;
    sum += p;
  }
  var r = rng.nextDouble() * sum;
  for (var i = 0; i < k; i++) {
    r -= probs[i];
    if (r <= 0) return idx[i];
  }
  return idx[k - 1];
}
