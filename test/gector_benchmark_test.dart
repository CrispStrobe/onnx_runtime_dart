/// Exercise the qualification CLI with a tiny graph; no network or large model.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:test/test.dart';

import '../tool/gector_benchmark.dart' as benchmark;

void main() {
  late Directory directory;
  setUp(() => directory = Directory.systemTemp.createTempSync('gector-test-'));
  tearDown(() => directory.deleteSync(recursive: true));

  List<String> arguments({String? narrowDirect}) => [
        '--model',
        '${directory.path}/model.onnx',
        '--tokenizer',
        '${directory.path}/tokenizer.json',
        '--fixtures',
        '${directory.path}/reference.json',
        '--report',
        '${directory.path}/report.json',
        '--warmup',
        '0',
        '--iterations',
        '1',
        '--workers',
        '0',
        if (narrowDirect != null) ...['--narrow-direct', narrowDirect],
      ];

  Future<Map<String, dynamic>> run(Map<String, dynamic> fixture,
      {String? narrowDirect}) async {
    File('${directory.path}/reference.json')
        .writeAsStringSync(jsonEncode(fixture));
    final previousExit = exitCode;
    try {
      await benchmark.main(arguments(narrowDirect: narrowDirect));
      return jsonDecode(
        File('${directory.path}/report.json').readAsStringSync(),
      ) as Map<String, dynamic>;
    } finally {
      exitCode = previousExit;
    }
  }

  test(
    'qualifies both heads and raw/thresholded tags on every saved pass',
    () async {
      final fixture = _fixture(directory);
      final report = await run(fixture);
      expect(report['passed'], isTrue, reason: report.toString());
      expect(report['experiments'], isEmpty);
      expect(report['complete'], isTrue);
      expect(report['cases_timed'], 1);
      expect(report['passes_validated'], 2);
      expect(report['cold_first_inference_seconds'], isNonNegative);
      expect(report['inference_peak_rss_bytes'], greaterThan(0));
      expect(
        (report['timings'] as List).single['samples_seconds'],
        hasLength(1),
      );
      expect((report['diagnostic_profile'] as Map)['operators'], isA<List>());
    },
  );

  for (final enabled in [false, true]) {
    test('records explicit narrow-direct=$enabled experiment selection',
        () async {
      final report = await run(_fixture(directory), narrowDirect: '$enabled');
      expect(report['passed'], isTrue, reason: report.toString());
      expect(report['experiments'], enabled ? ['narrowDirectGemm'] : isEmpty);
    });
  }

  test('rejects non-boolean narrow-direct values', () async {
    final fixture = _fixture(directory);
    File('${directory.path}/reference.json')
        .writeAsStringSync(jsonEncode(fixture));
    final previousExit = exitCode;
    try {
      for (final value in ['yes', 'TRUE', '1']) {
        exitCode = 0;
        await benchmark.main(arguments(narrowDirect: value));
        expect(exitCode, 1, reason: 'accepted invalid value $value');
      }
      expect(File('${directory.path}/report.json').existsSync(), isFalse);
    } finally {
      exitCode = previousExit;
    }
  });

  test('a missing graph head cannot silently pass', () async {
    final report = await run(_fixture(directory, missingDetectionHead: true),
        narrowDirect: 'true');
    expect(report['passed'], isFalse);
    expect(report['complete'], isFalse);
    expect(report['error'], contains('Missing graph output: logits_d'));
    expect(report['experiments'], ['narrowDirectGemm']);
  });

  test(
    'rejects corrupted reference bytes after preserving measured progress',
    () async {
      final fixture = _fixture(directory);
      File('${directory.path}/logits_d.f32le').writeAsBytesSync([0, 0, 0, 0]);
      final report = await run(fixture);
      expect(report['passed'], isFalse);
      expect(report['complete'], isFalse);
      expect(report['cases_timed'], 1);
      expect(report['error'], contains('SHA-256 mismatch'));
    },
  );

  test(
    'argmax or thresholded tag drift fails even with identical logits',
    () async {
      final fixture = _fixture(directory);
      final selected = fixture['cases'][0]['passes'][1]['selected'] as Map;
      selected['thresholded_label_ids'] = [1, 0, 0];
      final report = await run(fixture);
      expect(report['complete'], isTrue);
      expect(report['passed'], isFalse);
      final row = (report['validation'] as List)[1];
      expect(row['heads']['logits_labels']['passed'], isTrue);
      expect(row['selected']['thresholded_tag_mismatches'], 1);
    },
  );

  test(
    'tokenizer drift fails overall while retaining exact-input runtime evidence',
    () async {
      final fixture = _fixture(directory);
      fixture['cases'][0]['passes'][0]['word_ids'] = [null, 0, 0, null];
      final report = await run(fixture);
      expect(report['passed'], isFalse);
      expect(report['complete'], isTrue);
      expect(report['tokenizer_passed'], isFalse);
      expect(report['runtime_passed'], isTrue);
      expect(report['tokenizer_failures'][0]['error'],
          contains('word ids differ'));
    },
  );
}

Map<String, dynamic> _fixture(
  Directory directory, {
  bool missingDetectionHead = false,
}) {
  final tokenizer = File('test/data/tiny_wordpiece.json')
      .copySync('${directory.path}/tokenizer.json');
  final graph = GraphProto()..name = 'synthetic_dual_head';
  for (final name in ['input_ids', 'attention_mask']) {
    graph.input.add(
      ValueInfoProto()
        ..name = name
        ..type = (TypeProto()
          ..tensorType = (TypeProto_Tensor()
            ..elemType = 7
            ..shape = (TensorShapeProto()
              ..dim.addAll(
                [
                  1,
                  4,
                ].map((x) => TensorShapeProto_Dimension()..dimValue = Int64(x)),
              )))),
    );
  }
  final outputs = <String, dynamic>{};
  for (final name in ['logits_labels', 'logits_d']) {
    final width = name == 'logits_labels' ? 5000 : 2;
    final values = Float32List(4 * width);
    for (var row = 0; row < 4; row++) {
      values[row * width + 1] = width == 5000 ? 10 : 2;
    }
    final bytes = ByteData(values.length * 4);
    for (var i = 0; i < values.length; i++) {
      bytes.setFloat32(i * 4, values[i], Endian.little);
    }
    final file = File('${directory.path}/$name.f32le')
      ..writeAsBytesSync(bytes.buffer.asUint8List());
    outputs[name] = {
      'dtype': 'float32',
      'byte_order': 'little',
      'layout': 'row-major',
      'shape': [1, 4, width],
      'file': '$name.f32le',
      'size_bytes': file.lengthSync(),
      'sha256': _hash(file),
    };
    if (missingDetectionHead && name == 'logits_d') continue;
    graph.initializer.add(
      TensorProto()
        ..name = name
        ..dataType = 1
        ..dims.addAll([1, 4, width].map(Int64.new))
        ..rawData = bytes.buffer.asUint8List(),
    );
    graph.output.add(ValueInfoProto()..name = name);
  }
  final file = File('${directory.path}/model.onnx')
    ..writeAsBytesSync((ModelProto()..graph = graph).writeToBuffer());
  // The winning label's probability is below min_error_prob: raw id 1,
  // thresholded id 0. This catches confusing argmax with the gated decision.
  final probability = math.exp(10) / (math.exp(10) + 4999);
  final wrong = math.exp(2) / (1 + math.exp(2));
  Map<String, dynamic> pass() => {
        'words': ['Hello', 'world'],
        'word_ids': [null, 0, 1, null],
        'first_subword_indices': [0, 1, 2],
        'input_ids': [
          [2, 4, 6, 3],
        ],
        'attention_mask': [
          [1, 1, 1, 1],
        ],
        'outputs': outputs,
        'selected': {
          'label_ids': [1, 1, 1],
          'thresholded_label_ids': [0, 0, 0],
          'label_probabilities': [probability, probability, probability],
          'incorrect_probabilities': [wrong, wrong, wrong],
        },
      };
  return {
    'schema_version': 1,
    'kind': 'quds-gector-reference',
    'model': 'synthetic-test-only',
    'revision': 'test',
    'input_sha256': {
      'model.onnx': _hash(file),
      'tokenizer.json': _hash(tokenizer),
    },
    'inference_model': {
      'variant': 'synthetic',
      'sha256': _hash(file),
      'source_model_sha256': _hash(file),
      'size_bytes': file.lengthSync(),
    },
    'decoder': {
      'keep_label_id': 0,
      'incorrect_label_id': 1,
      'keep_confidence': 0.0,
      'min_error_prob': 0.9,
    },
    'cases': [
      {
        'id': 'tiny',
        'passes': [pass(), pass()],
      },
    ],
  };
}

String _hash(File file) => sha256.convert(file.readAsBytesSync()).toString();
