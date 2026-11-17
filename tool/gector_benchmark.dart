/// Exact-input, dual-head GECToR qualification against a pinned ORT reference.
///
/// Build AOT, then run (all paths are local; this tool never downloads models):
///   dart compile exe tool/gector_benchmark.dart -o build/gector_benchmark
///   build/gector_benchmark --model model.onnx --tokenizer tokenizer.json \
///     --fixtures reference/reference.json --report dart-report.json
///
/// Consumes schema_version 1 / kind quds-gector-reference emitted by the Quds
/// reference exporter. Times each case's first pass; validates every recorded
/// pass afterwards. Reference float buffers are not loaded during timing.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

const _heads = ['logits_labels', 'logits_d'];

Future<void> main(List<String> args) async {
  if (args.contains('--help')) {
    stdout.writeln(
      'gector_benchmark --model FILE --tokenizer FILE '
      '--fixtures FILE --report FILE [--warmup 2] [--iterations 10] '
      '[--workers 0] [--narrow-direct false]',
    );
    return;
  }
  String? reportPath;
  try {
    final options = _Options.parse(args);
    reportPath = options.report;
    _writeReport(reportPath, {
      'schema_version': 1,
      'kind': 'onnx-runtime-dart-gector-benchmark',
      'passed': false,
      'complete': false,
      'stage': 'validating_fixture',
      'experiments': options.experimentNames,
    });
    final report = await _benchmark(options);
    _writeReport(reportPath, report);
    if (report['passed'] != true) exitCode = 1;
  } catch (error, stack) {
    stderr.writeln('GECToR qualification failed: $error');
    stderr.writeln(stack);
    if (reportPath != null) {
      final existing = File(reportPath);
      Map<String, dynamic> previous = {};
      if (existing.existsSync()) {
        try {
          previous = _map(jsonDecode(existing.readAsStringSync()), 'report');
        } catch (_) {
          // Still emit a useful failure if a stale report was malformed.
        }
      }
      _writeReport(reportPath, {
        ...previous,
        'schema_version': 1,
        'kind': 'onnx-runtime-dart-gector-benchmark',
        'passed': false,
        'complete': false,
        'error': error.toString(),
      });
    }
    exitCode = 1;
  }
}

class _Options {
  final String model, tokenizer, fixtures, report;
  final int warmup, iterations, workers;
  final bool narrowDirect;
  const _Options(
    this.model,
    this.tokenizer,
    this.fixtures,
    this.report,
    this.warmup,
    this.iterations,
    this.workers,
    this.narrowDirect,
  );

  Set<OnnxExperiment> get experiments =>
      narrowDirect ? const {OnnxExperiment.narrowDirectGemm} : const {};
  List<String> get experimentNames =>
      [for (final value in experiments) value.name];

  factory _Options.parse(List<String> args) {
    const allowed = {
      '--model',
      '--tokenizer',
      '--fixtures',
      '--report',
      '--warmup',
      '--iterations',
      '--workers',
      '--narrow-direct',
    };
    final values = <String, String>{};
    for (var i = 0; i < args.length; i += 2) {
      if (!allowed.contains(args[i]) || i + 1 == args.length) {
        throw ArgumentError('Unknown or incomplete option: ${args[i]}');
      }
      if (values.containsKey(args[i])) {
        throw ArgumentError('Repeated option: ${args[i]}');
      }
      values[args[i]] = args[i + 1];
    }
    String required(String key) =>
        values[key] ?? (throw ArgumentError('$key is required'));
    final warmup = int.parse(values['--warmup'] ?? '2');
    final iterations = int.parse(values['--iterations'] ?? '10');
    final workers = int.parse(values['--workers'] ?? '0');
    final direct = values['--narrow-direct'] ?? 'false';
    if (direct != 'true' && direct != 'false') {
      throw ArgumentError('--narrow-direct must be true or false');
    }
    if (warmup < 0 || iterations < 1 || workers < 0) {
      throw ArgumentError(
        'warmup/workers must be >= 0; iterations must be > 0',
      );
    }
    return _Options(
      required('--model'),
      required('--tokenizer'),
      required('--fixtures'),
      required('--report'),
      warmup,
      iterations,
      workers,
      direct == 'true',
    );
  }
}

Future<Map<String, Object?>> _benchmark(_Options options) async {
  final fixtureFile = File(options.fixtures);
  final fixtureHash = await _fileHash(fixtureFile);
  final fixture = _map(
    jsonDecode(await fixtureFile.readAsString()),
    'manifest',
  );
  _require(
    fixture['schema_version'] == 1,
    'Unsupported fixture schema_version',
  );
  _require(
    fixture['kind'] == 'quds-gector-reference',
    'Unexpected fixture kind',
  );
  final hashes = _map(fixture['input_sha256'], 'input_sha256');
  final inferenceModel = fixture['inference_model'] == null
      ? null
      : _map(fixture['inference_model'], 'inference_model');
  if (inferenceModel != null) {
    _require(
      inferenceModel['source_model_sha256'] == hashes['model.onnx'],
      'Candidate source model differs from the pinned source',
    );
    _require(
      await File(options.model).length() == inferenceModel['size_bytes'],
      'Inference model byte count differs',
    );
  }
  final modelHash = await _verifyHash(
    File(options.model),
    inferenceModel?['sha256'] ?? hashes['model.onnx'],
  );
  final tokenizerHash = await _verifyHash(
    File(options.tokenizer),
    hashes['tokenizer.json'],
  );
  final tokenizer = WordPieceTokenizer.fromJson(
    await File(options.tokenizer).readAsString(),
  );
  final cases =
      (fixture['cases'] as List).map((value) => _map(value, 'case')).toList();
  _require(cases.isNotEmpty, 'No reference cases');
  final decoder = _map(fixture['decoder'], 'decoder');
  final keepLabelId = _integer(decoder['keep_label_id'], 'keep_label_id');
  final incorrectLabelId = _integer(
    decoder['incorrect_label_id'],
    'incorrect_label_id',
  );
  final keepConfidence = _number(decoder['keep_confidence'], 'keep_confidence');
  final minErrorProb = _number(decoder['min_error_prob'], 'min_error_prob');

  // Record tokenizer drift separately, then replay the exact saved inputs to
  // retain backend evidence. Overall qualification still fails on any drift.
  // Malformed feed structures remain fatal before loading the large model.
  var passCount = 0;
  final ids = <String>{};
  final tokenizerFailures = <Map<String, Object?>>[];
  for (final testCase in cases) {
    final id = testCase['id'] as String;
    _require(ids.add(id), 'Duplicate case id: $id');
    final passes = testCase['passes'] as List;
    _require(passes.isNotEmpty, '$id has no recorded passes');
    for (var p = 0; p < passes.length; p++) {
      final pass = _map(passes[p], '$id pass $p');
      try {
        _validateTokenization(tokenizer, pass, '$id pass ${p + 1}');
      } on StateError catch (error) {
        tokenizerFailures.add({
          'id': id,
          'pass': p + 1,
          'error': error.toString(),
        });
      }
      _inputs(pass);
      passCount++;
    }
  }

  final timing = <Map<String, Object?>>[];
  final validation = <Map<String, Object?>>[];
  final progress = <String, Object?>{
    'schema_version': 1,
    'kind': 'onnx-runtime-dart-gector-benchmark',
    'passed': false,
    'complete': false,
    'stage': 'loading_model',
    'model': fixture['model'],
    'revision': fixture['revision'],
    'model_sha256': modelHash,
    'model_variant': inferenceModel?['variant'] ?? 'float32',
    'tokenizer_sha256': tokenizerHash,
    'fixture_sha256': fixtureHash,
    'experiments': options.experimentNames,
    'expected_cases': cases.length,
    'expected_passes': passCount,
    'tokenizer_passed': tokenizerFailures.isEmpty,
    'tokenizer_failures': tokenizerFailures,
    'cases_timed': 0,
    'passes_validated': 0,
    'timings': timing,
    'validation': validation,
  };
  void checkpoint(String stage) {
    progress['stage'] = stage;
    progress['cases_timed'] = timing.length;
    progress['passes_validated'] = validation.length;
    progress['checkpoint_peak_rss_bytes'] = ProcessInfo.maxRss;
    _writeReport(options.report, progress);
  }

  checkpoint('loading_model');
  final rssBeforeLoad = ProcessInfo.currentRss;
  final setup = Stopwatch()..start();
  // Retain all token positions and both heads. Hardware-sensitive strategies
  // stay disabled unless explicitly selected by the benchmark caller.
  final model = loadOnnxModel(options.model, experiments: options.experiments);
  setup.stop();
  final rssAfterLoad = ProcessInfo.currentRss;
  _require(
    _same(model.inputSpecs.map((s) => s.name).toList()..sort(), [
      'attention_mask',
      'input_ids',
    ]),
    'Model inputs differ from reference contract',
  );
  _require(
    model.inputSpecs.every((s) => s.elemType == 7),
    'GECToR reference inputs must be ONNX int64',
  );
  for (final head in _heads) {
    _require(model.outputNames.contains(head), 'Missing graph output: $head');
  }

  final poolSetup = Stopwatch()..start();
  if (options.workers > 0) await model.parallelize(workers: options.workers);
  poolSetup.stop();
  final rssAfterPool = ProcessInfo.currentRss;
  progress.addAll({
    'graph_load_seconds': setup.elapsedMicroseconds / 1e6,
    'pool_setup_seconds': poolSetup.elapsedMicroseconds / 1e6,
    'rss_before_load_bytes': rssBeforeLoad,
    'rss_after_load_bytes': rssAfterLoad,
    'rss_after_pool_bytes': rssAfterPool,
    'runtime': 'onnx_runtime_dart',
    'backend': options.workers == 0 ? 'dart-cpu' : 'dart-cpu-isolate-pool',
    'workers': options.workers,
    'dart_version': Platform.version,
    'operating_system': Platform.operatingSystem,
    'operating_system_version': Platform.operatingSystemVersion,
    'processor_count': Platform.numberOfProcessors,
  });
  checkpoint('first_inference');
  Future<Map<String, Tensor>> run(
    Map<String, Tensor> inputs, {
    ExecutionProfile? profile,
  }) async =>
      options.workers > 0
          ? await model.runAsync(inputs, _heads, profile: profile)
          : model.run(inputs, _heads, profile: profile);

  var passed = true;
  late final int inferencePeakRss;
  late final double coldFirstInferenceSeconds;
  late final Map<String, Object?> profileReport;
  try {
    // This is the process's first forward pass after graph/pool setup. The
    // model file cache is already warm from hashing; this is not cold disk I/O.
    final firstPass = _map((cases.first['passes'] as List).first, 'first pass');
    final firstInputs = _inputs(firstPass);
    final cold = Stopwatch()..start();
    final coldOutputs = await run(firstInputs);
    cold.stop();
    coldFirstInferenceSeconds = cold.elapsedMicroseconds / 1e6;
    _validateOutputStructure(coldOutputs, firstPass);
    progress['cold_first_inference_seconds'] = coldFirstInferenceSeconds;
    checkpoint('timing');

    // Warm-up and measured inference use exact first-pass input ids, not
    // arbitrary BERT ids. No reference output arrays are read in this phase.
    for (final testCase in cases) {
      final pass = _map((testCase['passes'] as List).first, 'first pass');
      final inputs = _inputs(pass);
      for (var w = 0; w < options.warmup; w++) {
        _validateOutputStructure(await run(inputs), pass);
      }
      final micros = <int>[];
      for (var i = 0; i < options.iterations; i++) {
        final sw = Stopwatch()..start();
        final result = await run(inputs);
        sw.stop();
        _validateOutputStructure(result, pass);
        micros.add(sw.elapsedMicroseconds);
      }
      final stats = _timingStats(micros);
      timing.add({
        'id': testCase['id'],
        'pass': 1,
        'token_count': inputs['input_ids']!.shape[1],
        'warmup_iterations': options.warmup,
        'measured_iterations': options.iterations,
        ...stats,
      });
      stderr.writeln(
        '${testCase['id']}: median=${stats['median_seconds']}s '
        'p95=${stats['p95_seconds']}s',
      );
      checkpoint('timing');
    }
    inferencePeakRss = ProcessInfo.maxRss;
    progress['inference_peak_rss_bytes'] = inferencePeakRss;
    checkpoint('profiling');

    // One separate diagnostic pass; profiling never affects timed samples or
    // the inference-phase RSS high-water mark recorded above.
    final profile = ExecutionProfile();
    final profiled = Stopwatch()..start();
    final profiledOutputs = await run(firstInputs, profile: profile);
    profiled.stop();
    _validateOutputStructure(profiledOutputs, firstPass);
    final operators = profile.microsByOp.keys.toList()
      ..sort(
        (a, b) => profile.microsByOp[b]!.compareTo(profile.microsByOp[a]!),
      );
    profileReport = {
      'id': cases.first['id'],
      'pass': 1,
      'wall_seconds': profiled.elapsedMicroseconds / 1e6,
      'operator_total_seconds': profile.totalMicros / 1e6,
      'operators': [
        for (final op in operators)
          {
            'op': op,
            'calls': profile.callsByOp[op],
            'seconds': profile.microsByOp[op]! / 1e6,
          },
      ],
    };
    progress['diagnostic_profile'] = profileReport;
    checkpoint('validation');

    // Validation is deliberately after every timed case: parsing reference
    // float arrays cannot inflate the reported inference-phase high-water mark.
    for (final testCase in cases) {
      final passes = testCase['passes'] as List;
      for (var p = 0; p < passes.length; p++) {
        final pass = _map(passes[p], 'pass');
        final got = await run(_inputs(pass));
        _validateOutputStructure(got, pass);
        final references = _map(pass['outputs'], 'outputs');
        final headMetrics = <String, Object?>{};
        var passOk = true;
        for (final name in _heads) {
          final want = await _readReference(
            fixtureFile.parent,
            _map(references[name], name),
            name,
          );
          final metrics = _compare(got[name]!.asFloatList(), want);
          headMetrics[name] = metrics;
          if (metrics['passed'] != true) passOk = false;
        }
        final selected = _validateSelected(
          got,
          pass,
          keepLabelId: keepLabelId,
          incorrectLabelId: incorrectLabelId,
          keepConfidence: keepConfidence,
          minErrorProb: minErrorProb,
        );
        if (selected['passed'] != true) passOk = false;
        validation.add({
          'id': testCase['id'],
          'pass': p + 1,
          'passed': passOk,
          'heads': headMetrics,
          'selected': selected,
        });
        passed = passed && passOk;
        checkpoint('validation');
      }
    }
  } finally {
    model.dispose();
  }

  return {
    ...progress,
    'schema_version': 1,
    'kind': 'onnx-runtime-dart-gector-benchmark',
    'passed': passed && tokenizerFailures.isEmpty,
    'runtime_passed': passed,
    'complete': true,
    'stage': 'complete',
    'model': fixture['model'],
    'revision': fixture['revision'],
    'model_sha256': modelHash,
    'model_variant': inferenceModel?['variant'] ?? 'float32',
    'tokenizer_sha256': tokenizerHash,
    'fixture_sha256': fixtureHash,
    'runtime': 'onnx_runtime_dart',
    'backend': options.workers == 0 ? 'dart-cpu' : 'dart-cpu-isolate-pool',
    'workers': options.workers,
    'dart_version': Platform.version,
    'operating_system': Platform.operatingSystem,
    'operating_system_version': Platform.operatingSystemVersion,
    'processor_count': Platform.numberOfProcessors,
    'executed_at_utc': DateTime.now().toUtc().toIso8601String(),
    'graph_load_seconds': setup.elapsedMicroseconds / 1e6,
    'pool_setup_seconds': poolSetup.elapsedMicroseconds / 1e6,
    'cold_first_inference_seconds': coldFirstInferenceSeconds,
    'rss_before_load_bytes': rssBeforeLoad,
    'rss_after_load_bytes': rssAfterLoad,
    'rss_after_pool_bytes': rssAfterPool,
    'inference_peak_rss_bytes': inferencePeakRss,
    'post_validation_peak_rss_bytes': ProcessInfo.maxRss,
    'timing_scope': 'Exact first-pass inference only; tokenizer, hashes, '
        'reference buffers and decoding excluded. Graph load follows hash '
        'verification, so filesystem caches may already be warm.',
    'qualification_scope': 'Float32 output logits and selected tags on saved '
        'inputs against ORT for the identical model file; '
        'not an independently implemented complete correction/guard loop.',
    'thread_scope': 'workers controls the Dart GEMM isolate pool, not total '
        'process threads. Operators without pool support stay on the caller '
        'isolate, including MatMulNBits; it is not equivalent to ORT intra-op '
        'thread scheduling.',
    'cases_timed': timing.length,
    'passes_validated': passCount,
    'numeric_thresholds': {
      'minimum_cosine': 0.999999,
      'absolute_floor': 1e-3,
      'relative_to_max_magnitude': 2.5e-4,
      'selected_probability_absolute': 1e-4,
    },
    'timings': timing,
    'diagnostic_profile': profileReport,
    'validation': validation,
  };
}

void _validateTokenization(
  WordPieceTokenizer tokenizer,
  Map<String, dynamic> pass,
  String label,
) {
  final words = (pass['words'] as List).cast<String>();
  final actual = tokenizer.encodePretokenized(words);
  final expectedIds = _oneRow(pass['input_ids'], '$label input_ids');
  final expectedWords = (pass['word_ids'] as List).cast<int?>();
  _require(_same(actual.ids, expectedIds), '$label token ids differ');
  _require(_same(actual.wordIds, expectedWords), '$label word ids differ');
  _require(
    actual.ids.isNotEmpty && actual.ids.first == tokenizer.clsId,
    '$label must start with the configured CLS token',
  );
  final first = <int>[0]; // CLS represents the reference runner's $START.
  final seen = <int>{};
  for (var i = 1; i < actual.wordIds.length; i++) {
    final word = actual.wordIds[i];
    if (word != null && seen.add(word)) first.add(i);
  }
  _require(
    _same(first, (pass['first_subword_indices'] as List).cast<int>()),
    '$label first-subword positions differ',
  );
}

Map<String, Tensor> _inputs(Map<String, dynamic> pass) {
  final ids = _oneRow(pass['input_ids'], 'input_ids');
  final mask = _oneRow(pass['attention_mask'], 'attention_mask');
  _require(ids.isNotEmpty && ids.length == mask.length, 'Invalid input shapes');
  _require(ids.every((v) => v >= 0), 'Negative token id');
  _require(
    mask.every((v) => v == 1),
    'Reference must be unpadded batch-one text',
  );
  final shape = [1, ids.length];
  return {
    'input_ids': Tensor.int64(Int64List.fromList(ids), shape),
    'attention_mask': Tensor.int64(Int64List.fromList(mask), shape),
  };
}

void _validateOutputStructure(
  Map<String, Tensor> outputs,
  Map<String, dynamic> pass,
) {
  final references = _map(pass['outputs'], 'outputs');
  final tokens = _oneRow(pass['input_ids'], 'input_ids').length;
  for (final name in _heads) {
    final reference = _map(references[name], name);
    _require(
      reference['dtype'] == 'float32',
      '$name reference must be float32',
    );
    final shape = (reference['shape'] as List).cast<int>();
    _require(
      shape.length == 3 &&
          shape[0] == 1 &&
          shape[1] == tokens &&
          shape[2] == (name == 'logits_d' ? 2 : 5000),
      '$name reference shape invalid',
    );
    final actual =
        outputs[name] ?? (throw StateError('Missing required output: $name'));
    _require(actual.dtype == DType.float32, '$name dtype is not float32');
    _require(_same(actual.shape, shape), '$name shape differs from reference');
  }
}

Future<Float32List> _readReference(
  Directory directory,
  Map<String, dynamic> descriptor,
  String name,
) async {
  _require(
    descriptor['dtype'] == 'float32' &&
        descriptor['byte_order'] == 'little' &&
        descriptor['layout'] == 'row-major',
    '$name reference encoding differs',
  );
  final relative = descriptor['file'] as String;
  final uri = Uri.parse(relative);
  _require(
    !uri.isAbsolute &&
        !uri.hasAuthority &&
        !relative.startsWith('/') &&
        !relative.contains('\\') &&
        !uri.pathSegments.contains('..'),
    'Unsafe reference path: $relative',
  );
  final file = File.fromUri(directory.uri.resolveUri(uri));
  await _verifyHash(file, descriptor['sha256']);
  final bytes = await file.readAsBytes();
  final shape = (descriptor['shape'] as List).cast<int>();
  final count = shape.fold<int>(1, (a, b) => a * b);
  _require(
    bytes.length == descriptor['size_bytes'] && bytes.length == count * 4,
    '$name reference byte count differs',
  );
  final view = ByteData.sublistView(bytes);
  final result = Float32List(count);
  for (var i = 0; i < count; i++) {
    result[i] = view.getFloat32(i * 4, Endian.little);
  }
  return result;
}

Map<String, Object> _compare(Float32List actual, Float32List expected) {
  _require(actual.length == expected.length, 'Reference output length differs');
  double maxAbs = 0, maxMagnitude = 0, dot = 0, normA = 0, normB = 0;
  for (var i = 0; i < actual.length; i++) {
    final a = actual[i], b = expected[i];
    _require(a.isFinite && b.isFinite, 'Non-finite logit at index $i');
    maxAbs = math.max(maxAbs, (a - b).abs());
    maxMagnitude = math.max(maxMagnitude, b.abs());
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  final cosine = normA > 0 && normB > 0
      ? dot / math.sqrt(normA * normB)
      : (normA == normB ? 1.0 : 0.0);
  final tolerance = math.max(1e-3, 2.5e-4 * maxMagnitude);
  return {
    'passed': cosine >= 0.999999 && maxAbs <= tolerance,
    'cosine': cosine,
    'max_absolute_error': maxAbs,
    'allowed_absolute_error': tolerance,
  };
}

Map<String, Object> _validateSelected(
  Map<String, Tensor> outputs,
  Map<String, dynamic> pass, {
  required int keepLabelId,
  required int incorrectLabelId,
  required double keepConfidence,
  required double minErrorProb,
}) {
  final labels = outputs['logits_labels']!.asFloatList();
  final detections = outputs['logits_d']!.asFloatList();
  final positions = (pass['first_subword_indices'] as List).cast<int>();
  final selected = _map(pass['selected'], 'selected');
  final expectedIds = (selected['label_ids'] as List).cast<int>();
  final expectedThresholded =
      (selected['thresholded_label_ids'] as List).cast<int>();
  final expectedLabel = (selected['label_probabilities'] as List).cast<num>();
  final expectedWrong =
      (selected['incorrect_probabilities'] as List).cast<num>();
  _require(
    keepLabelId >= 0 &&
        keepLabelId < 5000 &&
        incorrectLabelId >= 0 &&
        incorrectLabelId < 2,
    'Invalid decoder label ids',
  );
  _require(
    positions.length == expectedIds.length &&
        positions.length == expectedThresholded.length &&
        positions.length == expectedLabel.length &&
        positions.length == expectedWrong.length,
    'Selected tag lengths differ',
  );
  final ids = <int>[];
  final thresholded = <int>[];
  final probabilities = <double>[];
  final wrong = <double>[];
  for (final position in positions) {
    _require(
      position >= 0 && position < outputs['logits_labels']!.shape[1],
      'Invalid first-subword position',
    );
    final row = _softmax(labels, position * 5000, 5000);
    row[keepLabelId] += keepConfidence;
    var best = 0;
    for (var i = 1; i < row.length; i++) {
      if (row[i] > row[best]) best = i;
    }
    ids.add(best);
    thresholded.add(row[best] >= minErrorProb ? best : keepLabelId);
    probabilities.add(row[best]);
    wrong.add(_softmax(detections, position * 2, 2)[incorrectLabelId]);
  }
  if (wrong.isNotEmpty && wrong.reduce(math.max) < minErrorProb) {
    thresholded.fillRange(0, thresholded.length, keepLabelId);
  }
  double maxLabelError = 0, maxWrongError = 0;
  var mismatches = 0, thresholdedMismatches = 0;
  for (var i = 0; i < positions.length; i++) {
    if (ids[i] != expectedIds[i]) mismatches++;
    if (thresholded[i] != expectedThresholded[i]) thresholdedMismatches++;
    maxLabelError = math.max(
      maxLabelError,
      (probabilities[i] - expectedLabel[i]).abs(),
    );
    maxWrongError = math.max(
      maxWrongError,
      (wrong[i] - expectedWrong[i]).abs(),
    );
  }
  return {
    'passed': mismatches == 0 &&
        thresholdedMismatches == 0 &&
        maxLabelError <= 1e-4 &&
        maxWrongError <= 1e-4,
    'tag_mismatches': mismatches,
    'thresholded_tag_mismatches': thresholdedMismatches,
    'max_label_probability_error': maxLabelError,
    'max_incorrect_probability_error': maxWrongError,
  };
}

List<double> _softmax(Float32List values, int start, int length) {
  var maximum = values[start];
  for (var i = 1; i < length; i++) {
    maximum = math.max(maximum, values[start + i]);
  }
  final result = List<double>.filled(length, 0);
  double sum = 0;
  for (var i = 0; i < length; i++) {
    result[i] = math.exp(values[start + i] - maximum);
    sum += result[i];
  }
  for (var i = 0; i < length; i++) {
    result[i] /= sum;
  }
  return result;
}

Map<String, Object> _timingStats(List<int> micros) {
  final sorted = [...micros]..sort();
  final n = sorted.length;
  final median = n.isOdd
      ? sorted[n ~/ 2].toDouble()
      : (sorted[n ~/ 2 - 1] + sorted[n ~/ 2]) / 2;
  return {
    'samples_seconds': [for (final sample in micros) sample / 1e6],
    'minimum_seconds': sorted.first / 1e6,
    'median_seconds': median / 1e6,
    'p95_seconds': sorted[(n * 0.95).ceil() - 1] / 1e6,
    'maximum_seconds': sorted.last / 1e6,
  };
}

List<int> _oneRow(Object? value, String label) {
  _require(
    value is List && value.length == 1 && value.first is List,
    '$label must have shape [1,T]',
  );
  return ((value as List).single as List)
      .map((item) => _integer(item, label))
      .toList();
}

Map<String, dynamic> _map(Object? value, String label) {
  _require(value is Map<String, dynamic>, '$label must be an object');
  return value as Map<String, dynamic>;
}

int _integer(Object? value, String label) {
  _require(value is int, '$label must be an integer');
  return value as int;
}

double _number(Object? value, String label) {
  _require(value is num && value.isFinite, '$label must be a finite number');
  return (value as num).toDouble();
}

bool _same(List<Object?> a, List<Object?> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

Future<String> _fileHash(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

Future<String> _verifyHash(File file, Object? expected) async {
  _require(
    expected is String && RegExp(r'^[a-f0-9]{64}$').hasMatch(expected),
    'Missing or invalid SHA-256 for ${file.path}',
  );
  final actual = await _fileHash(file);
  _require(actual == expected, 'SHA-256 mismatch for ${file.path}');
  return actual;
}

void _writeReport(String path, Map<String, Object?> report) {
  final file = File(path);
  file.parent.createSync(recursive: true);
  final temporary = File('$path.partial');
  temporary.writeAsStringSync(
    '${const JsonEncoder.withIndent('  ').convert(report)}\n',
  );
  temporary.renameSync(path);
}

void _require(bool condition, String message) {
  if (!condition) throw StateError(message);
}
