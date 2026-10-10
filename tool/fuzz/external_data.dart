/// Blind robustness fuzzing of the external-data guards in two phases.
///
/// Phase 1 (string level): `checkExternalRef` validates a model-declared
/// companion-file `location`/`offset`/`length` before any disk access.
///
/// Phase 2 (filesystem level): `loadOnnxModel` loads a rebuilt minimal model
/// whose companion `location` is a mutated string, from a directory seeded
/// with in-bounds/escaping/broken/looped symlinks, an unreadable companion
/// directory, and a missing companion. The contract for both phases: accept
/// a safe in-bounds reference or reject with FormatException — never leak
/// `FileSystemException`/`RangeError`/OOM, and never let a `..`/absolute
/// path or an out-of-bounds symlink through.
///
///   dart run tool/fuzz/external_data.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:covfuzz/covfuzz.dart';
import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';
import 'package:onnx_runtime_dart/src/onnx.pb.dart';

int _env(String k, int def) =>
    int.tryParse(Platform.environment[k] ?? '') ?? def;

const _seeds = <String>[
  'weights.bin',
  'model.onnx_data',
  'sub/dir/weights.bin',
  '../../etc/passwd', // traversal
  '/abs/path',
  r'C:\win',
  '', // empty
];

// final, not const: padRight is a runtime call.
final _stressors = ['..', '../', 'a/../b', 'x/'.padRight(4096, 'x')];

final _mutate = (String s, dynamic rng) => mutateString(s, rng, maxOps: 10);

/// Builds a minimal ModelProto whose single FLOAT [2,2] initializer `w`
/// lives in the companion named [location] at [offset], length 16 bytes.
Uint8List _externalModelBytes(String location, String offset) =>
    (ModelProto()
      ..graph = (GraphProto()
        ..initializer.add(TensorProto()
          ..name = 'w'
          ..dataType = 1 // FLOAT
          ..dims.addAll([Int64(2), Int64(2)])
          ..dataLocation = TensorProto_DataLocation.EXTERNAL
          ..externalData.add(StringStringEntryProto()
            ..key = 'location'
            ..value = location)
          ..externalData.add(StringStringEntryProto()
            ..key = 'offset'
            ..value = offset)
          ..externalData.add(StringStringEntryProto()
            ..key = 'length'
            ..value = '16'))))
        .writeToBuffer();

/// Seeds a model directory with one companion per behavior branch.
Directory _buildWorld(String parent) {
  final dir = Directory('$parent/model')..createSync();
  final out = Directory('$parent/outside')..createSync();
  File('${dir.path}/weights.bin').writeAsBytesSync(List.filled(16, 0));
  File('${out.path}/secret.bin').writeAsBytesSync(List.filled(16, 0));
  Link('${dir.path}/escape.bin').createSync('${out.path}/secret.bin');
  Link('${dir.path}/inside.bin').createSync('${dir.path}/weights.bin');
  Link('${dir.path}/dangle.bin').createSync('${dir.path}/nowhere.bin');
  Link('${dir.path}/loop.bin').createSync('${dir.path}/loop.bin');
  Directory('${dir.path}/subdir.bin').createSync();
  return dir;
}

void main() {
  print('phase=string');
  final stringReport = fuzz<String>(
    seeds: _seeds,
    entry: (loc) {
      // Vary offset/length/fileLen from the mutated location so the numeric
      // bounds logic is exercised alongside the path check.
      final n = loc.length;
      for (final (off, len, fl) in [
        (0, n, 1000),
        (n, n * 3, 500),
        (-n, n, 1 << 20),
        (n * 7, 1 << 40, 4096),
      ]) {
        try {
          checkExternalRef(loc, off, len, fl);
        } on FormatException {
          // documented clean reject
        }
      }
    },
    mutate: _mutate,
    isClean: (e) => e is FormatException,
    iterations: _env('FUZZ_ITERS', 300000),
    budgetMs: _env('FUZZ_BUDGET_MS', 30000),
    stressors: _stressors,
  );
  stringReport.report();

  print('phase=io');
  final tmp = Directory.systemTemp.createTempSync('fuzz_extdata_io_');
  FuzzReport? ioReport;
  try {
    final modelDir = _buildWorld(tmp.path);
    final modelPath = '${modelDir.path}${Platform.pathSeparator}model.onnx';
    ioReport = fuzz<String>(
      // Entry order: string guard, symlink resolve, containment, open,
      // live range check, read. 'escape.bin'/'inside.bin' hit resolve and
      // containment; 'dangle.bin'/'loop.bin' hit resolve failures;
      // 'subdir.bin' hits open/read; 'weights.bin' hits the range check.
      seeds: [
        'weights.bin',
        'inside.bin',
        'escape.bin',
        'dangle.bin',
        'loop.bin',
        'subdir.bin',
        'missing.bin',
        'sub/dir/weights.bin',
        'a.bin', // length % 5 == 0 like inside.bin
        'we\x00ights.bin',
      ],
      entry: (loc) {
        // Offset varies with the string so the live range check against the
        // real companion length rejects some iterations and accepts others.
        File(modelPath)
            .writeAsBytesSync(_externalModelBytes(loc, '${loc.length % 5}'));
        loadOnnxModel(modelPath);
      },
      mutate: _mutate,
      isClean: (e) => e is FormatException || e is UnsupportedError,
      iterations: _env('FUZZ_IO_ITERS', 15000),
      budgetMs: _env('FUZZ_IO_BUDGET_MS', 20000),
      stressors: _stressors,
    );
    ioReport.report();
  } finally {
    tmp.deleteSync(recursive: true);
  }

  final esc = stringReport.escapes.isNotEmpty || ioReport.escapes.isNotEmpty;
  final slow = stringReport.maxSingleMs > 3000 || ioReport.maxSingleMs > 3000;
  exit(esc ? 1 : (slow ? 2 : 0));
}
