/// Native (`dart:io`) helpers for loading ONNX models from disk, including
/// models whose weights live in a companion external-data file.
///
/// This library imports `dart:io`, so it is **not** available on the web —
/// keep using `package:onnx_runtime_dart/onnx_runtime_dart.dart` (and `OnnxModel.fromBytes`)
/// for web / WebAssembly targets.
library;

import 'dart:io';
import 'dart:typed_data';

import 'onnx_runtime_dart.dart';

/// Loads an ONNX model from [path], resolving any external-data weights from
/// the companion file(s) named in the model (relative to [path]'s directory).
///
/// External weights are read on demand with random access, so a model with a
/// multi-gigabyte `.onnx.data` file is not loaded into memory all at once.
OnnxModel loadOnnxModel(String path, {bool lastTokenLogits = false}) {
  final file = File(path);
  final dir = file.parent.path;
  final open = <String, RandomAccessFile>{};

  Uint8List resolve(String location, int offset, int length) {
    final raf = open.putIfAbsent(location, () {
      checkExternalRef(location, 0, 0, 0); // path safety before touching disk
      return File('$dir/$location').openSync();
    });
    checkExternalRef(location, offset, length, raf.lengthSync());
    raf.setPositionSync(offset);
    return raf.readSync(length);
  }

  try {
    return OnnxModel.fromBytes(file.readAsBytesSync(),
        externalData: resolve, lastTokenLogits: lastTokenLogits);
  } finally {
    for (final raf in open.values) {
      raf.closeSync();
    }
  }
}

/// Guards a model-declared external-data reference before it touches disk. A
/// hostile `.onnx` must not read arbitrary files or trigger a huge allocation:
/// [location] must be a plain relative path inside the model's own directory
/// (no `..`, no absolute path, no drive/volume), and `[offset, offset+length)`
/// must lie within the companion file's [fileLen] bytes. Violations reject with
/// [FormatException] — the documented reject type — never a leaked
/// `FileSystemException`/`RangeError`/OOM. (guard:extdata)
void checkExternalRef(String location, int offset, int length, int fileLen) {
  // GUARD:extdata >>>
  if (location.isEmpty ||
      location.contains('..') ||
      location.startsWith('/') ||
      location.startsWith(r'\') ||
      location.contains(':')) {
    throw FormatException('unsafe external-data location: "$location"');
  }
  if (offset < 0 ||
      length < 0 ||
      length > fileLen ||
      offset > fileLen - length) {
    throw FormatException('external-data range [$offset, +$length) lies outside '
        'the companion file ($fileLen bytes)');
  }
  // GUARD:extdata <<<
}
