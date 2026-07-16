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
OnnxModel loadOnnxModel(String path) {
  final file = File(path);
  final dir = file.parent.path;
  final open = <String, RandomAccessFile>{};

  Uint8List resolve(String location, int offset, int length) {
    final raf =
        open.putIfAbsent(location, () => File('$dir/$location').openSync());
    raf.setPositionSync(offset);
    return raf.readSync(length);
  }

  try {
    return OnnxModel.fromBytes(file.readAsBytesSync(), externalData: resolve);
  } finally {
    for (final raf in open.values) {
      raf.closeSync();
    }
  }
}
