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
OnnxModel loadOnnxModel(String path,
    {bool lastTokenLogits = false,
    Set<OnnxExperiment> experiments = const {}}) {
  final file = File(path);
  final dir = file.parent.path;
  final open = <String, RandomAccessFile>{};
  String? containmentDirPath;

  /// Canonical model directory, resolved once per load. (guard:extdata)
  String containmentDir() {
    try {
      return containmentDirPath ??= Directory(dir).resolveSymbolicLinksSync();
    } on FileSystemException catch (e) {
      throw FormatException('model directory unreadable ($e)');
    }
  }

  /// Opens the companion file after symlink containment. (guard:extdata)
  RandomAccessFile openExternal(String location) {
    // GUARD:extdata >>>
    checkExternalRef(location, 0, 0, 0); // path safety before touching disk
    final root = containmentDir();
    String resolved;
    try {
      resolved = File('$dir/$location').resolveSymbolicLinksSync();
    } on FileSystemException catch (e) {
      throw FormatException(
          'external-data file missing or unreadable "$location" ($e)');
    }
    final prefix = '$root${Platform.pathSeparator}';
    if (resolved != root && !resolved.startsWith(prefix)) {
      throw FormatException(
          'external-data path "$location" leaves the model directory '
          'after symlink resolution');
    }
    try {
      return File(resolved).openSync();
    } on FileSystemException catch (e) {
      throw FormatException('external-data file "$location" unreadable ($e)');
    }
    // GUARD:extdata <<<
  }

  /// Companion size read at range-check time. (guard:extdata)
  int rafLength(RandomAccessFile raf, String location) {
    try {
      return raf.lengthSync();
    } on FileSystemException catch (e) {
      throw FormatException(
          'external-data file "$location" vanished before the range check ($e)');
    }
  }

  /// Reads the declared range, rejecting a short read. (guard:extdata)
  Uint8List readBounded(
      RandomAccessFile raf, String location, int offset, int length) {
    try {
      raf.setPositionSync(offset);
      final data = raf.readSync(length);
      if (data.length < length) {
        throw FormatException(
            'external-data file "$location" shrank below the declared range');
      }
      return data;
    } on FileSystemException catch (e) {
      throw FormatException('external-data read failed for "$location" ($e)');
    }
  }

  Uint8List resolve(String location, int offset, int length) {
    final raf = open.putIfAbsent(location, () => openExternal(location));
    final fileLen = rafLength(raf, location);
    checkExternalRef(location, offset, length, fileLen);
    return readBounded(raf, location, offset, length);
  }

  try {
    return OnnxModel.fromBytes(file.readAsBytesSync(),
        externalData: resolve,
        lastTokenLogits: lastTokenLogits,
        experiments: experiments);
  } finally {
    for (final raf in open.values) {
      raf.closeSync();
    }
  }
}

/// Guards a model-declared external-data reference before it touches disk. A
/// hostile `.onnx` must not read arbitrary files or trigger a huge allocation:
/// [location] must be a plain relative path inside the model's own directory
/// (no `..`, no absolute path, no drive/volume, no NUL byte), and
/// `[offset, offset+length)` must lie within the companion file's [fileLen]
/// bytes. Violations reject with [FormatException] — the documented reject
/// type — never a leaked `FileSystemException`/`RangeError`/OOM.
/// (guard:extdata)
///
/// String level only. Filesystem-level checks run at open time.
void checkExternalRef(String location, int offset, int length, int fileLen) {
  // GUARD:extdata >>>
  if (location.isEmpty ||
      location.contains('\x00') ||
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
    throw FormatException(
        'external-data range [$offset, +$length) lies outside '
        'the companion file ($fileLen bytes)');
  }
  // GUARD:extdata <<<
}
