/// Reader-robustness regressions for the ONNX parser: malformed input must be
/// rejected with the single documented exception (FormatException), never leak
/// a protobuf-internal exception, a RangeError/StateError/TypeError, or hang.
/// Findings and reproducers come from `tool/fuzz/` (covfuzz).
@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';
import 'package:test/test.dart';

void main() {
  // Minimized reproducer from blind fuzzing: a 1-byte input used to surface
  // the protobuf decoder's InvalidProtocolBufferException; it must now be the
  // documented FormatException (guard:protobuf_leak).
  test('malformed bytes reject as FormatException, not a protobuf exception',
      () {
    expect(() => OnnxModel.fromBytes(Uint8List.fromList([5])),
        throwsFormatException);
  });

  test('malformed protobuf rejects as FormatException', () {
    final malformed = <Uint8List>[
      Uint8List.fromList([0x08]), // truncated varint
      Uint8List.fromList([0xFF, 0xFF, 0xFF, 0xFF]),
      Uint8List.fromList([0x08, ...List.filled(20, 0xFF), 0x7F]), // huge varint
    ];
    for (final c in malformed) {
      expect(() => OnnxModel.fromBytes(c), throwsFormatException,
          reason: 'input ${c.take(8).toList()} should be a clean reject');
    }
  });

  test('any input parses or clean-rejects — never leaks an Error', () {
    // The contract is "parse OR documented reject", not "always throw": an
    // empty message and loosely-tagged bytes are valid (trivial) protobuf.
    final inputs = <Uint8List>[
      Uint8List(0), // valid empty ModelProto
      Uint8List.fromList(List.filled(64, 0x0A)), // valid: skipped fields
      Uint8List.fromList([0x08]), // malformed
      Uint8List.fromList([0x12, 0x7F, 0x08]), // truncated length-delimited
    ];
    for (final c in inputs) {
      try {
        OnnxModel.fromBytes(c);
      } on FormatException {
        // documented clean reject
      } on UnsupportedError {
        // documented clean reject (unsupported op/dtype)
      } catch (e) {
        fail('leaked ${e.runtimeType} on ${c.take(8).toList()}: $e');
      }
    }
  });

  test('a large all-zero blob rejects quickly (no allocation bomb)', () {
    final sw = Stopwatch()..start();
    expect(() => OnnxModel.fromBytes(Uint8List(1024 * 1024)),
        throwsA(anything)); // rejects or parses; must not hang
    expect(sw.elapsedMilliseconds, lessThan(2000));
  });

  group('external-data reference is bounded and can\'t escape the directory',
      () {
    test('path traversal / absolute paths reject', () {
      for (final loc in [
        '../../etc/passwd',
        '/etc/passwd',
        'weights/../../../secret',
        r'C:\Windows\x',
        '',
      ]) {
        expect(() => checkExternalRef(loc, 0, 0, 1 << 20), throwsFormatException,
            reason: 'location "$loc" must be rejected');
      }
    });
    test('out-of-bounds / negative / oversized ranges reject', () {
      const fileLen = 1000;
      expect(() => checkExternalRef('w.bin', -1, 10, fileLen),
          throwsFormatException); // negative offset
      expect(() => checkExternalRef('w.bin', 0, -5, fileLen),
          throwsFormatException); // negative length
      expect(() => checkExternalRef('w.bin', 0, 1 << 40, fileLen),
          throwsFormatException); // huge length (would OOM)
      expect(() => checkExternalRef('w.bin', 990, 20, fileLen),
          throwsFormatException); // runs past EOF
    });
    test('a valid in-bounds relative reference passes', () {
      expect(() => checkExternalRef('weights.bin', 100, 400, 1000),
          returnsNormally);
      expect(() => checkExternalRef('sub/weights.bin', 0, 1000, 1000),
          returnsNormally);
    });
  });
}
