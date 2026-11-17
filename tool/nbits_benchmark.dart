/// Bounded, model-free A/B diagnostic for the 32-column INT4 dequantization.
///
/// dart compile exe tool/nbits_benchmark.dart -o build/nbits_benchmark
/// build/nbits_benchmark > build/nbits-benchmark.json
///
/// Compares the former full-matrix symmetric INT4 algorithm with the current
/// operator at BERT shapes. Alternates execution order; reports all samples.
/// Peak live matrix storage per case is below 20 MiB. This is not a complete
/// model/checker benchmark and does not establish a latency target.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/src/gemm_kernel_scalar.dart'
    if (dart.library.ffi) 'package:onnx_runtime_dart/src/gemm_kernel_simd.dart'
    as gemm;
import 'package:onnx_runtime_dart/src/onnx_qlinear_ops.dart' as quant;

void main() {
  const repeats = 3;
  const blockSize = 128;
  final reports = <Map<String, Object?>>[];
  for (final (m, k, n) in [
    (7, 768, 768),
    (7, 768, 3072),
    (7, 3072, 768),
    (10, 768, 768),
    (10, 768, 3072),
    (10, 3072, 768),
  ]) {
    final blocks = (k + blockSize - 1) ~/ blockSize;
    final a = Tensor.float(
        Float32List.fromList([
          for (var i = 0; i < m * k; i++) ((i * 31 % 257) - 128) / 128,
        ]),
        [1, m, k]);
    final packed = Uint8List(n * blocks * blockSize ~/ 2);
    for (var i = 0; i < packed.length; i++) {
      packed[i] = (i * 73 + 11) & 255;
    }
    final b = Tensor.uint8(packed, [n, blocks, blockSize ~/ 2]);
    final scales = Tensor.float(
        Float32List.fromList([
          for (var i = 0; i < n * blocks; i++) ((i % 13) - 6) * 0.01,
        ]),
        [n, blocks]);
    Tensor full() => _formerFullMatrix(a, b, scales, k, n, blockSize);
    Tensor tiled() => quant.opMatMulNBits(a, b, scales, null,
        k: k, n: n, bits: 4, blockSize: blockSize);

    // These first calls also warm both paths. Check exact output bits; the
    // optimized path must not introduce a new numerical tolerance.
    final expected = full().asFloatList();
    final actual = tiled().asFloatList();
    final expectedBits = Uint32List.view(
        expected.buffer, expected.offsetInBytes, expected.length);
    final actualBits =
        Uint32List.view(actual.buffer, actual.offsetInBytes, actual.length);
    for (var i = 0; i < expectedBits.length; i++) {
      if (expectedBits[i] != actualBits[i]) {
        throw StateError('INT4 bitwise difference at M=$m K=$k N=$n index=$i');
      }
    }
    final fullTimes = <int>[], tiledTimes = <int>[];
    void measure(Tensor Function() fn, List<int> samples) {
      final watch = Stopwatch()..start();
      final output = fn();
      watch.stop();
      if (output.length != m * n) throw StateError('Unexpected output length');
      samples.add(watch.elapsedMicroseconds);
    }

    for (var i = 0; i < repeats; i++) {
      if (i.isEven) {
        measure(full, fullTimes);
        measure(tiled, tiledTimes);
      } else {
        measure(tiled, tiledTimes);
        measure(full, fullTimes);
      }
    }
    final fullSorted = [...fullTimes]..sort();
    final tiledSorted = [...tiledTimes]..sort();
    reports.add({
      'm': m,
      'k': k,
      'n': n,
      'block_size': blockSize,
      'bitwise_equal': true,
      'full_samples_seconds': [for (final x in fullTimes) x / 1e6],
      'tiled_samples_seconds': [for (final x in tiledTimes) x / 1e6],
      'full_median_seconds': fullSorted[repeats ~/ 2] / 1e6,
      'tiled_median_seconds': tiledSorted[repeats ~/ 2] / 1e6,
      'median_speedup': fullSorted[repeats ~/ 2] / tiledSorted[repeats ~/ 2],
      'full_dequant_scratch_bytes': k * n * 4,
      'tiled_dequant_scratch_bytes': k * 32 * 4,
      'tiled_output_scratch_bytes': m * 32 * 4,
    });
  }
  stdout.writeln(const JsonEncoder.withIndent('  ').convert({
    'schema_version': 1,
    'kind': 'matmulnbits-synthetic-ab',
    'dart_version': Platform.version,
    'warmups_per_arm': 1,
    'measured_iterations_per_arm': repeats,
    'scope':
        'Synthetic symmetric INT4 operator only; no model or correction-loop qualification.',
    'cases': reports,
  }));
}

/// The symmetric three-input path from revision 0b49ca47, kept only as this
/// diagnostic's baseline. The production operator does not retain full weights.
Tensor _formerFullMatrix(
    Tensor a, Tensor b, Tensor scales, int k, int n, int blockSize) {
  final blocks = (k + blockSize - 1) ~/ blockSize;
  final bytesPerBlock = blockSize ~/ 2;
  final bytes = b.u8!, sf = scales.asFloatList();
  final weight = Float32List(k * n);
  for (var col = 0; col < n; col++) {
    for (var block = 0; block < blocks; block++) {
      final scale = sf[col * blocks + block];
      final offset = (col * blocks + block) * bytesPerBlock;
      final first = block * blockSize;
      final end = first + blockSize < k ? first + blockSize : k;
      for (var row = first; row < end; row++) {
        final inside = row - first;
        final byte = bytes[offset + (inside >> 1)];
        final value = (inside & 1) == 0 ? byte & 15 : byte >> 4;
        weight[row * n + col] = (value - 8.0) * scale;
      }
    }
  }
  final m = a.length ~/ k;
  final output = Float32List(m * n);
  gemm.matmulKernel(a.asFloatList(), 0, weight, 0, output, 0, m, k, n);
  return Tensor.float(output, [...a.shape.sublist(0, a.rank - 1), n]);
}
