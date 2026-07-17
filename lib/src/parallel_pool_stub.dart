/// Web stub for the isolate GEMM pool — parallel execution needs real
/// isolates, which only exist on native targets. `OnnxModel.parallelize`
/// throws here; the sync single-threaded path works everywhere.
library;

import 'dart:async';
import 'dart:typed_data';

class PartitionedWeight {
  final int k, n;
  final List<int> colCounts;
  PartitionedWeight(this.k, this.n, this.colCounts);
  int colStart(int w) => colCounts.take(w).fold(0, (a, b) => a + b);
}

class GemmPool {
  Map<String, PartitionedWeight> get weights => const {};
  int get workerCount => 0;

  static Future<GemmPool> spawn(
          int workers, Map<String, (Float32List, int, int)> toPartition) =>
      throw UnsupportedError(
          'Parallel execution requires isolates (native targets only)');

  Future<Float32List> matmul(String name, Float32List a, int m) =>
      throw UnsupportedError('no pool on this platform');

  void dispose() {}
}
