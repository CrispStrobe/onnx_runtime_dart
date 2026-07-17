/// Isolate worker pool for parallel GEMM (native targets).
///
/// Each worker permanently owns a contiguous **column slice** of every
/// partitioned weight matrix, assigned once at spawn — so a matmul call
/// only ships the (small) activation matrix to each worker and receives its
/// output-column slice back; the (large) weights never cross an isolate
/// boundary again. Column-sliced GEMM computes the exact same dot products
/// as the local kernel, so results are bitwise identical.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'gemm_kernel_scalar.dart' if (dart.library.ffi) 'gemm_kernel_simd.dart'
    as gemm;

class _WeightSlice {
  final Float32List data; // [k × nCount], row-major
  final int k, nCount;
  _WeightSlice(this.data, this.k, this.nCount);
}

/// One partitioned weight: k rows, n total columns, and each worker's count
/// (worker w owns columns [colStart(w), colStart(w)+colCount(w))).
class PartitionedWeight {
  final int k, n;
  final List<int> colCounts;
  PartitionedWeight(this.k, this.n, this.colCounts);
  int colStart(int w) =>
      colCounts.take(w).fold(0, (a, b) => a + b);
}

void _workerMain(SendPort toMain) {
  final fromMain = ReceivePort();
  toMain.send(fromMain.sendPort);
  final slices = <String, _WeightSlice>{};
  fromMain.listen((msg) {
    if (msg == null) {
      fromMain.close();
      return;
    }
    final map = msg as Map;
    if (map['op'] == 'load') {
      slices[map['name'] as String] = _WeightSlice(
          map['data'] as Float32List, map['k'] as int, map['n'] as int);
      (map['ack'] as SendPort).send(true);
      return;
    }
    // op == 'matmul'
    final s = slices[map['name'] as String]!;
    final a = map['a'] as Float32List;
    final m = map['m'] as int;
    final out = Float32List(m * s.nCount);
    gemm.matmulKernel(a, 0, s.data, 0, out, 0, m, s.k, s.nCount);
    (map['reply'] as SendPort).send({'id': map['id'], 'out': out});
  });
}

class GemmPool {
  final List<SendPort> _workers;
  final List<Isolate> _isolates;
  final ReceivePort _replies;
  final Map<String, PartitionedWeight> weights;
  final Map<int, void Function(int worker, Float32List out)> _pending = {};
  int _nextId = 0;

  GemmPool._(this._workers, this._isolates, this._replies, this.weights);

  int get workerCount => _workers.length;

  /// Spawns [workers] isolates and distributes column slices of every entry
  /// in [toPartition] (name -> row-major [k × n] weight).
  static Future<GemmPool> spawn(
      int workers, Map<String, (Float32List, int, int)> toPartition) async {
    final isolates = <Isolate>[];
    final ports = <SendPort>[];
    for (int w = 0; w < workers; w++) {
      final handshake = ReceivePort();
      isolates.add(await Isolate.spawn(_workerMain, handshake.sendPort));
      ports.add(await handshake.first as SendPort);
      handshake.close();
    }

    final weights = <String, PartitionedWeight>{};
    for (final entry in toPartition.entries) {
      final (data, k, n) = entry.value;
      final base = n ~/ workers, rem = n % workers;
      final counts = [for (int w = 0; w < workers; w++) base + (w < rem ? 1 : 0)];
      weights[entry.key] = PartitionedWeight(k, n, counts);
      int col = 0;
      for (int w = 0; w < workers; w++) {
        final nw = counts[w];
        final slice = Float32List(k * nw);
        for (int r = 0; r < k; r++) {
          slice.setRange(r * nw, (r + 1) * nw, data, r * n + col);
        }
        final ack = ReceivePort();
        ports[w].send({
          'op': 'load',
          'name': entry.key,
          'data': slice,
          'k': k,
          'n': nw,
          'ack': ack.sendPort,
        });
        await ack.first;
        ack.close();
        col += nw;
      }
    }

    final replies = ReceivePort();
    final pool = GemmPool._(ports, isolates, replies, weights);
    replies.listen((msg) {
      final map = msg as Map;
      final entry = pool._pending.remove(map['id'] as int);
      entry?.call(-1, map['out'] as Float32List); // worker id baked into id
    });
    return pool;
  }

  /// `out[m × n] = a[m × k] · W` for a partitioned weight, fanned out across
  /// the workers and reassembled by column range.
  Future<Float32List> matmul(String name, Float32List a, int m) {
    final w = weights[name]!;
    final out = Float32List(m * w.n);
    final completer = Completer<Float32List>();
    int remaining = _workers.length;
    for (int wk = 0; wk < _workers.length; wk++) {
      final id = _nextId++;
      final colStart = w.colStart(wk);
      final nCount = w.colCounts[wk];
      _pending[id] = (_, slice) {
        for (int i = 0; i < m; i++) {
          out.setRange(i * w.n + colStart, i * w.n + colStart + nCount, slice,
              i * nCount);
        }
        if (--remaining == 0) completer.complete(out);
      };
      _workers[wk].send({
        'op': 'matmul',
        'id': id,
        'name': name,
        'a': a,
        'm': m,
        'reply': _replies.sendPort,
      });
    }
    return completer.future;
  }

  void dispose() {
    for (final p in _workers) {
      p.send(null);
    }
    for (final iso in _isolates) {
      iso.kill(priority: Isolate.immediate);
    }
    _replies.close();
  }
}
