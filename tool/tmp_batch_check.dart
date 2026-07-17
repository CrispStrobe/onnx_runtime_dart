import 'dart:math' as math;
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

void main() {
  final model = loadOnnxModel(
      '/Users/christianstrobele/.cache/onnx_runtime_dart_models/maia3_5m_int32.onnx');
  final rng = math.Random(5);
  Float32List pos() =>
      Float32List.fromList(List.generate(64 * 96, (_) => rng.nextDouble()));
  final positions = [pos(), pos(), pos()];
  final selfElos = [1100, 1500, 1900];
  final oppoElos = [1900, 1500, 1100];

  // Sequential singles.
  final seq = <Float32List>[];
  for (int i = 0; i < 3; i++) {
    final out = model.run({
      'tokens': Tensor.float(positions[i], [1, 64, 96]),
      'self_elo': Tensor.int64(Int64List.fromList([selfElos[i]]), [1]),
      'oppo_elo': Tensor.int64(Int64List.fromList([oppoElos[i]]), [1]),
    }, ['logits_move']);
    seq.add(out['logits_move']!.f!);
  }

  // One batched run.
  final all = Float32List(3 * 64 * 96);
  for (int i = 0; i < 3; i++) {
    all.setAll(i * 64 * 96, positions[i]);
  }
  final batched = model.run({
    'tokens': Tensor.float(all, [3, 64, 96]),
    'self_elo': Tensor.int64(Int64List.fromList(selfElos), [3]),
    'oppo_elo': Tensor.int64(Int64List.fromList(oppoElos), [3]),
  }, ['logits_move'])['logits_move']!;
  print('batched shape: ${batched.shape}');
  final bf = batched.f!;
  for (int i = 0; i < 3; i++) {
    double maxAbs = 0, dot = 0, na = 0, nb = 0;
    final len = seq[i].length;
    for (int k = 0; k < len; k++) {
      final a = seq[i][k], b = bf[i * len + k];
      maxAbs = math.max(maxAbs, (a - b).abs());
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    print('pos $i: cosine=${(dot / (math.sqrt(na) * math.sqrt(nb))).toStringAsFixed(9)} max|Δ|=$maxAbs');
  }
}
