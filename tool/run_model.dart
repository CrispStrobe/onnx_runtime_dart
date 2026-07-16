import 'dart:io';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';

// run_model <model.onnx> <out.txt>  — feeds fixed BERT-style tokens.
void main(List<String> args) {
  final bytes = Uint8List.fromList(File(args[0]).readAsBytesSync());
  final meta = ModelProto.fromBuffer(bytes);
  final inputNames = meta.graph.input.map((i) => i.name).toSet();
  final outName = meta.graph.output.first.name;

  final ids = [101, 2054, 2003, 1996, 3007, 1997, 2605, 102];
  final seq = ids.length;
  Tensor i64(List<int> v) => Tensor.int64(Int64List.fromList(v), [1, seq]);
  final feed = <String, Tensor>{};
  if (inputNames.contains('input_ids')) feed['input_ids'] = i64(ids);
  if (inputNames.contains('attention_mask')) {
    feed['attention_mask'] = i64(List.filled(seq, 1));
  }
  if (inputNames.contains('token_type_ids')) {
    feed['token_type_ids'] = i64(List.filled(seq, 0));
  }
  if (inputNames.contains('position_ids')) {
    feed['position_ids'] = i64(List.generate(seq, (k) => k));
  }

  final sw = Stopwatch()..start();
  final out = loadOnnxModel(args[0]).run(feed, [outName]);
  sw.stop();
  final y = out[outName]!;
  final f = y.asFloatList();
  stderr
      .writeln('dart: $outName shape=${y.shape} in ${sw.elapsedMilliseconds}ms '
          'first4=${f.take(4).map((v) => v.toStringAsFixed(5)).toList()}');
  File(args[1]).writeAsStringSync(f.map((v) => v.toString()).join('\n'));
}
