import 'dart:io';
import 'package:onnx_runtime_dart/onnx_proto.dart';

void main(List<String> args) {
  final model = ModelProto.fromBuffer(File(args[0]).readAsBytesSync());
  final counts = <String, int>{};
  for (final n in model.graph.node) {
    counts[n.opType] = (counts[n.opType] ?? 0) + 1;
  }
  final sorted = counts.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));
  print('nodes=${model.graph.node.length}  unique_ops=${counts.length}');
  for (final e in sorted) {
    print('${e.value.toString().padLeft(5)}  ${e.key}');
  }
}
