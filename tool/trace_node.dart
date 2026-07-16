import 'dart:io';
import 'package:onnx_runtime_dart/onnx_proto.dart';

// trace_node <model> <nodeName> — prints the node + the producers of its inputs.
void main(List<String> args) {
  final m = ModelProto.fromBuffer(File(args[0]).readAsBytesSync());
  final byOutput = <String, NodeProto>{};
  for (final n in m.graph.node) {
    for (final o in n.output) {
      byOutput[o] = n;
    }
  }
  void show(String name, int depth) {
    if (depth > 8) return;
    final n = byOutput[name];
    final pad = '  ' * depth;
    if (n == null) {
      print('$pad$name  <- (input/initializer)');
      return;
    }
    print(
        '$pad${n.opType}("${n.name}") -> $name  [inputs: ${n.input.join(", ")}]');
    for (final inp in n.input) {
      if (inp.isNotEmpty) show(inp, depth + 1);
    }
  }

  final target = m.graph.node.firstWhere((n) => n.name == args[1]);
  print('TARGET ${target.opType}("${target.name}") inputs: ${target.input}');
  for (final inp in target.input) {
    if (inp.isNotEmpty) show(inp, 1);
  }
}
