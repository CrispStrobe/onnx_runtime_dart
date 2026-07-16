import 'dart:io';
import 'package:onnx_dart/onnx_proto.dart';

void main(List<String> args) {
  final m = ModelProto.fromBuffer(File(args[0]).readAsBytesSync());
  final n = m.graph.node.firstWhere((n) => n.name == args[1]);
  print('${n.opType}("${n.name}")');
  for (final a in n.attribute) {
    print('  attr ${a.name} type=${a.type} i=${a.i} ints=${a.ints}');
  }
}
