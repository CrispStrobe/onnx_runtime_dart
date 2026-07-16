import 'dart:io';
import 'package:onnx_dart/onnx_proto.dart';

void main(List<String> args) {
  final m = ModelProto.fromBuffer(File(args[0]).readAsBytesSync());
  print('INPUTS:');
  for (final i in m.graph.input) {
    final t = i.type.tensorType;
    final dims = t.shape.dim
        .map((d) => d.hasDimValue()
            ? '${d.dimValue}'
            : (d.dimParam.isNotEmpty ? d.dimParam : '?'))
        .join(',');
    print('  ${i.name}  dtype=${t.elemType}  shape=[$dims]');
  }
  print('OUTPUTS:');
  for (final o in m.graph.output) {
    final t = o.type.tensorType;
    final dims = t.shape.dim
        .map((d) => d.hasDimValue()
            ? '${d.dimValue}'
            : (d.dimParam.isNotEmpty ? d.dimParam : '?'))
        .join(',');
    print('  ${o.name}  dtype=${t.elemType}  shape=[$dims]');
  }
}
