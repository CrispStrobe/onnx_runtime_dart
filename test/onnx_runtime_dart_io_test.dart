// Tests the native (dart:io) external-data loader: filesystem failures
// normalize to FormatException and symlink containment is enforced.
library;

import 'dart:io';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';
import 'package:onnx_runtime_dart/src/onnx.pb.dart';
import 'package:test/test.dart';

/// Builds a minimal ModelProto whose single FLOAT [2,2] initializer `w` lives
/// in the companion file named [location], offset [offset], length 16 bytes.
List<int> buildModel({String location = 'weights.bin', String offset = '0'}) {
  final t = TensorProto()
    ..name = 'w'
    ..dataType = 1 // FLOAT
    ..dims.addAll([Int64(2), Int64(2)])
    ..dataLocation = TensorProto_DataLocation.EXTERNAL
    ..externalData.add(StringStringEntryProto()
      ..key = 'location'
      ..value = location)
    ..externalData.add(StringStringEntryProto()
      ..key = 'offset'
      ..value = offset)
    ..externalData.add(StringStringEntryProto()
      ..key = 'length'
      ..value = '16');
  return (ModelProto()..graph = (GraphProto()..initializer.add(t)))
      .writeToBuffer();
}

void main() {
  late Directory tmp;
  setUp(() {
    tmp = Directory.systemTemp.createTempSync('extdata_guard_');
  });
  tearDown(() {
    tmp.deleteSync(recursive: true);
  });

  test('happy path: plain companion file loads', () {
    final dir = Directory('${tmp.path}${Platform.pathSeparator}model')
      ..createSync();
    File('${dir.path}/model.onnx').writeAsBytesSync(buildModel());
    File('${dir.path}/weights.bin').writeAsBytesSync(List.filled(16, 0));
    expect(() => loadOnnxModel('${dir.path}/model.onnx'), returnsNormally);
  });

  test('symlink inside the model directory stays inside bounds', () {
    final dir = Directory('${tmp.path}${Platform.pathSeparator}model')
      ..createSync();
    File('${dir.path}/model.onnx').writeAsBytesSync(buildModel());
    File('${dir.path}/real.bin').writeAsBytesSync(List.filled(16, 0));
    Link('${dir.path}/weights.bin').createSync('${dir.path}/real.bin');
    expect(() => loadOnnxModel('${dir.path}/model.onnx'), returnsNormally);
  });

  test('symlink escape rejects with FormatException', () {
    final dir = Directory('${tmp.path}${Platform.pathSeparator}model')
      ..createSync();
    final outside = Directory('${tmp.path}${Platform.pathSeparator}outside')
      ..createSync();
    File('${outside.path}/secret.bin').writeAsBytesSync(List.filled(16, 0));
    File('${dir.path}/model.onnx').writeAsBytesSync(buildModel());
    Link('${dir.path}/weights.bin').createSync('${outside.path}/secret.bin');
    expect(() => loadOnnxModel('${dir.path}/model.onnx'),
        throwsA(isA<FormatException>()));
  });

  test('missing companion file rejects with FormatException', () {
    final dir = Directory('${tmp.path}${Platform.pathSeparator}model')
      ..createSync();
    File('${dir.path}/model.onnx').writeAsBytesSync(buildModel());
    expect(() => loadOnnxModel('${dir.path}/model.onnx'),
        throwsA(isA<FormatException>()));
  });

  test('companion shorter than the declared range rejects with FormatException',
      () {
    final dir = Directory('${tmp.path}${Platform.pathSeparator}model')
      ..createSync();
    File('${dir.path}/model.onnx').writeAsBytesSync(buildModel());
    File('${dir.path}/weights.bin').writeAsBytesSync(List.filled(8, 0));
    expect(() => loadOnnxModel('${dir.path}/model.onnx'),
        throwsA(isA<FormatException>()));
  });

  test('NUL byte in location rejects with FormatException', () {
    final dir = Directory('${tmp.path}${Platform.pathSeparator}model')
      ..createSync();
    File('${dir.path}/model.onnx')
        .writeAsBytesSync(buildModel(location: 'we\x00ights.bin'));
    expect(() => loadOnnxModel('${dir.path}/model.onnx'),
        throwsA(isA<FormatException>()));
  });
}
