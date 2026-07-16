/// A small, dependency-light **pure-Dart ONNX inference runtime**.
///
/// No FFI and no native `onnxruntime` — the graph is interpreted in plain
/// Dart, so the same code runs on every Dart/Flutter target, **including the
/// web and WebAssembly**. It implements the operator set needed for
/// transformer / attention style models (see the README for the list); it is
/// not a complete ONNX runtime (no convolution, pooling or RNN ops).
///
/// ```dart
/// import 'dart:typed_data';
/// import 'package:onnx_dart/onnx_dart.dart';
///
/// final model = OnnxModel.fromBytes(bytes); // bytes of a .onnx file
/// final out = model.run(
///   {'input': Tensor.float(Float32List.fromList([1, 2, 3]), [1, 3])},
///   ['output'],
/// );
/// print(out['output']!.asFloatList());
/// ```
///
/// The generated protobuf bindings (`onnx.pb.dart`) derive from the ONNX
/// project's `onnx.proto3` and are used under Apache-2.0; see `NOTICE.md`.
library;

import 'dart:typed_data';

import 'src/onnx.pb.dart';
import 'src/onnx_graph.dart';
import 'src/tensor.dart';

export 'src/tensor.dart' show Tensor, DType;
export 'src/onnx_graph.dart' show OnnxGraphExecutor;

/// A parsed ONNX model ready to run.
///
/// Wraps an [OnnxGraphExecutor]; construct one with [OnnxModel.fromBytes] from
/// the raw bytes of a `.onnx` file, then call [run].
class OnnxModel {
  final OnnxGraphExecutor _executor;

  OnnxModel._(this._executor);

  /// Parses a serialized ONNX model ([bytes] of a `.onnx` file) and prepares
  /// it for execution.
  factory OnnxModel.fromBytes(Uint8List bytes) =>
      OnnxModel._(OnnxGraphExecutor(ModelProto.fromBuffer(bytes)));

  /// Runs the graph with the given named [inputs] and returns the requested
  /// named [outputNames].
  Map<String, Tensor> run(
    Map<String, Tensor> inputs,
    List<String> outputNames,
  ) =>
      _executor.run(inputs, outputNames);
}
