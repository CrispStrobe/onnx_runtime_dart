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
/// import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
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
import 'src/onnx_proto_loader.dart';
import 'src/tensor.dart';

export 'src/tensor.dart' show Tensor, DType;
export 'src/onnx_graph.dart' show OnnxGraphExecutor, ExecutionProfile;
export 'src/onnx_proto_loader.dart' show ExternalDataResolver;

/// A parsed ONNX model ready to run.
///
/// Wraps an [OnnxGraphExecutor]; construct one with [OnnxModel.fromBytes] from
/// the raw bytes of a `.onnx` file, then call [run].
class OnnxModel {
  final OnnxGraphExecutor _executor;

  OnnxModel._(this._executor);

  /// Parses a serialized ONNX model ([bytes] of a `.onnx` file) and prepares
  /// it for execution.
  ///
  /// If the model stores its weights in a companion data file (large models,
  /// `dataLocation == EXTERNAL`), pass an [externalData] resolver that returns
  /// the bytes for a given `(location, offset, length)`. `OnnxModel.fromFile`
  /// in `package:onnx_runtime_dart/onnx_runtime_dart_io.dart` wires this up automatically for
  /// the native (`dart:io`) platforms.
  /// [fuse] controls the load-time pattern fusion (GELU/SDPA/RMSNorm);
  /// disable it to execute the graph exactly node-by-node (diagnostics).
  factory OnnxModel.fromBytes(Uint8List bytes,
          {ExternalDataResolver? externalData, bool fuse = true}) =>
      OnnxModel._(OnnxGraphExecutor(ModelProto.fromBuffer(bytes),
          externalData: externalData, fuse: fuse));

  /// Runs the graph with the given named [inputs] and returns the requested
  /// named [outputNames].
  ///
  /// Pass a [profile] to accumulate per-op-type wall time across the run.
  Map<String, Tensor> run(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
  }) =>
      _executor.run(inputs, outputNames, profile: profile);

  /// Spawns [workers] isolate workers and partitions the model's large
  /// `MatMul` weights across them by output column; afterwards [runAsync]
  /// executes those matmuls in parallel (results stay bitwise identical to
  /// [run]). Native targets only — throws [UnsupportedError] on the web.
  ///
  /// [poolConv] opts 2-D convolutions into the pool as well (output-row
  /// bands); measure before enabling — for typical CNN latencies the
  /// activation copying outweighs the parallel compute.
  ///
  /// Call [dispose] when done to shut the workers down.
  Future<void> parallelize(
          {required int workers,
          int minWeightElements = 65536,
          bool poolConv = false}) =>
      _executor.parallelize(
          workers: workers,
          minWeightElements: minWeightElements,
          poolConv: poolConv);

  /// Like [run], but partitioned matmuls execute on the isolate pool set up
  /// by [parallelize]. Without a prior [parallelize] call it behaves exactly
  /// like [run].
  Future<Map<String, Tensor>> runAsync(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
  }) =>
      _executor.runAsync(inputs, outputNames, profile: profile);

  /// Shuts down the isolate pool (no-op if [parallelize] was never called).
  void dispose() => _executor.dispose();
}
