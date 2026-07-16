/// Minimal N-dimensional tensor used by the ONNX graph interpreter.
///
/// Row-major (C order) storage, matching ONNX's own tensor layout — so
/// weight bytes read straight out of a TensorProto need no reshuffling,
/// only reinterpreting as the right element type. Supports float32 and
/// int64 since the graph mixes both (embeddings/indices are int64; the
/// actual math is float32).
library;

import 'dart:typed_data';

enum DType { float32, int64 }

class Tensor {
  final List<int> shape;
  final DType dtype;
  final Float32List? f;
  final Int64List? i;

  Tensor.float(Float32List data, this.shape)
      : dtype = DType.float32,
        f = data,
        i = null {
    _checkLength(data.length);
  }

  Tensor.int64(Int64List data, this.shape)
      : dtype = DType.int64,
        i = data,
        f = null {
    _checkLength(data.length);
  }

  factory Tensor.scalarFloat(double v) =>
      Tensor.float(Float32List.fromList([v]), const []);
  factory Tensor.scalarInt(int v) =>
      Tensor.int64(Int64List.fromList([v]), const []);

  factory Tensor.filledFloat(List<int> shape, double value) {
    final n = shape.fold<int>(1, (a, b) => a * b);
    return Tensor.float(Float32List(n)..fillRange(0, n, value), shape);
  }

  void _checkLength(int len) {
    final expected = shape.fold<int>(1, (a, b) => a * b);
    assert(len == expected,
        'Tensor data length $len != product of shape $shape ($expected)');
  }

  int get length => shape.fold<int>(1, (a, b) => a * b);
  int get rank => shape.length;
  bool get isFloat => dtype == DType.float32;

  double getD(int flatIdx) => isFloat ? f![flatIdx] : i![flatIdx].toDouble();
  int getI(int flatIdx) => isFloat ? f![flatIdx].toInt() : i![flatIdx];

  List<int> get strides {
    final s = List<int>.filled(shape.length, 1);
    for (int k = shape.length - 2; k >= 0; k--) {
      s[k] = s[k + 1] * shape[k + 1];
    }
    return s;
  }

  /// Returns a Float32-backed copy of this tensor's data (casting ints if needed).
  Float32List asFloatList() {
    if (isFloat) return f!;
    final out = Float32List(i!.length);
    for (int k = 0; k < i!.length; k++) {
      out[k] = i![k].toDouble();
    }
    return out;
  }

  /// Returns an Int64-backed copy of this tensor's data (truncating floats if needed).
  Int64List asIntList() {
    if (!isFloat) return i!;
    final out = Int64List(f!.length);
    for (int k = 0; k < f!.length; k++) {
      out[k] = f![k].toInt();
    }
    return out;
  }

  Tensor reshape(List<int> newShape) {
    final resolved = List<int>.from(newShape);
    final negIdx = resolved.indexOf(-1);
    if (negIdx != -1) {
      final knownProduct =
          resolved.where((d) => d != -1).fold<int>(1, (a, b) => a * b);
      resolved[negIdx] = length ~/ (knownProduct == 0 ? 1 : knownProduct);
    }
    // ONNX Reshape allows 0 to mean "copy dim from input shape at that position"
    for (int k = 0; k < resolved.length; k++) {
      if (resolved[k] == 0 && k < shape.length) resolved[k] = shape[k];
    }
    return isFloat ? Tensor.float(f!, resolved) : Tensor.int64(i!, resolved);
  }

  @override
  String toString() => 'Tensor($dtype, shape=$shape)';
}
