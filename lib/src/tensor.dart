/// Minimal N-dimensional tensor used by the ONNX graph interpreter.
///
/// Row-major (C order) storage, matching ONNX's own tensor layout — so
/// weight bytes read straight out of a TensorProto need no reshuffling,
/// only reinterpreting as the right element type. Supports float32 and
/// int64 since the graph mixes both (embeddings/indices are int64; the
/// actual math is float32).
library;

import 'dart:typed_data';

enum DType { float32, int64, uint8, int8 }

class Tensor {
  final List<int> shape;
  final DType dtype;
  final Float32List? f;
  final Int64List? i;

  /// Compact storage for quantized tensors — 1 byte/element instead of the
  /// 8x blow-up of widening to int64 (a 1 GB int8 model stays 1 GB).
  final Uint8List? u8;
  final Int8List? i8;

  Tensor.float(Float32List data, this.shape)
      : dtype = DType.float32,
        f = data,
        i = null,
        u8 = null,
        i8 = null {
    _checkLength(data.length);
  }

  Tensor.int64(Int64List data, this.shape)
      : dtype = DType.int64,
        i = data,
        f = null,
        u8 = null,
        i8 = null {
    _checkLength(data.length);
  }

  Tensor.uint8(Uint8List data, this.shape)
      : dtype = DType.uint8,
        u8 = data,
        f = null,
        i = null,
        i8 = null {
    _checkLength(data.length);
  }

  Tensor.int8(Int8List data, this.shape)
      : dtype = DType.int8,
        i8 = data,
        f = null,
        i = null,
        u8 = null {
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

  /// The integer backing store, whichever width it is — `Uint8List`,
  /// `Int8List` and `Int64List` all implement `List<int>`. Use this instead
  /// of `.i!` so compact quantized tensors work everywhere int64 ones do.
  List<int> get intData => (i ?? u8 ?? i8)!;

  double getD(int flatIdx) =>
      isFloat ? f![flatIdx] : intData[flatIdx].toDouble();
  int getI(int flatIdx) =>
      isFloat ? _floatToInt(f![flatIdx]) : intData[flatIdx];

  // int64 saturation bounds. Written as parsed values, not literals, so this
  // compiles under dart2js — on the web `int` is a 53-bit double and a 64-bit
  // literal (0x7FFFFFFFFFFFFFFF) is a compile error. Exact on the VM; the
  // nearest double on the web, which only matters when saturating a non-finite
  // Cast, so the web's imprecision there is harmless.
  static final int _int64Max = int.parse('9223372036854775807');
  static final int _int64Min = int.parse('-9223372036854775808');

  /// Float -> int with saturation for non-finite values (matching ORT's
  /// ARM Cast semantics: +-inf saturate, NaN -> 0) — Dart's toInt() throws.
  static int _floatToInt(double v) {
    if (v.isNaN) return 0;
    if (v.isInfinite) {
      return v > 0 ? _int64Max : _int64Min;
    }
    return v.toInt();
  }

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
    final src = intData;
    final out = Float32List(src.length);
    for (int k = 0; k < src.length; k++) {
      out[k] = src[k].toDouble();
    }
    return out;
  }

  /// Returns an Int64-backed copy of this tensor's data (truncating floats,
  /// widening compact quantized storage).
  Int64List asIntList() {
    if (i != null) return i!;
    if (!isFloat) {
      final src = intData;
      final out = Int64List(src.length);
      for (int k = 0; k < src.length; k++) {
        out[k] = src[k];
      }
      return out;
    }
    final out = Int64List(f!.length);
    for (int k = 0; k < f!.length; k++) {
      out[k] = _floatToInt(f![k]);
    }
    return out;
  }

  Tensor reshape(List<int> newShape) {
    final resolved = List<int>.from(newShape);
    // ONNX Reshape: 0 copies the dim from the input shape at that position.
    // Zeros MUST resolve before -1 inference — with a target like [0, 0, -1]
    // the copied dims are part of the known product (inferring first made
    // -1 swallow the whole length).
    for (int k = 0; k < resolved.length; k++) {
      if (resolved[k] == 0 && k < shape.length) resolved[k] = shape[k];
    }
    final negIdx = resolved.indexOf(-1);
    if (negIdx != -1) {
      final knownProduct =
          resolved.where((d) => d != -1).fold<int>(1, (a, b) => a * b);
      resolved[negIdx] = length ~/ (knownProduct == 0 ? 1 : knownProduct);
    }
    return switch (dtype) {
      DType.float32 => Tensor.float(f!, resolved),
      DType.int64 => Tensor.int64(i!, resolved),
      DType.uint8 => Tensor.uint8(u8!, resolved),
      DType.int8 => Tensor.int8(i8!, resolved),
    };
  }

  @override
  String toString() => 'Tensor($dtype, shape=$shape)';
}
