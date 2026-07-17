/// Decodes an ONNX ModelProto's weight tensors (TensorProto) into our
/// [Tensor] type. Handles both encodings ONNX allows: values inlined in the
/// typed repeated fields (float_data/int64_data) or packed as raw
/// little-endian bytes (raw_data) — exporters use either depending on
/// tensor size.
library;

import 'dart:typed_data';

import 'onnx.pb.dart';
import 'tensor.dart';

// ONNX TensorProto.DataType values this runtime handles. float32 and int64 are
// the core; int32 and bool are widened to our int64 tensor (bool as 0/1) since
// embedding / reranking graphs use them for shapes, indices and masks.
const int _kFloat = 1;
const int _kUint8 = 2;
const int _kInt8 = 3;
const int _kInt32 = 6;
const int _kInt64 = 7;
const int _kBool = 9;
const int _kFloat16 = 10;

/// Expands an IEEE-754 half-precision bit pattern [h] to the bit pattern of the
/// equivalent float32 (so fp16 weights can be widened without `dart:math`).
int halfToFloat32Bits(int h) {
  final sign = (h & 0x8000) << 16;
  var exp = (h >> 10) & 0x1F;
  var mant = h & 0x3FF;
  if (exp == 0) {
    if (mant == 0) return sign; // signed zero
    // Subnormal half → normalize into a float32 normal.
    var e = -1;
    do {
      mant <<= 1;
      e++;
    } while ((mant & 0x400) == 0);
    mant &= 0x3FF;
    return sign | ((127 - 15 - e) << 23) | (mant << 13);
  } else if (exp == 0x1F) {
    return sign | 0x7F800000 | (mant << 13); // inf / nan
  }
  return sign | ((exp - 15 + 127) << 23) | (mant << 13);
}

/// Reads the bytes for an external-data weight: `(location, offset, length)`
/// from a companion file. See [OnnxModel.fromFile].
typedef ExternalDataResolver = Uint8List Function(
    String location, int offset, int length);

/// The raw little-endian bytes of [t] — either inline `raw_data` or, for
/// external tensors, resolved from the companion file via [ext].
Uint8List _rawBytes(TensorProto t, ExternalDataResolver? ext) {
  if (t.dataLocation == TensorProto_DataLocation.EXTERNAL) {
    if (ext == null) {
      throw StateError('"${t.name}" stores its weights externally — load the '
          'model with OnnxModel.fromFile so the companion data file is found');
    }
    var location = '';
    var offset = 0;
    var length = 0;
    for (final e in t.externalData) {
      switch (e.key) {
        case 'location':
          location = e.value;
        case 'offset':
          offset = int.parse(e.value);
        case 'length':
          length = int.parse(e.value);
      }
    }
    return ext(location, offset, length);
  }
  return Uint8List.fromList(t.rawData);
}

Tensor tensorFromProto(TensorProto t, {ExternalDataResolver? ext}) {
  final shape = t.dims.map((d) => d.toInt()).toList();
  final n = shape.fold<int>(1, (a, b) => a * b);

  if (t.dataType == _kFloat) {
    if (t.floatData.isNotEmpty) {
      return Tensor.float(Float32List.fromList(t.floatData), shape);
    }
    final bd = ByteData.sublistView(_rawBytes(t, ext));
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getFloat32(i * 4, Endian.little);
    }
    return Tensor.float(out, shape);
  } else if (t.dataType == _kInt64) {
    if (t.int64Data.isNotEmpty) {
      return Tensor.int64(
          Int64List.fromList(t.int64Data.map((v) => v.toInt()).toList()),
          shape);
    }
    final bd = ByteData.sublistView(_rawBytes(t, ext));
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getInt64(i * 8, Endian.little);
    }
    return Tensor.int64(out, shape);
  } else if (t.dataType == _kInt32) {
    if (t.int32Data.isNotEmpty) {
      return Tensor.int64(Int64List.fromList(t.int32Data), shape);
    }
    final bd = ByteData.sublistView(_rawBytes(t, ext));
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getInt32(i * 4, Endian.little);
    }
    return Tensor.int64(out, shape);
  } else if (t.dataType == _kFloat16) {
    // Half-precision weights, widened to float32 (bit-exact).
    final src = ByteData.sublistView(_rawBytes(t, ext));
    final out = Float32List(n);
    final outBits = Uint32List.view(out.buffer);
    for (int i = 0; i < n; i++) {
      outBits[i] = halfToFloat32Bits(src.getUint16(i * 2, Endian.little));
    }
    return Tensor.float(out, shape);
  } else if (t.dataType == _kUint8 || t.dataType == _kInt8) {
    // Quantized weights / zero points, widened to int64. int32_data is the
    // inline carrier for both per the proto spec.
    if (t.int32Data.isNotEmpty) {
      return Tensor.int64(Int64List.fromList(t.int32Data), shape);
    }
    final bytes = _rawBytes(t, ext);
    final out = Int64List(n);
    final signed = t.dataType == _kInt8;
    for (int i = 0; i < n; i++) {
      final b = bytes[i];
      out[i] = signed && b > 127 ? b - 256 : b;
    }
    return Tensor.int64(out, shape);
  } else if (t.dataType == _kBool) {
    // BOOL is one byte per element, carried as int64 0/1.
    final bytes = _rawBytes(t, ext);
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = (i < bytes.length && bytes[i] != 0) ? 1 : 0;
    }
    return Tensor.int64(out, shape);
  }
  throw UnsupportedError(
      'Unsupported ONNX TensorProto dataType ${t.dataType} for "${t.name}"');
}
