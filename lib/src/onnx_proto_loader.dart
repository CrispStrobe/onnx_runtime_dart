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
const int _kDouble = 11;

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

/// Compresses a float32 bit pattern to the nearest half-precision bit
/// pattern (round to nearest, ties to even) — the inverse of
/// [halfToFloat32Bits], used by `Cast(to: FLOAT16)` to reproduce fp16
/// rounding semantics.
int float32ToHalfBits(int f) {
  final sign = (f >> 16) & 0x8000;
  final exp = (f >> 23) & 0xFF;
  var mant = f & 0x7FFFFF;
  if (exp == 0xFF) return sign | 0x7C00 | (mant != 0 ? 0x200 : 0); // inf/nan
  final e = exp - 127 + 15;
  if (e >= 0x1F) return sign | 0x7C00; // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign; // underflow -> signed zero
    mant |= 0x800000;
    final shift = 14 - e;
    var half = mant >> shift;
    final rem = mant & ((1 << shift) - 1);
    final halfway = 1 << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1) != 0)) half++;
    return sign | half;
  }
  var half = (e << 10) | (mant >> 13);
  final rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0)) half++;
  return sign | half; // mantissa carry rolls into the exponent correctly
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

/// Element count = product of [shape], validated. A malformed model can carry
/// a negative dimension (or dims whose product overflows to negative), which
/// would otherwise surface as an opaque RangeError when allocating the typed
/// output list. Reject it with a clear message instead.
int _elementCount(List<int> shape, TensorProto t) {
  var n = 1;
  for (final d in shape) {
    if (d < 0) {
      throw FormatException('Tensor "${t.name}" has a negative dimension $d');
    }
    n *= d;
  }
  if (n < 0) {
    throw FormatException('Tensor "${t.name}" shape ${shape} overflows');
  }
  return n;
}

/// Reject a raw_data buffer too short to hold [n] elements of [elemSize] bytes.
/// The division form avoids overflowing `n * elemSize` for a huge declared
/// shape. Without this, the per-element `getFloat32`/`getInt64`/… reads run
/// off the end of the buffer with an opaque RangeError.
void _needRawLen(TensorProto t, int haveBytes, int n, int elemSize) {
  if (n > haveBytes ~/ elemSize) {
    throw FormatException('Tensor "${t.name}" raw_data too short: $haveBytes '
        'bytes for $n elements of $elemSize bytes');
  }
}

/// Reject inline typed data (float_data/int64_data/…) whose length disagrees
/// with the shape product [n]. A well-formed tensor carries exactly [n]
/// values; a mismatch would otherwise build a Tensor whose data length and
/// declared shape disagree — a silent corruption that detonates downstream.
void _needInlineLen(TensorProto t, int have, int n) {
  if (have != n) {
    throw FormatException('Tensor "${t.name}" inline data length $have does '
        'not match shape product $n');
  }
}

Tensor tensorFromProto(TensorProto t, {ExternalDataResolver? ext}) {
  final shape = t.dims.map((d) => d.toInt()).toList();
  final n = _elementCount(shape, t);

  if (t.dataType == _kFloat) {
    if (t.floatData.isNotEmpty) {
      _needInlineLen(t, t.floatData.length, n);
      return Tensor.float(Float32List.fromList(t.floatData), shape);
    }
    final bytes = _rawBytes(t, ext);
    _needRawLen(t, bytes.length, n, 4);
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getFloat32(i * 4, Endian.little);
    }
    return Tensor.float(out, shape);
  } else if (t.dataType == _kInt64) {
    if (t.int64Data.isNotEmpty) {
      _needInlineLen(t, t.int64Data.length, n);
      return Tensor.int64(
          Int64List.fromList(t.int64Data.map((v) => v.toInt()).toList()),
          shape);
    }
    final bytes = _rawBytes(t, ext);
    _needRawLen(t, bytes.length, n, 8);
    final bd = ByteData.sublistView(bytes);
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getInt64(i * 8, Endian.little);
    }
    return Tensor.int64(out, shape);
  } else if (t.dataType == _kInt32) {
    if (t.int32Data.isNotEmpty) {
      _needInlineLen(t, t.int32Data.length, n);
      return Tensor.int64(Int64List.fromList(t.int32Data), shape);
    }
    final bytes = _rawBytes(t, ext);
    _needRawLen(t, bytes.length, n, 4);
    final bd = ByteData.sublistView(bytes);
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getInt32(i * 4, Endian.little);
    }
    return Tensor.int64(out, shape);
  } else if (t.dataType == _kDouble) {
    // float64 weights, narrowed to our float32 carrier.
    if (t.doubleData.isNotEmpty) {
      _needInlineLen(t, t.doubleData.length, n);
      final out = Float32List(n);
      for (int i = 0; i < n; i++) {
        out[i] = t.doubleData[i];
      }
      return Tensor.float(out, shape);
    }
    final bytes = _rawBytes(t, ext);
    _needRawLen(t, bytes.length, n, 8);
    final bd = ByteData.sublistView(bytes);
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getFloat64(i * 8, Endian.little);
    }
    return Tensor.float(out, shape);
  } else if (t.dataType == _kFloat16) {
    // Half-precision weights, widened to float32 (bit-exact).
    final bytes = _rawBytes(t, ext);
    _needRawLen(t, bytes.length, n, 2);
    final src = ByteData.sublistView(bytes);
    final out = Float32List(n);
    final outBits = Uint32List.view(out.buffer);
    for (int i = 0; i < n; i++) {
      outBits[i] = halfToFloat32Bits(src.getUint16(i * 2, Endian.little));
    }
    return Tensor.float(out, shape);
  } else if (t.dataType == _kUint8 || t.dataType == _kInt8) {
    // Quantized weights / zero points — kept in compact 1-byte storage
    // (int32_data is the inline carrier for both per the proto spec).
    final signed = t.dataType == _kInt8;
    if (t.int32Data.isNotEmpty) {
      _needInlineLen(t, t.int32Data.length, n);
      return signed
          ? Tensor.int8(Int8List.fromList(t.int32Data), shape)
          : Tensor.uint8(Uint8List.fromList(t.int32Data), shape);
    }
    final bytes = _rawBytes(t, ext);
    _needRawLen(t, bytes.length, n, 1);
    return signed
        ? Tensor.int8(Int8List.sublistView(bytes, 0, n), shape)
        : Tensor.uint8(Uint8List.sublistView(bytes, 0, n), shape);
  } else if (t.dataType == _kBool) {
    // BOOL is one byte per element, carried as int64 0/1.
    final bytes = _rawBytes(t, ext);
    _needRawLen(t, bytes.length, n, 1);
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bytes[i] != 0 ? 1 : 0;
    }
    return Tensor.int64(out, shape);
  }
  throw UnsupportedError(
      'Unsupported ONNX TensorProto dataType ${t.dataType} for "${t.name}"');
}
