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
const int _kInt32 = 6;
const int _kInt64 = 7;
const int _kBool = 9;

Tensor tensorFromProto(TensorProto t) {
  final shape = t.dims.map((d) => d.toInt()).toList();
  final n = shape.fold<int>(1, (a, b) => a * b);

  if (t.dataType == _kFloat) {
    if (t.floatData.isNotEmpty) {
      return Tensor.float(Float32List.fromList(t.floatData), shape);
    }
    final bd = ByteData.sublistView(Uint8List.fromList(t.rawData));
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
    final bd = ByteData.sublistView(Uint8List.fromList(t.rawData));
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getInt64(i * 8, Endian.little);
    }
    return Tensor.int64(out, shape);
  } else if (t.dataType == _kInt32) {
    if (t.int32Data.isNotEmpty) {
      return Tensor.int64(Int64List.fromList(t.int32Data), shape);
    }
    final bd = ByteData.sublistView(Uint8List.fromList(t.rawData));
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getInt32(i * 4, Endian.little);
    }
    return Tensor.int64(out, shape);
  } else if (t.dataType == _kBool) {
    // BOOL is one byte per element, carried as int64 0/1.
    final bytes = Uint8List.fromList(t.rawData);
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = (i < bytes.length && bytes[i] != 0) ? 1 : 0;
    }
    return Tensor.int64(out, shape);
  }
  throw UnsupportedError(
      'Unsupported ONNX TensorProto dataType ${t.dataType} for "${t.name}"');
}
