/// Compact uint8/int8 tensor storage: accessors, dtype-preserving reshape,
/// and the quantize ops producing/consuming compact tensors.
library;

import 'dart:typed_data';

import 'package:onnx_runtime_dart/src/onnx_ops.dart' as ops;
import 'package:onnx_runtime_dart/src/onnx_qlinear_ops.dart' as ql;
import 'package:onnx_runtime_dart/src/tensor.dart';
import 'package:test/test.dart';

void main() {
  test('uint8/int8 accessors and widening', () {
    final u = Tensor.uint8(Uint8List.fromList([0, 128, 255]), [3]);
    expect(u.dtype, DType.uint8);
    expect(u.isFloat, isFalse);
    expect(u.getI(1), 128);
    expect(u.getD(2), 255.0);
    expect(u.asIntList(), [0, 128, 255]);
    expect(u.asFloatList(), [0.0, 128.0, 255.0]);

    final s = Tensor.int8(Int8List.fromList([-128, -1, 127]), [3]);
    expect(s.getI(0), -128);
    expect(s.asIntList(), [-128, -1, 127]);
    expect(s.intData[1], -1);
  });

  test('reshape/squeeze/unsqueeze preserve compact dtype', () {
    final s = Tensor.int8(Int8List.fromList([1, -2, 3, -4]), [2, 2]);
    expect(s.reshape([4]).dtype, DType.int8);
    expect(ops.opUnsqueeze(s, [0]).dtype, DType.int8);
    expect(ops.opUnsqueeze(s, [0]).shape, [1, 2, 2]);
    expect(ops.opSqueeze(ops.opUnsqueeze(s, [0]), [0]).dtype, DType.int8);
  });

  test('QuantizeLinear produces compact output of the right signedness', () {
    final x = Tensor.float(Float32List.fromList([-1.0, 0.0, 1.0]), [3]);
    final scale = Tensor.scalarFloat(0.01);
    final qU = ops.opQuantizeLinear(x, scale, Tensor.scalarInt(128),
        lo: 0, hi: 255);
    expect(qU.dtype, DType.uint8);
    expect(qU.asIntList(), [28, 128, 228]);
    final qS = ops.opQuantizeLinear(x, scale, null, lo: -128, hi: 127);
    expect(qS.dtype, DType.int8);
    expect(qS.asIntList(), [-100, 0, 100]);
  });

  test('MatMulInteger consumes compact operands exactly', () {
    // a[1x3] u8, b[3x2] i8, exact integer expectations.
    final a = Tensor.uint8(Uint8List.fromList([10, 130, 250]), [1, 3]);
    final b =
        Tensor.int8(Int8List.fromList([-1, 2, 3, -4, 5, 6]), [3, 2]);
    final y = ql.opMatMulInteger(
        a, b, Tensor.scalarInt(10), Tensor.scalarInt(1));
    // (a-10) = [0,120,240]; (b-1) = [[-2,1],[2,-5],[4,5]]
    // y = [0*-2+120*2+240*4, 0*1+120*-5+240*5] = [1200, 600]
    expect(y.asIntList(), [1200, 600]);
  });

  test('DynamicQuantizeLinear output is uint8', () {
    final outs = ops.opDynamicQuantizeLinear(
        Tensor.float(Float32List.fromList([-1, 0, 2]), [3]));
    expect(outs[0].dtype, DType.uint8);
  });
}
