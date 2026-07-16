// This is a generated file - do not edit.
//
// Generated from onnx.proto3.

// @dart = 3.3

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names
// ignore_for_file: curly_braces_in_flow_control_structures
// ignore_for_file: deprecated_member_use_from_same_package, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_relative_imports

import 'dart:core' as $core;

import 'package:protobuf/protobuf.dart' as $pb;

/// Versioning
///
/// ONNX versioning is specified in docs/IR.md and elaborated on in docs/Versioning.md
///
/// To be compatible with both proto2 and proto3, we will use a version number
/// that is not defined by the default value but an explicit enum number.
class Version extends $pb.ProtobufEnum {
  /// proto3 requires the first enum value to be zero.
  /// We add this just to appease the compiler.
  static const Version START_VERSION_ =
      Version._(0, _omitEnumNames ? '' : '_START_VERSION');

  /// The version field is always serialized and we will use it to store the
  /// version that the  graph is generated from. This helps us set up version
  /// control.
  /// For the IR, we are using simple numbers starting with 0x00000001,
  /// which was the version we published on Oct 10, 2017.
  static const Version IR_VERSION_2017_10_10 =
      Version._(1, _omitEnumNames ? '' : 'IR_VERSION_2017_10_10');

  /// IR_VERSION 2 published on Oct 30, 2017
  /// - Added type discriminator to AttributeProto to support proto3 users
  static const Version IR_VERSION_2017_10_30 =
      Version._(2, _omitEnumNames ? '' : 'IR_VERSION_2017_10_30');

  /// IR VERSION 3 published on Nov 3, 2017
  /// - For operator versioning:
  ///    - Added new message OperatorSetIdProto
  ///    - Added opset_import in ModelProto
  /// - For vendor extensions, added domain in NodeProto
  static const Version IR_VERSION_2017_11_3 =
      Version._(3, _omitEnumNames ? '' : 'IR_VERSION_2017_11_3');

  /// IR VERSION 4 published on Jan 22, 2019
  /// - Relax constraint that initializers should be a subset of graph inputs
  /// - Add type BFLOAT16
  static const Version IR_VERSION_2019_1_22 =
      Version._(4, _omitEnumNames ? '' : 'IR_VERSION_2019_1_22');

  /// IR VERSION 5 published on March 18, 2019
  /// - Add message TensorAnnotation.
  /// - Add quantization annotation in GraphProto to map tensor with its scale and zero point quantization parameters.
  static const Version IR_VERSION_2019_3_18 =
      Version._(5, _omitEnumNames ? '' : 'IR_VERSION_2019_3_18');

  /// IR VERSION 6 published on Sep 19, 2019
  /// - Add support for sparse tensor constants stored in model.
  ///   - Add message SparseTensorProto
  ///   - Add sparse initializers
  static const Version IR_VERSION_2019_9_19 =
      Version._(6, _omitEnumNames ? '' : 'IR_VERSION_2019_9_19');

  /// IR VERSION 7 published on May 8, 2020
  /// - Add support to allow function body graph to rely on multiple external operator sets.
  /// - Add a list to promote inference graph's initializers to global and
  ///   mutable variables. Global variables are visible in all graphs of the
  ///   stored models.
  /// - Add message TrainingInfoProto to store initialization
  ///   method and training algorithm. The execution of TrainingInfoProto
  ///   can modify the values of mutable variables.
  /// - Implicitly add inference graph into each TrainingInfoProto's algorithm.
  static const Version IR_VERSION_2020_5_8 =
      Version._(7, _omitEnumNames ? '' : 'IR_VERSION_2020_5_8');

  /// IR VERSION 8 published on July 30, 2021
  /// Introduce TypeProto.SparseTensor
  /// Introduce TypeProto.Optional
  /// Added a list of FunctionProtos local to the model
  /// Deprecated since_version and operator status from FunctionProto
  static const Version IR_VERSION_2021_7_30 =
      Version._(8, _omitEnumNames ? '' : 'IR_VERSION_2021_7_30');

  /// IR VERSION 9 published on May 5, 2023
  /// Added AttributeProto to FunctionProto so that default attribute values can be set.
  /// Added FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ.
  static const Version IR_VERSION_2023_5_5 =
      Version._(9, _omitEnumNames ? '' : 'IR_VERSION_2023_5_5');

  /// IR VERSION 10 published on March 25, 2024
  /// Added UINT4, INT4, overload field for functions and metadata_props on multiple proto definitions.
  static const Version IR_VERSION_2024_3_25 =
      Version._(10, _omitEnumNames ? '' : 'IR_VERSION_2024_3_25');

  /// IR VERSION 11 published on May 12, 2025
  /// Added FLOAT4E2M1, multi-device protobuf classes.
  static const Version IR_VERSION_2025_05_12 =
      Version._(11, _omitEnumNames ? '' : 'IR_VERSION_2025_05_12');

  /// IR VERSION 12 published on August 26, 2025
  /// Added FLOAT8E8M0.
  static const Version IR_VERSION_2025_08_26 =
      Version._(12, _omitEnumNames ? '' : 'IR_VERSION_2025_08_26');

  /// IR VERSION 13 published on November 6, 2025
  /// Added UINT2, INT2.
  static const Version IR_VERSION =
      Version._(13, _omitEnumNames ? '' : 'IR_VERSION');

  static const $core.List<Version> values = <Version>[
    START_VERSION_,
    IR_VERSION_2017_10_10,
    IR_VERSION_2017_10_30,
    IR_VERSION_2017_11_3,
    IR_VERSION_2019_1_22,
    IR_VERSION_2019_3_18,
    IR_VERSION_2019_9_19,
    IR_VERSION_2020_5_8,
    IR_VERSION_2021_7_30,
    IR_VERSION_2023_5_5,
    IR_VERSION_2024_3_25,
    IR_VERSION_2025_05_12,
    IR_VERSION_2025_08_26,
    IR_VERSION,
  ];

  static final $core.List<Version?> _byValue =
      $pb.ProtobufEnum.$_initByValueList(values, 13);
  static Version? valueOf($core.int value) =>
      value < 0 || value >= _byValue.length ? null : _byValue[value];

  const Version._(super.value, super.name);
}

/// Operator/function status.
class OperatorStatus extends $pb.ProtobufEnum {
  static const OperatorStatus EXPERIMENTAL =
      OperatorStatus._(0, _omitEnumNames ? '' : 'EXPERIMENTAL');
  static const OperatorStatus STABLE =
      OperatorStatus._(1, _omitEnumNames ? '' : 'STABLE');

  static const $core.List<OperatorStatus> values = <OperatorStatus>[
    EXPERIMENTAL,
    STABLE,
  ];

  static final $core.List<OperatorStatus?> _byValue =
      $pb.ProtobufEnum.$_initByValueList(values, 1);
  static OperatorStatus? valueOf($core.int value) =>
      value < 0 || value >= _byValue.length ? null : _byValue[value];

  const OperatorStatus._(super.value, super.name);
}

/// Note: this enum is structurally identical to the OpSchema::AttrType
/// enum defined in schema.h.  If you rev one, you likely need to rev the other.
class AttributeProto_AttributeType extends $pb.ProtobufEnum {
  static const AttributeProto_AttributeType UNDEFINED =
      AttributeProto_AttributeType._(0, _omitEnumNames ? '' : 'UNDEFINED');
  static const AttributeProto_AttributeType FLOAT =
      AttributeProto_AttributeType._(1, _omitEnumNames ? '' : 'FLOAT');
  static const AttributeProto_AttributeType INT =
      AttributeProto_AttributeType._(2, _omitEnumNames ? '' : 'INT');
  static const AttributeProto_AttributeType STRING =
      AttributeProto_AttributeType._(3, _omitEnumNames ? '' : 'STRING');
  static const AttributeProto_AttributeType TENSOR =
      AttributeProto_AttributeType._(4, _omitEnumNames ? '' : 'TENSOR');
  static const AttributeProto_AttributeType GRAPH =
      AttributeProto_AttributeType._(5, _omitEnumNames ? '' : 'GRAPH');
  static const AttributeProto_AttributeType SPARSE_TENSOR =
      AttributeProto_AttributeType._(11, _omitEnumNames ? '' : 'SPARSE_TENSOR');
  static const AttributeProto_AttributeType TYPE_PROTO =
      AttributeProto_AttributeType._(13, _omitEnumNames ? '' : 'TYPE_PROTO');
  static const AttributeProto_AttributeType FLOATS =
      AttributeProto_AttributeType._(6, _omitEnumNames ? '' : 'FLOATS');
  static const AttributeProto_AttributeType INTS =
      AttributeProto_AttributeType._(7, _omitEnumNames ? '' : 'INTS');
  static const AttributeProto_AttributeType STRINGS =
      AttributeProto_AttributeType._(8, _omitEnumNames ? '' : 'STRINGS');
  static const AttributeProto_AttributeType TENSORS =
      AttributeProto_AttributeType._(9, _omitEnumNames ? '' : 'TENSORS');
  static const AttributeProto_AttributeType GRAPHS =
      AttributeProto_AttributeType._(10, _omitEnumNames ? '' : 'GRAPHS');
  static const AttributeProto_AttributeType SPARSE_TENSORS =
      AttributeProto_AttributeType._(
          12, _omitEnumNames ? '' : 'SPARSE_TENSORS');
  static const AttributeProto_AttributeType TYPE_PROTOS =
      AttributeProto_AttributeType._(14, _omitEnumNames ? '' : 'TYPE_PROTOS');

  static const $core.List<AttributeProto_AttributeType> values =
      <AttributeProto_AttributeType>[
    UNDEFINED,
    FLOAT,
    INT,
    STRING,
    TENSOR,
    GRAPH,
    SPARSE_TENSOR,
    TYPE_PROTO,
    FLOATS,
    INTS,
    STRINGS,
    TENSORS,
    GRAPHS,
    SPARSE_TENSORS,
    TYPE_PROTOS,
  ];

  static final $core.List<AttributeProto_AttributeType?> _byValue =
      $pb.ProtobufEnum.$_initByValueList(values, 14);
  static AttributeProto_AttributeType? valueOf($core.int value) =>
      value < 0 || value >= _byValue.length ? null : _byValue[value];

  const AttributeProto_AttributeType._(super.value, super.name);
}

class TensorProto_DataType extends $pb.ProtobufEnum {
  static const TensorProto_DataType UNDEFINED =
      TensorProto_DataType._(0, _omitEnumNames ? '' : 'UNDEFINED');

  /// Basic types.
  static const TensorProto_DataType FLOAT =
      TensorProto_DataType._(1, _omitEnumNames ? '' : 'FLOAT');
  static const TensorProto_DataType UINT8 =
      TensorProto_DataType._(2, _omitEnumNames ? '' : 'UINT8');
  static const TensorProto_DataType INT8 =
      TensorProto_DataType._(3, _omitEnumNames ? '' : 'INT8');
  static const TensorProto_DataType UINT16 =
      TensorProto_DataType._(4, _omitEnumNames ? '' : 'UINT16');
  static const TensorProto_DataType INT16 =
      TensorProto_DataType._(5, _omitEnumNames ? '' : 'INT16');
  static const TensorProto_DataType INT32 =
      TensorProto_DataType._(6, _omitEnumNames ? '' : 'INT32');
  static const TensorProto_DataType INT64 =
      TensorProto_DataType._(7, _omitEnumNames ? '' : 'INT64');
  static const TensorProto_DataType STRING =
      TensorProto_DataType._(8, _omitEnumNames ? '' : 'STRING');
  static const TensorProto_DataType BOOL =
      TensorProto_DataType._(9, _omitEnumNames ? '' : 'BOOL');

  /// IEEE754 half-precision floating-point format (16 bits wide).
  /// This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  static const TensorProto_DataType FLOAT16 =
      TensorProto_DataType._(10, _omitEnumNames ? '' : 'FLOAT16');
  static const TensorProto_DataType DOUBLE =
      TensorProto_DataType._(11, _omitEnumNames ? '' : 'DOUBLE');
  static const TensorProto_DataType UINT32 =
      TensorProto_DataType._(12, _omitEnumNames ? '' : 'UINT32');
  static const TensorProto_DataType UINT64 =
      TensorProto_DataType._(13, _omitEnumNames ? '' : 'UINT64');
  static const TensorProto_DataType COMPLEX64 =
      TensorProto_DataType._(14, _omitEnumNames ? '' : 'COMPLEX64');
  static const TensorProto_DataType COMPLEX128 =
      TensorProto_DataType._(15, _omitEnumNames ? '' : 'COMPLEX128');

  /// Non-IEEE floating-point format based on IEEE754 single-precision
  /// floating-point number truncated to 16 bits.
  /// This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  static const TensorProto_DataType BFLOAT16 =
      TensorProto_DataType._(16, _omitEnumNames ? '' : 'BFLOAT16');

  /// Non-IEEE floating-point format based on papers
  /// FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433,
  /// 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf.
  /// Operators supported FP8 are Cast, CastLike, QuantizeLinear, DequantizeLinear.
  /// The computation usually happens inside a block quantize / dequantize
  /// fused by the runtime.
  static const TensorProto_DataType FLOAT8E4M3FN =
      TensorProto_DataType._(17, _omitEnumNames ? '' : 'FLOAT8E4M3FN');
  static const TensorProto_DataType FLOAT8E4M3FNUZ =
      TensorProto_DataType._(18, _omitEnumNames ? '' : 'FLOAT8E4M3FNUZ');
  static const TensorProto_DataType FLOAT8E5M2 =
      TensorProto_DataType._(19, _omitEnumNames ? '' : 'FLOAT8E5M2');
  static const TensorProto_DataType FLOAT8E5M2FNUZ =
      TensorProto_DataType._(20, _omitEnumNames ? '' : 'FLOAT8E5M2FNUZ');

  /// 4-bit integer data types
  static const TensorProto_DataType UINT4 =
      TensorProto_DataType._(21, _omitEnumNames ? '' : 'UINT4');
  static const TensorProto_DataType INT4 =
      TensorProto_DataType._(22, _omitEnumNames ? '' : 'INT4');

  /// 4-bit floating point data types
  static const TensorProto_DataType FLOAT4E2M1 =
      TensorProto_DataType._(23, _omitEnumNames ? '' : 'FLOAT4E2M1');

  /// E8M0 type used as the scale for microscaling (MX) formats:
  /// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  static const TensorProto_DataType FLOAT8E8M0 =
      TensorProto_DataType._(24, _omitEnumNames ? '' : 'FLOAT8E8M0');

  /// 2-bit integer data type
  static const TensorProto_DataType UINT2 =
      TensorProto_DataType._(25, _omitEnumNames ? '' : 'UINT2');
  static const TensorProto_DataType INT2 =
      TensorProto_DataType._(26, _omitEnumNames ? '' : 'INT2');

  static const $core.List<TensorProto_DataType> values = <TensorProto_DataType>[
    UNDEFINED,
    FLOAT,
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    INT64,
    STRING,
    BOOL,
    FLOAT16,
    DOUBLE,
    UINT32,
    UINT64,
    COMPLEX64,
    COMPLEX128,
    BFLOAT16,
    FLOAT8E4M3FN,
    FLOAT8E4M3FNUZ,
    FLOAT8E5M2,
    FLOAT8E5M2FNUZ,
    UINT4,
    INT4,
    FLOAT4E2M1,
    FLOAT8E8M0,
    UINT2,
    INT2,
  ];

  static final $core.List<TensorProto_DataType?> _byValue =
      $pb.ProtobufEnum.$_initByValueList(values, 26);
  static TensorProto_DataType? valueOf($core.int value) =>
      value < 0 || value >= _byValue.length ? null : _byValue[value];

  const TensorProto_DataType._(super.value, super.name);
}

/// Location of the data for this tensor. MUST be one of:
/// - DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
/// - EXTERNAL - data stored in an external location as described by external_data field.
class TensorProto_DataLocation extends $pb.ProtobufEnum {
  static const TensorProto_DataLocation DEFAULT =
      TensorProto_DataLocation._(0, _omitEnumNames ? '' : 'DEFAULT');
  static const TensorProto_DataLocation EXTERNAL =
      TensorProto_DataLocation._(1, _omitEnumNames ? '' : 'EXTERNAL');

  static const $core.List<TensorProto_DataLocation> values =
      <TensorProto_DataLocation>[
    DEFAULT,
    EXTERNAL,
  ];

  static final $core.List<TensorProto_DataLocation?> _byValue =
      $pb.ProtobufEnum.$_initByValueList(values, 1);
  static TensorProto_DataLocation? valueOf($core.int value) =>
      value < 0 || value >= _byValue.length ? null : _byValue[value];

  const TensorProto_DataLocation._(super.value, super.name);
}

const $core.bool _omitEnumNames =
    $core.bool.fromEnvironment('protobuf.omit_enum_names');
