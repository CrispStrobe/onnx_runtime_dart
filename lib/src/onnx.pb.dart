// This is a generated file - do not edit.
//
// Generated from onnx.proto3.

// @dart = 3.3

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names
// ignore_for_file: curly_braces_in_flow_control_structures
// ignore_for_file: deprecated_member_use_from_same_package, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_relative_imports
// ignore_for_file: unintended_html_in_doc_comment

import 'dart:core' as $core;

import 'package:fixnum/fixnum.dart' as $fixnum;
import 'package:protobuf/protobuf.dart' as $pb;

import 'onnx.pbenum.dart';

export 'package:protobuf/protobuf.dart' show GeneratedMessageGenericExtensions;

export 'onnx.pbenum.dart';

/// Attributes
///
/// A named attribute containing either singular float, integer, string, graph,
/// and tensor values, or repeated float, integer, string, graph, and tensor values.
/// An AttributeProto MUST contain the name field, and *only one* of the
/// following content fields, effectively enforcing a C/C++ union equivalent.
class AttributeProto extends $pb.GeneratedMessage {
  factory AttributeProto({
    $core.String? name,
    $core.double? f,
    $fixnum.Int64? i,
    $core.List<$core.int>? s,
    TensorProto? t,
    GraphProto? g,
    $core.Iterable<$core.double>? floats,
    $core.Iterable<$fixnum.Int64>? ints,
    $core.Iterable<$core.List<$core.int>>? strings,
    $core.Iterable<TensorProto>? tensors,
    $core.Iterable<GraphProto>? graphs,
    $core.String? docString,
    TypeProto? tp,
    $core.Iterable<TypeProto>? typeProtos,
    AttributeProto_AttributeType? type,
    $core.String? refAttrName,
    SparseTensorProto? sparseTensor,
    $core.Iterable<SparseTensorProto>? sparseTensors,
  }) {
    final result = create();
    if (name != null) result.name = name;
    if (f != null) result.f = f;
    if (i != null) result.i = i;
    if (s != null) result.s = s;
    if (t != null) result.t = t;
    if (g != null) result.g = g;
    if (floats != null) result.floats.addAll(floats);
    if (ints != null) result.ints.addAll(ints);
    if (strings != null) result.strings.addAll(strings);
    if (tensors != null) result.tensors.addAll(tensors);
    if (graphs != null) result.graphs.addAll(graphs);
    if (docString != null) result.docString = docString;
    if (tp != null) result.tp = tp;
    if (typeProtos != null) result.typeProtos.addAll(typeProtos);
    if (type != null) result.type = type;
    if (refAttrName != null) result.refAttrName = refAttrName;
    if (sparseTensor != null) result.sparseTensor = sparseTensor;
    if (sparseTensors != null) result.sparseTensors.addAll(sparseTensors);
    return result;
  }

  AttributeProto._();

  factory AttributeProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory AttributeProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'AttributeProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'name')
    ..aD(2, _omitFieldNames ? '' : 'f', fieldType: $pb.PbFieldType.OF)
    ..aInt64(3, _omitFieldNames ? '' : 'i')
    ..a<$core.List<$core.int>>(
        4, _omitFieldNames ? '' : 's', $pb.PbFieldType.OY)
    ..aOM<TensorProto>(5, _omitFieldNames ? '' : 't',
        subBuilder: TensorProto.create)
    ..aOM<GraphProto>(6, _omitFieldNames ? '' : 'g',
        subBuilder: GraphProto.create)
    ..p<$core.double>(7, _omitFieldNames ? '' : 'floats', $pb.PbFieldType.KF)
    ..p<$fixnum.Int64>(8, _omitFieldNames ? '' : 'ints', $pb.PbFieldType.K6)
    ..p<$core.List<$core.int>>(
        9, _omitFieldNames ? '' : 'strings', $pb.PbFieldType.PY)
    ..pPM<TensorProto>(10, _omitFieldNames ? '' : 'tensors',
        subBuilder: TensorProto.create)
    ..pPM<GraphProto>(11, _omitFieldNames ? '' : 'graphs',
        subBuilder: GraphProto.create)
    ..aOS(13, _omitFieldNames ? '' : 'docString')
    ..aOM<TypeProto>(14, _omitFieldNames ? '' : 'tp',
        subBuilder: TypeProto.create)
    ..pPM<TypeProto>(15, _omitFieldNames ? '' : 'typeProtos',
        subBuilder: TypeProto.create)
    ..aE<AttributeProto_AttributeType>(20, _omitFieldNames ? '' : 'type',
        enumValues: AttributeProto_AttributeType.values)
    ..aOS(21, _omitFieldNames ? '' : 'refAttrName')
    ..aOM<SparseTensorProto>(22, _omitFieldNames ? '' : 'sparseTensor',
        subBuilder: SparseTensorProto.create)
    ..pPM<SparseTensorProto>(23, _omitFieldNames ? '' : 'sparseTensors',
        subBuilder: SparseTensorProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  AttributeProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  AttributeProto copyWith(void Function(AttributeProto) updates) =>
      super.copyWith((message) => updates(message as AttributeProto))
          as AttributeProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static AttributeProto create() => AttributeProto._();
  @$core.override
  AttributeProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static AttributeProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<AttributeProto>(create);
  static AttributeProto? _defaultInstance;

  /// The name field MUST be present for this version of the IR.
  @$pb.TagNumber(1)
  $core.String get name => $_getSZ(0);
  @$pb.TagNumber(1)
  set name($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasName() => $_has(0);
  @$pb.TagNumber(1)
  void clearName() => $_clearField(1);

  /// Exactly ONE of the following fields must be present for this version of the IR
  @$pb.TagNumber(2)
  $core.double get f => $_getN(1);
  @$pb.TagNumber(2)
  set f($core.double value) => $_setFloat(1, value);
  @$pb.TagNumber(2)
  $core.bool hasF() => $_has(1);
  @$pb.TagNumber(2)
  void clearF() => $_clearField(2);

  @$pb.TagNumber(3)
  $fixnum.Int64 get i => $_getI64(2);
  @$pb.TagNumber(3)
  set i($fixnum.Int64 value) => $_setInt64(2, value);
  @$pb.TagNumber(3)
  $core.bool hasI() => $_has(2);
  @$pb.TagNumber(3)
  void clearI() => $_clearField(3);

  @$pb.TagNumber(4)
  $core.List<$core.int> get s => $_getN(3);
  @$pb.TagNumber(4)
  set s($core.List<$core.int> value) => $_setBytes(3, value);
  @$pb.TagNumber(4)
  $core.bool hasS() => $_has(3);
  @$pb.TagNumber(4)
  void clearS() => $_clearField(4);

  @$pb.TagNumber(5)
  TensorProto get t => $_getN(4);
  @$pb.TagNumber(5)
  set t(TensorProto value) => $_setField(5, value);
  @$pb.TagNumber(5)
  $core.bool hasT() => $_has(4);
  @$pb.TagNumber(5)
  void clearT() => $_clearField(5);
  @$pb.TagNumber(5)
  TensorProto ensureT() => $_ensure(4);

  @$pb.TagNumber(6)
  GraphProto get g => $_getN(5);
  @$pb.TagNumber(6)
  set g(GraphProto value) => $_setField(6, value);
  @$pb.TagNumber(6)
  $core.bool hasG() => $_has(5);
  @$pb.TagNumber(6)
  void clearG() => $_clearField(6);
  @$pb.TagNumber(6)
  GraphProto ensureG() => $_ensure(5);

  @$pb.TagNumber(7)
  $pb.PbList<$core.double> get floats => $_getList(6);

  @$pb.TagNumber(8)
  $pb.PbList<$fixnum.Int64> get ints => $_getList(7);

  @$pb.TagNumber(9)
  $pb.PbList<$core.List<$core.int>> get strings => $_getList(8);

  @$pb.TagNumber(10)
  $pb.PbList<TensorProto> get tensors => $_getList(9);

  @$pb.TagNumber(11)
  $pb.PbList<GraphProto> get graphs => $_getList(10);

  /// A human-readable documentation for this attribute. Markdown is allowed.
  @$pb.TagNumber(13)
  $core.String get docString => $_getSZ(11);
  @$pb.TagNumber(13)
  set docString($core.String value) => $_setString(11, value);
  @$pb.TagNumber(13)
  $core.bool hasDocString() => $_has(11);
  @$pb.TagNumber(13)
  void clearDocString() => $_clearField(13);

  /// Do not use field below, it's deprecated.
  /// optional ValueProto v = 12;         // value - subsumes everything but graph
  @$pb.TagNumber(14)
  TypeProto get tp => $_getN(12);
  @$pb.TagNumber(14)
  set tp(TypeProto value) => $_setField(14, value);
  @$pb.TagNumber(14)
  $core.bool hasTp() => $_has(12);
  @$pb.TagNumber(14)
  void clearTp() => $_clearField(14);
  @$pb.TagNumber(14)
  TypeProto ensureTp() => $_ensure(12);

  @$pb.TagNumber(15)
  $pb.PbList<TypeProto> get typeProtos => $_getList(13);

  /// The type field MUST be present for this version of the IR.
  /// For 0.0.1 versions of the IR, this field was not defined, and
  /// implementations needed to use has_field heuristics to determine
  /// which value field was in use.  For IR_VERSION 0.0.2 or later, this
  /// field MUST be set and match the f|i|s|t|... field in use.  This
  /// change was made to accommodate proto3 implementations.
  @$pb.TagNumber(20)
  AttributeProto_AttributeType get type => $_getN(14);
  @$pb.TagNumber(20)
  set type(AttributeProto_AttributeType value) => $_setField(20, value);
  @$pb.TagNumber(20)
  $core.bool hasType() => $_has(14);
  @$pb.TagNumber(20)
  void clearType() => $_clearField(20);

  /// if ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
  /// In this case, this AttributeProto does not contain data, and it's a reference of attribute
  /// in parent scope.
  /// NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
  @$pb.TagNumber(21)
  $core.String get refAttrName => $_getSZ(15);
  @$pb.TagNumber(21)
  set refAttrName($core.String value) => $_setString(15, value);
  @$pb.TagNumber(21)
  $core.bool hasRefAttrName() => $_has(15);
  @$pb.TagNumber(21)
  void clearRefAttrName() => $_clearField(21);

  @$pb.TagNumber(22)
  SparseTensorProto get sparseTensor => $_getN(16);
  @$pb.TagNumber(22)
  set sparseTensor(SparseTensorProto value) => $_setField(22, value);
  @$pb.TagNumber(22)
  $core.bool hasSparseTensor() => $_has(16);
  @$pb.TagNumber(22)
  void clearSparseTensor() => $_clearField(22);
  @$pb.TagNumber(22)
  SparseTensorProto ensureSparseTensor() => $_ensure(16);

  @$pb.TagNumber(23)
  $pb.PbList<SparseTensorProto> get sparseTensors => $_getList(17);
}

/// Defines information on value, including the name, the type, and
/// the shape of the value.
class ValueInfoProto extends $pb.GeneratedMessage {
  factory ValueInfoProto({
    $core.String? name,
    TypeProto? type,
    $core.String? docString,
    $core.Iterable<StringStringEntryProto>? metadataProps,
  }) {
    final result = create();
    if (name != null) result.name = name;
    if (type != null) result.type = type;
    if (docString != null) result.docString = docString;
    if (metadataProps != null) result.metadataProps.addAll(metadataProps);
    return result;
  }

  ValueInfoProto._();

  factory ValueInfoProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory ValueInfoProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'ValueInfoProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'name')
    ..aOM<TypeProto>(2, _omitFieldNames ? '' : 'type',
        subBuilder: TypeProto.create)
    ..aOS(3, _omitFieldNames ? '' : 'docString')
    ..pPM<StringStringEntryProto>(4, _omitFieldNames ? '' : 'metadataProps',
        subBuilder: StringStringEntryProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ValueInfoProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ValueInfoProto copyWith(void Function(ValueInfoProto) updates) =>
      super.copyWith((message) => updates(message as ValueInfoProto))
          as ValueInfoProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static ValueInfoProto create() => ValueInfoProto._();
  @$core.override
  ValueInfoProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static ValueInfoProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<ValueInfoProto>(create);
  static ValueInfoProto? _defaultInstance;

  /// This field MUST be present in this version of the IR.
  @$pb.TagNumber(1)
  $core.String get name => $_getSZ(0);
  @$pb.TagNumber(1)
  set name($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasName() => $_has(0);
  @$pb.TagNumber(1)
  void clearName() => $_clearField(1);

  /// This field MUST be present in this version of the IR for
  /// inputs and outputs of the top-level graph.
  @$pb.TagNumber(2)
  TypeProto get type => $_getN(1);
  @$pb.TagNumber(2)
  set type(TypeProto value) => $_setField(2, value);
  @$pb.TagNumber(2)
  $core.bool hasType() => $_has(1);
  @$pb.TagNumber(2)
  void clearType() => $_clearField(2);
  @$pb.TagNumber(2)
  TypeProto ensureType() => $_ensure(1);

  /// A human-readable documentation for this value. Markdown is allowed.
  @$pb.TagNumber(3)
  $core.String get docString => $_getSZ(2);
  @$pb.TagNumber(3)
  set docString($core.String value) => $_setString(2, value);
  @$pb.TagNumber(3)
  $core.bool hasDocString() => $_has(2);
  @$pb.TagNumber(3)
  void clearDocString() => $_clearField(3);

  /// Named metadata values; keys should be distinct.
  @$pb.TagNumber(4)
  $pb.PbList<StringStringEntryProto> get metadataProps => $_getList(3);
}

/// Nodes
///
/// Computation graphs are made up of a DAG of nodes, which represent what is
/// commonly called a "layer" or "pipeline stage" in machine learning frameworks.
///
/// For example, it can be a node of type "Conv" that takes in an image, a filter
/// tensor and a bias tensor, and produces the convolved output.
class NodeProto extends $pb.GeneratedMessage {
  factory NodeProto({
    $core.Iterable<$core.String>? input,
    $core.Iterable<$core.String>? output,
    $core.String? name,
    $core.String? opType,
    $core.Iterable<AttributeProto>? attribute,
    $core.String? docString,
    $core.String? domain,
    $core.String? overload,
    $core.Iterable<StringStringEntryProto>? metadataProps,
    $core.Iterable<NodeDeviceConfigurationProto>? deviceConfigurations,
  }) {
    final result = create();
    if (input != null) result.input.addAll(input);
    if (output != null) result.output.addAll(output);
    if (name != null) result.name = name;
    if (opType != null) result.opType = opType;
    if (attribute != null) result.attribute.addAll(attribute);
    if (docString != null) result.docString = docString;
    if (domain != null) result.domain = domain;
    if (overload != null) result.overload = overload;
    if (metadataProps != null) result.metadataProps.addAll(metadataProps);
    if (deviceConfigurations != null)
      result.deviceConfigurations.addAll(deviceConfigurations);
    return result;
  }

  NodeProto._();

  factory NodeProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory NodeProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'NodeProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..pPS(1, _omitFieldNames ? '' : 'input')
    ..pPS(2, _omitFieldNames ? '' : 'output')
    ..aOS(3, _omitFieldNames ? '' : 'name')
    ..aOS(4, _omitFieldNames ? '' : 'opType')
    ..pPM<AttributeProto>(5, _omitFieldNames ? '' : 'attribute',
        subBuilder: AttributeProto.create)
    ..aOS(6, _omitFieldNames ? '' : 'docString')
    ..aOS(7, _omitFieldNames ? '' : 'domain')
    ..aOS(8, _omitFieldNames ? '' : 'overload')
    ..pPM<StringStringEntryProto>(9, _omitFieldNames ? '' : 'metadataProps',
        subBuilder: StringStringEntryProto.create)
    ..pPM<NodeDeviceConfigurationProto>(
        10, _omitFieldNames ? '' : 'deviceConfigurations',
        subBuilder: NodeDeviceConfigurationProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  NodeProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  NodeProto copyWith(void Function(NodeProto) updates) =>
      super.copyWith((message) => updates(message as NodeProto)) as NodeProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static NodeProto create() => NodeProto._();
  @$core.override
  NodeProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static NodeProto getDefault() =>
      _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<NodeProto>(create);
  static NodeProto? _defaultInstance;

  @$pb.TagNumber(1)
  $pb.PbList<$core.String> get input => $_getList(0);

  @$pb.TagNumber(2)
  $pb.PbList<$core.String> get output => $_getList(1);

  /// An optional identifier for this node in a graph.
  /// This field MAY be absent in this version of the IR.
  @$pb.TagNumber(3)
  $core.String get name => $_getSZ(2);
  @$pb.TagNumber(3)
  set name($core.String value) => $_setString(2, value);
  @$pb.TagNumber(3)
  $core.bool hasName() => $_has(2);
  @$pb.TagNumber(3)
  void clearName() => $_clearField(3);

  /// The symbolic identifier of the Operator to execute.
  @$pb.TagNumber(4)
  $core.String get opType => $_getSZ(3);
  @$pb.TagNumber(4)
  set opType($core.String value) => $_setString(3, value);
  @$pb.TagNumber(4)
  $core.bool hasOpType() => $_has(3);
  @$pb.TagNumber(4)
  void clearOpType() => $_clearField(4);

  /// Additional named attributes.
  @$pb.TagNumber(5)
  $pb.PbList<AttributeProto> get attribute => $_getList(4);

  /// A human-readable documentation for this node. Markdown is allowed.
  @$pb.TagNumber(6)
  $core.String get docString => $_getSZ(5);
  @$pb.TagNumber(6)
  set docString($core.String value) => $_setString(5, value);
  @$pb.TagNumber(6)
  $core.bool hasDocString() => $_has(5);
  @$pb.TagNumber(6)
  void clearDocString() => $_clearField(6);

  /// The domain of the OperatorSet that specifies the operator named by op_type.
  @$pb.TagNumber(7)
  $core.String get domain => $_getSZ(6);
  @$pb.TagNumber(7)
  set domain($core.String value) => $_setString(6, value);
  @$pb.TagNumber(7)
  $core.bool hasDomain() => $_has(6);
  @$pb.TagNumber(7)
  void clearDomain() => $_clearField(7);

  /// Overload identifier, used only to map this to a model-local function.
  @$pb.TagNumber(8)
  $core.String get overload => $_getSZ(7);
  @$pb.TagNumber(8)
  set overload($core.String value) => $_setString(7, value);
  @$pb.TagNumber(8)
  $core.bool hasOverload() => $_has(7);
  @$pb.TagNumber(8)
  void clearOverload() => $_clearField(8);

  /// Named metadata values; keys should be distinct.
  @$pb.TagNumber(9)
  $pb.PbList<StringStringEntryProto> get metadataProps => $_getList(8);

  /// Configuration of multi-device annotations.
  @$pb.TagNumber(10)
  $pb.PbList<NodeDeviceConfigurationProto> get deviceConfigurations =>
      $_getList(9);
}

/// IntIntListEntryProto follows the pattern for cross-proto-version maps.
/// See https://developers.google.com/protocol-buffers/docs/proto3#maps
class IntIntListEntryProto extends $pb.GeneratedMessage {
  factory IntIntListEntryProto({
    $fixnum.Int64? key,
    $core.Iterable<$fixnum.Int64>? value,
  }) {
    final result = create();
    if (key != null) result.key = key;
    if (value != null) result.value.addAll(value);
    return result;
  }

  IntIntListEntryProto._();

  factory IntIntListEntryProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory IntIntListEntryProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'IntIntListEntryProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aInt64(1, _omitFieldNames ? '' : 'key')
    ..p<$fixnum.Int64>(2, _omitFieldNames ? '' : 'value', $pb.PbFieldType.K6)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  IntIntListEntryProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  IntIntListEntryProto copyWith(void Function(IntIntListEntryProto) updates) =>
      super.copyWith((message) => updates(message as IntIntListEntryProto))
          as IntIntListEntryProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static IntIntListEntryProto create() => IntIntListEntryProto._();
  @$core.override
  IntIntListEntryProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static IntIntListEntryProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<IntIntListEntryProto>(create);
  static IntIntListEntryProto? _defaultInstance;

  @$pb.TagNumber(1)
  $fixnum.Int64 get key => $_getI64(0);
  @$pb.TagNumber(1)
  set key($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasKey() => $_has(0);
  @$pb.TagNumber(1)
  void clearKey() => $_clearField(1);

  @$pb.TagNumber(2)
  $pb.PbList<$fixnum.Int64> get value => $_getList(1);
}

/// Multi-device configuration proto for NodeProto.
class NodeDeviceConfigurationProto extends $pb.GeneratedMessage {
  factory NodeDeviceConfigurationProto({
    $core.String? configurationId,
    $core.Iterable<ShardingSpecProto>? shardingSpec,
    $core.int? pipelineStage,
  }) {
    final result = create();
    if (configurationId != null) result.configurationId = configurationId;
    if (shardingSpec != null) result.shardingSpec.addAll(shardingSpec);
    if (pipelineStage != null) result.pipelineStage = pipelineStage;
    return result;
  }

  NodeDeviceConfigurationProto._();

  factory NodeDeviceConfigurationProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory NodeDeviceConfigurationProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'NodeDeviceConfigurationProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'configurationId')
    ..pPM<ShardingSpecProto>(2, _omitFieldNames ? '' : 'shardingSpec',
        subBuilder: ShardingSpecProto.create)
    ..aI(3, _omitFieldNames ? '' : 'pipelineStage')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  NodeDeviceConfigurationProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  NodeDeviceConfigurationProto copyWith(
          void Function(NodeDeviceConfigurationProto) updates) =>
      super.copyWith(
              (message) => updates(message as NodeDeviceConfigurationProto))
          as NodeDeviceConfigurationProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static NodeDeviceConfigurationProto create() =>
      NodeDeviceConfigurationProto._();
  @$core.override
  NodeDeviceConfigurationProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static NodeDeviceConfigurationProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<NodeDeviceConfigurationProto>(create);
  static NodeDeviceConfigurationProto? _defaultInstance;

  /// This field MUST be present for this version of the IR.
  /// ID of the configuration. MUST match the name of a DeviceConfigurationProto.
  @$pb.TagNumber(1)
  $core.String get configurationId => $_getSZ(0);
  @$pb.TagNumber(1)
  set configurationId($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasConfigurationId() => $_has(0);
  @$pb.TagNumber(1)
  void clearConfigurationId() => $_clearField(1);

  /// Sharding spec for the node.
  @$pb.TagNumber(2)
  $pb.PbList<ShardingSpecProto> get shardingSpec => $_getList(1);

  /// Pipeline stage of this node.
  @$pb.TagNumber(3)
  $core.int get pipelineStage => $_getIZ(2);
  @$pb.TagNumber(3)
  set pipelineStage($core.int value) => $_setSignedInt32(2, value);
  @$pb.TagNumber(3)
  $core.bool hasPipelineStage() => $_has(2);
  @$pb.TagNumber(3)
  void clearPipelineStage() => $_clearField(3);
}

/// ShardingSpecProto: This describes the sharding spec for a specific
/// input or output tensor of a node.
class ShardingSpecProto extends $pb.GeneratedMessage {
  factory ShardingSpecProto({
    $core.String? tensorName,
    $core.Iterable<$fixnum.Int64>? device,
    $core.Iterable<IntIntListEntryProto>? indexToDeviceGroupMap,
    $core.Iterable<ShardedDimProto>? shardedDim,
  }) {
    final result = create();
    if (tensorName != null) result.tensorName = tensorName;
    if (device != null) result.device.addAll(device);
    if (indexToDeviceGroupMap != null)
      result.indexToDeviceGroupMap.addAll(indexToDeviceGroupMap);
    if (shardedDim != null) result.shardedDim.addAll(shardedDim);
    return result;
  }

  ShardingSpecProto._();

  factory ShardingSpecProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory ShardingSpecProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'ShardingSpecProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'tensorName')
    ..p<$fixnum.Int64>(2, _omitFieldNames ? '' : 'device', $pb.PbFieldType.K6)
    ..pPM<IntIntListEntryProto>(
        3, _omitFieldNames ? '' : 'indexToDeviceGroupMap',
        subBuilder: IntIntListEntryProto.create)
    ..pPM<ShardedDimProto>(4, _omitFieldNames ? '' : 'shardedDim',
        subBuilder: ShardedDimProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ShardingSpecProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ShardingSpecProto copyWith(void Function(ShardingSpecProto) updates) =>
      super.copyWith((message) => updates(message as ShardingSpecProto))
          as ShardingSpecProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static ShardingSpecProto create() => ShardingSpecProto._();
  @$core.override
  ShardingSpecProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static ShardingSpecProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<ShardingSpecProto>(create);
  static ShardingSpecProto? _defaultInstance;

  /// This field MUST be present for this version of the IR.
  /// Identifies the input or output of the node that is being sharded.
  /// Required to match a name specified in the node's input or output list of ValueInfoProtos.
  /// It is called `logical tensor` in subsequent descriptions.
  @$pb.TagNumber(1)
  $core.String get tensorName => $_getSZ(0);
  @$pb.TagNumber(1)
  set tensorName($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasTensorName() => $_has(0);
  @$pb.TagNumber(1)
  void clearTensorName() => $_clearField(1);

  /// The following is the list of devices across which the logical
  /// tensor is sharded or replicated.
  @$pb.TagNumber(2)
  $pb.PbList<$fixnum.Int64> get device => $_getList(1);

  /// Each element v in above field devices may represent either a
  /// device or a set of devices (when we want the same shard/tensor
  /// to be replicated across a subset of devices), as indicated by
  /// the following optional map. If the map contains an entry for v,
  /// then v represents a device group, and the map indicates the set
  /// of devices in that group.
  @$pb.TagNumber(3)
  $pb.PbList<IntIntListEntryProto> get indexToDeviceGroupMap => $_getList(2);

  /// The following is the sharded-shape of the tensor, consisting of
  /// the sharding-spec for each axis of the tensor.
  @$pb.TagNumber(4)
  $pb.PbList<ShardedDimProto> get shardedDim => $_getList(3);
}

/// ShardedDimProto: This describes the sharding spec for a single
/// axis of a sharded tensor.
class ShardedDimProto extends $pb.GeneratedMessage {
  factory ShardedDimProto({
    $fixnum.Int64? axis,
    $core.Iterable<SimpleShardedDimProto>? simpleSharding,
  }) {
    final result = create();
    if (axis != null) result.axis = axis;
    if (simpleSharding != null) result.simpleSharding.addAll(simpleSharding);
    return result;
  }

  ShardedDimProto._();

  factory ShardedDimProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory ShardedDimProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'ShardedDimProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aInt64(1, _omitFieldNames ? '' : 'axis')
    ..pPM<SimpleShardedDimProto>(2, _omitFieldNames ? '' : 'simpleSharding',
        subBuilder: SimpleShardedDimProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ShardedDimProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ShardedDimProto copyWith(void Function(ShardedDimProto) updates) =>
      super.copyWith((message) => updates(message as ShardedDimProto))
          as ShardedDimProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static ShardedDimProto create() => ShardedDimProto._();
  @$core.override
  ShardedDimProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static ShardedDimProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<ShardedDimProto>(create);
  static ShardedDimProto? _defaultInstance;

  /// This field MUST be present for this version of the IR.
  /// The axis this sharding corresponds to. Must be in the range of
  /// [-r, r - 1], where r is the rank of the tensor. Negative axis values means
  /// counting from the back.
  @$pb.TagNumber(1)
  $fixnum.Int64 get axis => $_getI64(0);
  @$pb.TagNumber(1)
  set axis($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasAxis() => $_has(0);
  @$pb.TagNumber(1)
  void clearAxis() => $_clearField(1);

  /// Describes how the tensor on the provided axis is sharded.
  /// The common-case is described by a single instance of SimpleShardedDimProto.
  /// Multiple instances can be used to handle cases where a sharded
  /// tensor is reshaped, fusing multiple axes into one.
  @$pb.TagNumber(2)
  $pb.PbList<SimpleShardedDimProto> get simpleSharding => $_getList(1);
}

enum SimpleShardedDimProto_Dim { dimValue, dimParam, notSet }

/// SimpleShardedDimProto: Indicates that N blocks are divided into M shards.
/// N is allowed to be symbolic where M is required to be a constant.
class SimpleShardedDimProto extends $pb.GeneratedMessage {
  factory SimpleShardedDimProto({
    $fixnum.Int64? dimValue,
    $core.String? dimParam,
    $fixnum.Int64? numShards,
  }) {
    final result = create();
    if (dimValue != null) result.dimValue = dimValue;
    if (dimParam != null) result.dimParam = dimParam;
    if (numShards != null) result.numShards = numShards;
    return result;
  }

  SimpleShardedDimProto._();

  factory SimpleShardedDimProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory SimpleShardedDimProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static const $core.Map<$core.int, SimpleShardedDimProto_Dim>
      _SimpleShardedDimProto_DimByTag = {
    1: SimpleShardedDimProto_Dim.dimValue,
    2: SimpleShardedDimProto_Dim.dimParam,
    0: SimpleShardedDimProto_Dim.notSet
  };
  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'SimpleShardedDimProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..oo(0, [1, 2])
    ..aInt64(1, _omitFieldNames ? '' : 'dimValue')
    ..aOS(2, _omitFieldNames ? '' : 'dimParam')
    ..aInt64(3, _omitFieldNames ? '' : 'numShards')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  SimpleShardedDimProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  SimpleShardedDimProto copyWith(
          void Function(SimpleShardedDimProto) updates) =>
      super.copyWith((message) => updates(message as SimpleShardedDimProto))
          as SimpleShardedDimProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static SimpleShardedDimProto create() => SimpleShardedDimProto._();
  @$core.override
  SimpleShardedDimProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static SimpleShardedDimProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<SimpleShardedDimProto>(create);
  static SimpleShardedDimProto? _defaultInstance;

  @$pb.TagNumber(1)
  @$pb.TagNumber(2)
  SimpleShardedDimProto_Dim whichDim() =>
      _SimpleShardedDimProto_DimByTag[$_whichOneof(0)]!;
  @$pb.TagNumber(1)
  @$pb.TagNumber(2)
  void clearDim() => $_clearField($_whichOneof(0));

  @$pb.TagNumber(1)
  $fixnum.Int64 get dimValue => $_getI64(0);
  @$pb.TagNumber(1)
  set dimValue($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasDimValue() => $_has(0);
  @$pb.TagNumber(1)
  void clearDimValue() => $_clearField(1);

  @$pb.TagNumber(2)
  $core.String get dimParam => $_getSZ(1);
  @$pb.TagNumber(2)
  set dimParam($core.String value) => $_setString(1, value);
  @$pb.TagNumber(2)
  $core.bool hasDimParam() => $_has(1);
  @$pb.TagNumber(2)
  void clearDimParam() => $_clearField(2);

  /// This field MUST be present for this version of the IR.
  /// Number of shards to split dim into.
  @$pb.TagNumber(3)
  $fixnum.Int64 get numShards => $_getI64(2);
  @$pb.TagNumber(3)
  set numShards($fixnum.Int64 value) => $_setInt64(2, value);
  @$pb.TagNumber(3)
  $core.bool hasNumShards() => $_has(2);
  @$pb.TagNumber(3)
  void clearNumShards() => $_clearField(3);
}

/// Training information
/// TrainingInfoProto stores information for training a model.
/// In particular, this defines two functionalities: an initialization-step
/// and a training-algorithm-step. Initialization resets the model
/// back to its original state as if no training has been performed.
/// Training algorithm improves the model based on input data.
///
/// The semantics of the initialization-step is that the initializers
/// in ModelProto.graph and in TrainingInfoProto.algorithm are first
/// initialized as specified by the initializers in the graph, and then
/// updated by the "initialization_binding" in every instance in
/// ModelProto.training_info.
///
/// The field "algorithm" defines a computation graph which represents a
/// training algorithm's step. After the execution of a
/// TrainingInfoProto.algorithm, the initializers specified by "update_binding"
/// may be immediately updated. If the targeted training algorithm contains
/// consecutive update steps (such as block coordinate descent methods),
/// the user needs to create a TrainingInfoProto for each step.
class TrainingInfoProto extends $pb.GeneratedMessage {
  factory TrainingInfoProto({
    GraphProto? initialization,
    GraphProto? algorithm,
    $core.Iterable<StringStringEntryProto>? initializationBinding,
    $core.Iterable<StringStringEntryProto>? updateBinding,
  }) {
    final result = create();
    if (initialization != null) result.initialization = initialization;
    if (algorithm != null) result.algorithm = algorithm;
    if (initializationBinding != null)
      result.initializationBinding.addAll(initializationBinding);
    if (updateBinding != null) result.updateBinding.addAll(updateBinding);
    return result;
  }

  TrainingInfoProto._();

  factory TrainingInfoProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TrainingInfoProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TrainingInfoProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOM<GraphProto>(1, _omitFieldNames ? '' : 'initialization',
        subBuilder: GraphProto.create)
    ..aOM<GraphProto>(2, _omitFieldNames ? '' : 'algorithm',
        subBuilder: GraphProto.create)
    ..pPM<StringStringEntryProto>(
        3, _omitFieldNames ? '' : 'initializationBinding',
        subBuilder: StringStringEntryProto.create)
    ..pPM<StringStringEntryProto>(4, _omitFieldNames ? '' : 'updateBinding',
        subBuilder: StringStringEntryProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TrainingInfoProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TrainingInfoProto copyWith(void Function(TrainingInfoProto) updates) =>
      super.copyWith((message) => updates(message as TrainingInfoProto))
          as TrainingInfoProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TrainingInfoProto create() => TrainingInfoProto._();
  @$core.override
  TrainingInfoProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TrainingInfoProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TrainingInfoProto>(create);
  static TrainingInfoProto? _defaultInstance;

  /// This field describes a graph to compute the initial tensors
  /// upon starting the training process. Initialization graph has no input
  /// and can have multiple outputs. Usually, trainable tensors in neural
  /// networks are randomly initialized. To achieve that, for each tensor,
  /// the user can put a random number operator such as RandomNormal or
  /// RandomUniform in TrainingInfoProto.initialization.node and assign its
  /// random output to the specific tensor using "initialization_binding".
  /// This graph can also set the initializers in "algorithm" in the same
  /// TrainingInfoProto; a use case is resetting the number of training
  /// iteration to zero.
  ///
  /// By default, this field is an empty graph and its evaluation does not
  /// produce any output. Thus, no initializer would be changed by default.
  @$pb.TagNumber(1)
  GraphProto get initialization => $_getN(0);
  @$pb.TagNumber(1)
  set initialization(GraphProto value) => $_setField(1, value);
  @$pb.TagNumber(1)
  $core.bool hasInitialization() => $_has(0);
  @$pb.TagNumber(1)
  void clearInitialization() => $_clearField(1);
  @$pb.TagNumber(1)
  GraphProto ensureInitialization() => $_ensure(0);

  /// This field represents a training algorithm step. Given required inputs,
  /// it computes outputs to update initializers in its own or inference graph's
  /// initializer lists. In general, this field contains loss node, gradient node,
  /// optimizer node, increment of iteration count.
  ///
  /// An execution of the training algorithm step is performed by executing the
  /// graph obtained by combining the inference graph (namely "ModelProto.graph")
  /// and the "algorithm" graph. That is, the actual
  /// input/initializer/output/node/value_info/sparse_initializer list of
  /// the training graph is the concatenation of
  /// "ModelProto.graph.input/initializer/output/node/value_info/sparse_initializer"
  /// and "algorithm.input/initializer/output/node/value_info/sparse_initializer"
  /// in that order. This combined graph must satisfy the normal ONNX conditions.
  /// Now, let's provide a visualization of graph combination for clarity.
  /// Let the inference graph (i.e., "ModelProto.graph") be
  ///    tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d
  /// and the "algorithm" graph be
  ///    tensor_d -> Add -> tensor_e
  /// The combination process results
  ///    tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d -> Add -> tensor_e
  ///
  /// Notice that an input of a node in the "algorithm" graph may reference the
  /// output of a node in the inference graph (but not the other way round). Also, inference
  /// node cannot reference inputs of "algorithm". With these restrictions, inference graph
  /// can always be run independently without training information.
  ///
  /// By default, this field is an empty graph and its evaluation does not
  /// produce any output. Evaluating the default training step never
  /// update any initializers.
  @$pb.TagNumber(2)
  GraphProto get algorithm => $_getN(1);
  @$pb.TagNumber(2)
  set algorithm(GraphProto value) => $_setField(2, value);
  @$pb.TagNumber(2)
  $core.bool hasAlgorithm() => $_has(1);
  @$pb.TagNumber(2)
  void clearAlgorithm() => $_clearField(2);
  @$pb.TagNumber(2)
  GraphProto ensureAlgorithm() => $_ensure(1);

  /// This field specifies the bindings from the outputs of "initialization" to
  /// some initializers in "ModelProto.graph.initializer" and
  /// the "algorithm.initializer" in the same TrainingInfoProto.
  /// See "update_binding" below for details.
  ///
  /// By default, this field is empty and no initializer would be changed
  /// by the execution of "initialization".
  @$pb.TagNumber(3)
  $pb.PbList<StringStringEntryProto> get initializationBinding => $_getList(2);

  /// Gradient-based training is usually an iterative procedure. In one gradient
  /// descent iteration, we apply
  ///
  /// x = x - r * g
  ///
  /// where "x" is the optimized tensor, "r" stands for learning rate, and "g" is
  /// gradient of "x" with respect to a chosen loss. To avoid adding assignments
  /// into the training graph, we split the update equation into
  ///
  /// y = x - r * g
  /// x = y
  ///
  /// The user needs to save "y = x - r * g" into TrainingInfoProto.algorithm. To
  /// tell that "y" should be assigned to "x", the field "update_binding" may
  /// contain a key-value pair of strings, "x" (key of StringStringEntryProto)
  /// and "y" (value of StringStringEntryProto).
  /// For a neural network with multiple trainable (mutable) tensors, there can
  /// be multiple key-value pairs in "update_binding".
  ///
  /// The initializers appears as keys in "update_binding" are considered
  /// mutable variables. This implies some behaviors
  /// as described below.
  ///
  ///  1. We have only unique keys in all "update_binding"s so that two
  ///     variables may not have the same name. This ensures that one
  ///     variable is assigned up to once.
  ///  2. The keys must appear in names of "ModelProto.graph.initializer" or
  ///     "TrainingInfoProto.algorithm.initializer".
  ///  3. The values must be output names of "algorithm" or "ModelProto.graph.output".
  ///  4. Mutable variables are initialized to the value specified by the
  ///     corresponding initializer, and then potentially updated by
  ///     "initializer_binding"s and "update_binding"s in "TrainingInfoProto"s.
  ///
  /// This field usually contains names of trainable tensors
  /// (in ModelProto.graph), optimizer states such as momentums in advanced
  /// stochastic gradient methods (in TrainingInfoProto.graph),
  /// and number of training iterations (in TrainingInfoProto.graph).
  ///
  /// By default, this field is empty and no initializer would be changed
  /// by the execution of "algorithm".
  @$pb.TagNumber(4)
  $pb.PbList<StringStringEntryProto> get updateBinding => $_getList(3);
}

/// Models
///
/// ModelProto is a top-level file/container format for bundling a ML model and
/// associating its computation graph with metadata.
///
/// The semantics of the model are described by the associated GraphProto's.
class ModelProto extends $pb.GeneratedMessage {
  factory ModelProto({
    $fixnum.Int64? irVersion,
    $core.String? producerName,
    $core.String? producerVersion,
    $core.String? domain,
    $fixnum.Int64? modelVersion,
    $core.String? docString,
    GraphProto? graph,
    $core.Iterable<OperatorSetIdProto>? opsetImport,
    $core.Iterable<StringStringEntryProto>? metadataProps,
    $core.Iterable<TrainingInfoProto>? trainingInfo,
    $core.Iterable<FunctionProto>? functions,
    $core.Iterable<DeviceConfigurationProto>? configuration,
  }) {
    final result = create();
    if (irVersion != null) result.irVersion = irVersion;
    if (producerName != null) result.producerName = producerName;
    if (producerVersion != null) result.producerVersion = producerVersion;
    if (domain != null) result.domain = domain;
    if (modelVersion != null) result.modelVersion = modelVersion;
    if (docString != null) result.docString = docString;
    if (graph != null) result.graph = graph;
    if (opsetImport != null) result.opsetImport.addAll(opsetImport);
    if (metadataProps != null) result.metadataProps.addAll(metadataProps);
    if (trainingInfo != null) result.trainingInfo.addAll(trainingInfo);
    if (functions != null) result.functions.addAll(functions);
    if (configuration != null) result.configuration.addAll(configuration);
    return result;
  }

  ModelProto._();

  factory ModelProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory ModelProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'ModelProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aInt64(1, _omitFieldNames ? '' : 'irVersion')
    ..aOS(2, _omitFieldNames ? '' : 'producerName')
    ..aOS(3, _omitFieldNames ? '' : 'producerVersion')
    ..aOS(4, _omitFieldNames ? '' : 'domain')
    ..aInt64(5, _omitFieldNames ? '' : 'modelVersion')
    ..aOS(6, _omitFieldNames ? '' : 'docString')
    ..aOM<GraphProto>(7, _omitFieldNames ? '' : 'graph',
        subBuilder: GraphProto.create)
    ..pPM<OperatorSetIdProto>(8, _omitFieldNames ? '' : 'opsetImport',
        subBuilder: OperatorSetIdProto.create)
    ..pPM<StringStringEntryProto>(14, _omitFieldNames ? '' : 'metadataProps',
        subBuilder: StringStringEntryProto.create)
    ..pPM<TrainingInfoProto>(20, _omitFieldNames ? '' : 'trainingInfo',
        subBuilder: TrainingInfoProto.create)
    ..pPM<FunctionProto>(25, _omitFieldNames ? '' : 'functions',
        subBuilder: FunctionProto.create)
    ..pPM<DeviceConfigurationProto>(26, _omitFieldNames ? '' : 'configuration',
        subBuilder: DeviceConfigurationProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ModelProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ModelProto copyWith(void Function(ModelProto) updates) =>
      super.copyWith((message) => updates(message as ModelProto)) as ModelProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static ModelProto create() => ModelProto._();
  @$core.override
  ModelProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static ModelProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<ModelProto>(create);
  static ModelProto? _defaultInstance;

  /// The version of the IR this model targets. See Version enum above.
  /// This field MUST be present.
  @$pb.TagNumber(1)
  $fixnum.Int64 get irVersion => $_getI64(0);
  @$pb.TagNumber(1)
  set irVersion($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasIrVersion() => $_has(0);
  @$pb.TagNumber(1)
  void clearIrVersion() => $_clearField(1);

  /// The name of the framework or tool used to generate this model.
  /// This field SHOULD be present to indicate which implementation/tool/framework
  /// emitted the model.
  @$pb.TagNumber(2)
  $core.String get producerName => $_getSZ(1);
  @$pb.TagNumber(2)
  set producerName($core.String value) => $_setString(1, value);
  @$pb.TagNumber(2)
  $core.bool hasProducerName() => $_has(1);
  @$pb.TagNumber(2)
  void clearProducerName() => $_clearField(2);

  /// The version of the framework or tool used to generate this model.
  /// This field SHOULD be present to indicate which implementation/tool/framework
  /// emitted the model.
  @$pb.TagNumber(3)
  $core.String get producerVersion => $_getSZ(2);
  @$pb.TagNumber(3)
  set producerVersion($core.String value) => $_setString(2, value);
  @$pb.TagNumber(3)
  $core.bool hasProducerVersion() => $_has(2);
  @$pb.TagNumber(3)
  void clearProducerVersion() => $_clearField(3);

  /// Domain name of the model.
  /// We use reverse domain names as name space indicators. For example:
  /// `com.facebook.fair` or `com.microsoft.cognitiveservices`
  ///
  /// Together with `model_version` and GraphProto.name, this forms the unique identity of
  /// the graph.
  @$pb.TagNumber(4)
  $core.String get domain => $_getSZ(3);
  @$pb.TagNumber(4)
  set domain($core.String value) => $_setString(3, value);
  @$pb.TagNumber(4)
  $core.bool hasDomain() => $_has(3);
  @$pb.TagNumber(4)
  void clearDomain() => $_clearField(4);

  /// The version of the graph encoded. See Version enum below.
  @$pb.TagNumber(5)
  $fixnum.Int64 get modelVersion => $_getI64(4);
  @$pb.TagNumber(5)
  set modelVersion($fixnum.Int64 value) => $_setInt64(4, value);
  @$pb.TagNumber(5)
  $core.bool hasModelVersion() => $_has(4);
  @$pb.TagNumber(5)
  void clearModelVersion() => $_clearField(5);

  /// A human-readable documentation for this model. Markdown is allowed.
  @$pb.TagNumber(6)
  $core.String get docString => $_getSZ(5);
  @$pb.TagNumber(6)
  set docString($core.String value) => $_setString(5, value);
  @$pb.TagNumber(6)
  $core.bool hasDocString() => $_has(5);
  @$pb.TagNumber(6)
  void clearDocString() => $_clearField(6);

  /// The parameterized graph that is evaluated to execute the model.
  @$pb.TagNumber(7)
  GraphProto get graph => $_getN(6);
  @$pb.TagNumber(7)
  set graph(GraphProto value) => $_setField(7, value);
  @$pb.TagNumber(7)
  $core.bool hasGraph() => $_has(6);
  @$pb.TagNumber(7)
  void clearGraph() => $_clearField(7);
  @$pb.TagNumber(7)
  GraphProto ensureGraph() => $_ensure(6);

  /// The OperatorSets this model relies on.
  /// All ModelProtos MUST have at least one entry that
  /// specifies which version of the ONNX OperatorSet is
  /// being imported.
  ///
  /// All nodes in the ModelProto's graph will bind against the operator
  /// with the same-domain/same-op_type operator with the HIGHEST version
  /// in the referenced operator sets.
  @$pb.TagNumber(8)
  $pb.PbList<OperatorSetIdProto> get opsetImport => $_getList(7);

  /// Named metadata values; keys should be distinct.
  @$pb.TagNumber(14)
  $pb.PbList<StringStringEntryProto> get metadataProps => $_getList(8);

  /// Training-specific information. Sequentially executing all stored
  /// `TrainingInfoProto.algorithm`s and assigning their outputs following
  /// the corresponding `TrainingInfoProto.update_binding`s is one training
  /// iteration. Similarly, to initialize the model
  /// (as if training hasn't happened), the user should sequentially execute
  /// all stored `TrainingInfoProto.initialization`s and assigns their outputs
  /// using `TrainingInfoProto.initialization_binding`s.
  ///
  /// If this field is empty, the training behavior of the model is undefined.
  @$pb.TagNumber(20)
  $pb.PbList<TrainingInfoProto> get trainingInfo => $_getList(9);

  /// A list of function protos local to the model.
  ///
  /// The (domain, name, overload) tuple must be unique across the function protos in this list.
  /// In case of any conflicts the behavior (whether the model local functions are given higher priority,
  /// or standard operator sets are given higher priority or this is treated as error) is defined by
  /// the runtimes.
  ///
  /// The operator sets imported by FunctionProto should be compatible with the ones
  /// imported by ModelProto and other model local FunctionProtos.
  /// Example, if same operator set say 'A' is imported by a FunctionProto and ModelProto
  /// or by 2 FunctionProtos then versions for the operator set may be different but,
  /// the operator schema returned for op_type, domain, version combination
  /// for both the versions should be same for every node in the function body.
  ///
  /// One FunctionProto can reference other FunctionProto in the model, however, recursive reference
  /// is not allowed.
  @$pb.TagNumber(25)
  $pb.PbList<FunctionProto> get functions => $_getList(10);

  /// Describes different target configurations for a multi-device use case.
  /// A model MAY describe multiple multi-device configurations for execution.
  @$pb.TagNumber(26)
  $pb.PbList<DeviceConfigurationProto> get configuration => $_getList(11);
}

/// DeviceConfigurationProto describes a multi-device configuration for a model.
class DeviceConfigurationProto extends $pb.GeneratedMessage {
  factory DeviceConfigurationProto({
    $core.String? name,
    $core.int? numDevices,
    $core.Iterable<$core.String>? device,
  }) {
    final result = create();
    if (name != null) result.name = name;
    if (numDevices != null) result.numDevices = numDevices;
    if (device != null) result.device.addAll(device);
    return result;
  }

  DeviceConfigurationProto._();

  factory DeviceConfigurationProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory DeviceConfigurationProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'DeviceConfigurationProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'name')
    ..aI(2, _omitFieldNames ? '' : 'numDevices')
    ..pPS(3, _omitFieldNames ? '' : 'device')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  DeviceConfigurationProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  DeviceConfigurationProto copyWith(
          void Function(DeviceConfigurationProto) updates) =>
      super.copyWith((message) => updates(message as DeviceConfigurationProto))
          as DeviceConfigurationProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static DeviceConfigurationProto create() => DeviceConfigurationProto._();
  @$core.override
  DeviceConfigurationProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static DeviceConfigurationProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<DeviceConfigurationProto>(create);
  static DeviceConfigurationProto? _defaultInstance;

  /// This field MUST be present for this version of the IR.
  /// Name of the configuration.
  @$pb.TagNumber(1)
  $core.String get name => $_getSZ(0);
  @$pb.TagNumber(1)
  set name($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasName() => $_has(0);
  @$pb.TagNumber(1)
  void clearName() => $_clearField(1);

  /// This field MUST be present for this version of the IR.
  /// Number of devices inside this configuration.
  @$pb.TagNumber(2)
  $core.int get numDevices => $_getIZ(1);
  @$pb.TagNumber(2)
  set numDevices($core.int value) => $_setSignedInt32(1, value);
  @$pb.TagNumber(2)
  $core.bool hasNumDevices() => $_has(1);
  @$pb.TagNumber(2)
  void clearNumDevices() => $_clearField(2);

  /// Optional names of the devices. MUST be length of num_devices if provided.
  @$pb.TagNumber(3)
  $pb.PbList<$core.String> get device => $_getList(2);
}

/// StringStringEntryProto follows the pattern for cross-proto-version maps.
/// See https://developers.google.com/protocol-buffers/docs/proto3#maps
class StringStringEntryProto extends $pb.GeneratedMessage {
  factory StringStringEntryProto({
    $core.String? key,
    $core.String? value,
  }) {
    final result = create();
    if (key != null) result.key = key;
    if (value != null) result.value = value;
    return result;
  }

  StringStringEntryProto._();

  factory StringStringEntryProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory StringStringEntryProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'StringStringEntryProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'key')
    ..aOS(2, _omitFieldNames ? '' : 'value')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  StringStringEntryProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  StringStringEntryProto copyWith(
          void Function(StringStringEntryProto) updates) =>
      super.copyWith((message) => updates(message as StringStringEntryProto))
          as StringStringEntryProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static StringStringEntryProto create() => StringStringEntryProto._();
  @$core.override
  StringStringEntryProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static StringStringEntryProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<StringStringEntryProto>(create);
  static StringStringEntryProto? _defaultInstance;

  @$pb.TagNumber(1)
  $core.String get key => $_getSZ(0);
  @$pb.TagNumber(1)
  set key($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasKey() => $_has(0);
  @$pb.TagNumber(1)
  void clearKey() => $_clearField(1);

  @$pb.TagNumber(2)
  $core.String get value => $_getSZ(1);
  @$pb.TagNumber(2)
  set value($core.String value) => $_setString(1, value);
  @$pb.TagNumber(2)
  $core.bool hasValue() => $_has(1);
  @$pb.TagNumber(2)
  void clearValue() => $_clearField(2);
}

class TensorAnnotation extends $pb.GeneratedMessage {
  factory TensorAnnotation({
    $core.String? tensorName,
    $core.Iterable<StringStringEntryProto>? quantParameterTensorNames,
  }) {
    final result = create();
    if (tensorName != null) result.tensorName = tensorName;
    if (quantParameterTensorNames != null)
      result.quantParameterTensorNames.addAll(quantParameterTensorNames);
    return result;
  }

  TensorAnnotation._();

  factory TensorAnnotation.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TensorAnnotation.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TensorAnnotation',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'tensorName')
    ..pPM<StringStringEntryProto>(
        2, _omitFieldNames ? '' : 'quantParameterTensorNames',
        subBuilder: StringStringEntryProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorAnnotation clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorAnnotation copyWith(void Function(TensorAnnotation) updates) =>
      super.copyWith((message) => updates(message as TensorAnnotation))
          as TensorAnnotation;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TensorAnnotation create() => TensorAnnotation._();
  @$core.override
  TensorAnnotation createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TensorAnnotation getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TensorAnnotation>(create);
  static TensorAnnotation? _defaultInstance;

  @$pb.TagNumber(1)
  $core.String get tensorName => $_getSZ(0);
  @$pb.TagNumber(1)
  set tensorName($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasTensorName() => $_has(0);
  @$pb.TagNumber(1)
  void clearTensorName() => $_clearField(1);

  /// <key, value> pairs to annotate tensor specified by <tensor_name> above.
  /// The keys used in the mapping below must be pre-defined in ONNX spec.
  /// For example, for 8-bit linear quantization case, 'SCALE_TENSOR', 'ZERO_POINT_TENSOR' will be pre-defined as
  /// quantization parameter keys.
  @$pb.TagNumber(2)
  $pb.PbList<StringStringEntryProto> get quantParameterTensorNames =>
      $_getList(1);
}

/// Graphs
///
/// A graph defines the computational logic of a model and is comprised of a parameterized
/// list of nodes that form a directed acyclic graph based on their inputs and outputs.
/// This is the equivalent of the "network" or "graph" in many deep learning
/// frameworks.
class GraphProto extends $pb.GeneratedMessage {
  factory GraphProto({
    $core.Iterable<NodeProto>? node,
    $core.String? name,
    $core.Iterable<TensorProto>? initializer,
    $core.String? docString,
    $core.Iterable<ValueInfoProto>? input,
    $core.Iterable<ValueInfoProto>? output,
    $core.Iterable<ValueInfoProto>? valueInfo,
    $core.Iterable<TensorAnnotation>? quantizationAnnotation,
    $core.Iterable<SparseTensorProto>? sparseInitializer,
    $core.Iterable<StringStringEntryProto>? metadataProps,
  }) {
    final result = create();
    if (node != null) result.node.addAll(node);
    if (name != null) result.name = name;
    if (initializer != null) result.initializer.addAll(initializer);
    if (docString != null) result.docString = docString;
    if (input != null) result.input.addAll(input);
    if (output != null) result.output.addAll(output);
    if (valueInfo != null) result.valueInfo.addAll(valueInfo);
    if (quantizationAnnotation != null)
      result.quantizationAnnotation.addAll(quantizationAnnotation);
    if (sparseInitializer != null)
      result.sparseInitializer.addAll(sparseInitializer);
    if (metadataProps != null) result.metadataProps.addAll(metadataProps);
    return result;
  }

  GraphProto._();

  factory GraphProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory GraphProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'GraphProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..pPM<NodeProto>(1, _omitFieldNames ? '' : 'node',
        subBuilder: NodeProto.create)
    ..aOS(2, _omitFieldNames ? '' : 'name')
    ..pPM<TensorProto>(5, _omitFieldNames ? '' : 'initializer',
        subBuilder: TensorProto.create)
    ..aOS(10, _omitFieldNames ? '' : 'docString')
    ..pPM<ValueInfoProto>(11, _omitFieldNames ? '' : 'input',
        subBuilder: ValueInfoProto.create)
    ..pPM<ValueInfoProto>(12, _omitFieldNames ? '' : 'output',
        subBuilder: ValueInfoProto.create)
    ..pPM<ValueInfoProto>(13, _omitFieldNames ? '' : 'valueInfo',
        subBuilder: ValueInfoProto.create)
    ..pPM<TensorAnnotation>(14, _omitFieldNames ? '' : 'quantizationAnnotation',
        subBuilder: TensorAnnotation.create)
    ..pPM<SparseTensorProto>(15, _omitFieldNames ? '' : 'sparseInitializer',
        subBuilder: SparseTensorProto.create)
    ..pPM<StringStringEntryProto>(16, _omitFieldNames ? '' : 'metadataProps',
        subBuilder: StringStringEntryProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  GraphProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  GraphProto copyWith(void Function(GraphProto) updates) =>
      super.copyWith((message) => updates(message as GraphProto)) as GraphProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static GraphProto create() => GraphProto._();
  @$core.override
  GraphProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static GraphProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<GraphProto>(create);
  static GraphProto? _defaultInstance;

  /// The nodes in the graph, sorted topologically.
  @$pb.TagNumber(1)
  $pb.PbList<NodeProto> get node => $_getList(0);

  /// The name of the graph.
  @$pb.TagNumber(2)
  $core.String get name => $_getSZ(1);
  @$pb.TagNumber(2)
  set name($core.String value) => $_setString(1, value);
  @$pb.TagNumber(2)
  $core.bool hasName() => $_has(1);
  @$pb.TagNumber(2)
  void clearName() => $_clearField(2);

  /// A list of named tensor values, used to specify constant inputs of the graph.
  /// Each initializer (both TensorProto as well SparseTensorProto) MUST have a name.
  /// The name MUST be unique across both initializer and sparse_initializer,
  /// but the name MAY also appear in the input list.
  @$pb.TagNumber(5)
  $pb.PbList<TensorProto> get initializer => $_getList(2);

  /// A human-readable documentation for this graph. Markdown is allowed.
  @$pb.TagNumber(10)
  $core.String get docString => $_getSZ(3);
  @$pb.TagNumber(10)
  set docString($core.String value) => $_setString(3, value);
  @$pb.TagNumber(10)
  $core.bool hasDocString() => $_has(3);
  @$pb.TagNumber(10)
  void clearDocString() => $_clearField(10);

  /// The inputs and outputs of the graph.
  @$pb.TagNumber(11)
  $pb.PbList<ValueInfoProto> get input => $_getList(4);

  @$pb.TagNumber(12)
  $pb.PbList<ValueInfoProto> get output => $_getList(5);

  /// Information for the values in the graph. The ValueInfoProto.name's
  /// must be distinct. It is optional for a value to appear in value_info list.
  @$pb.TagNumber(13)
  $pb.PbList<ValueInfoProto> get valueInfo => $_getList(6);

  /// This field carries information to indicate the mapping among a tensor and its
  /// quantization parameter tensors. For example:
  /// For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated,
  /// which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.
  @$pb.TagNumber(14)
  $pb.PbList<TensorAnnotation> get quantizationAnnotation => $_getList(7);

  /// Initializers (see above) stored in sparse format.
  @$pb.TagNumber(15)
  $pb.PbList<SparseTensorProto> get sparseInitializer => $_getList(8);

  /// Named metadata values; keys should be distinct.
  @$pb.TagNumber(16)
  $pb.PbList<StringStringEntryProto> get metadataProps => $_getList(9);
}

/// For very large tensors, we may want to store them in chunks, in which
/// case the following fields will specify the segment that is stored in
/// the current TensorProto.
class TensorProto_Segment extends $pb.GeneratedMessage {
  factory TensorProto_Segment({
    $fixnum.Int64? begin,
    $fixnum.Int64? end,
  }) {
    final result = create();
    if (begin != null) result.begin = begin;
    if (end != null) result.end = end;
    return result;
  }

  TensorProto_Segment._();

  factory TensorProto_Segment.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TensorProto_Segment.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TensorProto.Segment',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aInt64(1, _omitFieldNames ? '' : 'begin')
    ..aInt64(2, _omitFieldNames ? '' : 'end')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorProto_Segment clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorProto_Segment copyWith(void Function(TensorProto_Segment) updates) =>
      super.copyWith((message) => updates(message as TensorProto_Segment))
          as TensorProto_Segment;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TensorProto_Segment create() => TensorProto_Segment._();
  @$core.override
  TensorProto_Segment createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TensorProto_Segment getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TensorProto_Segment>(create);
  static TensorProto_Segment? _defaultInstance;

  @$pb.TagNumber(1)
  $fixnum.Int64 get begin => $_getI64(0);
  @$pb.TagNumber(1)
  set begin($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasBegin() => $_has(0);
  @$pb.TagNumber(1)
  void clearBegin() => $_clearField(1);

  @$pb.TagNumber(2)
  $fixnum.Int64 get end => $_getI64(1);
  @$pb.TagNumber(2)
  set end($fixnum.Int64 value) => $_setInt64(1, value);
  @$pb.TagNumber(2)
  $core.bool hasEnd() => $_has(1);
  @$pb.TagNumber(2)
  void clearEnd() => $_clearField(2);
}

/// Tensors
///
/// A serialized tensor value.
class TensorProto extends $pb.GeneratedMessage {
  factory TensorProto({
    $core.Iterable<$fixnum.Int64>? dims,
    $core.int? dataType,
    TensorProto_Segment? segment,
    $core.Iterable<$core.double>? floatData,
    $core.Iterable<$core.int>? int32Data,
    $core.Iterable<$core.List<$core.int>>? stringData,
    $core.Iterable<$fixnum.Int64>? int64Data,
    $core.String? name,
    $core.List<$core.int>? rawData,
    $core.Iterable<$core.double>? doubleData,
    $core.Iterable<$fixnum.Int64>? uint64Data,
    $core.String? docString,
    $core.Iterable<StringStringEntryProto>? externalData,
    TensorProto_DataLocation? dataLocation,
    $core.Iterable<StringStringEntryProto>? metadataProps,
  }) {
    final result = create();
    if (dims != null) result.dims.addAll(dims);
    if (dataType != null) result.dataType = dataType;
    if (segment != null) result.segment = segment;
    if (floatData != null) result.floatData.addAll(floatData);
    if (int32Data != null) result.int32Data.addAll(int32Data);
    if (stringData != null) result.stringData.addAll(stringData);
    if (int64Data != null) result.int64Data.addAll(int64Data);
    if (name != null) result.name = name;
    if (rawData != null) result.rawData = rawData;
    if (doubleData != null) result.doubleData.addAll(doubleData);
    if (uint64Data != null) result.uint64Data.addAll(uint64Data);
    if (docString != null) result.docString = docString;
    if (externalData != null) result.externalData.addAll(externalData);
    if (dataLocation != null) result.dataLocation = dataLocation;
    if (metadataProps != null) result.metadataProps.addAll(metadataProps);
    return result;
  }

  TensorProto._();

  factory TensorProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TensorProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TensorProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..p<$fixnum.Int64>(1, _omitFieldNames ? '' : 'dims', $pb.PbFieldType.K6)
    ..aI(2, _omitFieldNames ? '' : 'dataType')
    ..aOM<TensorProto_Segment>(3, _omitFieldNames ? '' : 'segment',
        subBuilder: TensorProto_Segment.create)
    ..p<$core.double>(4, _omitFieldNames ? '' : 'floatData', $pb.PbFieldType.KF)
    ..p<$core.int>(5, _omitFieldNames ? '' : 'int32Data', $pb.PbFieldType.K3)
    ..p<$core.List<$core.int>>(
        6, _omitFieldNames ? '' : 'stringData', $pb.PbFieldType.PY)
    ..p<$fixnum.Int64>(
        7, _omitFieldNames ? '' : 'int64Data', $pb.PbFieldType.K6)
    ..aOS(8, _omitFieldNames ? '' : 'name')
    ..a<$core.List<$core.int>>(
        9, _omitFieldNames ? '' : 'rawData', $pb.PbFieldType.OY)
    ..p<$core.double>(
        10, _omitFieldNames ? '' : 'doubleData', $pb.PbFieldType.KD)
    ..p<$fixnum.Int64>(
        11, _omitFieldNames ? '' : 'uint64Data', $pb.PbFieldType.KU6)
    ..aOS(12, _omitFieldNames ? '' : 'docString')
    ..pPM<StringStringEntryProto>(13, _omitFieldNames ? '' : 'externalData',
        subBuilder: StringStringEntryProto.create)
    ..aE<TensorProto_DataLocation>(14, _omitFieldNames ? '' : 'dataLocation',
        enumValues: TensorProto_DataLocation.values)
    ..pPM<StringStringEntryProto>(16, _omitFieldNames ? '' : 'metadataProps',
        subBuilder: StringStringEntryProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorProto copyWith(void Function(TensorProto) updates) =>
      super.copyWith((message) => updates(message as TensorProto))
          as TensorProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TensorProto create() => TensorProto._();
  @$core.override
  TensorProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TensorProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TensorProto>(create);
  static TensorProto? _defaultInstance;

  /// The shape of the tensor.
  @$pb.TagNumber(1)
  $pb.PbList<$fixnum.Int64> get dims => $_getList(0);

  /// The data type of the tensor.
  /// This field MUST have a valid TensorProto.DataType value
  @$pb.TagNumber(2)
  $core.int get dataType => $_getIZ(1);
  @$pb.TagNumber(2)
  set dataType($core.int value) => $_setSignedInt32(1, value);
  @$pb.TagNumber(2)
  $core.bool hasDataType() => $_has(1);
  @$pb.TagNumber(2)
  void clearDataType() => $_clearField(2);

  @$pb.TagNumber(3)
  TensorProto_Segment get segment => $_getN(2);
  @$pb.TagNumber(3)
  set segment(TensorProto_Segment value) => $_setField(3, value);
  @$pb.TagNumber(3)
  $core.bool hasSegment() => $_has(2);
  @$pb.TagNumber(3)
  void clearSegment() => $_clearField(3);
  @$pb.TagNumber(3)
  TensorProto_Segment ensureSegment() => $_ensure(2);

  /// For float and complex64 values
  /// Complex64 tensors are encoded as a single array of floats,
  /// with the real components appearing in odd numbered positions,
  /// and the corresponding imaginary component appearing in the
  /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
  /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
  /// When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
  @$pb.TagNumber(4)
  $pb.PbList<$core.double> get floatData => $_getList(3);

  /// For int32, uint8, int8, uint16, int16, uint4, int4, uint2, int2, bool, (b)float16, float8, and float4:
  /// - (b)float16 and float8 values MUST be converted bit-wise into an unsigned integer
  ///   representation before being written to the buffer.
  /// - Each pair of uint4, int4, and float4 values MUST be packed as two 4-bit elements into a single byte.
  ///   The first element is stored in the 4 least significant bits (LSB),
  ///   and the second element is stored in the 4 most significant bits (MSB).
  /// - Each group of four uint2, int2 values MUST be packed as four 2-bit elements into a single byte.
  ///   The elements are packed from LSB to MSB, with the first element in bits 0-1, second element in bits 2-3,
  ///   third element in bits 4-5, and fourth element in bits 6-7.
  ///
  /// Consequently:
  /// - For data types with a bit-width of 8 or greater, each `int32_data` stores one element.
  /// - For 4-bit data types, each `int32_data` stores two elements.
  /// - For 2-bit data types, each `int32_data` stores four elements.
  ///
  /// When this field is present, the data_type field MUST be
  /// INT32, INT16, INT8, INT4, INT2, UINT16, UINT8, UINT4, UINT2, BOOL, FLOAT16, BFLOAT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, FLOAT8E8M0, FLOAT4E2M1
  @$pb.TagNumber(5)
  $pb.PbList<$core.int> get int32Data => $_getList(4);

  /// For strings.
  /// Each element of string_data is a UTF-8 encoded Unicode
  /// string. No trailing null, no leading BOM. The protobuf "string"
  /// scalar type is not used to match ML community conventions.
  /// When this field is present, the data_type field MUST be STRING
  @$pb.TagNumber(6)
  $pb.PbList<$core.List<$core.int>> get stringData => $_getList(5);

  /// For int64.
  /// When this field is present, the data_type field MUST be INT64
  @$pb.TagNumber(7)
  $pb.PbList<$fixnum.Int64> get int64Data => $_getList(6);

  /// Optionally, a name for the tensor.
  @$pb.TagNumber(8)
  $core.String get name => $_getSZ(7);
  @$pb.TagNumber(8)
  set name($core.String value) => $_setString(7, value);
  @$pb.TagNumber(8)
  $core.bool hasName() => $_has(7);
  @$pb.TagNumber(8)
  void clearName() => $_clearField(8);

  /// Serializations can either use one of the fields above, or use this
  /// raw bytes field. The only exception is the string case, where one is
  /// required to store the content in the repeated bytes string_data field.
  ///
  /// When this raw_data field is used to store tensor value, elements MUST
  /// be stored in as fixed-width, little-endian order.
  /// Floating-point data types MUST be stored in IEEE 754 format.
  /// Complex64 elements must be written as two consecutive FLOAT values, real component first.
  /// Complex128 elements must be written as two consecutive DOUBLE values, real component first.
  /// Boolean type MUST be written one byte per tensor element (00000001 for true, 00000000 for false).
  /// uint4 and int4 values must be packed to 4bitx2, the first element is stored in the 4 LSB and the second element is stored in the 4 MSB.
  /// uint2 and int2 values must be packed to 2bitx4, with elements packed from LSB to MSB in a single byte as: x0 | (x1 << 2) | (x2 << 4) | (x3 << 6)
  /// where x0, x1, x2, x3 are consecutive elements.
  ///
  /// Note: the advantage of specific field rather than the raw_data field is
  /// that in some cases (e.g. int data), protobuf does a better packing via
  /// variable length storage, and may lead to smaller binary footprint.
  /// When this field is present, the data_type field MUST NOT be STRING or UNDEFINED
  @$pb.TagNumber(9)
  $core.List<$core.int> get rawData => $_getN(8);
  @$pb.TagNumber(9)
  set rawData($core.List<$core.int> value) => $_setBytes(8, value);
  @$pb.TagNumber(9)
  $core.bool hasRawData() => $_has(8);
  @$pb.TagNumber(9)
  void clearRawData() => $_clearField(9);

  /// For double
  /// Complex128 tensors are encoded as a single array of doubles,
  /// with the real components appearing in odd numbered positions,
  /// and the corresponding imaginary component appearing in the
  /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
  /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
  /// When this field is present, the data_type field MUST be DOUBLE or COMPLEX128
  @$pb.TagNumber(10)
  $pb.PbList<$core.double> get doubleData => $_getList(9);

  /// For uint64 and uint32 values
  /// When this field is present, the data_type field MUST be
  /// UINT32 or UINT64
  @$pb.TagNumber(11)
  $pb.PbList<$fixnum.Int64> get uint64Data => $_getList(10);

  /// A human-readable documentation for this tensor. Markdown is allowed.
  @$pb.TagNumber(12)
  $core.String get docString => $_getSZ(11);
  @$pb.TagNumber(12)
  set docString($core.String value) => $_setString(11, value);
  @$pb.TagNumber(12)
  $core.bool hasDocString() => $_has(11);
  @$pb.TagNumber(12)
  void clearDocString() => $_clearField(12);

  /// Data can be stored inside the protobuf file using type-specific fields or raw_data.
  /// Alternatively, raw bytes data can be stored in an external file, using the external_data field.
  /// external_data stores key-value pairs describing data location. Recognized keys are:
  /// - "location" (required) - POSIX filesystem path relative to the directory where the ONNX
  ///                           protobuf model was stored
  /// - "offset" (optional) - position of byte at which stored data begins. Integer stored as string.
  ///                         Offset values SHOULD be multiples 4096 (page size) to enable mmap support.
  /// - "length" (optional) - number of bytes containing data. Integer stored as string.
  /// - "checksum" (optional) - SHA1 digest of file specified in under 'location' key.
  @$pb.TagNumber(13)
  $pb.PbList<StringStringEntryProto> get externalData => $_getList(12);

  /// If value not set, data is stored in raw_data (if set) otherwise in type-specified field.
  @$pb.TagNumber(14)
  TensorProto_DataLocation get dataLocation => $_getN(13);
  @$pb.TagNumber(14)
  set dataLocation(TensorProto_DataLocation value) => $_setField(14, value);
  @$pb.TagNumber(14)
  $core.bool hasDataLocation() => $_has(13);
  @$pb.TagNumber(14)
  void clearDataLocation() => $_clearField(14);

  /// Named metadata values; keys should be distinct.
  @$pb.TagNumber(16)
  $pb.PbList<StringStringEntryProto> get metadataProps => $_getList(14);
}

/// A serialized sparse-tensor value
class SparseTensorProto extends $pb.GeneratedMessage {
  factory SparseTensorProto({
    TensorProto? values,
    TensorProto? indices,
    $core.Iterable<$fixnum.Int64>? dims,
  }) {
    final result = create();
    if (values != null) result.values = values;
    if (indices != null) result.indices = indices;
    if (dims != null) result.dims.addAll(dims);
    return result;
  }

  SparseTensorProto._();

  factory SparseTensorProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory SparseTensorProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'SparseTensorProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOM<TensorProto>(1, _omitFieldNames ? '' : 'values',
        subBuilder: TensorProto.create)
    ..aOM<TensorProto>(2, _omitFieldNames ? '' : 'indices',
        subBuilder: TensorProto.create)
    ..p<$fixnum.Int64>(3, _omitFieldNames ? '' : 'dims', $pb.PbFieldType.K6)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  SparseTensorProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  SparseTensorProto copyWith(void Function(SparseTensorProto) updates) =>
      super.copyWith((message) => updates(message as SparseTensorProto))
          as SparseTensorProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static SparseTensorProto create() => SparseTensorProto._();
  @$core.override
  SparseTensorProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static SparseTensorProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<SparseTensorProto>(create);
  static SparseTensorProto? _defaultInstance;

  /// The sequence of non-default values are encoded as a tensor of shape [NNZ].
  /// The default-value is zero for numeric tensors, and empty-string for string tensors.
  /// values must have a non-empty name present which serves as a name for SparseTensorProto
  /// when used in sparse_initializer list.
  @$pb.TagNumber(1)
  TensorProto get values => $_getN(0);
  @$pb.TagNumber(1)
  set values(TensorProto value) => $_setField(1, value);
  @$pb.TagNumber(1)
  $core.bool hasValues() => $_has(0);
  @$pb.TagNumber(1)
  void clearValues() => $_clearField(1);
  @$pb.TagNumber(1)
  TensorProto ensureValues() => $_ensure(0);

  /// The indices of the non-default values, which may be stored in one of two formats.
  /// (a) Indices can be a tensor of shape [NNZ, rank] with the [i,j]-th value
  /// corresponding to the j-th index of the i-th value (in the values tensor).
  /// (b) Indices can be a tensor of shape [NNZ], in which case the i-th value
  /// must be the linearized-index of the i-th value (in the values tensor).
  /// The linearized-index can be converted into an index tuple (k_1,...,k_rank)
  /// using the shape provided below.
  /// The indices must appear in ascending order without duplication.
  /// In the first format, the ordering is lexicographic-ordering:
  /// e.g., index-value [1,4] must appear before [2,1]
  @$pb.TagNumber(2)
  TensorProto get indices => $_getN(1);
  @$pb.TagNumber(2)
  set indices(TensorProto value) => $_setField(2, value);
  @$pb.TagNumber(2)
  $core.bool hasIndices() => $_has(1);
  @$pb.TagNumber(2)
  void clearIndices() => $_clearField(2);
  @$pb.TagNumber(2)
  TensorProto ensureIndices() => $_ensure(1);

  /// The shape of the underlying dense-tensor: [dim_1, dim_2, ... dim_rank]
  @$pb.TagNumber(3)
  $pb.PbList<$fixnum.Int64> get dims => $_getList(2);
}

enum TensorShapeProto_Dimension_Value { dimValue, dimParam, notSet }

class TensorShapeProto_Dimension extends $pb.GeneratedMessage {
  factory TensorShapeProto_Dimension({
    $fixnum.Int64? dimValue,
    $core.String? dimParam,
    $core.String? denotation,
  }) {
    final result = create();
    if (dimValue != null) result.dimValue = dimValue;
    if (dimParam != null) result.dimParam = dimParam;
    if (denotation != null) result.denotation = denotation;
    return result;
  }

  TensorShapeProto_Dimension._();

  factory TensorShapeProto_Dimension.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TensorShapeProto_Dimension.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static const $core.Map<$core.int, TensorShapeProto_Dimension_Value>
      _TensorShapeProto_Dimension_ValueByTag = {
    1: TensorShapeProto_Dimension_Value.dimValue,
    2: TensorShapeProto_Dimension_Value.dimParam,
    0: TensorShapeProto_Dimension_Value.notSet
  };
  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TensorShapeProto.Dimension',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..oo(0, [1, 2])
    ..aInt64(1, _omitFieldNames ? '' : 'dimValue')
    ..aOS(2, _omitFieldNames ? '' : 'dimParam')
    ..aOS(3, _omitFieldNames ? '' : 'denotation')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorShapeProto_Dimension clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorShapeProto_Dimension copyWith(
          void Function(TensorShapeProto_Dimension) updates) =>
      super.copyWith(
              (message) => updates(message as TensorShapeProto_Dimension))
          as TensorShapeProto_Dimension;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TensorShapeProto_Dimension create() => TensorShapeProto_Dimension._();
  @$core.override
  TensorShapeProto_Dimension createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TensorShapeProto_Dimension getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TensorShapeProto_Dimension>(create);
  static TensorShapeProto_Dimension? _defaultInstance;

  @$pb.TagNumber(1)
  @$pb.TagNumber(2)
  TensorShapeProto_Dimension_Value whichValue() =>
      _TensorShapeProto_Dimension_ValueByTag[$_whichOneof(0)]!;
  @$pb.TagNumber(1)
  @$pb.TagNumber(2)
  void clearValue() => $_clearField($_whichOneof(0));

  @$pb.TagNumber(1)
  $fixnum.Int64 get dimValue => $_getI64(0);
  @$pb.TagNumber(1)
  set dimValue($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasDimValue() => $_has(0);
  @$pb.TagNumber(1)
  void clearDimValue() => $_clearField(1);

  @$pb.TagNumber(2)
  $core.String get dimParam => $_getSZ(1);
  @$pb.TagNumber(2)
  set dimParam($core.String value) => $_setString(1, value);
  @$pb.TagNumber(2)
  $core.bool hasDimParam() => $_has(1);
  @$pb.TagNumber(2)
  void clearDimParam() => $_clearField(2);

  /// Standard denotation can optionally be used to denote tensor
  /// dimensions with standard semantic descriptions to ensure
  /// that operations are applied to the correct axis of a tensor.
  /// Refer to https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
  /// for pre-defined dimension denotations.
  @$pb.TagNumber(3)
  $core.String get denotation => $_getSZ(2);
  @$pb.TagNumber(3)
  set denotation($core.String value) => $_setString(2, value);
  @$pb.TagNumber(3)
  $core.bool hasDenotation() => $_has(2);
  @$pb.TagNumber(3)
  void clearDenotation() => $_clearField(3);
}

/// Defines a tensor shape. A dimension can be either an integer value
/// or a symbolic variable. A symbolic variable represents an unknown
/// dimension.
class TensorShapeProto extends $pb.GeneratedMessage {
  factory TensorShapeProto({
    $core.Iterable<TensorShapeProto_Dimension>? dim,
  }) {
    final result = create();
    if (dim != null) result.dim.addAll(dim);
    return result;
  }

  TensorShapeProto._();

  factory TensorShapeProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TensorShapeProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TensorShapeProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..pPM<TensorShapeProto_Dimension>(1, _omitFieldNames ? '' : 'dim',
        subBuilder: TensorShapeProto_Dimension.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorShapeProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TensorShapeProto copyWith(void Function(TensorShapeProto) updates) =>
      super.copyWith((message) => updates(message as TensorShapeProto))
          as TensorShapeProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TensorShapeProto create() => TensorShapeProto._();
  @$core.override
  TensorShapeProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TensorShapeProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TensorShapeProto>(create);
  static TensorShapeProto? _defaultInstance;

  @$pb.TagNumber(1)
  $pb.PbList<TensorShapeProto_Dimension> get dim => $_getList(0);
}

class TypeProto_Tensor extends $pb.GeneratedMessage {
  factory TypeProto_Tensor({
    $core.int? elemType,
    TensorShapeProto? shape,
  }) {
    final result = create();
    if (elemType != null) result.elemType = elemType;
    if (shape != null) result.shape = shape;
    return result;
  }

  TypeProto_Tensor._();

  factory TypeProto_Tensor.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TypeProto_Tensor.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TypeProto.Tensor',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aI(1, _omitFieldNames ? '' : 'elemType')
    ..aOM<TensorShapeProto>(2, _omitFieldNames ? '' : 'shape',
        subBuilder: TensorShapeProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Tensor clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Tensor copyWith(void Function(TypeProto_Tensor) updates) =>
      super.copyWith((message) => updates(message as TypeProto_Tensor))
          as TypeProto_Tensor;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TypeProto_Tensor create() => TypeProto_Tensor._();
  @$core.override
  TypeProto_Tensor createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TypeProto_Tensor getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TypeProto_Tensor>(create);
  static TypeProto_Tensor? _defaultInstance;

  /// This field MUST NOT have the value of UNDEFINED
  /// This field MUST have a valid TensorProto.DataType value
  /// This field MUST be present for this version of the IR.
  @$pb.TagNumber(1)
  $core.int get elemType => $_getIZ(0);
  @$pb.TagNumber(1)
  set elemType($core.int value) => $_setSignedInt32(0, value);
  @$pb.TagNumber(1)
  $core.bool hasElemType() => $_has(0);
  @$pb.TagNumber(1)
  void clearElemType() => $_clearField(1);

  @$pb.TagNumber(2)
  TensorShapeProto get shape => $_getN(1);
  @$pb.TagNumber(2)
  set shape(TensorShapeProto value) => $_setField(2, value);
  @$pb.TagNumber(2)
  $core.bool hasShape() => $_has(1);
  @$pb.TagNumber(2)
  void clearShape() => $_clearField(2);
  @$pb.TagNumber(2)
  TensorShapeProto ensureShape() => $_ensure(1);
}

/// repeated T
class TypeProto_Sequence extends $pb.GeneratedMessage {
  factory TypeProto_Sequence({
    TypeProto? elemType,
  }) {
    final result = create();
    if (elemType != null) result.elemType = elemType;
    return result;
  }

  TypeProto_Sequence._();

  factory TypeProto_Sequence.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TypeProto_Sequence.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TypeProto.Sequence',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOM<TypeProto>(1, _omitFieldNames ? '' : 'elemType',
        subBuilder: TypeProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Sequence clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Sequence copyWith(void Function(TypeProto_Sequence) updates) =>
      super.copyWith((message) => updates(message as TypeProto_Sequence))
          as TypeProto_Sequence;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TypeProto_Sequence create() => TypeProto_Sequence._();
  @$core.override
  TypeProto_Sequence createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TypeProto_Sequence getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TypeProto_Sequence>(create);
  static TypeProto_Sequence? _defaultInstance;

  /// The type and optional shape of each element of the sequence.
  /// This field MUST be present for this version of the IR.
  @$pb.TagNumber(1)
  TypeProto get elemType => $_getN(0);
  @$pb.TagNumber(1)
  set elemType(TypeProto value) => $_setField(1, value);
  @$pb.TagNumber(1)
  $core.bool hasElemType() => $_has(0);
  @$pb.TagNumber(1)
  void clearElemType() => $_clearField(1);
  @$pb.TagNumber(1)
  TypeProto ensureElemType() => $_ensure(0);
}

/// map<K,V>
class TypeProto_Map extends $pb.GeneratedMessage {
  factory TypeProto_Map({
    $core.int? keyType,
    TypeProto? valueType,
  }) {
    final result = create();
    if (keyType != null) result.keyType = keyType;
    if (valueType != null) result.valueType = valueType;
    return result;
  }

  TypeProto_Map._();

  factory TypeProto_Map.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TypeProto_Map.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TypeProto.Map',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aI(1, _omitFieldNames ? '' : 'keyType')
    ..aOM<TypeProto>(2, _omitFieldNames ? '' : 'valueType',
        subBuilder: TypeProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Map clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Map copyWith(void Function(TypeProto_Map) updates) =>
      super.copyWith((message) => updates(message as TypeProto_Map))
          as TypeProto_Map;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TypeProto_Map create() => TypeProto_Map._();
  @$core.override
  TypeProto_Map createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TypeProto_Map getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TypeProto_Map>(create);
  static TypeProto_Map? _defaultInstance;

  /// This field MUST have a valid TensorProto.DataType value
  /// This field MUST be present for this version of the IR.
  /// This field MUST refer to an integral type ([U]INT{8|16|32|64}) or STRING
  @$pb.TagNumber(1)
  $core.int get keyType => $_getIZ(0);
  @$pb.TagNumber(1)
  set keyType($core.int value) => $_setSignedInt32(0, value);
  @$pb.TagNumber(1)
  $core.bool hasKeyType() => $_has(0);
  @$pb.TagNumber(1)
  void clearKeyType() => $_clearField(1);

  /// This field MUST be present for this version of the IR.
  @$pb.TagNumber(2)
  TypeProto get valueType => $_getN(1);
  @$pb.TagNumber(2)
  set valueType(TypeProto value) => $_setField(2, value);
  @$pb.TagNumber(2)
  $core.bool hasValueType() => $_has(1);
  @$pb.TagNumber(2)
  void clearValueType() => $_clearField(2);
  @$pb.TagNumber(2)
  TypeProto ensureValueType() => $_ensure(1);
}

/// wrapper for Tensor, Sequence, or Map
class TypeProto_Optional extends $pb.GeneratedMessage {
  factory TypeProto_Optional({
    TypeProto? elemType,
  }) {
    final result = create();
    if (elemType != null) result.elemType = elemType;
    return result;
  }

  TypeProto_Optional._();

  factory TypeProto_Optional.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TypeProto_Optional.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TypeProto.Optional',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOM<TypeProto>(1, _omitFieldNames ? '' : 'elemType',
        subBuilder: TypeProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Optional clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_Optional copyWith(void Function(TypeProto_Optional) updates) =>
      super.copyWith((message) => updates(message as TypeProto_Optional))
          as TypeProto_Optional;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TypeProto_Optional create() => TypeProto_Optional._();
  @$core.override
  TypeProto_Optional createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TypeProto_Optional getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TypeProto_Optional>(create);
  static TypeProto_Optional? _defaultInstance;

  /// The type and optional shape of the element wrapped.
  /// This field MUST be present for this version of the IR.
  /// Possible values correspond to OptionalProto.DataType enum
  @$pb.TagNumber(1)
  TypeProto get elemType => $_getN(0);
  @$pb.TagNumber(1)
  set elemType(TypeProto value) => $_setField(1, value);
  @$pb.TagNumber(1)
  $core.bool hasElemType() => $_has(0);
  @$pb.TagNumber(1)
  void clearElemType() => $_clearField(1);
  @$pb.TagNumber(1)
  TypeProto ensureElemType() => $_ensure(0);
}

class TypeProto_SparseTensor extends $pb.GeneratedMessage {
  factory TypeProto_SparseTensor({
    $core.int? elemType,
    TensorShapeProto? shape,
  }) {
    final result = create();
    if (elemType != null) result.elemType = elemType;
    if (shape != null) result.shape = shape;
    return result;
  }

  TypeProto_SparseTensor._();

  factory TypeProto_SparseTensor.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TypeProto_SparseTensor.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TypeProto.SparseTensor',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aI(1, _omitFieldNames ? '' : 'elemType')
    ..aOM<TensorShapeProto>(2, _omitFieldNames ? '' : 'shape',
        subBuilder: TensorShapeProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_SparseTensor clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto_SparseTensor copyWith(
          void Function(TypeProto_SparseTensor) updates) =>
      super.copyWith((message) => updates(message as TypeProto_SparseTensor))
          as TypeProto_SparseTensor;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TypeProto_SparseTensor create() => TypeProto_SparseTensor._();
  @$core.override
  TypeProto_SparseTensor createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TypeProto_SparseTensor getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<TypeProto_SparseTensor>(create);
  static TypeProto_SparseTensor? _defaultInstance;

  /// This field MUST NOT have the value of UNDEFINED
  /// This field MUST have a valid TensorProto.DataType value
  /// This field MUST be present for this version of the IR.
  @$pb.TagNumber(1)
  $core.int get elemType => $_getIZ(0);
  @$pb.TagNumber(1)
  set elemType($core.int value) => $_setSignedInt32(0, value);
  @$pb.TagNumber(1)
  $core.bool hasElemType() => $_has(0);
  @$pb.TagNumber(1)
  void clearElemType() => $_clearField(1);

  @$pb.TagNumber(2)
  TensorShapeProto get shape => $_getN(1);
  @$pb.TagNumber(2)
  set shape(TensorShapeProto value) => $_setField(2, value);
  @$pb.TagNumber(2)
  $core.bool hasShape() => $_has(1);
  @$pb.TagNumber(2)
  void clearShape() => $_clearField(2);
  @$pb.TagNumber(2)
  TensorShapeProto ensureShape() => $_ensure(1);
}

enum TypeProto_Value {
  tensorType,
  sequenceType,
  mapType,
  sparseTensorType,
  optionalType,
  notSet
}

/// Types
///
/// The standard ONNX data types.
class TypeProto extends $pb.GeneratedMessage {
  factory TypeProto({
    TypeProto_Tensor? tensorType,
    TypeProto_Sequence? sequenceType,
    TypeProto_Map? mapType,
    $core.String? denotation,
    TypeProto_SparseTensor? sparseTensorType,
    TypeProto_Optional? optionalType,
  }) {
    final result = create();
    if (tensorType != null) result.tensorType = tensorType;
    if (sequenceType != null) result.sequenceType = sequenceType;
    if (mapType != null) result.mapType = mapType;
    if (denotation != null) result.denotation = denotation;
    if (sparseTensorType != null) result.sparseTensorType = sparseTensorType;
    if (optionalType != null) result.optionalType = optionalType;
    return result;
  }

  TypeProto._();

  factory TypeProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory TypeProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static const $core.Map<$core.int, TypeProto_Value> _TypeProto_ValueByTag = {
    1: TypeProto_Value.tensorType,
    4: TypeProto_Value.sequenceType,
    5: TypeProto_Value.mapType,
    8: TypeProto_Value.sparseTensorType,
    9: TypeProto_Value.optionalType,
    0: TypeProto_Value.notSet
  };
  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'TypeProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..oo(0, [1, 4, 5, 8, 9])
    ..aOM<TypeProto_Tensor>(1, _omitFieldNames ? '' : 'tensorType',
        subBuilder: TypeProto_Tensor.create)
    ..aOM<TypeProto_Sequence>(4, _omitFieldNames ? '' : 'sequenceType',
        subBuilder: TypeProto_Sequence.create)
    ..aOM<TypeProto_Map>(5, _omitFieldNames ? '' : 'mapType',
        subBuilder: TypeProto_Map.create)
    ..aOS(6, _omitFieldNames ? '' : 'denotation')
    ..aOM<TypeProto_SparseTensor>(8, _omitFieldNames ? '' : 'sparseTensorType',
        subBuilder: TypeProto_SparseTensor.create)
    ..aOM<TypeProto_Optional>(9, _omitFieldNames ? '' : 'optionalType',
        subBuilder: TypeProto_Optional.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  TypeProto copyWith(void Function(TypeProto) updates) =>
      super.copyWith((message) => updates(message as TypeProto)) as TypeProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static TypeProto create() => TypeProto._();
  @$core.override
  TypeProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static TypeProto getDefault() =>
      _defaultInstance ??= $pb.GeneratedMessage.$_defaultFor<TypeProto>(create);
  static TypeProto? _defaultInstance;

  @$pb.TagNumber(1)
  @$pb.TagNumber(4)
  @$pb.TagNumber(5)
  @$pb.TagNumber(8)
  @$pb.TagNumber(9)
  TypeProto_Value whichValue() => _TypeProto_ValueByTag[$_whichOneof(0)]!;
  @$pb.TagNumber(1)
  @$pb.TagNumber(4)
  @$pb.TagNumber(5)
  @$pb.TagNumber(8)
  @$pb.TagNumber(9)
  void clearValue() => $_clearField($_whichOneof(0));

  /// The type of a tensor.
  @$pb.TagNumber(1)
  TypeProto_Tensor get tensorType => $_getN(0);
  @$pb.TagNumber(1)
  set tensorType(TypeProto_Tensor value) => $_setField(1, value);
  @$pb.TagNumber(1)
  $core.bool hasTensorType() => $_has(0);
  @$pb.TagNumber(1)
  void clearTensorType() => $_clearField(1);
  @$pb.TagNumber(1)
  TypeProto_Tensor ensureTensorType() => $_ensure(0);

  /// The type of a sequence.
  @$pb.TagNumber(4)
  TypeProto_Sequence get sequenceType => $_getN(1);
  @$pb.TagNumber(4)
  set sequenceType(TypeProto_Sequence value) => $_setField(4, value);
  @$pb.TagNumber(4)
  $core.bool hasSequenceType() => $_has(1);
  @$pb.TagNumber(4)
  void clearSequenceType() => $_clearField(4);
  @$pb.TagNumber(4)
  TypeProto_Sequence ensureSequenceType() => $_ensure(1);

  /// The type of a map.
  @$pb.TagNumber(5)
  TypeProto_Map get mapType => $_getN(2);
  @$pb.TagNumber(5)
  set mapType(TypeProto_Map value) => $_setField(5, value);
  @$pb.TagNumber(5)
  $core.bool hasMapType() => $_has(2);
  @$pb.TagNumber(5)
  void clearMapType() => $_clearField(5);
  @$pb.TagNumber(5)
  TypeProto_Map ensureMapType() => $_ensure(2);

  /// An optional denotation can be used to denote the whole
  /// type with a standard semantic description as to what is
  /// stored inside. Refer to https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition
  /// for pre-defined type denotations.
  @$pb.TagNumber(6)
  $core.String get denotation => $_getSZ(3);
  @$pb.TagNumber(6)
  set denotation($core.String value) => $_setString(3, value);
  @$pb.TagNumber(6)
  $core.bool hasDenotation() => $_has(3);
  @$pb.TagNumber(6)
  void clearDenotation() => $_clearField(6);

  /// Type of the sparse tensor
  @$pb.TagNumber(8)
  TypeProto_SparseTensor get sparseTensorType => $_getN(4);
  @$pb.TagNumber(8)
  set sparseTensorType(TypeProto_SparseTensor value) => $_setField(8, value);
  @$pb.TagNumber(8)
  $core.bool hasSparseTensorType() => $_has(4);
  @$pb.TagNumber(8)
  void clearSparseTensorType() => $_clearField(8);
  @$pb.TagNumber(8)
  TypeProto_SparseTensor ensureSparseTensorType() => $_ensure(4);

  /// The type of an optional.
  @$pb.TagNumber(9)
  TypeProto_Optional get optionalType => $_getN(5);
  @$pb.TagNumber(9)
  set optionalType(TypeProto_Optional value) => $_setField(9, value);
  @$pb.TagNumber(9)
  $core.bool hasOptionalType() => $_has(5);
  @$pb.TagNumber(9)
  void clearOptionalType() => $_clearField(9);
  @$pb.TagNumber(9)
  TypeProto_Optional ensureOptionalType() => $_ensure(5);
}

/// Operator Sets
///
/// OperatorSets are uniquely identified by a (domain, opset_version) pair.
class OperatorSetIdProto extends $pb.GeneratedMessage {
  factory OperatorSetIdProto({
    $core.String? domain,
    $fixnum.Int64? version,
  }) {
    final result = create();
    if (domain != null) result.domain = domain;
    if (version != null) result.version = version;
    return result;
  }

  OperatorSetIdProto._();

  factory OperatorSetIdProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory OperatorSetIdProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'OperatorSetIdProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'domain')
    ..aInt64(2, _omitFieldNames ? '' : 'version')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  OperatorSetIdProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  OperatorSetIdProto copyWith(void Function(OperatorSetIdProto) updates) =>
      super.copyWith((message) => updates(message as OperatorSetIdProto))
          as OperatorSetIdProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static OperatorSetIdProto create() => OperatorSetIdProto._();
  @$core.override
  OperatorSetIdProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static OperatorSetIdProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<OperatorSetIdProto>(create);
  static OperatorSetIdProto? _defaultInstance;

  /// The domain of the operator set being identified.
  /// The empty string ("") or absence of this field implies the operator
  /// set that is defined as part of the ONNX specification.
  /// This field MUST be present in this version of the IR when referring to any other operator set.
  @$pb.TagNumber(1)
  $core.String get domain => $_getSZ(0);
  @$pb.TagNumber(1)
  set domain($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasDomain() => $_has(0);
  @$pb.TagNumber(1)
  void clearDomain() => $_clearField(1);

  /// The version of the operator set being identified.
  /// This field MUST be present in this version of the IR.
  @$pb.TagNumber(2)
  $fixnum.Int64 get version => $_getI64(1);
  @$pb.TagNumber(2)
  set version($fixnum.Int64 value) => $_setInt64(1, value);
  @$pb.TagNumber(2)
  $core.bool hasVersion() => $_has(1);
  @$pb.TagNumber(2)
  void clearVersion() => $_clearField(2);
}

class FunctionProto extends $pb.GeneratedMessage {
  factory FunctionProto({
    $core.String? name,
    $core.Iterable<$core.String>? input,
    $core.Iterable<$core.String>? output,
    $core.Iterable<$core.String>? attribute,
    $core.Iterable<NodeProto>? node,
    $core.String? docString,
    $core.Iterable<OperatorSetIdProto>? opsetImport,
    $core.String? domain,
    $core.Iterable<AttributeProto>? attributeProto,
    $core.Iterable<ValueInfoProto>? valueInfo,
    $core.String? overload,
    $core.Iterable<StringStringEntryProto>? metadataProps,
  }) {
    final result = create();
    if (name != null) result.name = name;
    if (input != null) result.input.addAll(input);
    if (output != null) result.output.addAll(output);
    if (attribute != null) result.attribute.addAll(attribute);
    if (node != null) result.node.addAll(node);
    if (docString != null) result.docString = docString;
    if (opsetImport != null) result.opsetImport.addAll(opsetImport);
    if (domain != null) result.domain = domain;
    if (attributeProto != null) result.attributeProto.addAll(attributeProto);
    if (valueInfo != null) result.valueInfo.addAll(valueInfo);
    if (overload != null) result.overload = overload;
    if (metadataProps != null) result.metadataProps.addAll(metadataProps);
    return result;
  }

  FunctionProto._();

  factory FunctionProto.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory FunctionProto.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'FunctionProto',
      package: const $pb.PackageName(_omitMessageNames ? '' : 'onnx'),
      createEmptyInstance: create)
    ..aOS(1, _omitFieldNames ? '' : 'name')
    ..pPS(4, _omitFieldNames ? '' : 'input')
    ..pPS(5, _omitFieldNames ? '' : 'output')
    ..pPS(6, _omitFieldNames ? '' : 'attribute')
    ..pPM<NodeProto>(7, _omitFieldNames ? '' : 'node',
        subBuilder: NodeProto.create)
    ..aOS(8, _omitFieldNames ? '' : 'docString')
    ..pPM<OperatorSetIdProto>(9, _omitFieldNames ? '' : 'opsetImport',
        subBuilder: OperatorSetIdProto.create)
    ..aOS(10, _omitFieldNames ? '' : 'domain')
    ..pPM<AttributeProto>(11, _omitFieldNames ? '' : 'attributeProto',
        subBuilder: AttributeProto.create)
    ..pPM<ValueInfoProto>(12, _omitFieldNames ? '' : 'valueInfo',
        subBuilder: ValueInfoProto.create)
    ..aOS(13, _omitFieldNames ? '' : 'overload')
    ..pPM<StringStringEntryProto>(14, _omitFieldNames ? '' : 'metadataProps',
        subBuilder: StringStringEntryProto.create)
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  FunctionProto clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  FunctionProto copyWith(void Function(FunctionProto) updates) =>
      super.copyWith((message) => updates(message as FunctionProto))
          as FunctionProto;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static FunctionProto create() => FunctionProto._();
  @$core.override
  FunctionProto createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static FunctionProto getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<FunctionProto>(create);
  static FunctionProto? _defaultInstance;

  /// The name of the function, similar to op_type in NodeProto.
  /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
  @$pb.TagNumber(1)
  $core.String get name => $_getSZ(0);
  @$pb.TagNumber(1)
  set name($core.String value) => $_setString(0, value);
  @$pb.TagNumber(1)
  $core.bool hasName() => $_has(0);
  @$pb.TagNumber(1)
  void clearName() => $_clearField(1);

  /// The inputs and outputs of the function.
  @$pb.TagNumber(4)
  $pb.PbList<$core.String> get input => $_getList(1);

  @$pb.TagNumber(5)
  $pb.PbList<$core.String> get output => $_getList(2);

  /// The attribute parameters of the function.
  /// It is for function parameters without default values.
  @$pb.TagNumber(6)
  $pb.PbList<$core.String> get attribute => $_getList(3);

  /// The nodes in the function.
  @$pb.TagNumber(7)
  $pb.PbList<NodeProto> get node => $_getList(4);

  /// A human-readable documentation for this function. Markdown is allowed.
  @$pb.TagNumber(8)
  $core.String get docString => $_getSZ(5);
  @$pb.TagNumber(8)
  set docString($core.String value) => $_setString(5, value);
  @$pb.TagNumber(8)
  $core.bool hasDocString() => $_has(5);
  @$pb.TagNumber(8)
  void clearDocString() => $_clearField(8);

  @$pb.TagNumber(9)
  $pb.PbList<OperatorSetIdProto> get opsetImport => $_getList(6);

  /// The domain which this function belongs to.
  /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
  @$pb.TagNumber(10)
  $core.String get domain => $_getSZ(7);
  @$pb.TagNumber(10)
  set domain($core.String value) => $_setString(7, value);
  @$pb.TagNumber(10)
  $core.bool hasDomain() => $_has(7);
  @$pb.TagNumber(10)
  void clearDomain() => $_clearField(10);

  /// The attribute protos of the function.
  /// It is for function attributes with default values.
  /// A function attribute shall be represented either as
  /// a string attribute or an AttributeProto, not both.
  @$pb.TagNumber(11)
  $pb.PbList<AttributeProto> get attributeProto => $_getList(8);

  /// Information for the values in the function. The ValueInfoProto.name's
  /// must be distinct and refer to names in the function (including inputs,
  /// outputs, and intermediate values). It is optional for a value to appear
  /// in value_info list.
  @$pb.TagNumber(12)
  $pb.PbList<ValueInfoProto> get valueInfo => $_getList(9);

  /// The overload identifier of the function.
  /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
  @$pb.TagNumber(13)
  $core.String get overload => $_getSZ(10);
  @$pb.TagNumber(13)
  set overload($core.String value) => $_setString(10, value);
  @$pb.TagNumber(13)
  $core.bool hasOverload() => $_has(10);
  @$pb.TagNumber(13)
  void clearOverload() => $_clearField(13);

  /// Named metadata values; keys should be distinct.
  @$pb.TagNumber(14)
  $pb.PbList<StringStringEntryProto> get metadataProps => $_getList(11);
}

const $core.bool _omitFieldNames =
    $core.bool.fromEnvironment('protobuf.omit_field_names');
const $core.bool _omitMessageNames =
    $core.bool.fromEnvironment('protobuf.omit_message_names');
