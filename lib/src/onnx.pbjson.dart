// This is a generated file - do not edit.
//
// Generated from onnx.proto3.

// @dart = 3.3

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names
// ignore_for_file: curly_braces_in_flow_control_structures
// ignore_for_file: deprecated_member_use_from_same_package, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_relative_imports
// ignore_for_file: unused_import

import 'dart:convert' as $convert;
import 'dart:core' as $core;
import 'dart:typed_data' as $typed_data;

@$core.Deprecated('Use versionDescriptor instead')
const Version$json = {
  '1': 'Version',
  '2': [
    {'1': '_START_VERSION', '2': 0},
    {'1': 'IR_VERSION_2017_10_10', '2': 1},
    {'1': 'IR_VERSION_2017_10_30', '2': 2},
    {'1': 'IR_VERSION_2017_11_3', '2': 3},
    {'1': 'IR_VERSION_2019_1_22', '2': 4},
    {'1': 'IR_VERSION_2019_3_18', '2': 5},
    {'1': 'IR_VERSION_2019_9_19', '2': 6},
    {'1': 'IR_VERSION_2020_5_8', '2': 7},
    {'1': 'IR_VERSION_2021_7_30', '2': 8},
    {'1': 'IR_VERSION_2023_5_5', '2': 9},
    {'1': 'IR_VERSION_2024_3_25', '2': 10},
    {'1': 'IR_VERSION_2025_05_12', '2': 11},
    {'1': 'IR_VERSION_2025_08_26', '2': 12},
    {'1': 'IR_VERSION', '2': 13},
  ],
};

/// Descriptor for `Version`. Decode as a `google.protobuf.EnumDescriptorProto`.
final $typed_data.Uint8List versionDescriptor = $convert.base64Decode(
    'CgdWZXJzaW9uEhIKDl9TVEFSVF9WRVJTSU9OEAASGQoVSVJfVkVSU0lPTl8yMDE3XzEwXzEwEA'
    'ESGQoVSVJfVkVSU0lPTl8yMDE3XzEwXzMwEAISGAoUSVJfVkVSU0lPTl8yMDE3XzExXzMQAxIY'
    'ChRJUl9WRVJTSU9OXzIwMTlfMV8yMhAEEhgKFElSX1ZFUlNJT05fMjAxOV8zXzE4EAUSGAoUSV'
    'JfVkVSU0lPTl8yMDE5XzlfMTkQBhIXChNJUl9WRVJTSU9OXzIwMjBfNV84EAcSGAoUSVJfVkVS'
    'U0lPTl8yMDIxXzdfMzAQCBIXChNJUl9WRVJTSU9OXzIwMjNfNV81EAkSGAoUSVJfVkVSU0lPTl'
    '8yMDI0XzNfMjUQChIZChVJUl9WRVJTSU9OXzIwMjVfMDVfMTIQCxIZChVJUl9WRVJTSU9OXzIw'
    'MjVfMDhfMjYQDBIOCgpJUl9WRVJTSU9OEA0=');

@$core.Deprecated('Use operatorStatusDescriptor instead')
const OperatorStatus$json = {
  '1': 'OperatorStatus',
  '2': [
    {'1': 'EXPERIMENTAL', '2': 0},
    {'1': 'STABLE', '2': 1},
  ],
};

/// Descriptor for `OperatorStatus`. Decode as a `google.protobuf.EnumDescriptorProto`.
final $typed_data.Uint8List operatorStatusDescriptor = $convert.base64Decode(
    'Cg5PcGVyYXRvclN0YXR1cxIQCgxFWFBFUklNRU5UQUwQABIKCgZTVEFCTEUQAQ==');

@$core.Deprecated('Use attributeProtoDescriptor instead')
const AttributeProto$json = {
  '1': 'AttributeProto',
  '2': [
    {'1': 'name', '3': 1, '4': 1, '5': 9, '10': 'name'},
    {'1': 'ref_attr_name', '3': 21, '4': 1, '5': 9, '10': 'refAttrName'},
    {'1': 'doc_string', '3': 13, '4': 1, '5': 9, '10': 'docString'},
    {
      '1': 'type',
      '3': 20,
      '4': 1,
      '5': 14,
      '6': '.onnx.AttributeProto.AttributeType',
      '10': 'type'
    },
    {'1': 'f', '3': 2, '4': 1, '5': 2, '10': 'f'},
    {'1': 'i', '3': 3, '4': 1, '5': 3, '10': 'i'},
    {'1': 's', '3': 4, '4': 1, '5': 12, '10': 's'},
    {'1': 't', '3': 5, '4': 1, '5': 11, '6': '.onnx.TensorProto', '10': 't'},
    {'1': 'g', '3': 6, '4': 1, '5': 11, '6': '.onnx.GraphProto', '10': 'g'},
    {
      '1': 'sparse_tensor',
      '3': 22,
      '4': 1,
      '5': 11,
      '6': '.onnx.SparseTensorProto',
      '10': 'sparseTensor'
    },
    {'1': 'tp', '3': 14, '4': 1, '5': 11, '6': '.onnx.TypeProto', '10': 'tp'},
    {'1': 'floats', '3': 7, '4': 3, '5': 2, '10': 'floats'},
    {'1': 'ints', '3': 8, '4': 3, '5': 3, '10': 'ints'},
    {'1': 'strings', '3': 9, '4': 3, '5': 12, '10': 'strings'},
    {
      '1': 'tensors',
      '3': 10,
      '4': 3,
      '5': 11,
      '6': '.onnx.TensorProto',
      '10': 'tensors'
    },
    {
      '1': 'graphs',
      '3': 11,
      '4': 3,
      '5': 11,
      '6': '.onnx.GraphProto',
      '10': 'graphs'
    },
    {
      '1': 'sparse_tensors',
      '3': 23,
      '4': 3,
      '5': 11,
      '6': '.onnx.SparseTensorProto',
      '10': 'sparseTensors'
    },
    {
      '1': 'type_protos',
      '3': 15,
      '4': 3,
      '5': 11,
      '6': '.onnx.TypeProto',
      '10': 'typeProtos'
    },
  ],
  '4': [AttributeProto_AttributeType$json],
  '9': [
    {'1': 12, '2': 13},
    {'1': 16, '2': 20},
  ],
  '10': ['v'],
};

@$core.Deprecated('Use attributeProtoDescriptor instead')
const AttributeProto_AttributeType$json = {
  '1': 'AttributeType',
  '2': [
    {'1': 'UNDEFINED', '2': 0},
    {'1': 'FLOAT', '2': 1},
    {'1': 'INT', '2': 2},
    {'1': 'STRING', '2': 3},
    {'1': 'TENSOR', '2': 4},
    {'1': 'GRAPH', '2': 5},
    {'1': 'SPARSE_TENSOR', '2': 11},
    {'1': 'TYPE_PROTO', '2': 13},
    {'1': 'FLOATS', '2': 6},
    {'1': 'INTS', '2': 7},
    {'1': 'STRINGS', '2': 8},
    {'1': 'TENSORS', '2': 9},
    {'1': 'GRAPHS', '2': 10},
    {'1': 'SPARSE_TENSORS', '2': 12},
    {'1': 'TYPE_PROTOS', '2': 14},
  ],
};

/// Descriptor for `AttributeProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List attributeProtoDescriptor = $convert.base64Decode(
    'Cg5BdHRyaWJ1dGVQcm90bxISCgRuYW1lGAEgASgJUgRuYW1lEiIKDXJlZl9hdHRyX25hbWUYFS'
    'ABKAlSC3JlZkF0dHJOYW1lEh0KCmRvY19zdHJpbmcYDSABKAlSCWRvY1N0cmluZxI2CgR0eXBl'
    'GBQgASgOMiIub25ueC5BdHRyaWJ1dGVQcm90by5BdHRyaWJ1dGVUeXBlUgR0eXBlEgwKAWYYAi'
    'ABKAJSAWYSDAoBaRgDIAEoA1IBaRIMCgFzGAQgASgMUgFzEh8KAXQYBSABKAsyES5vbm54LlRl'
    'bnNvclByb3RvUgF0Eh4KAWcYBiABKAsyEC5vbm54LkdyYXBoUHJvdG9SAWcSPAoNc3BhcnNlX3'
    'RlbnNvchgWIAEoCzIXLm9ubnguU3BhcnNlVGVuc29yUHJvdG9SDHNwYXJzZVRlbnNvchIfCgJ0'
    'cBgOIAEoCzIPLm9ubnguVHlwZVByb3RvUgJ0cBIWCgZmbG9hdHMYByADKAJSBmZsb2F0cxISCg'
    'RpbnRzGAggAygDUgRpbnRzEhgKB3N0cmluZ3MYCSADKAxSB3N0cmluZ3MSKwoHdGVuc29ycxgK'
    'IAMoCzIRLm9ubnguVGVuc29yUHJvdG9SB3RlbnNvcnMSKAoGZ3JhcGhzGAsgAygLMhAub25ueC'
    '5HcmFwaFByb3RvUgZncmFwaHMSPgoOc3BhcnNlX3RlbnNvcnMYFyADKAsyFy5vbm54LlNwYXJz'
    'ZVRlbnNvclByb3RvUg1zcGFyc2VUZW5zb3JzEjAKC3R5cGVfcHJvdG9zGA8gAygLMg8ub25ueC'
    '5UeXBlUHJvdG9SCnR5cGVQcm90b3Mi2QEKDUF0dHJpYnV0ZVR5cGUSDQoJVU5ERUZJTkVEEAAS'
    'CQoFRkxPQVQQARIHCgNJTlQQAhIKCgZTVFJJTkcQAxIKCgZURU5TT1IQBBIJCgVHUkFQSBAFEh'
    'EKDVNQQVJTRV9URU5TT1IQCxIOCgpUWVBFX1BST1RPEA0SCgoGRkxPQVRTEAYSCAoESU5UUxAH'
    'EgsKB1NUUklOR1MQCBILCgdURU5TT1JTEAkSCgoGR1JBUEhTEAoSEgoOU1BBUlNFX1RFTlNPUl'
    'MQDBIPCgtUWVBFX1BST1RPUxAOSgQIDBANSgQIEBAUUgF2');

@$core.Deprecated('Use valueInfoProtoDescriptor instead')
const ValueInfoProto$json = {
  '1': 'ValueInfoProto',
  '2': [
    {'1': 'name', '3': 1, '4': 1, '5': 9, '10': 'name'},
    {
      '1': 'type',
      '3': 2,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto',
      '10': 'type'
    },
    {'1': 'doc_string', '3': 3, '4': 1, '5': 9, '10': 'docString'},
    {
      '1': 'metadata_props',
      '3': 4,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'metadataProps'
    },
  ],
};

/// Descriptor for `ValueInfoProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List valueInfoProtoDescriptor = $convert.base64Decode(
    'Cg5WYWx1ZUluZm9Qcm90bxISCgRuYW1lGAEgASgJUgRuYW1lEiMKBHR5cGUYAiABKAsyDy5vbm'
    '54LlR5cGVQcm90b1IEdHlwZRIdCgpkb2Nfc3RyaW5nGAMgASgJUglkb2NTdHJpbmcSQwoObWV0'
    'YWRhdGFfcHJvcHMYBCADKAsyHC5vbm54LlN0cmluZ1N0cmluZ0VudHJ5UHJvdG9SDW1ldGFkYX'
    'RhUHJvcHM=');

@$core.Deprecated('Use nodeProtoDescriptor instead')
const NodeProto$json = {
  '1': 'NodeProto',
  '2': [
    {'1': 'input', '3': 1, '4': 3, '5': 9, '10': 'input'},
    {'1': 'output', '3': 2, '4': 3, '5': 9, '10': 'output'},
    {'1': 'name', '3': 3, '4': 1, '5': 9, '10': 'name'},
    {'1': 'op_type', '3': 4, '4': 1, '5': 9, '10': 'opType'},
    {'1': 'domain', '3': 7, '4': 1, '5': 9, '10': 'domain'},
    {'1': 'overload', '3': 8, '4': 1, '5': 9, '10': 'overload'},
    {
      '1': 'attribute',
      '3': 5,
      '4': 3,
      '5': 11,
      '6': '.onnx.AttributeProto',
      '10': 'attribute'
    },
    {'1': 'doc_string', '3': 6, '4': 1, '5': 9, '10': 'docString'},
    {
      '1': 'metadata_props',
      '3': 9,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'metadataProps'
    },
    {
      '1': 'device_configurations',
      '3': 10,
      '4': 3,
      '5': 11,
      '6': '.onnx.NodeDeviceConfigurationProto',
      '10': 'deviceConfigurations'
    },
  ],
};

/// Descriptor for `NodeProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List nodeProtoDescriptor = $convert.base64Decode(
    'CglOb2RlUHJvdG8SFAoFaW5wdXQYASADKAlSBWlucHV0EhYKBm91dHB1dBgCIAMoCVIGb3V0cH'
    'V0EhIKBG5hbWUYAyABKAlSBG5hbWUSFwoHb3BfdHlwZRgEIAEoCVIGb3BUeXBlEhYKBmRvbWFp'
    'bhgHIAEoCVIGZG9tYWluEhoKCG92ZXJsb2FkGAggASgJUghvdmVybG9hZBIyCglhdHRyaWJ1dG'
    'UYBSADKAsyFC5vbm54LkF0dHJpYnV0ZVByb3RvUglhdHRyaWJ1dGUSHQoKZG9jX3N0cmluZxgG'
    'IAEoCVIJZG9jU3RyaW5nEkMKDm1ldGFkYXRhX3Byb3BzGAkgAygLMhwub25ueC5TdHJpbmdTdH'
    'JpbmdFbnRyeVByb3RvUg1tZXRhZGF0YVByb3BzElcKFWRldmljZV9jb25maWd1cmF0aW9ucxgK'
    'IAMoCzIiLm9ubnguTm9kZURldmljZUNvbmZpZ3VyYXRpb25Qcm90b1IUZGV2aWNlQ29uZmlndX'
    'JhdGlvbnM=');

@$core.Deprecated('Use intIntListEntryProtoDescriptor instead')
const IntIntListEntryProto$json = {
  '1': 'IntIntListEntryProto',
  '2': [
    {'1': 'key', '3': 1, '4': 1, '5': 3, '10': 'key'},
    {'1': 'value', '3': 2, '4': 3, '5': 3, '10': 'value'},
  ],
};

/// Descriptor for `IntIntListEntryProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List intIntListEntryProtoDescriptor = $convert.base64Decode(
    'ChRJbnRJbnRMaXN0RW50cnlQcm90bxIQCgNrZXkYASABKANSA2tleRIUCgV2YWx1ZRgCIAMoA1'
    'IFdmFsdWU=');

@$core.Deprecated('Use nodeDeviceConfigurationProtoDescriptor instead')
const NodeDeviceConfigurationProto$json = {
  '1': 'NodeDeviceConfigurationProto',
  '2': [
    {'1': 'configuration_id', '3': 1, '4': 1, '5': 9, '10': 'configurationId'},
    {
      '1': 'sharding_spec',
      '3': 2,
      '4': 3,
      '5': 11,
      '6': '.onnx.ShardingSpecProto',
      '10': 'shardingSpec'
    },
    {'1': 'pipeline_stage', '3': 3, '4': 1, '5': 5, '10': 'pipelineStage'},
  ],
};

/// Descriptor for `NodeDeviceConfigurationProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List nodeDeviceConfigurationProtoDescriptor = $convert.base64Decode(
    'ChxOb2RlRGV2aWNlQ29uZmlndXJhdGlvblByb3RvEikKEGNvbmZpZ3VyYXRpb25faWQYASABKA'
    'lSD2NvbmZpZ3VyYXRpb25JZBI8Cg1zaGFyZGluZ19zcGVjGAIgAygLMhcub25ueC5TaGFyZGlu'
    'Z1NwZWNQcm90b1IMc2hhcmRpbmdTcGVjEiUKDnBpcGVsaW5lX3N0YWdlGAMgASgFUg1waXBlbG'
    'luZVN0YWdl');

@$core.Deprecated('Use shardingSpecProtoDescriptor instead')
const ShardingSpecProto$json = {
  '1': 'ShardingSpecProto',
  '2': [
    {'1': 'tensor_name', '3': 1, '4': 1, '5': 9, '10': 'tensorName'},
    {'1': 'device', '3': 2, '4': 3, '5': 3, '10': 'device'},
    {
      '1': 'index_to_device_group_map',
      '3': 3,
      '4': 3,
      '5': 11,
      '6': '.onnx.IntIntListEntryProto',
      '10': 'indexToDeviceGroupMap'
    },
    {
      '1': 'sharded_dim',
      '3': 4,
      '4': 3,
      '5': 11,
      '6': '.onnx.ShardedDimProto',
      '10': 'shardedDim'
    },
  ],
};

/// Descriptor for `ShardingSpecProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List shardingSpecProtoDescriptor = $convert.base64Decode(
    'ChFTaGFyZGluZ1NwZWNQcm90bxIfCgt0ZW5zb3JfbmFtZRgBIAEoCVIKdGVuc29yTmFtZRIWCg'
    'ZkZXZpY2UYAiADKANSBmRldmljZRJUChlpbmRleF90b19kZXZpY2VfZ3JvdXBfbWFwGAMgAygL'
    'Mhoub25ueC5JbnRJbnRMaXN0RW50cnlQcm90b1IVaW5kZXhUb0RldmljZUdyb3VwTWFwEjYKC3'
    'NoYXJkZWRfZGltGAQgAygLMhUub25ueC5TaGFyZGVkRGltUHJvdG9SCnNoYXJkZWREaW0=');

@$core.Deprecated('Use shardedDimProtoDescriptor instead')
const ShardedDimProto$json = {
  '1': 'ShardedDimProto',
  '2': [
    {'1': 'axis', '3': 1, '4': 1, '5': 3, '10': 'axis'},
    {
      '1': 'simple_sharding',
      '3': 2,
      '4': 3,
      '5': 11,
      '6': '.onnx.SimpleShardedDimProto',
      '10': 'simpleSharding'
    },
  ],
};

/// Descriptor for `ShardedDimProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List shardedDimProtoDescriptor = $convert.base64Decode(
    'Cg9TaGFyZGVkRGltUHJvdG8SEgoEYXhpcxgBIAEoA1IEYXhpcxJECg9zaW1wbGVfc2hhcmRpbm'
    'cYAiADKAsyGy5vbm54LlNpbXBsZVNoYXJkZWREaW1Qcm90b1IOc2ltcGxlU2hhcmRpbmc=');

@$core.Deprecated('Use simpleShardedDimProtoDescriptor instead')
const SimpleShardedDimProto$json = {
  '1': 'SimpleShardedDimProto',
  '2': [
    {'1': 'dim_value', '3': 1, '4': 1, '5': 3, '9': 0, '10': 'dimValue'},
    {'1': 'dim_param', '3': 2, '4': 1, '5': 9, '9': 0, '10': 'dimParam'},
    {'1': 'num_shards', '3': 3, '4': 1, '5': 3, '10': 'numShards'},
  ],
  '8': [
    {'1': 'dim'},
  ],
};

/// Descriptor for `SimpleShardedDimProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List simpleShardedDimProtoDescriptor = $convert.base64Decode(
    'ChVTaW1wbGVTaGFyZGVkRGltUHJvdG8SHQoJZGltX3ZhbHVlGAEgASgDSABSCGRpbVZhbHVlEh'
    '0KCWRpbV9wYXJhbRgCIAEoCUgAUghkaW1QYXJhbRIdCgpudW1fc2hhcmRzGAMgASgDUgludW1T'
    'aGFyZHNCBQoDZGlt');

@$core.Deprecated('Use trainingInfoProtoDescriptor instead')
const TrainingInfoProto$json = {
  '1': 'TrainingInfoProto',
  '2': [
    {
      '1': 'initialization',
      '3': 1,
      '4': 1,
      '5': 11,
      '6': '.onnx.GraphProto',
      '10': 'initialization'
    },
    {
      '1': 'algorithm',
      '3': 2,
      '4': 1,
      '5': 11,
      '6': '.onnx.GraphProto',
      '10': 'algorithm'
    },
    {
      '1': 'initialization_binding',
      '3': 3,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'initializationBinding'
    },
    {
      '1': 'update_binding',
      '3': 4,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'updateBinding'
    },
  ],
};

/// Descriptor for `TrainingInfoProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List trainingInfoProtoDescriptor = $convert.base64Decode(
    'ChFUcmFpbmluZ0luZm9Qcm90bxI4Cg5pbml0aWFsaXphdGlvbhgBIAEoCzIQLm9ubnguR3JhcG'
    'hQcm90b1IOaW5pdGlhbGl6YXRpb24SLgoJYWxnb3JpdGhtGAIgASgLMhAub25ueC5HcmFwaFBy'
    'b3RvUglhbGdvcml0aG0SUwoWaW5pdGlhbGl6YXRpb25fYmluZGluZxgDIAMoCzIcLm9ubnguU3'
    'RyaW5nU3RyaW5nRW50cnlQcm90b1IVaW5pdGlhbGl6YXRpb25CaW5kaW5nEkMKDnVwZGF0ZV9i'
    'aW5kaW5nGAQgAygLMhwub25ueC5TdHJpbmdTdHJpbmdFbnRyeVByb3RvUg11cGRhdGVCaW5kaW'
    '5n');

@$core.Deprecated('Use modelProtoDescriptor instead')
const ModelProto$json = {
  '1': 'ModelProto',
  '2': [
    {'1': 'ir_version', '3': 1, '4': 1, '5': 3, '10': 'irVersion'},
    {
      '1': 'opset_import',
      '3': 8,
      '4': 3,
      '5': 11,
      '6': '.onnx.OperatorSetIdProto',
      '10': 'opsetImport'
    },
    {'1': 'producer_name', '3': 2, '4': 1, '5': 9, '10': 'producerName'},
    {'1': 'producer_version', '3': 3, '4': 1, '5': 9, '10': 'producerVersion'},
    {'1': 'domain', '3': 4, '4': 1, '5': 9, '10': 'domain'},
    {'1': 'model_version', '3': 5, '4': 1, '5': 3, '10': 'modelVersion'},
    {'1': 'doc_string', '3': 6, '4': 1, '5': 9, '10': 'docString'},
    {
      '1': 'graph',
      '3': 7,
      '4': 1,
      '5': 11,
      '6': '.onnx.GraphProto',
      '10': 'graph'
    },
    {
      '1': 'metadata_props',
      '3': 14,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'metadataProps'
    },
    {
      '1': 'training_info',
      '3': 20,
      '4': 3,
      '5': 11,
      '6': '.onnx.TrainingInfoProto',
      '10': 'trainingInfo'
    },
    {
      '1': 'functions',
      '3': 25,
      '4': 3,
      '5': 11,
      '6': '.onnx.FunctionProto',
      '10': 'functions'
    },
    {
      '1': 'configuration',
      '3': 26,
      '4': 3,
      '5': 11,
      '6': '.onnx.DeviceConfigurationProto',
      '10': 'configuration'
    },
  ],
};

/// Descriptor for `ModelProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List modelProtoDescriptor = $convert.base64Decode(
    'CgpNb2RlbFByb3RvEh0KCmlyX3ZlcnNpb24YASABKANSCWlyVmVyc2lvbhI7CgxvcHNldF9pbX'
    'BvcnQYCCADKAsyGC5vbm54Lk9wZXJhdG9yU2V0SWRQcm90b1ILb3BzZXRJbXBvcnQSIwoNcHJv'
    'ZHVjZXJfbmFtZRgCIAEoCVIMcHJvZHVjZXJOYW1lEikKEHByb2R1Y2VyX3ZlcnNpb24YAyABKA'
    'lSD3Byb2R1Y2VyVmVyc2lvbhIWCgZkb21haW4YBCABKAlSBmRvbWFpbhIjCg1tb2RlbF92ZXJz'
    'aW9uGAUgASgDUgxtb2RlbFZlcnNpb24SHQoKZG9jX3N0cmluZxgGIAEoCVIJZG9jU3RyaW5nEi'
    'YKBWdyYXBoGAcgASgLMhAub25ueC5HcmFwaFByb3RvUgVncmFwaBJDCg5tZXRhZGF0YV9wcm9w'
    'cxgOIAMoCzIcLm9ubnguU3RyaW5nU3RyaW5nRW50cnlQcm90b1INbWV0YWRhdGFQcm9wcxI8Cg'
    '10cmFpbmluZ19pbmZvGBQgAygLMhcub25ueC5UcmFpbmluZ0luZm9Qcm90b1IMdHJhaW5pbmdJ'
    'bmZvEjEKCWZ1bmN0aW9ucxgZIAMoCzITLm9ubnguRnVuY3Rpb25Qcm90b1IJZnVuY3Rpb25zEk'
    'QKDWNvbmZpZ3VyYXRpb24YGiADKAsyHi5vbm54LkRldmljZUNvbmZpZ3VyYXRpb25Qcm90b1IN'
    'Y29uZmlndXJhdGlvbg==');

@$core.Deprecated('Use deviceConfigurationProtoDescriptor instead')
const DeviceConfigurationProto$json = {
  '1': 'DeviceConfigurationProto',
  '2': [
    {'1': 'name', '3': 1, '4': 1, '5': 9, '10': 'name'},
    {'1': 'num_devices', '3': 2, '4': 1, '5': 5, '10': 'numDevices'},
    {'1': 'device', '3': 3, '4': 3, '5': 9, '10': 'device'},
  ],
};

/// Descriptor for `DeviceConfigurationProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List deviceConfigurationProtoDescriptor =
    $convert.base64Decode(
        'ChhEZXZpY2VDb25maWd1cmF0aW9uUHJvdG8SEgoEbmFtZRgBIAEoCVIEbmFtZRIfCgtudW1fZG'
        'V2aWNlcxgCIAEoBVIKbnVtRGV2aWNlcxIWCgZkZXZpY2UYAyADKAlSBmRldmljZQ==');

@$core.Deprecated('Use stringStringEntryProtoDescriptor instead')
const StringStringEntryProto$json = {
  '1': 'StringStringEntryProto',
  '2': [
    {'1': 'key', '3': 1, '4': 1, '5': 9, '10': 'key'},
    {'1': 'value', '3': 2, '4': 1, '5': 9, '10': 'value'},
  ],
};

/// Descriptor for `StringStringEntryProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List stringStringEntryProtoDescriptor =
    $convert.base64Decode(
        'ChZTdHJpbmdTdHJpbmdFbnRyeVByb3RvEhAKA2tleRgBIAEoCVIDa2V5EhQKBXZhbHVlGAIgAS'
        'gJUgV2YWx1ZQ==');

@$core.Deprecated('Use tensorAnnotationDescriptor instead')
const TensorAnnotation$json = {
  '1': 'TensorAnnotation',
  '2': [
    {'1': 'tensor_name', '3': 1, '4': 1, '5': 9, '10': 'tensorName'},
    {
      '1': 'quant_parameter_tensor_names',
      '3': 2,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'quantParameterTensorNames'
    },
  ],
};

/// Descriptor for `TensorAnnotation`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List tensorAnnotationDescriptor = $convert.base64Decode(
    'ChBUZW5zb3JBbm5vdGF0aW9uEh8KC3RlbnNvcl9uYW1lGAEgASgJUgp0ZW5zb3JOYW1lEl0KHH'
    'F1YW50X3BhcmFtZXRlcl90ZW5zb3JfbmFtZXMYAiADKAsyHC5vbm54LlN0cmluZ1N0cmluZ0Vu'
    'dHJ5UHJvdG9SGXF1YW50UGFyYW1ldGVyVGVuc29yTmFtZXM=');

@$core.Deprecated('Use graphProtoDescriptor instead')
const GraphProto$json = {
  '1': 'GraphProto',
  '2': [
    {
      '1': 'node',
      '3': 1,
      '4': 3,
      '5': 11,
      '6': '.onnx.NodeProto',
      '10': 'node'
    },
    {'1': 'name', '3': 2, '4': 1, '5': 9, '10': 'name'},
    {
      '1': 'initializer',
      '3': 5,
      '4': 3,
      '5': 11,
      '6': '.onnx.TensorProto',
      '10': 'initializer'
    },
    {
      '1': 'sparse_initializer',
      '3': 15,
      '4': 3,
      '5': 11,
      '6': '.onnx.SparseTensorProto',
      '10': 'sparseInitializer'
    },
    {'1': 'doc_string', '3': 10, '4': 1, '5': 9, '10': 'docString'},
    {
      '1': 'input',
      '3': 11,
      '4': 3,
      '5': 11,
      '6': '.onnx.ValueInfoProto',
      '10': 'input'
    },
    {
      '1': 'output',
      '3': 12,
      '4': 3,
      '5': 11,
      '6': '.onnx.ValueInfoProto',
      '10': 'output'
    },
    {
      '1': 'value_info',
      '3': 13,
      '4': 3,
      '5': 11,
      '6': '.onnx.ValueInfoProto',
      '10': 'valueInfo'
    },
    {
      '1': 'quantization_annotation',
      '3': 14,
      '4': 3,
      '5': 11,
      '6': '.onnx.TensorAnnotation',
      '10': 'quantizationAnnotation'
    },
    {
      '1': 'metadata_props',
      '3': 16,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'metadataProps'
    },
  ],
  '9': [
    {'1': 3, '2': 4},
    {'1': 4, '2': 5},
    {'1': 6, '2': 10},
  ],
  '10': ['ir_version', 'producer_version', 'producer_tag', 'domain'],
};

/// Descriptor for `GraphProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List graphProtoDescriptor = $convert.base64Decode(
    'CgpHcmFwaFByb3RvEiMKBG5vZGUYASADKAsyDy5vbm54Lk5vZGVQcm90b1IEbm9kZRISCgRuYW'
    '1lGAIgASgJUgRuYW1lEjMKC2luaXRpYWxpemVyGAUgAygLMhEub25ueC5UZW5zb3JQcm90b1IL'
    'aW5pdGlhbGl6ZXISRgoSc3BhcnNlX2luaXRpYWxpemVyGA8gAygLMhcub25ueC5TcGFyc2VUZW'
    '5zb3JQcm90b1IRc3BhcnNlSW5pdGlhbGl6ZXISHQoKZG9jX3N0cmluZxgKIAEoCVIJZG9jU3Ry'
    'aW5nEioKBWlucHV0GAsgAygLMhQub25ueC5WYWx1ZUluZm9Qcm90b1IFaW5wdXQSLAoGb3V0cH'
    'V0GAwgAygLMhQub25ueC5WYWx1ZUluZm9Qcm90b1IGb3V0cHV0EjMKCnZhbHVlX2luZm8YDSAD'
    'KAsyFC5vbm54LlZhbHVlSW5mb1Byb3RvUgl2YWx1ZUluZm8STwoXcXVhbnRpemF0aW9uX2Fubm'
    '90YXRpb24YDiADKAsyFi5vbm54LlRlbnNvckFubm90YXRpb25SFnF1YW50aXphdGlvbkFubm90'
    'YXRpb24SQwoObWV0YWRhdGFfcHJvcHMYECADKAsyHC5vbm54LlN0cmluZ1N0cmluZ0VudHJ5UH'
    'JvdG9SDW1ldGFkYXRhUHJvcHNKBAgDEARKBAgEEAVKBAgGEApSCmlyX3ZlcnNpb25SEHByb2R1'
    'Y2VyX3ZlcnNpb25SDHByb2R1Y2VyX3RhZ1IGZG9tYWlu');

@$core.Deprecated('Use tensorProtoDescriptor instead')
const TensorProto$json = {
  '1': 'TensorProto',
  '2': [
    {'1': 'dims', '3': 1, '4': 3, '5': 3, '10': 'dims'},
    {'1': 'data_type', '3': 2, '4': 1, '5': 5, '10': 'dataType'},
    {
      '1': 'segment',
      '3': 3,
      '4': 1,
      '5': 11,
      '6': '.onnx.TensorProto.Segment',
      '10': 'segment'
    },
    {
      '1': 'float_data',
      '3': 4,
      '4': 3,
      '5': 2,
      '8': {'2': true},
      '10': 'floatData',
    },
    {
      '1': 'int32_data',
      '3': 5,
      '4': 3,
      '5': 5,
      '8': {'2': true},
      '10': 'int32Data',
    },
    {'1': 'string_data', '3': 6, '4': 3, '5': 12, '10': 'stringData'},
    {
      '1': 'int64_data',
      '3': 7,
      '4': 3,
      '5': 3,
      '8': {'2': true},
      '10': 'int64Data',
    },
    {'1': 'name', '3': 8, '4': 1, '5': 9, '10': 'name'},
    {'1': 'doc_string', '3': 12, '4': 1, '5': 9, '10': 'docString'},
    {'1': 'raw_data', '3': 9, '4': 1, '5': 12, '10': 'rawData'},
    {
      '1': 'external_data',
      '3': 13,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'externalData'
    },
    {
      '1': 'data_location',
      '3': 14,
      '4': 1,
      '5': 14,
      '6': '.onnx.TensorProto.DataLocation',
      '10': 'dataLocation'
    },
    {
      '1': 'double_data',
      '3': 10,
      '4': 3,
      '5': 1,
      '8': {'2': true},
      '10': 'doubleData',
    },
    {
      '1': 'uint64_data',
      '3': 11,
      '4': 3,
      '5': 4,
      '8': {'2': true},
      '10': 'uint64Data',
    },
    {
      '1': 'metadata_props',
      '3': 16,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'metadataProps'
    },
  ],
  '3': [TensorProto_Segment$json],
  '4': [TensorProto_DataType$json, TensorProto_DataLocation$json],
};

@$core.Deprecated('Use tensorProtoDescriptor instead')
const TensorProto_Segment$json = {
  '1': 'Segment',
  '2': [
    {'1': 'begin', '3': 1, '4': 1, '5': 3, '10': 'begin'},
    {'1': 'end', '3': 2, '4': 1, '5': 3, '10': 'end'},
  ],
};

@$core.Deprecated('Use tensorProtoDescriptor instead')
const TensorProto_DataType$json = {
  '1': 'DataType',
  '2': [
    {'1': 'UNDEFINED', '2': 0},
    {'1': 'FLOAT', '2': 1},
    {'1': 'UINT8', '2': 2},
    {'1': 'INT8', '2': 3},
    {'1': 'UINT16', '2': 4},
    {'1': 'INT16', '2': 5},
    {'1': 'INT32', '2': 6},
    {'1': 'INT64', '2': 7},
    {'1': 'STRING', '2': 8},
    {'1': 'BOOL', '2': 9},
    {'1': 'FLOAT16', '2': 10},
    {'1': 'DOUBLE', '2': 11},
    {'1': 'UINT32', '2': 12},
    {'1': 'UINT64', '2': 13},
    {'1': 'COMPLEX64', '2': 14},
    {'1': 'COMPLEX128', '2': 15},
    {'1': 'BFLOAT16', '2': 16},
    {'1': 'FLOAT8E4M3FN', '2': 17},
    {'1': 'FLOAT8E4M3FNUZ', '2': 18},
    {'1': 'FLOAT8E5M2', '2': 19},
    {'1': 'FLOAT8E5M2FNUZ', '2': 20},
    {'1': 'UINT4', '2': 21},
    {'1': 'INT4', '2': 22},
    {'1': 'FLOAT4E2M1', '2': 23},
    {'1': 'FLOAT8E8M0', '2': 24},
    {'1': 'UINT2', '2': 25},
    {'1': 'INT2', '2': 26},
  ],
};

@$core.Deprecated('Use tensorProtoDescriptor instead')
const TensorProto_DataLocation$json = {
  '1': 'DataLocation',
  '2': [
    {'1': 'DEFAULT', '2': 0},
    {'1': 'EXTERNAL', '2': 1},
  ],
};

/// Descriptor for `TensorProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List tensorProtoDescriptor = $convert.base64Decode(
    'CgtUZW5zb3JQcm90bxISCgRkaW1zGAEgAygDUgRkaW1zEhsKCWRhdGFfdHlwZRgCIAEoBVIIZG'
    'F0YVR5cGUSMwoHc2VnbWVudBgDIAEoCzIZLm9ubnguVGVuc29yUHJvdG8uU2VnbWVudFIHc2Vn'
    'bWVudBIhCgpmbG9hdF9kYXRhGAQgAygCQgIQAVIJZmxvYXREYXRhEiEKCmludDMyX2RhdGEYBS'
    'ADKAVCAhABUglpbnQzMkRhdGESHwoLc3RyaW5nX2RhdGEYBiADKAxSCnN0cmluZ0RhdGESIQoK'
    'aW50NjRfZGF0YRgHIAMoA0ICEAFSCWludDY0RGF0YRISCgRuYW1lGAggASgJUgRuYW1lEh0KCm'
    'RvY19zdHJpbmcYDCABKAlSCWRvY1N0cmluZxIZCghyYXdfZGF0YRgJIAEoDFIHcmF3RGF0YRJB'
    'Cg1leHRlcm5hbF9kYXRhGA0gAygLMhwub25ueC5TdHJpbmdTdHJpbmdFbnRyeVByb3RvUgxleH'
    'Rlcm5hbERhdGESQwoNZGF0YV9sb2NhdGlvbhgOIAEoDjIeLm9ubnguVGVuc29yUHJvdG8uRGF0'
    'YUxvY2F0aW9uUgxkYXRhTG9jYXRpb24SIwoLZG91YmxlX2RhdGEYCiADKAFCAhABUgpkb3VibG'
    'VEYXRhEiMKC3VpbnQ2NF9kYXRhGAsgAygEQgIQAVIKdWludDY0RGF0YRJDCg5tZXRhZGF0YV9w'
    'cm9wcxgQIAMoCzIcLm9ubnguU3RyaW5nU3RyaW5nRW50cnlQcm90b1INbWV0YWRhdGFQcm9wcx'
    'oxCgdTZWdtZW50EhQKBWJlZ2luGAEgASgDUgViZWdpbhIQCgNlbmQYAiABKANSA2VuZCLuAgoI'
    'RGF0YVR5cGUSDQoJVU5ERUZJTkVEEAASCQoFRkxPQVQQARIJCgVVSU5UOBACEggKBElOVDgQAx'
    'IKCgZVSU5UMTYQBBIJCgVJTlQxNhAFEgkKBUlOVDMyEAYSCQoFSU5UNjQQBxIKCgZTVFJJTkcQ'
    'CBIICgRCT09MEAkSCwoHRkxPQVQxNhAKEgoKBkRPVUJMRRALEgoKBlVJTlQzMhAMEgoKBlVJTl'
    'Q2NBANEg0KCUNPTVBMRVg2NBAOEg4KCkNPTVBMRVgxMjgQDxIMCghCRkxPQVQxNhAQEhAKDEZM'
    'T0FUOEU0TTNGThAREhIKDkZMT0FUOEU0TTNGTlVaEBISDgoKRkxPQVQ4RTVNMhATEhIKDkZMT0'
    'FUOEU1TTJGTlVaEBQSCQoFVUlOVDQQFRIICgRJTlQ0EBYSDgoKRkxPQVQ0RTJNMRAXEg4KCkZM'
    'T0FUOEU4TTAQGBIJCgVVSU5UMhAZEggKBElOVDIQGiIpCgxEYXRhTG9jYXRpb24SCwoHREVGQV'
    'VMVBAAEgwKCEVYVEVSTkFMEAE=');

@$core.Deprecated('Use sparseTensorProtoDescriptor instead')
const SparseTensorProto$json = {
  '1': 'SparseTensorProto',
  '2': [
    {
      '1': 'values',
      '3': 1,
      '4': 1,
      '5': 11,
      '6': '.onnx.TensorProto',
      '10': 'values'
    },
    {
      '1': 'indices',
      '3': 2,
      '4': 1,
      '5': 11,
      '6': '.onnx.TensorProto',
      '10': 'indices'
    },
    {'1': 'dims', '3': 3, '4': 3, '5': 3, '10': 'dims'},
  ],
};

/// Descriptor for `SparseTensorProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List sparseTensorProtoDescriptor = $convert.base64Decode(
    'ChFTcGFyc2VUZW5zb3JQcm90bxIpCgZ2YWx1ZXMYASABKAsyES5vbm54LlRlbnNvclByb3RvUg'
    'Z2YWx1ZXMSKwoHaW5kaWNlcxgCIAEoCzIRLm9ubnguVGVuc29yUHJvdG9SB2luZGljZXMSEgoE'
    'ZGltcxgDIAMoA1IEZGltcw==');

@$core.Deprecated('Use tensorShapeProtoDescriptor instead')
const TensorShapeProto$json = {
  '1': 'TensorShapeProto',
  '2': [
    {
      '1': 'dim',
      '3': 1,
      '4': 3,
      '5': 11,
      '6': '.onnx.TensorShapeProto.Dimension',
      '10': 'dim'
    },
  ],
  '3': [TensorShapeProto_Dimension$json],
};

@$core.Deprecated('Use tensorShapeProtoDescriptor instead')
const TensorShapeProto_Dimension$json = {
  '1': 'Dimension',
  '2': [
    {'1': 'dim_value', '3': 1, '4': 1, '5': 3, '9': 0, '10': 'dimValue'},
    {'1': 'dim_param', '3': 2, '4': 1, '5': 9, '9': 0, '10': 'dimParam'},
    {'1': 'denotation', '3': 3, '4': 1, '5': 9, '10': 'denotation'},
  ],
  '8': [
    {'1': 'value'},
  ],
};

/// Descriptor for `TensorShapeProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List tensorShapeProtoDescriptor = $convert.base64Decode(
    'ChBUZW5zb3JTaGFwZVByb3RvEjIKA2RpbRgBIAMoCzIgLm9ubnguVGVuc29yU2hhcGVQcm90by'
    '5EaW1lbnNpb25SA2RpbRpyCglEaW1lbnNpb24SHQoJZGltX3ZhbHVlGAEgASgDSABSCGRpbVZh'
    'bHVlEh0KCWRpbV9wYXJhbRgCIAEoCUgAUghkaW1QYXJhbRIeCgpkZW5vdGF0aW9uGAMgASgJUg'
    'pkZW5vdGF0aW9uQgcKBXZhbHVl');

@$core.Deprecated('Use typeProtoDescriptor instead')
const TypeProto$json = {
  '1': 'TypeProto',
  '2': [
    {
      '1': 'tensor_type',
      '3': 1,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto.Tensor',
      '9': 0,
      '10': 'tensorType'
    },
    {
      '1': 'sequence_type',
      '3': 4,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto.Sequence',
      '9': 0,
      '10': 'sequenceType'
    },
    {
      '1': 'map_type',
      '3': 5,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto.Map',
      '9': 0,
      '10': 'mapType'
    },
    {
      '1': 'optional_type',
      '3': 9,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto.Optional',
      '9': 0,
      '10': 'optionalType'
    },
    {
      '1': 'sparse_tensor_type',
      '3': 8,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto.SparseTensor',
      '9': 0,
      '10': 'sparseTensorType'
    },
    {'1': 'denotation', '3': 6, '4': 1, '5': 9, '10': 'denotation'},
  ],
  '3': [
    TypeProto_Tensor$json,
    TypeProto_Sequence$json,
    TypeProto_Map$json,
    TypeProto_Optional$json,
    TypeProto_SparseTensor$json
  ],
  '8': [
    {'1': 'value'},
  ],
};

@$core.Deprecated('Use typeProtoDescriptor instead')
const TypeProto_Tensor$json = {
  '1': 'Tensor',
  '2': [
    {'1': 'elem_type', '3': 1, '4': 1, '5': 5, '10': 'elemType'},
    {
      '1': 'shape',
      '3': 2,
      '4': 1,
      '5': 11,
      '6': '.onnx.TensorShapeProto',
      '10': 'shape'
    },
  ],
};

@$core.Deprecated('Use typeProtoDescriptor instead')
const TypeProto_Sequence$json = {
  '1': 'Sequence',
  '2': [
    {
      '1': 'elem_type',
      '3': 1,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto',
      '10': 'elemType'
    },
  ],
};

@$core.Deprecated('Use typeProtoDescriptor instead')
const TypeProto_Map$json = {
  '1': 'Map',
  '2': [
    {'1': 'key_type', '3': 1, '4': 1, '5': 5, '10': 'keyType'},
    {
      '1': 'value_type',
      '3': 2,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto',
      '10': 'valueType'
    },
  ],
};

@$core.Deprecated('Use typeProtoDescriptor instead')
const TypeProto_Optional$json = {
  '1': 'Optional',
  '2': [
    {
      '1': 'elem_type',
      '3': 1,
      '4': 1,
      '5': 11,
      '6': '.onnx.TypeProto',
      '10': 'elemType'
    },
  ],
};

@$core.Deprecated('Use typeProtoDescriptor instead')
const TypeProto_SparseTensor$json = {
  '1': 'SparseTensor',
  '2': [
    {'1': 'elem_type', '3': 1, '4': 1, '5': 5, '10': 'elemType'},
    {
      '1': 'shape',
      '3': 2,
      '4': 1,
      '5': 11,
      '6': '.onnx.TensorShapeProto',
      '10': 'shape'
    },
  ],
};

/// Descriptor for `TypeProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List typeProtoDescriptor = $convert.base64Decode(
    'CglUeXBlUHJvdG8SOQoLdGVuc29yX3R5cGUYASABKAsyFi5vbm54LlR5cGVQcm90by5UZW5zb3'
    'JIAFIKdGVuc29yVHlwZRI/Cg1zZXF1ZW5jZV90eXBlGAQgASgLMhgub25ueC5UeXBlUHJvdG8u'
    'U2VxdWVuY2VIAFIMc2VxdWVuY2VUeXBlEjAKCG1hcF90eXBlGAUgASgLMhMub25ueC5UeXBlUH'
    'JvdG8uTWFwSABSB21hcFR5cGUSPwoNb3B0aW9uYWxfdHlwZRgJIAEoCzIYLm9ubnguVHlwZVBy'
    'b3RvLk9wdGlvbmFsSABSDG9wdGlvbmFsVHlwZRJMChJzcGFyc2VfdGVuc29yX3R5cGUYCCABKA'
    'syHC5vbm54LlR5cGVQcm90by5TcGFyc2VUZW5zb3JIAFIQc3BhcnNlVGVuc29yVHlwZRIeCgpk'
    'ZW5vdGF0aW9uGAYgASgJUgpkZW5vdGF0aW9uGlMKBlRlbnNvchIbCgllbGVtX3R5cGUYASABKA'
    'VSCGVsZW1UeXBlEiwKBXNoYXBlGAIgASgLMhYub25ueC5UZW5zb3JTaGFwZVByb3RvUgVzaGFw'
    'ZRo4CghTZXF1ZW5jZRIsCgllbGVtX3R5cGUYASABKAsyDy5vbm54LlR5cGVQcm90b1IIZWxlbV'
    'R5cGUaUAoDTWFwEhkKCGtleV90eXBlGAEgASgFUgdrZXlUeXBlEi4KCnZhbHVlX3R5cGUYAiAB'
    'KAsyDy5vbm54LlR5cGVQcm90b1IJdmFsdWVUeXBlGjgKCE9wdGlvbmFsEiwKCWVsZW1fdHlwZR'
    'gBIAEoCzIPLm9ubnguVHlwZVByb3RvUghlbGVtVHlwZRpZCgxTcGFyc2VUZW5zb3ISGwoJZWxl'
    'bV90eXBlGAEgASgFUghlbGVtVHlwZRIsCgVzaGFwZRgCIAEoCzIWLm9ubnguVGVuc29yU2hhcG'
    'VQcm90b1IFc2hhcGVCBwoFdmFsdWU=');

@$core.Deprecated('Use operatorSetIdProtoDescriptor instead')
const OperatorSetIdProto$json = {
  '1': 'OperatorSetIdProto',
  '2': [
    {'1': 'domain', '3': 1, '4': 1, '5': 9, '10': 'domain'},
    {'1': 'version', '3': 2, '4': 1, '5': 3, '10': 'version'},
  ],
};

/// Descriptor for `OperatorSetIdProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List operatorSetIdProtoDescriptor = $convert.base64Decode(
    'ChJPcGVyYXRvclNldElkUHJvdG8SFgoGZG9tYWluGAEgASgJUgZkb21haW4SGAoHdmVyc2lvbh'
    'gCIAEoA1IHdmVyc2lvbg==');

@$core.Deprecated('Use functionProtoDescriptor instead')
const FunctionProto$json = {
  '1': 'FunctionProto',
  '2': [
    {'1': 'name', '3': 1, '4': 1, '5': 9, '10': 'name'},
    {'1': 'input', '3': 4, '4': 3, '5': 9, '10': 'input'},
    {'1': 'output', '3': 5, '4': 3, '5': 9, '10': 'output'},
    {'1': 'attribute', '3': 6, '4': 3, '5': 9, '10': 'attribute'},
    {
      '1': 'attribute_proto',
      '3': 11,
      '4': 3,
      '5': 11,
      '6': '.onnx.AttributeProto',
      '10': 'attributeProto'
    },
    {
      '1': 'node',
      '3': 7,
      '4': 3,
      '5': 11,
      '6': '.onnx.NodeProto',
      '10': 'node'
    },
    {'1': 'doc_string', '3': 8, '4': 1, '5': 9, '10': 'docString'},
    {
      '1': 'opset_import',
      '3': 9,
      '4': 3,
      '5': 11,
      '6': '.onnx.OperatorSetIdProto',
      '10': 'opsetImport'
    },
    {'1': 'domain', '3': 10, '4': 1, '5': 9, '10': 'domain'},
    {'1': 'overload', '3': 13, '4': 1, '5': 9, '10': 'overload'},
    {
      '1': 'value_info',
      '3': 12,
      '4': 3,
      '5': 11,
      '6': '.onnx.ValueInfoProto',
      '10': 'valueInfo'
    },
    {
      '1': 'metadata_props',
      '3': 14,
      '4': 3,
      '5': 11,
      '6': '.onnx.StringStringEntryProto',
      '10': 'metadataProps'
    },
  ],
  '9': [
    {'1': 2, '2': 3},
    {'1': 3, '2': 4},
  ],
  '10': ['since_version', 'status'],
};

/// Descriptor for `FunctionProto`. Decode as a `google.protobuf.DescriptorProto`.
final $typed_data.Uint8List functionProtoDescriptor = $convert.base64Decode(
    'Cg1GdW5jdGlvblByb3RvEhIKBG5hbWUYASABKAlSBG5hbWUSFAoFaW5wdXQYBCADKAlSBWlucH'
    'V0EhYKBm91dHB1dBgFIAMoCVIGb3V0cHV0EhwKCWF0dHJpYnV0ZRgGIAMoCVIJYXR0cmlidXRl'
    'Ej0KD2F0dHJpYnV0ZV9wcm90bxgLIAMoCzIULm9ubnguQXR0cmlidXRlUHJvdG9SDmF0dHJpYn'
    'V0ZVByb3RvEiMKBG5vZGUYByADKAsyDy5vbm54Lk5vZGVQcm90b1IEbm9kZRIdCgpkb2Nfc3Ry'
    'aW5nGAggASgJUglkb2NTdHJpbmcSOwoMb3BzZXRfaW1wb3J0GAkgAygLMhgub25ueC5PcGVyYX'
    'RvclNldElkUHJvdG9SC29wc2V0SW1wb3J0EhYKBmRvbWFpbhgKIAEoCVIGZG9tYWluEhoKCG92'
    'ZXJsb2FkGA0gASgJUghvdmVybG9hZBIzCgp2YWx1ZV9pbmZvGAwgAygLMhQub25ueC5WYWx1ZU'
    'luZm9Qcm90b1IJdmFsdWVJbmZvEkMKDm1ldGFkYXRhX3Byb3BzGA4gAygLMhwub25ueC5TdHJp'
    'bmdTdHJpbmdFbnRyeVByb3RvUg1tZXRhZGF0YVByb3BzSgQIAhADSgQIAxAEUg1zaW5jZV92ZX'
    'JzaW9uUgZzdGF0dXM=');
