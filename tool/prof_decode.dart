import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';
void main(List<String> a) {
  final m = loadOnnxModel(a[0]);
  final past = <String, Tensor>{};
  final pastKeys = m.inputSpecs.where((s)=>s.name.startsWith('past_key_values')&&s.name.endsWith('.key')).toList();
  final nL = pastKeys.length; final kvh = pastKeys.first.shape[1]; final hs = pastKeys.first.shape[3];
  final outs = ['logits', for (var l=0;l<nL;l++) ...['present.$l.key','present.$l.value']];
  final pastLen = 10;
  for (var l=0;l<nL;l++){ past['past_key_values.$l.key']=Tensor.float(Float32List(kvh*pastLen*hs),[1,kvh,pastLen,hs]); past['past_key_values.$l.value']=Tensor.float(Float32List(kvh*pastLen*hs),[1,kvh,pastLen,hs]); }
  final inp = <String,Tensor>{
    'input_ids': Tensor.int64(Int64List.fromList([100]),[1,1]),
    'attention_mask': Tensor.int64(Int64List(pastLen+1)..fillRange(0,pastLen+1,1),[1,pastLen+1]),
    'position_ids': Tensor.int64(Int64List.fromList([pastLen]),[1,1]),
    ...past,
  };
  m.run(inp, outs);
  final p = ExecutionProfile();
  final sw = Stopwatch()..start();
  for (var i=0;i<5;i++) { m.run(inp, outs, profile: p); }
  sw.stop();
  print('5 decode steps: ${sw.elapsedMilliseconds}ms  (${sw.elapsedMilliseconds/5}ms/step)');
  print(p.report());
}
