/// End-to-end text -> embedding in pure Dart: WordPiece tokenize -> ONNX ->
/// masked mean-pool -> L2 normalize, checked against a sentence-transformers
/// reference. Proves the embedding family is usable with no external tokenizer.
///   dart run tool/embed_e2e.dart MODEL.onnx TOKENIZER.json REF.json
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';
void main(List<String> a){
  final model=loadOnnxModel(a[0]);
  final tok=WordPieceTokenizer.fromFile(a[1]);
  final ref=jsonDecode(File(a[2]).readAsStringSync()) as List;
  var worst=1.0;
  for(final c in ref){
    final text=c['text'] as String;
    final ids=tok.encode(text);
    final refIds=(c['ids'] as List).cast<int>();
    final idsOk=ids.length==refIds.length && List.generate(ids.length,(i)=>ids[i]==refIds[i]).every((x)=>x);
    final n=ids.length;
    final out=model.run({
      'input_ids':Tensor.int64(Int64List.fromList(ids),[1,n]),
      'attention_mask':Tensor.int64(Int64List(n)..fillRange(0,n,1),[1,n]),
      'token_type_ids':Tensor.int64(Int64List(n),[1,n]),
    },['last_hidden_state'])['last_hidden_state']!;
    final h=out.f??out.asFloatList(); final dim=out.shape[2];
    final emb=Float64List(dim);
    for(var t=0;t<n;t++){for(var d=0;d<dim;d++){emb[d]+=h[t*dim+d];}}
    var norm=0.0; for(var d=0;d<dim;d++){emb[d]/=n; norm+=emb[d]*emb[d];}
    norm=math.sqrt(norm); for(var d=0;d<dim;d++){emb[d]/=norm;}
    final re=(c['emb'] as List).map((v)=>(v as num).toDouble()).toList();
    var dot=0.0; for(var d=0;d<dim;d++){dot+=emb[d]*re[d];}
    worst=math.min(worst,dot);
    print('${idsOk?"ids✓":"ids✗"} cosine=${dot.toStringAsFixed(9)}  "${text.length>32?text.substring(0,32):text}"');
  }
  print('worst cosine=${worst.toStringAsFixed(9)}');
  if(worst<0.9999){stderr.writeln('FAIL');exit(1);}
  print('PASS: pure-Dart embeddings match sentence-transformers');
}
