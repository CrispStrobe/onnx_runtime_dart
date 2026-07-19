import 'dart:convert';
import 'dart:io';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
void main(List<String> a){
  final tok=WordPieceTokenizer.fromFile(a[0]);
  final ref=jsonDecode(File(a[1]).readAsStringSync()) as List;
  var pass=0,fail=0;
  for(final c in ref){
    final text=c['text'] as String;
    // Reference is padded to 128 with [PAD]; trim to through [SEP].
    final refToks=(c['toks'] as List).cast<String>();
    final sep=refToks.indexOf('[SEP]');
    final want=sep>=0?refToks.sublist(0,sep+1):refToks;
    final got=tok.tokens(text);
    final ok=got.length==want.length && List.generate(got.length,(i)=>got[i]==want[i]).every((x)=>x);
    if(ok){pass++;}else{fail++;
      stderr.writeln('MISMATCH: ${jsonEncode(text)}');
      stderr.writeln('  want: $want');
      stderr.writeln('  got:  $got');
    }
  }
  print('wordpiece: $pass/${pass+fail} exact');
  if(fail>0) exit(1);
  print('PASS');
}
