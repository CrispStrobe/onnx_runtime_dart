import 'dart:convert';
import 'dart:io';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
void main(List<String> a){
  final tok=WordPieceTokenizer.fromFile(a[0]);
  final ref=jsonDecode(File(a[1]).readAsStringSync()) as List;
  var pass=0,fail=0;
  for(final c in ref){
    final text=c['text'] as String;
    final want=(c['ids'] as List).cast<int>();
    final got=tok.encode(text);
    final ok=got.length==want.length && List.generate(got.length,(i)=>got[i]==want[i]).every((x)=>x);
    if(ok){pass++;}else{fail++;
      stderr.writeln('MISMATCH ${jsonEncode(text)}\n  want ${want}\n  got  ${got}');}
  }
  print('wordpiece: $pass/${pass+fail} exact'); if(fail>0) exit(1); print('PASS');
}
