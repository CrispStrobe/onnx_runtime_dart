/// Canonical NFC composition helper (base + combining mark → precomposed),
/// which the Unigram and BPE tokenizers use to match reference normalization
/// on decomposed input. Inputs are built from explicit code points so the
/// decomposed vs precomposed distinction is unambiguous (source-encoding safe).
@TestOn('vm')
library;

import 'package:onnx_runtime_dart/src/nfc.dart';
import 'package:test/test.dart';

void main() {
  const acute = 0x0301, diaeresis = 0x0308;
  String cp(List<int> c) => String.fromCharCodes(c);

  test('composes base + combining mark into precomposed form', () {
    expect(composeNfc(cp([0x65, acute])), cp([0xE9])); // e +  ́ -> é
    expect(composeNfc(cp([0x69, diaeresis])), cp([0xEF])); // i +  ̈ -> ï
    expect(composeNfc('cafe${cp([acute])}'), cp([0x63, 0x61, 0x66, 0xE9]));
  });
  test('leaves already-composed and mark-free text untouched', () {
    expect(composeNfc(cp([0xE9])), cp([0xE9])); // precomposed é stays
    expect(composeNfc('hello world 123'), 'hello world 123');
    expect(composeNfc(cp([0x65, acute])).length, 1); // collapsed to one rune
  });
  test('non-composing base+mark pair is left as-is', () {
    // 'b' + combining acute has no precomposed form → stays two code points.
    expect(composeNfc(cp([0x62, acute])).runes.length, 2);
  });
}
