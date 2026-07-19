/// Canonical NFC composition of a string, using the generated [nfcCompose]
/// pair table. Composes a base character followed by combining mark(s) into
/// the precomposed form (e.g. `e` + U+0301 → `é`), the piece of Unicode NFC
/// the per-codepoint compatibility tables can't do. Dart core has no Unicode
/// normalization, so tokenizers whose `tokenizer.json` declares an NFC/NFKC
/// normalizer call this to match the reference on decomposed input.
library;

import 'nfc_compose.dart';

String composeNfc(String s) {
  // Fast path: nothing to compose unless a combining-range char is present.
  var hasMark = false;
  for (final cp in s.runes) {
    if (cp >= 0x0300 && cp <= 0x1DFF || (cp >= 0x20D0 && cp <= 0x20FF)) {
      hasMark = true;
      break;
    }
  }
  if (!hasMark) return s;
  final out = <int>[];
  for (final cp in s.runes) {
    if (out.isNotEmpty) {
      final composed = nfcCompose[(out.last << 21) | cp];
      if (composed != null) {
        out[out.length - 1] = composed; // may itself take a further mark
        continue;
      }
    }
    out.add(cp);
  }
  return String.fromCharCodes(out);
}
