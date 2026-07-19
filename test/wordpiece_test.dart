/// BERT WordPiece tokenizer: normalization (accent-strip, CJK spacing,
/// punctuation isolation, lowercase), greedy ## continuation, [CLS]/[SEP]
/// wrapping, and [UNK] fallback — against a tiny hand-built vocab.
@TestOn('vm')
library;

import 'package:test/test.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

void main() {
  final tok = WordPieceTokenizer.fromFile('test/data/tiny_wordpiece.json');
  test('punctuation split + lowercase + specials', () {
    expect(tok.tokens('Hello, world!'),
        ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']);
  });
  test('accent stripping + ## continuation', () {
    // "Café déjà" -> cafe de ##ja  (é/à stripped, "deja" split de + ##ja)
    expect(tok.tokens('Café déjà'), ['[CLS]', 'cafe', 'de', '##ja', '[SEP]']);
  });
  test('CJK characters are spaced into individual tokens', () {
    expect(tok.tokens('北京'), ['[CLS]', '北', '京', '[SEP]']);
  });
  test('unmatched word falls back to [UNK]', () {
    expect(tok.tokens('xyzzy'), ['[CLS]', '[UNK]', '[SEP]']);
  });
  test('greedy longest-match subwords', () {
    expect(tok.tokens('tokenization'),
        ['[CLS]', 'token', '##ization', '[SEP]']);
  });

  test('cased config preserves case and accents', () {
    final cased = WordPieceTokenizer.fromFile('test/data/tiny_wordpiece_cased.json');
    // lowercase=false, strip_accents=false: "Hello" and "Café" stay intact.
    expect(cased.tokens('Hello, World!'),
        ['[CLS]', 'Hello', ',', 'World', '!', '[SEP]']);
    expect(cased.tokens('Caf\u00e9'), ['[CLS]', 'Caf\u00e9', '[SEP]']);
  });
}
