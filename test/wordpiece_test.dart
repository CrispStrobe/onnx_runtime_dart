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

  test('encodePair follows the pair template with segment ids', () {
    // [CLS] hello [SEP] world [SEP], token_type_ids 0,0,0,1,1
    final (ids, types) = tok.encodePair('hello', 'world');
    expect(ids.map((i) => tok.idToToken[i]).toList(),
        ['[CLS]', 'hello', '[SEP]', 'world', '[SEP]']);
    expect(types, [0, 0, 0, 1, 1]);
  });

  test('maxLength truncates content, reserving special tokens', () {
    const text = 'hello world hello world !';
    // budget = maxLength(4) - 2 specials = 2 content tokens.
    expect(
        tok.encode('hello world', maxLength: 5).map((i) => tok.idToToken[i]),
        ['[CLS]', 'hello', 'world', '[SEP]']); // fits, unchanged
    final right = tok.encode(text, maxLength: 4);
    expect(right.length, 4);
    expect(right.first, tok.vocab['[CLS]']);
    expect(right.last, tok.vocab['[SEP]']);
    expect(right.sublist(1, 3), [tok.vocab['hello'], tok.vocab['world']]);
    // left keeps the tail instead of the front.
    final left =
        tok.encode(text, maxLength: 4, direction: TruncationDirection.left);
    expect(left.sublist(1, 3), [tok.vocab['world'], tok.vocab['!']]);
  });

  test('encodePair longest_first keeps the pair within maxLength', () {
    final (ids, _) = tok.encodePair('hello world hello', 'world',
        maxLength: 5); // 3 specials -> 2 content across both sides
    expect(ids.length, lessThanOrEqualTo(5));
    expect(ids.first, tok.vocab['[CLS]']);
    expect(ids.last, tok.vocab['[SEP]']);
  });
}
