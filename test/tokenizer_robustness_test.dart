/// Reader-robustness regressions for the tokenizers (findings from
/// `tool/fuzz/`): a malformed `tokenizer.json` must reject with FormatException
/// (never a leaked cast/type error), and `encode` must never throw on any text.
@TestOn('vm')
library;

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

void main() {
  group('config loaders reject malformed json as FormatException', () {
    // Structurally-wrong (but valid JSON) configs that would leak a
    // TypeError/CastError without the *_config guards.
    const badVocabType = '{"model":{"vocab":42}}';
    const badTop = '[1,2,3]';
    const notJson = '{not json';

    test('WordPiece', () {
      expect(() => WordPieceTokenizer.fromJson(badVocabType),
          throwsFormatException);
      expect(() => WordPieceTokenizer.fromJson(badTop), throwsFormatException);
      expect(() => WordPieceTokenizer.fromJson(notJson), throwsFormatException);
    });
    test('Unigram', () {
      expect(() => UnigramTokenizer.fromJson('{"model":{"vocab":"x"}}'),
          throwsFormatException);
      expect(() => UnigramTokenizer.fromJson(badTop), throwsFormatException);
    });
    test('BPE', () {
      expect(() => BpeTokenizer.fromJson('{"model":{"vocab":[],"merges":7}}'),
          throwsFormatException);
      expect(() => BpeTokenizer.fromJson(badTop), throwsFormatException);
    });
  });

  test('encode never throws on hostile text', () {
    final wp = WordPieceTokenizer.fromFile('test/data/tiny_wordpiece.json');
    final uni = UnigramTokenizer.fromFile('test/data/tiny_unigram.json');
    final hostile = [
      '',
      '   \t\n',
      String.fromCharCodes(List.filled(300, 0x0301)), // combining flood
      '\u{10FFFF}\u{FEFF}\u{200D}', // max cp, BOM, ZWJ
      'a' * 5000,
    ];
    for (final t in hostile) {
      expect(() => wp.encode(t), returnsNormally);
      expect(() => wp.encodePair(t, t), returnsNormally);
      expect(() => uni.encode(t), returnsNormally);
      expect(() => uni.encodePair(t, t), returnsNormally);
    }
  });
}
