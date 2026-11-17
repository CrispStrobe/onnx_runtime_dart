/// Source-word alignment needed by token classifiers and edit taggers.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

void main() {
  final tok = WordPieceTokenizer.fromFile('test/data/tiny_wordpiece.json');

  test('subwords and punctuation retain the supplied word id', () {
    final enc = tok.encodePretokenized(['Hello,', 'tokenization', 'déjà']);
    // [CLS] hello , token ##ization de ##ja [SEP]. The comma belongs to
    // supplied word 0, rather than shifting every later word's index.
    expect(enc.ids, [2, 4, 5, 14, 15, 9, 10, 3]);
    expect(enc.wordIds, [null, 0, 0, 1, 1, 2, 2, null]);
  });

  test('empty and normalized-away words do not renumber following words', () {
    final enc = tok.encodePretokenized(['', 'Hello', '\u0000', 'world', '']);
    expect(enc.ids, [2, 4, 6, 3]);
    expect(enc.wordIds, [null, 1, 3, null]);
  });

  test('unknown words and CJK pieces each retain their source word', () {
    final enc = tok.encodePretokenized(['xyzzy', '北京']);
    expect(enc.ids, [2, 1, 12, 13, 3]);
    expect(enc.wordIds, [null, 0, 1, 1, null]);
  });

  test(
    'cased fallback template preserves accents and nonstandard special ids',
    () {
      final cased = WordPieceTokenizer.fromFile(
        'test/data/tiny_wordpiece_cased.json',
      );
      final enc = cased.encodePretokenized(['Hello', 'Café']);
      expect(enc.ids, [2, 4, 8, 3]);
      expect(enc.wordIds, [null, 0, 1, null]);
    },
  );

  test('right truncation preserves word ids through a partial word', () {
    final enc = tok.encodePretokenized([
      'Hello',
      'tokenization',
      'world',
    ], maxLength: 4);
    expect(enc.ids, [2, 4, 14, 3]);
    expect(enc.wordIds, [null, 0, 1, null]);
  });

  test('left truncation preserves the original word indices', () {
    final enc = tok.encodePretokenized(
      ['Hello', 'tokenization', 'world'],
      maxLength: 4,
      direction: TruncationDirection.left,
    );
    expect(enc.ids, [2, 15, 6, 3]);
    expect(enc.wordIds, [null, 1, 2, null]);
  });

  test('bare ids and empty sequences have matching alignment lengths', () {
    final enc = tok.encodePretokenized(
      ['Hello', 'tokenization'],
      addSpecial: false,
      maxLength: 2,
    );
    expect(enc.ids, [4, 14]);
    expect(enc.wordIds, [0, 1]);
    final empty = tok.encodePretokenized([]);
    expect(empty.ids, [2, 3]);
    expect(empty.wordIds, [null, null]);
    final bareEmpty = tok.encodePretokenized([], addSpecial: false);
    expect(bareEmpty.ids, isEmpty);
    expect(bareEmpty.wordIds, isEmpty);
  });

  test('custom single template controls special-token placement', () {
    final config = jsonDecode(
      File('test/data/tiny_wordpiece.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    (config['post_processor'] as Map<String, dynamic>)['single'] = [
      {
        'SpecialToken': {'id': '[SEP]', 'type_id': 0},
      },
      {
        'Sequence': {'id': 'A', 'type_id': 0},
      },
      {
        'SpecialToken': {'id': '[CLS]', 'type_id': 0},
      },
      {
        'SpecialToken': {'id': '[SEP]', 'type_id': 0},
      },
    ];
    final custom = WordPieceTokenizer.fromJson(jsonEncode(config));
    final enc = custom.encodePretokenized(['Hello', 'world'], maxLength: 4);
    expect(enc.ids, [3, 4, 2, 3]);
    expect(enc.wordIds, [null, 0, null, null]);
  });
}
