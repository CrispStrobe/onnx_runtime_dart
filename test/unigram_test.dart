/// SentencePiece Unigram tokenizer: Metaspace prefix, Viterbi segmentation over
/// vocab log-probs, NFKC compatibility fold, [<s>…</s>], and UNK — against a
/// tiny hand-built vocab.
@TestOn('vm')
library;

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

void main() {
  final tok = UnigramTokenizer.fromFile('test/data/tiny_unigram.json');
  test('metaspace prefix + highest-score segmentation', () {
    // "hello" -> ▁hello (score -3) beats ▁he+llo (-4.5); wrapped in <s>/</s>.
    expect(tok.tokens('hello'), ['<s>', '▁hello', '</s>']);
  });
  test('word boundary becomes ▁ and segments greedily by score', () {
    expect(tok.tokens('he world'),
        ['<s>', '▁he', '▁wor', 'ld', '</s>']);
  });
  test('NFKC folds full-width digits before segmentation', () {
    // Full-width １２ -> 1 2
    expect(tok.tokens('１２'), ['<s>', '▁', '1', '2', '</s>']);
  });
  test('unknown character falls back to <unk>', () {
    expect(tok.encode('hz').contains(2), isTrue); // 'z' has no piece -> <unk>
  });
}
