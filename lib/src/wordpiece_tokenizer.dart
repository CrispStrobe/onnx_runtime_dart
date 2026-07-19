/// Minimal BERT WordPiece tokenizer (the `WordPiece` model in a HuggingFace
/// `tokenizer.json`), so the embedding/reranker family (BERT / MiniLM / MPNet /
/// GTE / E5 / mxbai …) is text-in / vector-out in pure Dart. Reproduces the
/// `BertNormalizer` + `BertPreTokenizer` + greedy WordPiece + `[CLS]…[SEP]`
/// pipeline; accent stripping uses a precomputed NFD table (see
/// `bert_strip_accents.dart`) since Dart core has no Unicode normalization.
library;

import 'dart:convert';
import 'dart:io';

import 'bert_strip_accents.dart';
import 'token_template.dart';

class WordPieceTokenizer {
  final Map<String, int> vocab;
  final Map<int, String> idToToken;
  final String prefix; // continuing-subword marker, "##"
  final String unk;
  final int maxChars;
  final bool lowercase;
  final bool stripAccents;
  final int clsId, sepId;
  final List<TemplateItem>? singleTpl, pairTpl; // post-processor templates

  WordPieceTokenizer._(this.vocab, this.idToToken, this.prefix, this.unk,
      this.maxChars, this.lowercase, this.stripAccents, this.clsId, this.sepId,
      this.singleTpl, this.pairTpl);

  factory WordPieceTokenizer.fromFile(String path) {
    final j = jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    final model = j['model'] as Map<String, dynamic>;
    final vocab = (model['vocab'] as Map<String, dynamic>)
        .map((k, v) => MapEntry(k, (v as num).toInt()));
    final idToToken = {for (final e in vocab.entries) e.value: e.key};
    final norm = j['normalizer'] as Map<String, dynamic>?;
    final lower = norm?['lowercase'] as bool? ?? true;
    // strip_accents defaults to the lowercase flag when null (BERT uncased).
    final strip = norm?['strip_accents'] as bool? ?? lower;
    final pp = j['post_processor'];
    final single = pp is Map<String, dynamic>
        ? parseTemplate(pp['single'], (s) => vocab[s])
        : null;
    final pair = pp is Map<String, dynamic>
        ? parseTemplate(pp['pair'], (s) => vocab[s])
        : null;
    return WordPieceTokenizer._(
      vocab,
      idToToken,
      model['continuing_subword_prefix'] as String? ?? '##',
      model['unk_token'] as String? ?? '[UNK]',
      model['max_input_chars_per_word'] as int? ?? 100,
      lower,
      strip,
      vocab['[CLS]'] ?? -1,
      vocab['[SEP]'] ?? -1,
      single,
      pair,
    );
  }

  static bool _isWhitespace(int cp) =>
      cp == 0x20 ||
      cp == 0x09 ||
      cp == 0x0A ||
      cp == 0x0D ||
      cp == 0x0C ||
      // Unicode Zs separators + line/paragraph separators.
      cp == 0xA0 ||
      cp == 0x1680 ||
      (cp >= 0x2000 && cp <= 0x200A) ||
      cp == 0x2028 ||
      cp == 0x2029 ||
      cp == 0x202F ||
      cp == 0x205F ||
      cp == 0x3000;

  static bool _isControl(int cp) {
    if (cp == 0x09 || cp == 0x0A || cp == 0x0D) return false; // treated as space
    return cp == 0 ||
        cp == 0xFFFD ||
        (cp < 0x20) ||
        (cp >= 0x7F && cp <= 0x9F) ||
        // Cf format chars (zero-width, bidi marks) commonly stripped by BERT.
        cp == 0xAD ||
        (cp >= 0x200B && cp <= 0x200F) ||
        (cp >= 0x2028 && cp <= 0x202E && cp != 0x2028 && cp != 0x2029) ||
        cp == 0xFEFF;
  }

  static bool _isCjk(int cp) =>
      (cp >= 0x4E00 && cp <= 0x9FFF) ||
      (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) ||
      (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) ||
      (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0x2F800 && cp <= 0x2FA1F);

  static bool _isPunct(int cp) {
    // ASCII punctuation ranges + common Unicode punctuation blocks. BERT also
    // treats these as standalone pre-tokens.
    if ((cp >= 33 && cp <= 47) ||
        (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) ||
        (cp >= 123 && cp <= 126)) {
      return true;
    }
    return (cp >= 0x2000 && cp <= 0x206F) || // general punctuation
        (cp >= 0x3000 && cp <= 0x303F) || // CJK symbols/punctuation
        (cp >= 0xFF00 && cp <= 0xFF0F) ||
        (cp >= 0xFF1A && cp <= 0xFF20);
  }

  /// BertNormalizer: clean control/whitespace, space out CJK, lowercase, then
  /// strip accents; followed by BertPreTokenizer (split on whitespace, isolate
  /// punctuation). Returns the pre-token strings.
  List<String> _normalizeAndSplit(String text) {
    final buf = StringBuffer();
    for (final cp in text.runes) {
      if (_isControl(cp)) continue;
      if (_isWhitespace(cp)) {
        buf.write(' ');
        continue;
      }
      if (_isCjk(cp)) {
        buf..write(' ')..writeCharCode(cp)..write(' ');
        continue;
      }
      buf.writeCharCode(cp);
    }
    var s = buf.toString();
    if (lowercase) s = s.toLowerCase();
    if (stripAccents) {
      final sb = StringBuffer();
      for (final cp in s.runes) {
        final rep = bertStripAccents[cp];
        if (rep != null) {
          sb.write(rep); // may be '' for a pure combining mark
        } else {
          sb.writeCharCode(cp);
        }
      }
      s = sb.toString();
    }
    // Pre-tokenize: whitespace split, then peel punctuation into own tokens.
    final tokens = <String>[];
    for (final word in s.split(RegExp(r'\s+'))) {
      if (word.isEmpty) continue;
      final cur = StringBuffer();
      for (final cp in word.runes) {
        if (_isPunct(cp)) {
          if (cur.isNotEmpty) {
            tokens.add(cur.toString());
            cur.clear();
          }
          tokens.add(String.fromCharCode(cp));
        } else {
          cur.writeCharCode(cp);
        }
      }
      if (cur.isNotEmpty) tokens.add(cur.toString());
    }
    return tokens;
  }

  /// Greedy longest-match-first WordPiece over one pre-token's characters.
  List<String> _wordpiece(String word) {
    final chars = word.runes.toList();
    if (chars.length > maxChars) return [unk];
    final out = <String>[];
    var start = 0;
    while (start < chars.length) {
      var end = chars.length;
      String? cur;
      while (start < end) {
        var sub = String.fromCharCodes(chars.sublist(start, end));
        if (start > 0) sub = '$prefix$sub';
        if (vocab.containsKey(sub)) {
          cur = sub;
          break;
        }
        end--;
      }
      if (cur == null) return [unk]; // any unmatchable piece → whole word UNK
      out.add(cur);
      start = end;
    }
    return out;
  }

  /// Piece ids for [text] with no special tokens.
  List<int> _rawIds(String text) {
    final ids = <int>[];
    for (final word in _normalizeAndSplit(text)) {
      for (final piece in _wordpiece(word)) {
        ids.add(vocab[piece] ?? vocab[unk]!);
      }
    }
    return ids;
  }

  /// Encode [text] to token ids, wrapped per the `single` post-processor
  /// template (`[CLS] … [SEP]` for BERT). `addSpecial: false` returns the bare
  /// piece ids.
  List<int> encode(String text, {bool addSpecial = true}) {
    final raw = _rawIds(text);
    if (!addSpecial) return raw;
    if (singleTpl != null) return applyTemplate(singleTpl!, raw, null).$1;
    return [if (clsId >= 0) clsId, ...raw, if (sepId >= 0) sepId];
  }

  /// Encode a sentence pair (query, document) for cross-encoders, following the
  /// `pair` post-processor template (`[CLS] A [SEP] B [SEP]`, segments 0/1 for
  /// BERT). Returns the token ids and matching `token_type_ids`.
  (List<int>, List<int>) encodePair(String a, String b) {
    final aIds = _rawIds(a), bIds = _rawIds(b);
    if (pairTpl != null) return applyTemplate(pairTpl!, aIds, bIds);
    // Default BERT pair layout when no template is present.
    final ids = [if (clsId >= 0) clsId, ...aIds, sepId, ...bIds, sepId];
    final types = [
      for (var i = 0; i < aIds.length + (clsId >= 0 ? 2 : 1); i++) 0,
      for (var i = 0; i < bIds.length + 1; i++) 1,
    ];
    return (ids, types);
  }

  List<String> tokens(String text, {bool addSpecial = true}) =>
      [for (final id in encode(text, addSpecial: addSpecial)) idToToken[id]!];
}
