/// Minimal byte-level BPE tokenizer (GPT-2 / Qwen / Llama-BPE family), loading
/// a HuggingFace `tokenizer.json` directly. Applies a declared NFC/NFKC
/// normalizer (Qwen uses NFC) before byte-level encoding; validated exact vs
/// the reference `tokenizers` library over a broad corpus.
///
/// The pipeline mirrors GPT-2: split on the standard pre-tokenization regex,
/// map each UTF-8 byte to a printable unicode char (`bytes_to_unicode`), run
/// rank-ordered BPE merges per chunk, then look up vocab ids. Special/added
/// tokens are matched verbatim before splitting so ChatML markers
/// (`<|im_start|>` …) map to their single ids.
library;

import 'dart:convert';
import 'dart:io';

import 'nfc.dart';
import 'nfkc_compat.dart';
import 'token_template.dart';

class BpeTokenizer {
  final Map<String, int> vocab;
  final Map<int, String> idToToken;
  final Map<String, int> mergeRank; // "a b" -> rank
  final Map<String, int> specials; // content -> id
  final Map<int, int> byteEncoder; // byte -> unicode code point
  final Map<int, int> byteDecoder; // unicode code point -> byte
  final RegExp _splitRe;
  final RegExp? _specialRe;
  final bool _nfc, _nfkc; // declared normalizer (e.g. Qwen uses NFC)

  BpeTokenizer._(this.vocab, this.idToToken, this.mergeRank, this.specials,
      this.byteEncoder, this.byteDecoder, this._splitRe, this._specialRe,
      this._nfc, this._nfkc);

  factory BpeTokenizer.fromFile(String path) =>
      BpeTokenizer.fromJson(File(path).readAsStringSync());

  /// Build from a `tokenizer.json` string (web / in-memory). A malformed or
  /// structurally-wrong config is rejected with [FormatException] — never a
  /// leaked cast/type error (guard:bpe_config, verified by tool/fuzz/).
  factory BpeTokenizer.fromJson(String source) {
    try {
      return BpeTokenizer._fromJson(source);
    } on FormatException {
      rethrow;
      // GUARD:bpe_config >>>
    } catch (e) {
      throw FormatException('Invalid BPE tokenizer.json: $e');
      // GUARD:bpe_config <<<
    }
  }

  factory BpeTokenizer._fromJson(String source) {
    final j = jsonDecode(source) as Map<String, dynamic>;
    final model = j['model'] as Map<String, dynamic>;
    final vocab = (model['vocab'] as Map<String, dynamic>)
        .map((k, v) => MapEntry(k, (v as num).toInt()));
    final idToToken = {for (final e in vocab.entries) e.value: e.key};
    final mergeRank = <String, int>{};
    final merges = model['merges'] as List;
    for (var i = 0; i < merges.length; i++) {
      final m = merges[i];
      // tokenizer.json stores merges either as "a b" or ["a","b"].
      mergeRank[m is List ? '${m[0]} ${m[1]}' : m as String] = i;
    }
    final specials = <String, int>{};
    for (final a in (j['added_tokens'] as List? ?? const [])) {
      final am = a as Map<String, dynamic>;
      final content = am['content'] as String;
      final id = (am['id'] as num).toInt();
      specials[content] = id;
      // Added tokens aren't in model.vocab, so seed idToToken for decode.
      idToToken[id] = content;
    }

    final (enc, dec) = _byteMaps();
    // GPT-2 pre-tokenization regex; Dart has no inline (?i:) group, so the
    // case-insensitive contraction alternation is expanded to both cases.
    final splitRe = RegExp(
        r"'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s+(?!\S)|\s+",
        unicode: true);
    RegExp? specialRe;
    if (specials.isNotEmpty) {
      final alt = (specials.keys.toList()
            ..sort((a, b) => b.length.compareTo(a.length)))
          .map(RegExp.escape)
          .join('|');
      specialRe = RegExp(alt);
    }
    // Declared unicode normalizer (Qwen/Llama BPE commonly use NFC).
    var nfc = false, nfkc = false;
    void scan(Object? n) {
      if (n is! Map) return;
      if (n['type'] == 'NFC') nfc = true;
      if (n['type'] == 'NFKC') nfkc = true;
      if (n['type'] == 'Sequence') {
        for (final x in (n['normalizers'] as List? ?? const [])) {
          scan(x);
        }
      }
    }

    scan(j['normalizer']);
    return BpeTokenizer._(vocab, idToToken, mergeRank, specials, enc, dec,
        splitRe, specialRe, nfc, nfkc);
  }

  /// Apply the declared unicode normalizer to a (non-special) text chunk.
  String _normalize(String text) {
    if (_nfkc) {
      final sb = StringBuffer();
      for (final cp in text.runes) {
        final m = nfkcCompat[cp];
        if (m != null) {
          sb.write(m);
        } else {
          sb.writeCharCode(cp);
        }
      }
      return composeNfc(sb.toString());
    }
    return _nfc ? composeNfc(text) : text;
  }

  /// GPT-2 `bytes_to_unicode`: a reversible map from each of the 256 byte
  /// values to a printable unicode code point.
  static (Map<int, int>, Map<int, int>) _byteMaps() {
    final bs = <int>[];
    for (var i = 0x21; i <= 0x7E; i++) {
      bs.add(i); // '!'..'~'
    }
    for (var i = 0xA1; i <= 0xAC; i++) {
      bs.add(i); // '¡'..'¬'
    }
    for (var i = 0xAE; i <= 0xFF; i++) {
      bs.add(i); // '®'..'ÿ'
    }
    final enc = <int, int>{};
    for (final b in bs) {
      enc[b] = b;
    }
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (!enc.containsKey(b)) {
        enc[b] = 256 + n;
        n++;
      }
    }
    final dec = {for (final e in enc.entries) e.value: e.key};
    return (enc, dec);
  }

  /// Rank-ordered BPE over the unicode-mapped chars of one pre-token: greedily
  /// merge the lowest-rank adjacent pair (all its occurrences) until none of
  /// the remaining pairs is a known merge.
  List<String> _bpe(List<String> parts) {
    if (parts.length < 2) return parts;
    while (true) {
      var bestRank = 1 << 30;
      var bestI = -1;
      for (var i = 0; i < parts.length - 1; i++) {
        final r = mergeRank['${parts[i]} ${parts[i + 1]}'];
        if (r != null && r < bestRank) {
          bestRank = r;
          bestI = i;
        }
      }
      if (bestI < 0) break;
      final a = parts[bestI], b = parts[bestI + 1];
      final merged = <String>[];
      for (var i = 0; i < parts.length;) {
        if (i < parts.length - 1 && parts[i] == a && parts[i + 1] == b) {
          merged.add(a + b);
          i += 2;
        } else {
          merged.add(parts[i]);
          i++;
        }
      }
      parts = merged;
    }
    return parts;
  }

  List<int> _encodeChunk(String rawText) {
    final text = _normalize(rawText);
    final ids = <int>[];
    for (final m in _splitRe.allMatches(text)) {
      final piece = m.group(0)!;
      // Byte-level: UTF-8 bytes -> unicode chars, one char per byte.
      final chars = [for (final byte in utf8.encode(piece)) String.fromCharCode(byteEncoder[byte]!)];
      for (final tok in _bpe(chars)) {
        final id = vocab[tok];
        if (id != null) ids.add(id);
      }
    }
    return ids;
  }

  /// Encode [text] to token ids, splitting out special/added tokens first.
  /// [maxLength] truncates the result (byte-level BPE has no template specials);
  /// [direction] keeps the front (`right`) or tail.
  List<int> encode(String text,
      {int? maxLength, TruncationDirection direction = TruncationDirection.right}) {
    final ids = <int>[];
    if (_specialRe == null) {
      ids.addAll(_encodeChunk(text));
    } else {
      var last = 0;
      for (final m in _specialRe.allMatches(text)) {
        if (m.start > last) {
          ids.addAll(_encodeChunk(text.substring(last, m.start)));
        }
        ids.add(specials[m.group(0)!]!);
        last = m.end;
      }
      if (last < text.length) ids.addAll(_encodeChunk(text.substring(last)));
    }
    return maxLength == null ? ids : truncateSingle(ids, maxLength, direction);
  }

  /// Decode token ids back to text (reverse the byte-level mapping and UTF-8
  /// decode). Unknown ids are skipped.
  String decode(List<int> ids, {bool skipSpecial = true}) {
    final bytes = <int>[];
    for (final id in ids) {
      final tok = idToToken[id];
      if (tok == null) continue;
      if (skipSpecial && specials.containsValue(id)) continue;
      for (final cp in tok.runes) {
        final b = byteDecoder[cp];
        if (b != null) bytes.add(b);
      }
    }
    return utf8.decode(bytes, allowMalformed: true);
  }
}
