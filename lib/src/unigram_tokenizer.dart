/// Minimal SentencePiece **Unigram** tokenizer (the `Unigram` model in a
/// HuggingFace `tokenizer.json`), so the multilingual embedder family
/// (XLM-RoBERTa: multilingual-e5 / bge-m3 / paraphrase-multilingual …) is
/// text-in / vector-out in pure Dart. Reproduces `Metaspace` pre-tokenization
/// + Viterbi segmentation over the vocab log-probs + `<s>…</s>` templating.
///
/// Normalization: SentencePiece's `Precompiled` charsmap is approximated by a
/// per-codepoint NFKC compatibility fold (`nfkc_compat.dart` — full-width
/// forms, ligatures, fractions, circled numbers …), canonical NFC composition
/// of base+combining sequences (`nfc.dart`), zero-width/BOM → space, and
/// whitespace collapsing. Validated exact vs the reference `tokenizers`
/// library over a broad multilingual corpus. Byte-fallback vocabularies are
/// not supported.
library;

import 'dart:convert';
import 'dart:io';

import 'nfc.dart';
import 'nfkc_compat.dart';
import 'token_template.dart';

class UnigramTokenizer {
  final Map<String, int> pieceToId;
  final Map<String, double> pieceScore;
  final Map<int, String> idToPiece;
  final int unkId;
  final int maxPieceLen; // in runes, the Viterbi look-back window
  final double unkScore;
  final String replacement; // metaspace marker, "▁"
  final bool addPrefixSpace;
  final int? bosId, eosId;
  final List<TemplateItem>? singleTpl, pairTpl;

  UnigramTokenizer._(
      this.pieceToId,
      this.pieceScore,
      this.idToPiece,
      this.unkId,
      this.maxPieceLen,
      this.unkScore,
      this.replacement,
      this.addPrefixSpace,
      this.bosId,
      this.eosId,
      this.singleTpl,
      this.pairTpl);

  factory UnigramTokenizer.fromFile(String path) =>
      UnigramTokenizer.fromJson(File(path).readAsStringSync());

  /// Build from a `tokenizer.json` string (web / in-memory). A malformed or
  /// structurally-wrong config is rejected with [FormatException] — never a
  /// leaked cast/type error (guard:uni_config, verified by tool/fuzz/).
  factory UnigramTokenizer.fromJson(String source) {
    try {
      return UnigramTokenizer._fromJson(source);
    } on FormatException {
      rethrow;
      // GUARD:uni_config >>>
    } catch (e) {
      throw FormatException('Invalid Unigram tokenizer.json: $e');
      // GUARD:uni_config <<<
    }
  }

  factory UnigramTokenizer._fromJson(String source) {
    final j = jsonDecode(source) as Map<String, dynamic>;
    final model = j['model'] as Map<String, dynamic>;
    final pieceToId = <String, int>{};
    final pieceScore = <String, double>{};
    final idToPiece = <int, String>{};
    var maxLen = 1;
    var minScore = 0.0;
    final vocab = model['vocab'] as List;
    for (var i = 0; i < vocab.length; i++) {
      final entry = vocab[i] as List;
      final piece = entry[0] as String;
      final score = (entry[1] as num).toDouble();
      pieceToId[piece] = i;
      pieceScore[piece] = score;
      idToPiece[i] = piece;
      final len = piece.runes.length;
      if (len > maxLen) maxLen = len;
      if (score < minScore) minScore = score;
    }
    final unkId = (model['unk_id'] as num?)?.toInt() ?? 0;
    // Metaspace config (replacement char + prefix-space behaviour).
    var replacement = '▁';
    var addPrefix = true;
    final pt = j['pre_tokenizer'];
    void readMeta(Map<String, dynamic> m) {
      replacement = m['replacement'] as String? ?? replacement;
      addPrefix =
          (m['add_prefix_space'] as bool?) ?? (m['prepend_scheme'] != 'never');
    }

    if (pt is Map<String, dynamic>) {
      if (pt['type'] == 'Metaspace') readMeta(pt);
      if (pt['type'] == 'Sequence') {
        for (final p in (pt['pretokenizers'] as List? ?? const [])) {
          if ((p as Map<String, dynamic>)['type'] == 'Metaspace') readMeta(p);
        }
      }
    }
    // Post-processor templates (special-token placement + segment ids).
    final pp = j['post_processor'];
    final single = pp is Map<String, dynamic>
        ? parseTemplate(pp['single'], (s) => pieceToId[s])
        : null;
    final pair = pp is Map<String, dynamic>
        ? parseTemplate(pp['pair'], (s) => pieceToId[s])
        : null;
    // Fallback <s>/</s> from the single template's specials.
    int? bos, eos;
    if (single != null) {
      for (final it in single) {
        if (it.specialId != null) {
          bos ??= it.specialId;
          eos = it.specialId;
        }
      }
    }
    return UnigramTokenizer._(pieceToId, pieceScore, idToPiece, unkId, maxLen,
        minScore - 10.0, replacement, addPrefix, bos, eos, single, pair);
  }

  /// Whitespace → space, collapse runs (the `Replace " {2,}"→" "` step after
  /// the NFKC normalizer, which also folds tabs/newlines to spaces), then
  /// Metaspace: spaces → ▁ and a leading ▁ when `add_prefix_space`.
  String _normalize(String text) {
    final sb = StringBuffer();
    var prevSpace = false;
    for (final raw in composeNfc(text).runes) {
      // Per-char NFKC compatibility fold (full-width, ligatures, …); may
      // expand one codepoint to several, each re-checked for whitespace.
      final mapped = nfkcCompat[raw];
      for (final cp in mapped == null ? [raw] : mapped.runes) {
        final isSpace = cp == 0x20 ||
            cp == 0x09 ||
            cp == 0x0A ||
            cp == 0x0D ||
            cp == 0x0C ||
            cp == 0xA0 ||
            (cp >= 0x2000 && cp <= 0x200A) ||
            // zero-width / BOM: SentencePiece's normalizer maps these to space.
            cp == 0x200B ||
            cp == 0x200C ||
            cp == 0x200D ||
            cp == 0xFEFF ||
            cp == 0x3000;
        if (isSpace) {
          if (!prevSpace) sb.write(' ');
          prevSpace = true;
        } else {
          sb.writeCharCode(cp);
          prevSpace = false;
        }
      }
    }
    var s = sb.toString().replaceAll(' ', replacement);
    // Don't synthesize a lone ▁ for empty input (SentencePiece emits nothing).
    if (addPrefixSpace && s.isNotEmpty && !s.startsWith(replacement)) {
      s = '$replacement$s';
    }
    return s;
  }

  /// Viterbi: pick the segmentation of [text] into vocab pieces maximizing the
  /// summed log-probs; unmatchable characters become a single `<unk>`.
  List<int> _rawIds(String text) {
    final runes = _normalize(text).runes.toList();
    final n = runes.length;
    final best = List<double>.filled(n + 1, double.negativeInfinity);
    final backStart = List<int>.filled(n + 1, -1);
    final backId = List<int>.filled(n + 1, -1); // -1 marks an <unk> segment
    best[0] = 0;
    for (var end = 1; end <= n; end++) {
      final lo = end - maxPieceLen < 0 ? 0 : end - maxPieceLen;
      for (var start = lo; start < end; start++) {
        if (best[start] == double.negativeInfinity) continue;
        final sub = String.fromCharCodes(runes.getRange(start, end));
        final sc = pieceScore[sub];
        if (sc == null) continue;
        final cand = best[start] + sc;
        if (cand > best[end]) {
          best[end] = cand;
          backStart[end] = start;
          backId[end] = pieceToId[sub]!;
        }
      }
      // No piece reaches here: consume one char as <unk>.
      if (best[end] == double.negativeInfinity) {
        best[end] = best[end - 1] + unkScore;
        backStart[end] = end - 1;
        backId[end] = -1;
      }
    }
    final ids = <int>[];
    for (var pos = n; pos > 0;) {
      final start = backStart[pos];
      ids.add(backId[pos] < 0 ? unkId : backId[pos]);
      pos = start;
    }
    return ids.reversed.toList();
  }

  int get _singleSpecials =>
      singleTpl != null ? countSpecials(singleTpl!) : ((bosId != null ? 1 : 0) + (eosId != null ? 1 : 0));
  int get _pairSpecials => pairTpl != null ? countSpecials(pairTpl!) : 4;

  /// Encode [text] to token ids, wrapped per the `single` post-processor
  /// template (`<s> … </s>`). `addSpecial: false` returns the bare piece ids.
  /// [maxLength] truncates the total (reserving special tokens); [direction]
  /// keeps the front (`right`) or tail.
  List<int> encode(String text,
      {bool addSpecial = true,
      int? maxLength,
      TruncationDirection direction = TruncationDirection.right}) {
    var raw = _rawIds(text);
    if (maxLength != null) {
      raw = truncateSingle(
          raw, maxLength - (addSpecial ? _singleSpecials : 0), direction);
    }
    if (!addSpecial) return raw;
    if (singleTpl != null) return applyTemplate(singleTpl!, raw, null).$1;
    return [if (bosId != null) bosId!, ...raw, if (eosId != null) eosId!];
  }

  /// Encode a sentence pair for cross-encoders, following the `pair`
  /// post-processor template (`<s> A </s></s> B </s>` for XLM-R). Returns the
  /// token ids and matching `token_type_ids`. [maxLength] bounds the total;
  /// [strategy]/[direction] choose which side and end to trim.
  (List<int>, List<int>) encodePair(String a, String b,
      {int? maxLength,
      TruncationStrategy strategy = TruncationStrategy.longestFirst,
      TruncationDirection direction = TruncationDirection.right}) {
    var aIds = _rawIds(a), bIds = _rawIds(b);
    if (maxLength != null) {
      (aIds, bIds) = truncatePair(
          aIds, bIds, maxLength - _pairSpecials, strategy, direction);
    }
    if (pairTpl != null) return applyTemplate(pairTpl!, aIds, bIds);
    final ids = [
      if (bosId != null) bosId!,
      ...aIds,
      if (eosId != null) eosId!,
      if (eosId != null) eosId!,
      ...bIds,
      if (eosId != null) eosId!
    ];
    return (ids, List<int>.filled(ids.length, 0));
  }

  List<String> tokens(String text, {bool addSpecial = true}) =>
      [for (final id in encode(text, addSpecial: addSpecial)) idToPiece[id]!];
}
