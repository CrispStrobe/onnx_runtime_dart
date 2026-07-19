/// Minimal SentencePiece **Unigram** tokenizer (the `Unigram` model in a
/// HuggingFace `tokenizer.json`), so the multilingual embedder family
/// (XLM-RoBERTa: multilingual-e5 / bge-m3 / paraphrase-multilingual …) is
/// text-in / vector-out in pure Dart. Reproduces `Metaspace` pre-tokenization
/// + Viterbi segmentation over the vocab log-probs + `<s>…</s>` templating.
///
/// Normalization: SentencePiece's `Precompiled` charsmap is approximated by a
/// per-codepoint NFKC compatibility fold (`nfkc_compat.dart` — full-width
/// forms, ligatures, fractions, circled numbers …) plus whitespace collapsing.
/// This matches the reference exactly for NFC-normalized input across scripts.
/// The one unhandled case is base+combining *composition* (e.g. an "e" plus a
/// separate combining acute instead of a precomposed "é"); feed NFC text for
/// those. Byte-fallback vocabularies are not supported.
library;

import 'dart:convert';
import 'dart:io';

import 'nfkc_compat.dart';

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
      this.eosId);

  factory UnigramTokenizer.fromFile(String path) {
    final j = jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
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
    // Special <s>/</s> ids from the post-processor template, if present.
    int? bos, eos;
    final pp = j['post_processor'];
    if (pp is Map<String, dynamic> && pp['single'] is List) {
      for (final item in pp['single'] as List) {
        final st = (item as Map<String, dynamic>)['SpecialToken'];
        if (st is Map<String, dynamic>) {
          final id = pieceToId[st['id']];
          if (id != null) {
            bos ??= id;
            eos = id;
          }
        }
      }
    }
    return UnigramTokenizer._(pieceToId, pieceScore, idToPiece, unkId, maxLen,
        minScore - 10.0, replacement, addPrefix, bos, eos);
  }

  /// Whitespace → space, collapse runs (the `Replace " {2,}"→" "` step after
  /// the NFKC normalizer, which also folds tabs/newlines to spaces), then
  /// Metaspace: spaces → ▁ and a leading ▁ when `add_prefix_space`.
  String _normalize(String text) {
    final sb = StringBuffer();
    var prevSpace = false;
    for (final raw in text.runes) {
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
    if (addPrefixSpace && !s.startsWith(replacement)) s = '$replacement$s';
    return s;
  }

  /// Viterbi: pick the segmentation of [text] into vocab pieces maximizing the
  /// summed log-probs; unmatchable characters become a single `<unk>`.
  List<int> encode(String text, {bool addSpecial = true}) {
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
    final rev = ids.reversed.toList();
    if (!addSpecial) return rev;
    return [if (bosId != null) bosId!, ...rev, if (eosId != null) eosId!];
  }

  List<String> tokens(String text, {bool addSpecial = true}) =>
      [for (final id in encode(text, addSpecial: addSpecial)) idToPiece[id]!];
}
