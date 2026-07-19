/// Shared `TemplateProcessing` post-processor support for the tokenizers:
/// turns the `single` / `pair` templates in a HuggingFace `tokenizer.json`
/// into the final token-id + segment-id (`token_type_ids`) sequence, so
/// special-token placement is exact for BERT (`[CLS] A [SEP] B [SEP]`,
/// segments 0/1), RoBERTa/XLM-R (`<s> A </s></s> B </s>`), and variants.
library;

/// Which end of a too-long sequence to drop: `right` keeps the front (drop the
/// tail — the usual default), `left` keeps the tail (drop the front).
enum TruncationDirection { right, left }

/// How to shrink a sentence pair to fit: remove from whichever side is longer
/// ([longestFirst], the HF default), or only from the first / second sequence.
enum TruncationStrategy { longestFirst, onlyFirst, onlySecond }

/// Number of literal special tokens a template contributes (the budget a
/// truncation limit must reserve, e.g. 2 for `[CLS] A [SEP]`, 3 for a pair).
int countSpecials(List<TemplateItem> tpl) =>
    tpl.where((i) => i.specialId != null).length;

/// Truncate one sequence to at most [budget] ids, keeping the front ([right])
/// or the tail ([left]).
List<int> truncateSingle(List<int> ids, int budget, TruncationDirection dir) {
  if (budget < 0) budget = 0;
  if (ids.length <= budget) return ids;
  return dir == TruncationDirection.right
      ? ids.sublist(0, budget)
      : ids.sublist(ids.length - budget);
}

/// Shrink a pair `(a, b)` so `a.length + b.length <= budget`, per [strategy]
/// and [dir] — mirrors HuggingFace `truncate_sequences`: for [longestFirst],
/// each removed token comes from whichever list is currently longer (ties →
/// the second), dropped from the tail ([right]) or front ([left]).
(List<int>, List<int>) truncatePair(List<int> a, List<int> b, int budget,
    TruncationStrategy strategy, TruncationDirection dir) {
  if (budget < 0) budget = 0;
  var an = a.length, bn = b.length;
  var remove = an + bn - budget;
  if (remove <= 0) return (a, b);
  switch (strategy) {
    case TruncationStrategy.onlyFirst:
      an = (an - remove).clamp(0, an);
    case TruncationStrategy.onlySecond:
      bn = (bn - remove).clamp(0, bn);
    case TruncationStrategy.longestFirst:
      while (remove > 0 && (an > 0 || bn > 0)) {
        if (an > bn) {
          an--;
        } else {
          bn--;
        }
        remove--;
      }
  }
  List<int> cut(List<int> s, int keep) => keep >= s.length
      ? s
      : (dir == TruncationDirection.right
          ? s.sublist(0, keep)
          : s.sublist(s.length - keep));
  return (cut(a, an), cut(b, bn));
}

/// One entry of a post-processor template: either a literal special token
/// (`specialId`) or a placeholder for sequence A/B (`seqIsB`), with its
/// segment id.
class TemplateItem {
  final int? specialId; // non-null → emit this token id
  final bool seqIsB; // when specialId == null: false = A, true = B
  final int typeId;
  const TemplateItem.special(this.specialId, this.typeId) : seqIsB = false;
  const TemplateItem.sequence({required this.seqIsB, required this.typeId})
      : specialId = null;
}

/// Parse a template array (the `single`/`pair` value of a `TemplateProcessing`
/// post-processor) into [TemplateItem]s. [idOf] resolves a special token's
/// content to its id; entries whose special can't be resolved are skipped.
/// Returns null if [tpl] isn't a usable list.
List<TemplateItem>? parseTemplate(Object? tpl, int? Function(String) idOf) {
  if (tpl is! List) return null;
  final out = <TemplateItem>[];
  for (final raw in tpl) {
    if (raw is! Map) continue;
    final special = raw['SpecialToken'];
    final seq = raw['Sequence'];
    if (special is Map) {
      final id = idOf(special['id'] as String);
      final typeId = (special['type_id'] as num?)?.toInt() ?? 0;
      if (id != null) out.add(TemplateItem.special(id, typeId));
    } else if (seq is Map) {
      final typeId = (seq['type_id'] as num?)?.toInt() ?? 0;
      out.add(TemplateItem.sequence(seqIsB: seq['id'] == 'B', typeId: typeId));
    }
  }
  return out.isEmpty ? null : out;
}

/// Apply a parsed template, substituting sequence A/B placeholders with
/// [aIds]/[bIds]. Returns the token ids and matching segment ids.
(List<int>, List<int>) applyTemplate(
    List<TemplateItem> tpl, List<int> aIds, List<int>? bIds) {
  final ids = <int>[];
  final types = <int>[];
  for (final item in tpl) {
    if (item.specialId != null) {
      ids.add(item.specialId!);
      types.add(item.typeId);
    } else {
      final seq = item.seqIsB ? (bIds ?? const <int>[]) : aIds;
      for (final id in seq) {
        ids.add(id);
        types.add(item.typeId);
      }
    }
  }
  return (ids, types);
}
