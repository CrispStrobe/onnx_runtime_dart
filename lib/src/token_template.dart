/// Shared `TemplateProcessing` post-processor support for the tokenizers:
/// turns the `single` / `pair` templates in a HuggingFace `tokenizer.json`
/// into the final token-id + segment-id (`token_type_ids`) sequence, so
/// special-token placement is exact for BERT (`[CLS] A [SEP] B [SEP]`,
/// segments 0/1), RoBERTa/XLM-R (`<s> A </s></s> B </s>`), and variants.
library;

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
