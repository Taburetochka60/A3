# Checkpoint error analysis and handoff insights

## Scope and integrity

This is a post-hoc diagnosis of the **frozen 12,000-text BiGRU pilot**, selected at epoch 8, on all 4,000 checkpoint texts. There was **no retraining, threshold optimization, calibration evaluation, final audit evaluation, or test-target access**. The saved model was used only to recover original predicted character offsets. Every re-inferred surface list is asserted identical to the saved pilot prediction CSV.

The complete official-code counts remain **TP=887, FP=917, FN=431, F1=0.5682**, with errors in 886 texts. The analysis distinguishes normalized per-text surface-set errors from raw occurrence/span errors. Those denominators must not be mixed.

Model SHA-256: `ca15bc6e9e03560b6674355fa7070f86a32ea1a320d22955fd4c1d486a71e6c4`. Pilot selection SHA-256: `253bf1f3aab0a7f5e74dc2fbec4cb735b875782ca593fe834024cf91ef67d828`. Frequencies and vocabulary checks use the actual **12,000 fit texts**, not all one million source texts and not validation labels.

## 1. Known versus new location surfaces

A surface is **seen** if its official-normalized string appeared at least once as a LOC annotation in the pilot fit subset. This measures surface familiarity, not whether the underlying real place or another inflected form was known. Each gold or predicted normalized string is assigned to a slice independently: an unseen erroneous prediction counts as unseen FP even when it overlaps a seen gold entity. Thus slice TP/FP/FN sum exactly to the full score, but these are not scores from simply selecting documents with seen/unseen gold.

| Official surface slice | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Seen in pilot fit | 756 | 182 | 125 | 0.806 | 0.858 | 0.831 |
| New to pilot fit | 131 | 735 | 306 | 0.151 | 0.300 | 0.201 |

**Seen recall is 85.8%, versus 30.0% for new surfaces.** New surfaces account for 437 / 1318 = 33.16% of unique per-text gold entities, 71.00% of FN, and 80.15% of FP. The low unseen precision also includes ordinary words incorrectly proposed as places; it is not an estimate of precision restricted to real new locations.

A 2,000-replicate document bootstrap gives recall intervals [0.834, 0.881] for seen surfaces and [0.251, 0.348] for new ones. All checkpoint canonical groups are unique. Intervals condition on this selected pilot, do not include retraining/model-selection/domain-shift uncertainty, and do not prove novelty is the cause of a miss.

Strict raw occurrence recall confirms the direction: 0.853 seen versus 0.292 unseen. Raw supports are 889 and 439 mentions, distinct from deduplicated official supports.

### Frequency gradient

| Pilot LOC mention frequency | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| unseen_0 | 131 | 735 | 306 | 0.151 | 0.300 | 0.201 |
| seen_1 | 63 | 42 | 53 | 0.600 | 0.543 | 0.570 |
| seen_2_to_4 | 112 | 37 | 37 | 0.752 | 0.752 | 0.752 |
| seen_5_plus | 581 | 103 | 35 | 0.849 | 0.943 | 0.894 |

The model is substantially stronger on repeated names. More fit data, subword-aware embeddings or a recurrent character encoder are plausible next experiments. Simply training more epochs on the same small vocabulary is not an evidence-based fix. Do not enforce a closed dictionary: reviewed cases include common words used as **names of rocks**, where context makes them valid locations.

### Token OOV is a different quantity

The earlier 14.36% checkpoint token OOV rate is not the fraction of new place names. Here the gold-entity slices are:

| Gold normalized entity slice | Gold support | Correct | Recall |
|---|---:|---:|---:|
| All overlapping tokens are in pilot vocabulary | 947 | 763 | 0.806 |
| At least one overlapping token is UNK | 371 | 124 | 0.334 |

Gold supports use one normalized entity per text; repeated occurrences share the slice if any occurrence contains UNK. A surface can have appeared as LOC in training while a low-frequency component was dropped by the 30,000-word vocabulary cap. Conversely, a new multiword name can consist entirely of known tokens. These are associated failure factors, not a causal decomposition of every miss.

### Normalization sensitivity

The published regex can alter Latin text and retain unexpected punctuation. As a diagnostic, the same analysis was recomputed using conservative NFC + case-folding + whitespace collapse. Overall F1 becomes 0.5624; seen/unseen recall is 0.855/0.290. The qualitative novelty gap persists. The conservative metric is **not substituted** for the official one. Full values are in `checkpoint_normalization_sensitivity.json`.

## 2. False positives and misses: full-census taxonomy

These categories cover all **1348 official error events**, not just the selected 200 cases. They are assigned from actual span overlaps, saved vocabulary membership and auxiliary annotations, in this precedence: LOC overlap first, then PER overlap, then ORG overlap, then other FP. For non-overlap FN, UNK is checked first, then unseen/seen surface. Multi-cause errors therefore appear once in the table; counts are not independent causal contributions.

| Error | Observed category | Count | Fraction within FP or FN |
|---|---|---:|---:|
| FP | other_unannotated_FP | 622 | 67.83% |
| FN | missed_seen_surface | 70 | 16.24% |
| FN | missed_with_UNK | 196 | 45.48% |
| FP | PER_annotation_overlap | 52 | 5.67% |
| FP | boundary_overlap | 117 | 12.76% |
| FN | boundary_overlap | 108 | 25.06% |
| FP | ORG_annotation_overlap | 126 | 13.74% |
| FN | missed_unseen_surface_known_tokens | 57 | 13.23% |

Important interpretations:

- **117 FP and 108 FN overlap an annotated/predicted location.** They are boundary-related candidates. The remaining 800 FP and 323 FN have no such overlap. Even perfect handling of the observed boundary cases would leave substantial detection/type errors; no counterfactual improved F1 is claimed.
- 126 FP overlap ORG annotations and 52 overlap PER. This does **not** mean all 178 are unambiguous entity-type confusions. Country names embedded in an institution may be sensible geographic substrings that the source's non-nested annotation policy excludes. Auxiliary ORG/PER spans also contain suspected noise.
- The 622 other FP are not supported by LOC gold, but should not all be called hallucinations. Context review finds both obvious ordinary-word errors and plausible unannotated locations.
- Missing with UNK, new surface and known surface are observed properties. A high class weight, decoder boundary, context, or annotation defect can coexist with any of them.

## 3. Character-boundary diagnosis

Exact span matches are removed first. Remaining gold/prediction occurrences form a character-overlap graph. One-to-one assignment maximizes overlap-match cardinality and then intersection-over-union; one prediction cannot repair multiple missed entities in the diagnostic count.

Strict occurrence totals: 886 exact matches, 936 unmatched predicted spans and 442 unmatched gold spans; exact-span F1=0.5625. There are 810 unmatched predictions with no gold overlap and 325 unmatched gold mentions with no prediction overlap.

| One-to-one unmatched overlap pair | Count |
|---|---:|
| missing_name_material | 49 |
| extra_neighbor_material | 42 |
| punctuation_or_whitespace | 15 |
| shifted_partial_overlap | 1 |
| word_ending_fragment | 1 |

**Multiword truncation and neighboring-word capture dominate the paired boundary errors.** There is only one ending-fragment pair in this pilot. This does not show that morphology is unimportant overall: inflection also appears through OOV and new surfaces. It shows that a suffix-trimming rule would address little of the measured boundary subset.

| Overlap component topology | Components |
|---|---:|
| split_location | 17 |
| one_to_one_boundary | 82 |
| merged_locations | 9 |

A split/merge component can contain several error events. Component counts, overlap pairs and official FP/FN therefore have different units. Punctuation/whitespace mismatches sometimes reflect gold including a quote/comma. Splits can reflect inconsistent gold granularity: `Івано-Франківську` is sometimes one entity and sometimes two.

## 4. Contextual review of 200 errors

The supplement contains **200 focal errors from 200 distinct checkpoint texts: 100 FP and 100 FN**, with 93 boundary-overlap focal cases. Sampling deliberately balances kinds and emphasizes boundary coverage (seed 2026); selection is not representative of prevalence. Full-census percentages above are the source of quantitative conclusions.

All 200 contexts were read by Codex and given a contextual note. This is **AI-assisted review, not independent expert adjudication**. Focal labels remain errors relative to the unchanged original LOC gold. Suspected label defects or policy ambiguities are flagged for human checking; none were automatically corrected, removed from checkpoint scoring, or copied into fit.

| Case ID | Focal error | Surface | Review issue |
|---|---|---|---|
| 1 | FN | `Покровському районі` | missing_name_material |
| 8 | FN | `Москви, у тому чи` | possible_annotation_error |
| 30 | FN | `Нідерландів Аранчі Рус` | possible_annotation_error |
| 38 | FN | `-Франківську` | hyphen_granularity |
| 52 | FN | `Путіна` | possible_annotation_error |
| 64 | FN | `Схід Сокола` | possible_annotation_error |
| 84 | FN | `Шлапак` | possible_annotation_error |
| 87 | FN | `Орегоні` | possible_annotation_omission |
| 93 | FN | `Шуршун` | unusual_landmark_names |
| 116 | FP | `-Франківська` | hyphen_granularity |
| 138 | FP | `Діми` | possible_type_error |
| 150 | FP | `Казахстану` | nested_ORG_policy |
| 168 | FP | `талліннському аеропорту` | possible_annotation_omission |
| 175 | FP | `Мохаммеда` | PER_as_LOC |

The full [contextual catalog](CHECKPOINT_REVIEW_200.md) includes original text, complete gold/prediction lists, seen/UNK evidence and a note per case. The annotated CSV/JSON additionally retain auxiliary types and original offsets for filtering.

### Data insights that change the handoff

1. **Potential type errors:** `Путіна`, `Діми Кийкова`, a film title and full sports-team names appear as LOC in their contexts. These require review of the source labels; a low score on such a case is not sufficient evidence that the model misunderstood geography.
2. **Granularity conflicts:** hyphenated cities may be one or two gold entities; coordinated regions may be one long span; a city and its region may be merged or separate. Decide and document the target convention before changing BIO decoding.
3. **Potential span/text corruption:** examples include `Москви, у тому чи`, `Росію, ховаюч`, `України окр`, and fused `Кримужурнал`. Bounds-valid annotations can still be semantically broken.
4. **Possible incomplete annotation or nesting policy:** some country mentions inside organizations and titles are excluded, while similar mentions elsewhere are included. An airport and part of a list of US states look unannotated. Some cases are policy choices rather than obvious errors.
5. **Genuine generic-word FP:** ordinary verbs, anatomical words, numbers and unnamed houses/streets are extracted. Their presence argues for precision-oriented development checks, not just better name coverage.

These are findings in a **diagnostic sample selected from model errors**, not estimates of corpus-wide annotation noise. An independently sampled, blinded annotation audit is needed before reporting a global label-error rate.

## 5. Recommended next experiments and safeguards

| Priority | Action | Evidence / safeguard |
|---|---|---|
| 1 | Write a short LOC annotation contract: nested names, ORG/PER, buildings, addresses, hyphens, coordination and punctuation | Several model failures depend on gold conventions. Keep raw labels; any cleaned version needs a versioned mapping and honest comparisons. |
| 2 | Expand fit coverage and compare fastText/subword embeddings or a character BiGRU | New-surface recall is about 30%; names containing UNK are much harder. Retain original offsets. No transformer is needed. |
| 3 | Compare a few prespecified entity-loss weights and fit-only hard-negative examples | Many genuine generic-word FP; current B/I weights are high. Never insert checkpoint examples into training. |
| 4 | Test constrained BIO/CRF decoding or offset-aware boundary improvements | Most paired boundary errors omit parts of names or absorb adjacent words. Fix conventions before adding generic split/merge rules. |
| 5 | Tune a small final decoding threshold grid on calibration after model selection | Do not optimize thresholds on this error-review sample or final audit. Track precision/recall and slices, not only overall F1. |

Do not turn reviewed checkpoint names into a whitelist, silently relabel held-out answers to match the model, or infer that source-domain gains guarantee Public LB gains. This analysis is development information; final audit remains unscored and target-domain shift remains.

## Reproduction and outputs

```powershell
.venv/Scripts/python.exe assignment3/analyze_checkpoint_errors.py
.venv/Scripts/python.exe assignment3/review_checkpoint_cases.py
.venv/Scripts/python.exe assignment3/build_checkpoint_error_report.py
.venv/Scripts/python.exe -m unittest discover -s assignment3 -p "test_*.py"
```

The review-note script checks the fingerprint of the exact 200-case set and refuses to apply contextual notes to another model/sample. Data tables: `checkpoint_novelty_metrics.csv`, `checkpoint_frequency_metrics.csv`, `checkpoint_gold_OOV_recall.csv`, `checkpoint_review_200_annotated.csv`. All events and matched boundary pairs are retained in JSON. The notebook displays computed results and the plot without retraining.

AI assistance: Codex generated/executed the analysis and reviewed these contexts. Preserve the actual chat transcript/link in the assignment disclosure; the review is not presented as independent human ground truth.
