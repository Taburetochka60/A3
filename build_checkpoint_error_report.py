"""Build error-analysis handoff, sensitivity checks, and a results notebook."""
import ast
from collections import Counter
import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

from metrics import conservative_processing,score_details
from analyze_checkpoint_errors import count_scores

BASE=Path(__file__).resolve().parent


def main():
    r=json.loads((BASE/'reports/checkpoint_error_analysis.json').read_text(encoding='utf-8'))
    reviewed=json.loads((BASE/'reports/checkpoint_review_200_annotated.json').read_text(encoding='utf-8'))
    raw=pd.read_csv(BASE/'data/uk_geo_dataset.csv',usecols=['text','loc_markers'])
    selected=pd.read_csv(BASE/'data/ner_pilot_selection.csv')
    counts=Counter()
    for row in raw.loc[selected.loc[selected.role.eq('pilot_fit'),'row_id']].itertuples():
        for a,b in ast.literal_eval(row.loc_markers):counts[conservative_processing(row.text[a:b])]+=1
    preds=pd.read_csv(BASE/'reports/ner_pilot_checkpoint_predictions.csv')
    cg=[];cp=[];sensitivity={key:np.zeros(3,dtype=int) for key in ['seen','unseen']}
    for row in preds.itertuples():
        source=raw.loc[row.row_id]
        g=[source.text[a:b] for a,b in ast.literal_eval(source.loc_markers)]
        p=ast.literal_eval(row.locations);cg.append(g);cp.append(p)
        gg=set(map(conservative_processing,g));pp=set(map(conservative_processing,p))
        for norm in gg|pp:
            key='seen' if counts[norm] else 'unseen'
            k=0 if norm in gg&pp else 1 if norm in pp else 2
            sensitivity[key][k]+=1
    sensitive=dict(overall=score_details(cg,cp,conservative_processing),
                   novelty={key:count_scores(*value) for key,value in sensitivity.items()},
                   definition='NFC, casefold, whitespace collapse; digits/punctuation retained. Diagnostic, not official score.')
    (BASE/'reports/checkpoint_normalization_sensitivity.json').write_text(json.dumps(sensitive,indent=2),encoding='utf-8')
    seen,new=r['novelty']['seen'],r['novelty']['unseen'];o=r['overall']
    total=o['fp']+o['fn'];cats=r['official_error_categories']
    catrows='\n'.join(f"| {x['kind']} | {x['category']} | {x['count']} | {100*x['count']/o['fp' if x['kind']=='FP' else 'fn']:.2f}% |" for x in cats)
    freqrows='\n'.join(f"| {x['frequency_bin']} | {x['tp']} | {x['fp']} | {x['fn']} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} |" for x in r['frequency_slices'])
    pairrows='\n'.join(f'| {key} | {value} |' for key,value in r['overlap_pair_relations'].items())
    toporows='\n'.join(f'| {key} | {value} |' for key,value in r['overlap_component_topology'].items())
    cpairs=pd.DataFrame(r['official_error_categories'])
    fpover=int(cpairs.loc[(cpairs.kind=='FP')&(cpairs.category=='boundary_overlap'),'count'].iloc[0])
    fnover=int(cpairs.loc[(cpairs.kind=='FN')&(cpairs.category=='boundary_overlap'),'count'].iloc[0])
    selected_boundary=sum(x['category']=='boundary_overlap' for x in reviewed)
    review_lookup={x['case_id']:x for x in reviewed}
    examples=[]
    for case_id in [1,8,30,38,52,64,84,87,93,116,138,150,168,175]:
        e=review_lookup[case_id]
        examples.append(f"| {case_id} | {e['kind']} | `{e['surface']}` | {e['context_review_tag']} |")
    table='\n'.join(examples)
    oov=r['gold_token_OOV_slices']
    report=f'''# Checkpoint error analysis and handoff insights

## Scope and integrity

This is a post-hoc diagnosis of the **frozen 12,000-text BiGRU pilot**, selected at epoch 8, on all 4,000 checkpoint texts. There was **no retraining, threshold optimization, calibration evaluation, final audit evaluation, or test-target access**. The saved model was used only to recover original predicted character offsets. Every re-inferred surface list is asserted identical to the saved pilot prediction CSV.

The complete official-code counts remain **TP={o['tp']}, FP={o['fp']}, FN={o['fn']}, F1={o['f1']:.4f}**, with errors in {r['erroneous_texts']} texts. The analysis distinguishes normalized per-text surface-set errors from raw occurrence/span errors. Those denominators must not be mixed.

Model SHA-256: `{r['model_sha256']}`. Pilot selection SHA-256: `{r['pilot_selection_sha256']}`. Frequencies and vocabulary checks use the actual **12,000 fit texts**, not all one million source texts and not validation labels.

## 1. Known versus new location surfaces

A surface is **seen** if its official-normalized string appeared at least once as a LOC annotation in the pilot fit subset. This measures surface familiarity, not whether the underlying real place or another inflected form was known. Each gold or predicted normalized string is assigned to a slice independently: an unseen erroneous prediction counts as unseen FP even when it overlaps a seen gold entity. Thus slice TP/FP/FN sum exactly to the full score, but these are not scores from simply selecting documents with seen/unseen gold.

| Official surface slice | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Seen in pilot fit | {seen['tp']} | {seen['fp']} | {seen['fn']} | {seen['precision']:.3f} | {seen['recall']:.3f} | {seen['f1']:.3f} |
| New to pilot fit | {new['tp']} | {new['fp']} | {new['fn']} | {new['precision']:.3f} | {new['recall']:.3f} | {new['f1']:.3f} |

**Seen recall is {100*seen['recall']:.1f}%, versus {100*new['recall']:.1f}% for new surfaces.** New surfaces account for {new['tp']+new['fn']} / {o['tp']+o['fn']} = {100*(new['tp']+new['fn'])/(o['tp']+o['fn']):.2f}% of unique per-text gold entities, {100*new['fn']/o['fn']:.2f}% of FN, and {100*new['fp']/o['fp']:.2f}% of FP. The low unseen precision also includes ordinary words incorrectly proposed as places; it is not an estimate of precision restricted to real new locations.

A 2,000-replicate document bootstrap gives recall intervals [{seen['recall_95ci'][0]:.3f}, {seen['recall_95ci'][1]:.3f}] for seen surfaces and [{new['recall_95ci'][0]:.3f}, {new['recall_95ci'][1]:.3f}] for new ones. All checkpoint canonical groups are unique. Intervals condition on this selected pilot, do not include retraining/model-selection/domain-shift uncertainty, and do not prove novelty is the cause of a miss.

Strict raw occurrence recall confirms the direction: {seen['strict_exact_mention_recall']:.3f} seen versus {new['strict_exact_mention_recall']:.3f} unseen. Raw supports are {seen['raw_gold_mentions']} and {new['raw_gold_mentions']} mentions, distinct from deduplicated official supports.

### Frequency gradient

| Pilot LOC mention frequency | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{freqrows}

The model is substantially stronger on repeated names. More fit data, subword-aware embeddings or a recurrent character encoder are plausible next experiments. Simply training more epochs on the same small vocabulary is not an evidence-based fix. Do not enforce a closed dictionary: reviewed cases include common words used as **names of rocks**, where context makes them valid locations.

### Token OOV is a different quantity

The earlier 14.36% checkpoint token OOV rate is not the fraction of new place names. Here the gold-entity slices are:

| Gold normalized entity slice | Gold support | Correct | Recall |
|---|---:|---:|---:|
| All overlapping tokens are in pilot vocabulary | {oov[0]['gold']} | {oov[0]['tp']} | {oov[0]['recall']:.3f} |
| At least one overlapping token is UNK | {oov[1]['gold']} | {oov[1]['tp']} | {oov[1]['recall']:.3f} |

Gold supports use one normalized entity per text; repeated occurrences share the slice if any occurrence contains UNK. A surface can have appeared as LOC in training while a low-frequency component was dropped by the 30,000-word vocabulary cap. Conversely, a new multiword name can consist entirely of known tokens. These are associated failure factors, not a causal decomposition of every miss.

### Normalization sensitivity

The published regex can alter Latin text and retain unexpected punctuation. As a diagnostic, the same analysis was recomputed using conservative NFC + case-folding + whitespace collapse. Overall F1 becomes {sensitive['overall']['f1']:.4f}; seen/unseen recall is {sensitive['novelty']['seen']['recall']:.3f}/{sensitive['novelty']['unseen']['recall']:.3f}. The qualitative novelty gap persists. The conservative metric is **not substituted** for the official one. Full values are in `checkpoint_normalization_sensitivity.json`.

## 2. False positives and misses: full-census taxonomy

These categories cover all **{total} official error events**, not just the selected 200 cases. They are assigned from actual span overlaps, saved vocabulary membership and auxiliary annotations, in this precedence: LOC overlap first, then PER overlap, then ORG overlap, then other FP. For non-overlap FN, UNK is checked first, then unseen/seen surface. Multi-cause errors therefore appear once in the table; counts are not independent causal contributions.

| Error | Observed category | Count | Fraction within FP or FN |
|---|---|---:|---:|
{catrows}

Important interpretations:

- **{fpover} FP and {fnover} FN overlap an annotated/predicted location.** They are boundary-related candidates. The remaining {o['fp']-fpover} FP and {o['fn']-fnover} FN have no such overlap. Even perfect handling of the observed boundary cases would leave substantial detection/type errors; no counterfactual improved F1 is claimed.
- 126 FP overlap ORG annotations and 52 overlap PER. This does **not** mean all 178 are unambiguous entity-type confusions. Country names embedded in an institution may be sensible geographic substrings that the source's non-nested annotation policy excludes. Auxiliary ORG/PER spans also contain suspected noise.
- The 622 other FP are not supported by LOC gold, but should not all be called hallucinations. Context review finds both obvious ordinary-word errors and plausible unannotated locations.
- Missing with UNK, new surface and known surface are observed properties. A high class weight, decoder boundary, context, or annotation defect can coexist with any of them.

## 3. Character-boundary diagnosis

Exact span matches are removed first. Remaining gold/prediction occurrences form a character-overlap graph. One-to-one assignment maximizes overlap-match cardinality and then intersection-over-union; one prediction cannot repair multiple missed entities in the diagnostic count.

Strict occurrence totals: {r['strict_span_totals']['exact_matches']} exact matches, {r['strict_span_totals']['unmatched_pred']} unmatched predicted spans and {r['strict_span_totals']['unmatched_gold']} unmatched gold spans; exact-span F1={r['strict_span_score']['f1']:.4f}. There are {r['strict_span_totals']['pred_with_no_gold_overlap']} unmatched predictions with no gold overlap and {r['strict_span_totals']['gold_with_no_pred_overlap']} unmatched gold mentions with no prediction overlap.

| One-to-one unmatched overlap pair | Count |
|---|---:|
{pairrows}

**Multiword truncation and neighboring-word capture dominate the paired boundary errors.** There is only one ending-fragment pair in this pilot. This does not show that morphology is unimportant overall: inflection also appears through OOV and new surfaces. It shows that a suffix-trimming rule would address little of the measured boundary subset.

| Overlap component topology | Components |
|---|---:|
{toporows}

A split/merge component can contain several error events. Component counts, overlap pairs and official FP/FN therefore have different units. Punctuation/whitespace mismatches sometimes reflect gold including a quote/comma. Splits can reflect inconsistent gold granularity: `Івано-Франківську` is sometimes one entity and sometimes two.

## 4. Contextual review of 200 errors

The supplement contains **200 focal errors from 200 distinct checkpoint texts: 100 FP and 100 FN**, with {selected_boundary} boundary-overlap focal cases. Sampling deliberately balances kinds and emphasizes boundary coverage (seed 2026); selection is not representative of prevalence. Full-census percentages above are the source of quantitative conclusions.

All 200 contexts were read by Codex and given a contextual note. This is **AI-assisted review, not independent expert adjudication**. Focal labels remain errors relative to the unchanged original LOC gold. Suspected label defects or policy ambiguities are flagged for human checking; none were automatically corrected, removed from checkpoint scoring, or copied into fit.

| Case ID | Focal error | Surface | Review issue |
|---|---|---|---|
{table}

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
'''
    (BASE/'reports/CHECKPOINT_ERROR_ANALYSIS.md').write_text(report,encoding='utf-8')
    nb=nbf.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
    nb.cells=[nbf.v4.new_markdown_cell('''# Frozen checkpoint: errors, novelty, boundaries

4000 checkpoint texts; same saved 12k-fit pilot. No retraining, calibration or final audit scoring. Counts from the full checkpoint and the 200-case diagnostic review are kept separate. Contextual notes are AI-assisted hypotheses, not corrected ground truth.'''),
        nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
BASE = Path.cwd() if (Path.cwd() / 'reports/checkpoint_error_analysis.json').exists() else Path.cwd() / 'assignment3'
BASE = BASE.resolve()
result = json.loads((BASE / 'reports/checkpoint_error_analysis.json').read_text(encoding='utf-8'))
display(pd.read_csv(BASE / 'reports/checkpoint_novelty_metrics.csv'))
display(pd.read_csv(BASE / 'reports/checkpoint_gold_OOV_recall.csv'))"""),
        nbf.v4.new_code_cell("display(Image(filename=str(BASE / 'figures/checkpoint_error_analysis.png')))"),
        nbf.v4.new_code_cell("display(pd.DataFrame(result['official_error_categories']))\ndisplay(pd.Series(result['overlap_pair_relations']))"),
        nbf.v4.new_code_cell("""review = pd.read_csv(BASE / 'reports/checkpoint_review_200_annotated.csv')
assert len(review) == 200 and review.row_id.nunique() == 200
display(review[['case_id', 'kind', 'surface', 'seen_in_pilot', 'contains_UNK', 'context_review_tag', 'context_review_note']].head(20))
print('Full 200-case contextual notes are in reports/CHECKPOINT_REVIEW_200.md')"""),
        nbf.v4.new_markdown_cell('## Interpretation\n\nNew surface is defined relative to LOC labels in the actual 12k pilot fit, not the whole million-row corpus. OOV is different. Boundary overlaps are diagnostic, not proof that a decoder fix would resolve every error. See the full report for label-policy ambiguities and safeguards.')]
    nbf.validate(nb);nbf.write(nb,BASE/'checkpoint_error_analysis.ipynb')
    print('Built checkpoint error report and notebook; no model changes')


if __name__=='__main__':main()
