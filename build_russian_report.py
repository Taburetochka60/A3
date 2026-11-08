"""Assemble full-corpus findings and the three-seed auxiliary-data comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FuncFormatter
import nbformat as nbf
import numpy as np
import pandas as pd


def weighted_quantile(hist,q):
    counts=sorted((int(k),int(v)) for k,v in hist.items());target=q*sum(v for _,v in counts);acc=0
    for value,n in counts:
        acc+=n
        if acc>=target:return value
    return counts[-1][0]


def mean_hist(hist):return sum(int(k)*int(v) for k,v in hist.items())/sum(map(int,hist.values()))


SPECIAL = {
    4: ('metonymy', 'Tehran is used as a state actor. This common news metonymy needs a convention consistent with the target labels.'),
    5: ('possible_type_error', 'The European Commission is an institution here but is labeled LOC. This is a candidate ORG/LOC type error; gold is unchanged.'),
    6: ('metonymy', 'Vatican can denote the city-state or its governing institution. A single context does not define a universal rule.'),
    11: ('text_fusion', 'Two sentences are fused without a space after the period; the Petersburg LOC looks plausible.'),
    14: ('text_fusion', 'A sentence boundary lacks a space. Offsets are valid, but the row is not necessarily one clean sentence.'),
    28: ('union_policy', 'The European Union is labeled LOC. This may be the convention for country groupings and should be aligned with Ukrainian annotations.'),
    33: ('institution_geography_policy', 'Missouri is labeled LOC, while the university head is separately ORG. Geographic components are not absorbed into one full institution span here.'),
    37: ('union_policy', 'The EU is labeled LOC alongside the USA. Political-union conventions should be consistent across both corpora.'),
    38: ('hyphen_name', 'New York is preserved as one hyphenated LOC, contrasting with mixed whole/split treatment in the Ukrainian corpus.'),
    39: ('text_fusion', 'A period joins two sentences without a space; the Kyrgyzstan LOC appears consistent with context.'),
    44: ('text_fusion_and_aux', 'The news-agency name is fused with the preceding verb and is missing from ORG. The city and region LOCs look plausible.'),
    46: ('institution_geography_policy', 'Saint Petersburg is separated as LOC from the head of a religious-institution name, which matters for type transfer.'),
    51: ('elliptical_name', 'The second square has an elliptical name. The annotation contract should define whether such shortened names are included.'),
    57: ('institution_geography_policy', 'The USA is labeled LOC separately from the defense ministry head (ORG). Do not assume the Ukrainian nesting policy is identical.'),
    61: ('institution_geography_policy', 'Alaska and the USA are geographic components; an institution need not absorb all such words into its span.'),
    68: ('abbreviation_and_quote', 'The abbreviation needs document context for interpretation; the auxiliary ORG includes an opening quote. Not every label should be assumed perfect.'),
    69: ('address_scope', 'LOC includes the street name but not building number 41. Ukrainian address spans sometimes include numbers, so scope conventions differ.'),
    73: ('nonlocation_name', 'The movement name remains ORG, while Samara is LOC. This is a useful negative example of a geographic word inside another entity name.'),
    87: ('text_fusion', 'Two statements are fused without a space. There is no LOC; the generic region phrase is not a named place.'),
    93: ('auxiliary_span_quality', 'PER includes an opening bracket after the name. LOC is absent, but auxiliary annotation quality also needs care.'),
}


def semantic_review(root):
    path=root/'reports/ru_annotation_review_sample.json'
    cases=json.loads(path.read_text(encoding='utf-8'))
    fingerprint=hashlib.sha256(json.dumps([e['row_id'] for e in cases]).encode()).hexdigest()
    assert fingerprint=='d02cffdb6620ec702db2571cc8a36a61cb941744303ec44153c7d866d6563452'
    for i,e in enumerate(cases,1):
        tag,note=SPECIAL.get(i,('context_plausible_LOC','The geographic labels look consistent with this context. This is AI review, not independent gold annotation.') if e['LOC'] else ('context_plausible_negative','The context does not clearly require a named location; the absence of LOC looks plausible.'))
        e.update(case_id=i,review_tag=tag,review_note=note,reviewer='Codex, AI-assisted contextual review',gold_changed=False)
    (root/'reports/ru_annotation_review_100.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in e.items()} for e in cases]).to_csv(root/'reports/ru_annotation_review_100.csv',index=False,encoding='utf-8-sig')
    lines=['# Contextual review of 100 Russian examples','',
           '80 LOC-positive and 20 LOC-negative texts: separate full-file reservoir samples, seed 42. This ratio does not reflect positive prevalence. Codex performed the review, not independent experts; global semantic accuracy was not estimated. Gold is unchanged. Original NER text and entity strings are preserved.','']
    for e in cases:
        lines.extend([f"## {e['case_id']:03d} · row_id={e['row_id']} · doc_id={e['doc_id']} · {e['review_tag']}",'',e['text'],'',
                      '**LOC:** '+json.dumps(e['LOC'],ensure_ascii=False),'',
                      '**ORG:** '+json.dumps(e['ORG'],ensure_ascii=False),'',
                      '**PER:** '+json.dumps(e['PER'],ensure_ascii=False),'','**Review:** '+e['review_note'],''])
    (root/'reports/RU_ANNOTATION_REVIEW_100.md').write_text('\n'.join(lines),encoding='utf-8')
    return cases



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
    root=args.project_root.resolve();reports=root/'reports'
    uk=json.loads((reports/'uk_full_structural_audit.json').read_text())
    ru=json.loads((reports/'ru_full_structural_audit.json').read_text())
    split=json.loads((reports/'ru_corpus_split_audit.json').read_text(encoding='utf-8'))
    exp=json.loads((reports/'ru_augmentation_experiment.json').read_text())
    cases=semantic_review(root);uc,rc=uk['counts'],ru['counts']
    runs=pd.DataFrame(exp['runs']);summary=pd.read_csv(reports/'ru_augmentation_summary.csv')
    comparison=exp['comparisons'];delta_base=comparison[0];delta_control=comparison[2]
    if delta_base['mean_delta']<0 and delta_control['mean_delta']<0:
        decision='The 1:1 Russian mixture is not recommended as the default for this random-embedding word-BiGRU pilot: its average Ukrainian checkpoint result is below both controls.'
    elif delta_base['mean_delta']>0 and delta_control['mean_delta']<0:
        decision='Adding Russian data improves this pilot over the smaller Ukrainian baseline on average, but extra Ukrainian data is the stronger matched-update control. This does not establish a specific advantage for Russian augmentation.'
    elif delta_base['mean_delta']>0 and delta_control['mean_delta']>0:
        decision='The Russian mixture has the highest mean Ukrainian development score in this pilot. Check the paired intervals and seed signs before claiming a reliable gain; no Public LB advantage is established.'
    else:
        decision='The pilot does not give a clear consistent advantage for Russian augmentation; use seed-level deltas and matched-control comparisons rather than a single best run.'
    datarows=[]
    for name,d in [('Ukrainian',uk),('Russian',ru)]:
        c=d['counts'];n=c['rows']
        datarows.append(f"| {name} | {n:,} | {c['LOC_mentions']:,} | {100*(1-c['positive_rows']/n):.2f}% | {weighted_quantile(d['character_hist'],.5)} | {weighted_quantile(d['token_hist'],.5)} | {weighted_quantile(d['token_hist'],.95)} |")
    duprows='\n'.join(f"| {name} | {split[name]['exact_duplicate_groups']:,} | {split[name]['exact_duplicate_excess']:,} | {split[name]['canon_duplicate_groups']:,} | {split[name]['canon_duplicate_excess']:,} |" for name in ['uk','ru'])
    partrows='\n'.join(f"| {d['partition']} | {d['rows']:,} | {d['documents']:,} | {int(d['LOC_mentions']):,} |" for d in split['partitions'])
    runrows='\n'.join(f"| {r.seed} | {r.arm} | {r.best_epoch} | {r.checkpoint_f1:.4f} | {r.checkpoint_precision:.4f} | {r.checkpoint_recall:.4f} | {r.ru_diagnostic_f1:.4f} | {100*r.checkpoint_OOV:.2f}% |" for r in runs.itertuples())
    sumrows='\n'.join(f"| {r.arm} | {r.checkpoint_mean:.4f} | {r.checkpoint_sd:.4f} | {r.ru_mean:.4f} | {100*r.oov_mean:.2f}% |" for r in summary.itertuples())
    comprows='\n'.join(f"| {d['treatment']} − {d['reference']} | {d['mean_delta']:+.4f} | [{d['conditional_paired_document_95ci'][0]:+.4f}, {d['conditional_paired_document_95ci'][1]:+.4f}] | {', '.join(f'{x:+.4f}' for x in d['seed_deltas'])} |" for d in comparison)
    quality=['LOC_parse_failure','LOC_noninteger_or_shape','LOC_out_of_bounds','LOC_overlap','LOC_duplicate_spans','LOC_edge_whitespace','LOC_partial_token','LOC_ORG_overlaps','LOC_PER_overlaps','missing_doc_id','missing_sent_id']
    qualityrows='\n'.join(f"| {k} | {rc.get(k,0):,} |" for k in quality)
    entity=split['entity_surfaces'];shared=split['shared_conservative_LOC_surfaces']
    report=f'''# Russian auxiliary corpus: full audit and augmentation experiment

## Decision

{decision}

This covers the **entire downloaded Russian CSV**, not an extrapolation from a sample. Structural quality, duplicates, cross-corpus overlap and document-safe partitions were checked globally. Semantic review covers 100 sampled contexts. The training experiment is deliberately a small controlled pilot, not full training on eight million sentences or a Kaggle submission.

## 1. Provenance and task compatibility

The competition README identifies `ru_geo_dataset.csv` as derived from [Nerus](https://github.com/natasha/nerus). Its authors describe a silver-standard news corpus with automatically generated entity annotations, rather than independent human gold. Therefore valid offsets do not establish semantic truth. The competition CSV conversion is assessed here directly; upstream quality claims are not treated as measurements of this file.

The supplied file is {ru['source_bytes']:,} bytes, with **{rc['rows']:,} rows and {split['RU_documents']:,} documents**. Fields are text, LOC/ORG/PER character markers, `doc_id` and `sent_id`. We train only the LOC target using the existing recurrent backbone. Source labels being machine-generated does not introduce a transformer into our training architecture.

Russian CSV SHA-256: `{ru['source_sha256']}`. Ukrainian CSV SHA-256: `{uk['source_sha256']}`. The original Ukrainian manifest is unchanged.

## 2. Full-corpus distribution

| Corpus | Sentences | LOC mentions | No-LOC sentences | Median characters | Median tokens | P95 tokens |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(datarows)}

Tokens use the same original-offset word/punctuation/newline expression as the NER pilot. They differ from EDA whitespace counts. Empty Russian text rows: {rc['empty_text_rows']}. Russian source sentences with newlines: {rc['has_newline']}. Script-presence indicators are descriptive, not ground-truth language labels; Russian-specific letters appear in {100*rc['has_ru_specific']/rc['rows']:.2f}% of RU sentences, Latin letters in {100*rc['has_latin']/rc['rows']:.2f}%.

Russian LOC surface vocabulary (NFC/case-fold/whitespace only) has {entity['ru']['distinct_surfaces']:,} distinct strings, versus {entity['uk']['distinct_surfaces']:,} Ukrainian. Shared strings: **{shared:,}**, or {100*shared/entity['uk']['distinct_surfaces']:.2f}% of the Ukrainian vocabulary. This is literal surface overlap, not geographic identity overlap; translations and inflected forms remain distinct. Country/city coverage alone does not align independently learned embeddings across languages.

The Russian corpus contains relatively more LOC-positive sentences. Mixing it changes language, genre, vocabulary competition and positive-label density together. No experiment below isolates pure language causality.

## 3. LOC annotation quality and agreement

| Full Russian audit check | Findings |
|---|---:|
{qualityrows}

**{rc['rows']-rc['bio_eligible_rows']:,} rows ({100*(rc['rows']-rc['bio_eligible_rows'])/rc['rows']:.4f}%) are incompatible with this whole-word BIO representation.** Partial-word markers are not automatically expanded. Annotation edge whitespace is trimmed only for BIO targets and counted; original gold strings stay unchanged for scoring. Auxiliary type overlap flags may indicate nesting, conversion or annotation ambiguity; they are not automatically relabeled.

Within identical canonical texts, **{split['canonical_duplicate_groups_with_LOC_surface_conflicts']:,} groups disagree on conservative-normalized LOC surface sets**. This is an observable consistency problem, not proof of which annotation is right. The check does not identify every wrong occurrence when the same surface is repeated, and does not replace semantic adjudication. Such conflicting groups are excluded from the auxiliary fitting pool rather than resolved arbitrarily.

### Context review

We reviewed 100 full-file reservoir-sampled contexts: 80 positive and 20 negative, seed 42. The sampling is stratified for diagnosis, so its 80/20 ratio is not a prevalence estimate. This is AI-assisted contextual review, not independent expert annotation and not a semantic accuracy benchmark.

The reviewed geographic labels are mostly plausible in context, but important transfer issues remain:

- Case 5 labels the European Commission as LOC, a candidate ORG/LOC type error.
- Cases 33, 46 and 57 extract a geographic component separately from an institution head. The Ukrainian examples do not always follow the same policy.
- Case 38 preserves a hyphenated New York as one LOC; Ukrainian examples show mixed whole/split treatment of hyphenated places.
- Case 69 includes a street name but not the house number; Ukrainian address scopes sometimes include numbers and punctuation.
- Several contexts contain fused sentence/source fragments. There are also imperfect auxiliary PER/ORG boundaries.

See `RU_ANNOTATION_REVIEW_100.md` and the CSV/JSON for each context and note. Labels were not manually rewritten. Agree on a shared LOC contract before using either corpus as unquestioned gold.

## 4. Duplicates and Ukrainian overlap

| Corpus | Exact duplicate groups | Exact excess rows | Canonical groups | Canonical excess rows |
|---|---:|---:|---:|---:|
{duprows}

Canonical means NFC + case folding + whitespace collapse. Excess rows are copies beyond one per group, not the number of rows belonging to duplicate groups. Full-file joins find **no exact or canonical full-text overlap between the supplied Russian and Ukrainian corpora**. The separate protected key set also includes the unlabeled test texts and provided 26-example texts; active RU pools have {split['protected_UA_matches_in_active_RU']} protected matches.

This does **not** prove absence of translated, paraphrased, fuzzy or shared-source news. No translation-equivalence or semantic near-duplicate search was performed. Shared names are expected and are not sentence leakage by themselves.

## 5. Document-safe Russian validation

Base assignment is deterministic: `BLAKE2b("42:ru:" + doc_id) mod 10 == 0` selects Russian validation, otherwise Russian fit. All sentences of a document have the same base assignment. `doc_id` is namespaced to the Russian corpus; equal numbers in other sources are not the same document.

Document grouping alone is insufficient because duplicate sentences occur in different documents. We reserve canonical keys from retained Russian validation. If a Russian fit document contains any reserved validation sentence, **the entire fit document is quarantined**. Any document matching protected Ukrainian/external texts is similarly quarantined. Thus simply changing sentence folds cannot reintroduce document context leakage.

Further fit-only filtering excludes incompatible BIO rows, conflicting canonical LOC groups, redundant Ukrainian-fit sentences and within-fit canonical copies. Russian validation is not cleaned to make its entity score easier. No Ukrainian calibration/final audit score is used.

| Russian partition / exclusion | Rows | Documents represented | LOC mentions |
|---|---:|---:|---:|
{partrows}

Document counts in exclusion rows can overlap across row-level reasons and must not be summed as disjoint documents. The aggressive cross-document quarantine removes many sentences; this is a real cost of strict leakage control and can alter the retained distribution. More than five million usable fit sentences remain.

Checks on active partitions: **0 shared document IDs, 0 shared canonical sentences between RU fit and RU valid, and 0 protected Ukrainian/external matches**. Duplicate `(doc_id, sent_id)` groups in the source: {split['duplicate_doc_sent_groups']}. Missing IDs are reported in the quality table.

The full row manifest is `data/russian_audit/ru_split_manifest.parquet`; SHA-256 `{split['split_sha256']}`. Large metadata, the on-disk DuckDB and spill files remain under ignored `data/`. Counts, hashes and small tables are under `reports/` for review.

## 6. Controlled experiment: does Russian data help?

The protocol was recorded in `ru_experiment_plan.json` before training results:

1. **UA_12k:** the same original 12,000 Ukrainian pilot groups.
2. **UA_12k_plus_RU_12k:** the same base plus 12,000 eligible, deduplicated, document-safe Russian sentences.
3. **UA_12k_plus_UA_12k:** the same base plus 12,000 additional eligible unique Ukrainian fit groups.

The two 24k arms match sentence counts and optimizer-update counts. Added sentences are at most 128 tokens so one row gives one training window. Actual sequence lengths and FLOPs can differ. The 12k baseline receives fewer updates, so an improvement over it alone would not establish that Russian text is better than more Ukrainian text.

All arms use the original embedding64/BiGRU64x2/BIO configuration, dropout .2, AdamW lr=.002, weight decay .01, batch64 and eight epochs. Seeds are 42,43,44. Vocabulary is fit on each arm's training texts only, capped at 30k words plus PAD/UNK. Loss weights are fixed from the base Ukrainian 12k in every arm: {exp['fixed_loss_weights']}. This avoids changing both data and loss weighting at once, although resulting label density still differs.

Best epoch is selected on the **same 4,000 Ukrainian checkpoint texts**. The Russian diagnostic contains a fixed 2,000 sentences from held-out RU documents and is evaluated only after selecting a checkpoint. Models, vocabularies and predictions are saved separately from the original pilot.

### Every run

| Seed | Arm | Best epoch | UA checkpoint F1 | UA precision | UA recall | RU diagnostic F1 | UA token OOV |
|---|---|---:|---:|---:|---:|---:|---:|
{runrows}

### Means across three seeds

| Arm | Mean UA F1 | Seed SD | Mean RU F1 | Mean UA OOV |
|---|---:|---:|---:|---:|
{sumrows}

### Paired comparisons on the Ukrainian checkpoint

| Contrast | Mean F1 delta | Conditional 95% document interval | Per-seed deltas |
|---|---:|---|---|
{comprows}

Intervals resample the same 4,000 checkpoint groups jointly across models and fixed seeds, 2,000 times. They condition on the fitted/selected models and do not include unseen-seed, checkpoint-selection or domain-shift uncertainty. Three seeds are informative but not a broad robustness study. These are **development holdout scores, not NER CV, final audit or Public LB results**.

![Russian addition experiment](../figures/ru_augmentation_experiment.png)

## 7. Interpretation and handoff

{decision}

Use the paired contrasts rather than cherry-picking a best seed or the Russian diagnostic alone. Higher RU diagnostic F1 means better source-domain Russian NER, not automatically better performance on mixed Ukrainian competition posts.

The result applies to **a 1:1 data mixture with random word embeddings and a capped shared vocabulary**. Adding Russian can crowd Ukrainian tokens out of the vocabulary; compare the measured UA OOV rates. It can also change type conventions and entity density. These are plausible mechanisms, not causally isolated explanations.

Do not generalize this pilot into “all Russian data is useful/useless.” A lower mixing ratio, multilingual/subword embeddings, a character BiGRU, or Russian pretraining followed by Ukrainian fine-tuning remain separate experiments. Do not launch all eight million rows blindly or tune the final audit based on these results.

For training with this source, use the saved RU split manifest, preserve document namespaces and original offsets, keep the Ukrainian split fixed, and retain corpus provenance in experiment logs. Never reuse sentences from the RU diagnostic or Ukrainian checkpoint as fit examples. Any cleaned annotation version needs an explicit version and fresh, honest comparisons.

## Reproduction and delivery

```powershell
python russian_corpus_audit.py
python russian_augmentation_experiment.py
python build_russian_report.py
python -m unittest discover -s . -p "test_*.py"
```

Download the supplied `ru_geo_dataset.csv` separately with Kaggle. The scan is chunked; DuckDB uses a 256 MB limit and disk spill. `--stage split` can rebuild splits from complete metadata; `--stage samples` freezes the experiment selection without rescanning both CSV files. The large data, metadata and weights are excluded from Git; push code, reports, figures, sources and notebooks.

Full-corpus JSONs, partition table, experiment plan, nine run records, histories, predictions and paired comparisons are retained. The original Ukrainian manifest and pilot artifacts are not overwritten. No external messages or Kaggle submissions were sent.

AI assistance: Codex implemented/executed the audit and experiment and reviewed the 100 contexts. Keep the actual chat transcript/link for the assignment disclosure; semantic notes are not independent human gold.
'''
    (reports/'RUSSIAN_CORPUS_ANALYSIS.md').write_text(report,encoding='utf-8')
    # Compact distribution figure from exact histograms.
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,data in [('Ukrainian',uk),('Russian',ru)]:
        hist=sorted((int(k),int(v)) for k,v in data['token_hist'].items());xs=np.array([x for x,n in hist]);ns=np.array([n for x,n in hist])
        axes[0].plot(xs,np.cumsum(ns)/ns.sum(),label=name)
    axes[0].set(xscale='symlog',xlabel='NER tokens per sentence',ylabel='Cumulative fraction',title='Full-file length distributions');axes[0].legend()
    axes[1].bar(['UA','RU'],[100*(1-uc['positive_rows']/uc['rows']),100*(1-rc['positive_rows']/rc['rows'])]);axes[1].set(title='No LOC in annotation',ylabel='% of sentences',ylim=(0,100))
    top=entity['ru']['top20'][:8]
    axes[2].barh([x['surface'] for x in top][::-1],[x['n'] for x in top][::-1]);axes[2].set(title='Most frequent Russian LOC surfaces',xlabel='Mentions')
    axes[2].xaxis.set_major_locator(MaxNLocator(4))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'0' if value==0 else f'{value/1000:g}k'))
    fig.savefig(root/'figures/ru_corpus_overview.png',dpi=170);plt.close(fig)
    nb=nbf.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
    nb.cells=[nbf.v4.new_markdown_cell('''# Russian auxiliary corpus: audit and experiment

Full-file structural and duplicate census; 100-context AI review; nine controlled recurrent runs. No calibration, final audit, or test-label scores. See `reports/RUSSIAN_CORPUS_ANALYSIS.md` for definitions, costs and limits.'''),
              nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
BASE = Path.cwd() if (Path.cwd() / 'reports/ru_corpus_split_audit.json').exists() else Path.cwd() / 'assignment3'
BASE = BASE.resolve()
audit = json.loads((BASE / 'reports/ru_full_structural_audit.json').read_text())
print('Full Russian rows:', audit['counts']['rows'])
display(pd.Series(audit['counts']))
display(pd.read_csv(BASE / 'reports/ru_validation_partitions.csv'))"""),
              nbf.v4.new_code_cell("display(Image(filename=str(BASE / 'figures/ru_corpus_overview.png')))"),
              nbf.v4.new_code_cell("display(pd.read_csv(BASE / 'reports/ru_augmentation_runs.csv'))\ndisplay(pd.read_csv(BASE / 'reports/ru_augmentation_summary.csv'))"),
              nbf.v4.new_code_cell("display(Image(filename=str(BASE / 'figures/ru_augmentation_experiment.png')))")]
    nbf.validate(nb);nbf.write(nb,root/'russian_corpus_analysis.ipynb')
    (root/'sources/NERUS_ORIGIN.md').write_text('''# Russian source provenance

Competition README: the Russian CSV is derived from Nerus.

The [upstream authors](https://github.com/natasha/nerus) describe automatically produced entity labels over news articles. Treat them as silver supervision rather than independently human-adjudicated gold. Their upstream evaluation is not a validation of the competition CSV conversion. The full structural census and contextual sample in this project assess the supplied file separately.

Checked 8 November 2026. No upstream model is used in this project's recurrent training pipeline.
''',encoding='utf-8')
    print('Built full Russian report, 100-case review, figures and notebook',flush=True)


if __name__=='__main__':main()
