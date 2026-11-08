"""Render the measured validation protocol and adversarial results."""
import base64
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

BASE=Path(__file__).resolve().parent


def main():
    split=json.loads((BASE/'reports/validation_splits_summary.json').read_text())
    av=json.loads((BASE/'reports/adversarial_results.json').read_text())
    eda=json.loads((BASE/'reports/eda_summary.json').read_text(encoding='utf-8'))
    df=pd.DataFrame(split['partitions'])
    order=['fit','checkpoint','calibration','audit','excluded_overlap']
    df=df.set_index('partition').loc[order].reset_index()
    rows='\n'.join(f"| {r.partition} | {r.rows:,} | {r.groups:,} | {r.loc_mentions:,} | {r.positive_rows:,} | {100*r.no_location_fraction:.2f}% |" for r in df.itertuples())
    folds=pd.DataFrame(av['folds'])
    fold_rows='\n'.join(f"| {r.fold} | {r.train_rows} | {r.heldout_rows} | {r.auc:.4f} | {100*r.oov_token_fraction:.2f}% |" for r in folds.itertuples())
    ci=av['bootstrap_auc_95ci']
    report=f'''# Assignment 3 — Validation Strategy

## 1. Decision and motivation

Use a **frozen, group-aware source holdout with separate checkpoint, calibration and audit partitions**, complemented by the provided 26 labeled in-domain examples as a small external diagnostic. Use the same split file for GRU, LSTM, word/BPE, pretrained embeddings and ensembles. Compare the competition's published surface-set micro-F1, precision and recall, with exact span F1 and grouped uncertainty when offsets are available.

This design is motivated by measured problems, not a generic random 80/20 rule:

- There are 52 canonical duplicate groups shared by the supplied fit and validation sets. All 52 overlapping fit rows are excluded from fitting; their validation counterparts stay in validation. They are 0.52% of the official validation rows, so the leakage is real but its score effect is not established.
- Median source and test lengths differ: 91 versus 264 characters and 13 versus 36 whitespace tokens. The source has no newlines, while 85.12% of test posts contain them. Astral Unicode appears in 0% versus 53.25% of texts. The source-domain holdout alone is therefore unlikely to be a fully representative leaderboard proxy.
- 76.89% of source texts contain no LOC. Stratifying by entity count and length stabilizes the amount of positive evidence in each holdout without changing the distribution to match the uncertain 26-example sample.
- Test labels are hidden; its empty `locations` are placeholders. No test labels are created, inferred from placeholders or manually annotated.

## 2. Reproducible split construction

`validation.py` assigns every original Ukrainian CSV row a stable `row_id` (zero-based file row, excluding the header). The grouping key is SHA-256 of NFC-normalized, case-folded, whitespace-collapsed text. All equivalent texts must stay together. This catches exact/case/whitespace duplicates, **not all semantic or near duplicates**.

Preserve the official `is_valid=0/1` separation. Reserve all official validation groups, all provided labeled-sample groups and all unlabeled test groups from fit. The latter two checks currently remove no additional rows because EDA found no such overlaps. Partition only `is_valid=1` with 5-fold `StratifiedGroupKFold`, shuffle enabled and seed 42, stratified by LOC count (`0`, `1`, `2+`) and whitespace length (`≤15`, `16–30`, `>30`). Folds 0–1 become checkpoint, fold 2 calibration, folds 3–4 audit.

| Partition | Texts | Canonical groups | LOC mentions | LOC-positive texts | No-LOC texts |
|---|---:|---:|---:|---:|---:|
{rows}

The script asserts **zero canonical groups crossing active partitions**, full row coverage and unique row IDs. Repeated rows within fit remain permitted; no validation example is used to fit a vocabulary, embedding adapter, BPE tokenizer or neural weight.

The full assignment is saved in `data/validation_splits.csv` (excluded from Git with other data). Its SHA-256 is:

```text
{split['manifest_sha256']}
```

Small reviewable partition counts and hashes are in `reports/validation_splits_summary.json` and `reports/validation_partitions.csv`. Record the split hash with every experiment. Fixing a random seed does not guarantee bitwise-identical GPU training across devices or software versions; it does fix this split procedure.

### Roles of the three holdouts

1. **Checkpoint (4,000 texts):** choose the epoch/early stopping and compare architectures under a fixed decoding protocol. Report these as development holdout results; repeated selection makes them optimistic as a final estimate.
2. **Calibration (2,000 texts):** after selecting a checkpoint, choose a small prespecified threshold/merging grid using the official normalizer. Keep the budget small, save the selected settings and freeze them. The score optimized on this partition is a tuning score, not an unbiased final score.
3. **Audit (4,000 texts):** evaluate the frozen model, tokenizer, thresholds and merger once, after the experiment decision is locked. Do not select ensemble weights, thresholds or epochs from audit results. This is the clean source-domain final estimate, not proof of target-domain performance.

These are three **holdouts**, not three-fold NER CV. The five grouping folds are only a deterministic way to construct them. If actual NER cross-validation is subsequently run, train the NER model separately for every fold and call only those results CV. Ensembling uses models whose out-of-sample predictions are aligned to the same splits; avoid fitting ensemble weights on final audit predictions. All hyperparameter variants use identical partition IDs.

The 26 provided labeled in-domain examples contain only 32 mentions and ten negative texts. Keep them external to training and threshold tuning. They may diagnose missed addresses, morphology and long-post boundaries after development choices are frozen; repeated inspection and model changes would turn them into another development set. Do not describe their score as a stable target-domain estimate or tune a negative rate from them.

### Group and time factors

The Ukrainian corpus lacks document IDs and timestamps. A true document-disjoint or chronological validation cannot be justified from the available metadata; row order is not assumed to represent time. Canonical grouping is a defensible minimum. Before major experiments, audit near duplicates without using test target labels. If Russian auxiliary data are later added, group by `(corpus, doc_id)` so sentences from one document do not cross fit/holdout, and audit cross-corpus duplicates as well. Do not reuse the Ukrainian row-level split rule blindly for that corpus.

## 3. Scoring and uncertainty

The primary local score reproduces the **published competition code**, including its literal-regex normalizer. `metrics.py` already matches it on 500 randomized examples. The hidden Kaggle scorer has not been inspected. If the host clarifies or changes normalization, freeze a new metric version and rescore all compared experiments consistently.

Score decoded original-text substrings, aligned by row ID. Merge overlapping windows back into original offsets before scoring; do not score duplicate chunks as independent documents. Do not silently truncate long texts in NER evaluation. Maintain offsets through tokenization, Unicode, punctuation and emoji handling. The recurrent adversarial experiment below has its own explicitly limited 128-token prefix representation.

`evaluate_validation.py` reads prediction CSVs with `row_id,locations`, requires exact holdout coverage and unique IDs, and reports micro-F1, precision/recall, macro per-text F1, positive-text macro F1, exact-set accuracy and the false-positive text rate on gold-empty texts. Its 2,000-replicate bootstrap resamples **canonical groups**, carrying TP/FP/FN together, then recomputes micro-F1. It never averages per-text F1 to approximate micro-F1.

The interval is conditional on one trained model and this source-domain sample. It excludes retraining variation and domain mismatch. For close model comparisons, use paired group resampling of both models' predictions on the same texts and train several prespecified seeds; do not rank models from overlapping point estimates alone. A separate short NER pilot is documented in `NER_PREFLIGHT_AND_PILOT.md`; this strategy report does not present its development score as a final audit or leaderboard estimate.

Example commands, after real NER predictions exist:

```powershell
.venv/Scripts/python.exe assignment3/evaluate_validation.py checkpoint_predictions.csv --partition checkpoint
.venv/Scripts/python.exe assignment3/evaluate_validation.py calibration_predictions.csv --partition calibration
.venv/Scripts/python.exe assignment3/evaluate_validation.py audit_predictions.csv --partition audit --final-audit
```

The final flag is a protection against accidentally evaluating audit while tuning, not an access or permission requirement. A calibration-only scoring utility cannot ensure that the researcher has respected the model-selection protocol; the experiment ledger must document that decision.

## 4. Adversarial validation: performed with a recurrent encoder

Adversarial validation is justified by the observed source/test shift. Train a binary model to predict **file origin**, not NER labels: `0 = Ukrainian fit corpus`, `1 = unlabeled competition test`. Successful classification means the input distributions are distinguishable in this representation.

### Model consistency with the main task

`recurrent_model.py` defines a shared `RecurrentEncoder`: a 64-dimensional learned embedding, one bidirectional GRU layer with hidden size 64 per direction, and dropout 0.2. `NERModel` adds a per-token BIO classifier; `DomainModel` uses the **same encoder implementation and configuration** followed by masked mean/max pooling and a binary head. Padding is excluded by packed recurrent sequences and masked pooling. No transformer or self-attention is used.

The same NER head/encoder is used in a separate **12,000-text pilot**, documented in `NER_PREFLIGHT_AND_PILOT.md`; a final full-scale model is not fixed. Adversarial tokenization was rerun using the same offset-preserving word/punctuation/newline tokenizer and per-token normalization as that pilot. The task-specific head must differ because domain discrimination is a binary document task. If the final main model instead uses LSTM, another tokenizer/embedding initialization, or a different recurrent configuration, **rerun this experiment with that final encoder/preprocessing** to retain the assignment's same-model condition. Sharing only the broad label “RNN” would not be enough.

### Experimental protocol

- Uniformly sample one fit text per sampled canonical group: {av['source_rows']} unique source texts; use all {av['test_rows']} unique unlabeled test texts. Class balance prevents accuracy being dominated by corpus size. The 26 labeled examples and all three NER holdouts are excluded from adversarial fitting.
- Three stratified, group-disjoint folds; each held-out text receives a prediction from a model that did not train on it. Per fold, {av['folds'][0]['train_rows']} texts train the discriminator and {av['folds'][0]['heldout_rows']} are held out.
- Tokenization retains newlines and punctuation and uses NFC + case folding. Each text is represented by its first 128 lexical/punctuation/newline tokens. Vocabulary is learned on the **domain-training fold only**, capped at 12,000; unseen tokens map to `UNK`.
- AdamW, learning rate 0.002, weight decay 0.01, batch 64, ten fixed epochs, gradient clipping 1.0. No held-out epoch selection or hyperparameter search. Held-out curves are diagnostic; the final epoch is always used.
- Pool all 954 held-out predictions. Use ROC-AUC because the question is domain separability; 0.5 is chance discrimination. Bootstrap source and test texts separately 2,000 times for a conditional 95% interval.

| Domain fold | Training texts | Held-out texts | ROC-AUC | Held-out OOV token fraction |
|---|---:|---:|---:|---:|
{fold_rows}

**Pooled OOF ROC-AUC: {av['oof_roc_auc']:.4f}; 95% conditional interval [{ci[0]:.4f}, {ci[1]:.4f}].** OOF accuracy at threshold 0.5 is {100*av['oof_accuracy']:.2f}%; binary cross-entropy is {av['oof_bce']:.4f}. These are **domain-classification results, not NER or leaderboard scores**. The interval conditions on these three fitted models and this sampled source set; it does not include new random seeds, source resampling or retraining uncertainty. Vocabulary novelty is also part of the measured shift.

![Adversarial validation](../figures/adversarial_validation.png)

The AUC clearly exceeds chance: the recurrent model distinguishes source from test. This supports using source holdouts as controlled development checks while qualifying their expected leaderboard correlation. It does not measure the direction or size of a NER F1 gap, label-distribution shift, entity-boundary quality, or causal reasons for shift. An easy formatting cue can yield high domain AUC even when NER transfers well. Conversely, an AUC near 0.5 would only mean this discriminator did not detect shift; it would not prove distributions equal.

Do **not** automatically importance-weight source examples from these probabilities. The data show weak support overlap for features such as emoji and multiline posts, and the discriminator is learned from only 477 source examples. Reweighting cannot create missing target-domain patterns and can be unstable. First preserve negative examples, handle full long posts, inspect errors on permitted labeled data, and compare repeatable source improvements with the small external diagnostic.

## 5. Expected correlation with Public leaderboard

No empirical correlation can be computed yet: there are no paired NER/Public-LB results. A short NER pilot exists separately, but it has not been submitted to the leaderboard. The public subset size and its label distribution are unknown. The adversarial AUC motivates expecting **imperfect source-holdout-to-Public-LB agreement**, not asserting a measured low correlation.

Reasons for **higher** rank agreement include common task/annotation conventions, the identical scoring normalizer, consistent original-text decoding, larger genuine improvements across seeds, correctly grouped holdouts, and gains that generalize across length/novelty subsets and the external diagnostic. A model that genuinely reduces both false positives and missed entities may improve on both domains even when absolute scores differ.

Reasons for **lower or reversed** agreement include:

- Source sentences versus longer multiline/emoji-containing posts, mixed scripts and differing context lengths; truncation/window-merging errors may be almost absent locally.
- Different location density, addresses, unseen or inflected names and annotation conventions. The 26-example negative fraction is too uncertain to substitute for test prevalence.
- Duplicate or document leakage inflating local scores; this split fixes observed canonical overlap, while near-duplicate leakage remains a limitation.
- Overfitting checkpoint/calibration or the repeatedly inspected 26-example set; tuning too many thresholds or ensemble weights on a small holdout.
- Public-subset sampling noise, low entity support, leaderboard rounding and different seed sensitivity. A small score gap need not be a real improvement.
- A hidden normalizer differing from the published literal regex, or inconsistent list serialization/row alignment. Confirm scoring semantics before attributing a discrepancy entirely to the model.
- Selection bias from repeatedly choosing the best Public-LB submission. Public LB becomes another tuned development set and can disagree with the private final subset.

Keep `reports/experiment_ledger.csv` with experiment ID, actual model/tokenizer, seed, split hash, checkpoint precision/recall/F1, calibration parameters, final audit F1 where permitted, and observed Public-LB F1. Empty fields mean **not measured**, not zero. Record submitted configuration and public score once available. Do not retrain or choose thresholds from final audit results.

After enough genuinely different prespecified configurations have both local and public scores, compare paired local/public changes and Spearman rank correlation (Pearson only as an additional measure of linear score changes). State the number of paired experiments, confidence/uncertainty and the common test subset; five near-identical runs do not establish a dependable relationship. Seed replicas are not independent architecture comparisons. Never fabricate LB values or infer them from adversarial AUC.

## 6. What is completed and what remains for main-model experiments

**Completed:** fixed partition file and hash, observed-leakage removal, support counts and cross-partition assertions, a recurrent adversarial experiment with genuine out-of-fold predictions, optimization history and ROC plot, scoring/uncertainty utilities, and a ledger schema. Tests cover padding invariance, shared encoder shapes, additive count scoring and group-bootstrap behavior.

**For full-scale NER experiments:** train on fit only, choose checkpoints on checkpoint, tune limited decoding settings on calibration, freeze choices, run the audit evaluation, and record actual public submission results. A fit/checkpoint pilot is documented separately; final audit quality and local/LB correlation remain unmeasured. If the final recurrent encoder differs from this BiGRU baseline, repeat adversarial validation with it.

## Reproduction and provenance

```powershell
.venv/Scripts/python.exe assignment3/validation.py
.venv/Scripts/python.exe assignment3/build_validation_report.py
.venv/Scripts/python.exe -m unittest discover -s assignment3 -p "test_*.py"
```

`validation.py --splits-only` regenerates only the manifest. Source CSV hashes are preserved in the split summary and earlier EDA. Source requirements and official metric are archived in `sources/`; the earlier EDA report contains the data/metric analysis. CUDA was used for this run. Generated reports and logs contain no Kaggle credentials.

AI assistance: Codex generated and ran the validation code and drafted this report from measured outputs. Include the actual chat transcript/link in the assignment's AI disclosure; review and understand the protocol before defense.
'''
    (BASE/'reports/VALIDATION_STRATEGY.md').write_text(report,encoding='utf-8')
    nb=nbf.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
    nb.cells=[nbf.v4.new_markdown_cell('''# Validation strategy and recurrent adversarial validation

See `reports/VALIDATION_STRATEGY.md` for the frozen protocol and limitations. These are real domain-discrimination results; a separate NER pilot is documented in `NER_PREFLIGHT_AND_PILOT.md`. Public-LB correlations are not measured. The same BiGRU encoder/tokenizer is used by the pilot; rerun adversarial validation if the final main encoder/tokenizer changes.''')]
    nb.cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
BASE = Path.cwd() if (Path.cwd() / 'reports/validation_splits_summary.json').exists() else Path.cwd() / 'assignment3'
BASE = BASE.resolve()
partitions = pd.read_csv(BASE / 'reports/validation_partitions.csv')
display(partitions)"""))
    nb.cells.append(nbf.v4.new_markdown_cell('## Recurrent OOF domain discrimination\n\nDomain label 0 is source; 1 is unlabeled test. Vocabulary is fit inside each domain-training fold. Ten fixed epochs, no checkpoint selection on held-out AUC.'))
    nb.cells.append(nbf.v4.new_code_cell("""result = json.loads((BASE / 'reports/adversarial_results.json').read_text())
display(pd.DataFrame(result['folds']))
print('Pooled OOF AUC:', round(result['oof_roc_auc'], 4))
print('95% conditional bootstrap interval:', result['bootstrap_auc_95ci'])
print(result['bootstrap_caveat'])"""))
    nb.cells.append(nbf.v4.new_code_cell("display(Image(filename=str(BASE / 'figures/adversarial_validation.png')))"))
    nb.cells.append(nbf.v4.new_markdown_cell('## Optimization health\n\nFinite loss and gradients were checked during training. Held-out trajectories are diagnostic, not an epoch-selection criterion.'))
    nb.cells.append(nbf.v4.new_code_cell("history = pd.read_csv(BASE / 'reports/adversarial_history.csv')\ndisplay(history.groupby('fold').tail(1))"))
    nb.cells.append(nbf.v4.new_markdown_cell('## Reproduce (optional)\n\nThe full manifest is in `data/validation_splits.csv`. This recomputes the split and retrains only the adversarial models; it does not train or score NER.'))
    nb.cells.append(nbf.v4.new_code_cell("# import subprocess, sys\n# subprocess.run([sys.executable, str(BASE / 'validation.py')], check=True)"))
    nbf.validate(nb);nbf.write(nb,BASE/'validation.ipynb')
    print('Built validation report and notebook')


if __name__=='__main__':main()
