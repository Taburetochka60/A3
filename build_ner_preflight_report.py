"""Create the measured BIO/padding/long-input preflight and pilot report."""
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

BASE=Path(__file__).resolve().parent


def main():
    audit=json.loads((BASE/'reports/ner_preprocessing_audit.json').read_text(encoding='utf-8'))
    pilot=json.loads((BASE/'reports/ner_pilot_results.json').read_text())
    long=json.loads((BASE/'reports/ner_long_text_checks.json').read_text())
    av=json.loads((BASE/'reports/adversarial_results.json').read_text())
    fit=audit['by_partition']['fit'];checkpoint=audit['by_partition']['checkpoint']
    train,val=pilot['best_train_monitor'],pilot['best_checkpoint']
    history=pd.read_csv(BASE/'reports/ner_pilot_history.csv')
    independent=json.loads((BASE/'reports/ner_pilot_checkpoint_evaluation.json').read_text())
    assert independent['f1']==val['f1']
    metric_names=[('Official-code entity F1','f1'),('Precision','precision'),('Recall','recall'),
                  ('Exact raw span F1','exact_raw_span_f1'),('Unweighted token CE','unweighted_token_ce'),
                  ('False-positive text rate on gold-empty texts','empty_gold_text_fp_rate')]
    comparisons='\n'.join(f'| {name} | {train[key]:.4f} | {val[key]:.4f} |' for name,key in metric_names)
    counts='\n'.join(f"| {part} | {stats['rows']:,} | {stats['alignment_incompatible_rows']:,} | {stats['trimmed_whitespace_spans']:,} |" for part,stats in audit['by_partition'].items())
    report=f'''# NER preprocessing preflight and short training pilot

## Result

The recurrent NER pipeline has been implemented and exercised, including a full BIO/offset eligibility census, padding/loss checks, a tiny-set memorization check, eight real training epochs, and gold-free forward inference on every competition test text.

**The code checks pass, but this does not guarantee absence of overfitting or good test accuracy.** The pilot has a substantial train/development generalization gap, and the earlier source/test shift remains. No final audit, calibration, labeled in-domain sample, or leaderboard target score was used to choose or evaluate this pilot.

## 1. BIO labels and original character offsets

`ner_data.py` tokenizes original text into words, punctuation, underscores, combining marks and newline tokens. Token boundaries are computed **before** normalization. Only vocabulary keys are NFC-normalized and case-folded, so emoji, decomposed Unicode and case-folding length changes cannot shift the original text offsets. Every non-whitespace input character is retained. Tokenization does not inspect gold spans.

All 1,010,000 Ukrainian source rows were checked. For representable labels, BIO encoding followed by decoding must exactly reproduce the whitespace-trimmed character spans. Distinct adjacent locations receive separate `B-LOC` labels. Gold spans inside a word are not silently extended, and words are not split at label-dependent boundaries that would be unavailable at inference.

| Original partition | Rows | Whole-word BIO-incompatible rows | Whitespace-trimmed annotation spans |
|---|---:|---:|---:|
{counts}

The fit portion has **{fit['alignment_incompatible_rows']:,} incompatible rows ({100*fit['alignment_incompatible_rows']/fit['rows']:.3f}%)**, containing {fit['partial_token_boundary']:,} partial-token-boundary findings. This is an incompatibility with this **whole-word BIO representation**, not proof that every such annotation is semantically wrong. Some names are stems within inflected words; other examples extend partway into a following word. Arbitrary automatic repair could corrupt supervision.

Policy implemented: quarantine the entire incompatible fit row; never teach unrepresentable positive tokens as `O`. This leaves **{fit['roundtrip_passed_rows']:,} eligible fit rows before within-fit group deduplication**. The pilot then samples 12,000 unique eligible fit groups. The source CSV and frozen validation manifest remain unchanged. Original annotations remain accessible in the audit and example records.

Whitespace trimming affects only BIO targets and is counted explicitly. Original raw spans and original strings remain the scoring gold, so strict span scores can penalize whitespace differences even when the competition normalizer ignores them. The entire checkpoint partition remains in entity evaluation, including its {checkpoint['alignment_incompatible_rows']} incompatible rows; only those rows are omitted from token cross-entropy. Therefore the validation score is not inflated by dropping difficult examples.

The same eligibility checks were applied to calibration/audit annotations as a data-integrity census, with **no model predictions or model-selection scores** on those partitions. Inspecting format/support is distinct from using final audit performance to tune a model.

Files: `reports/ner_preprocessing_audit.json`, `reports/ner_preprocessing_counts.csv`, and the reusable eligibility manifest `data/ner_bio_eligibility.csv`.

## 2. Padding and loss masking

The shared BiGRU uses packed sequences with true lengths. Input padding is ID 0; padded labels are **-100**. Cross-entropy ignores those labels, and the padding embedding receives zero gradient. Regression tests verify that:

- Adding right padding does not change real-token NER logits.
- Loss with ignored padded positions equals loss computed only on real positions.
- Replacing padded logits with extreme values cannot change loss.
- The padding embedding's gradient is zero.

Every training batch is checked for finite loss and gradients. Gradient norms are clipped to 1.0; the saved history contains the pre-clipping norms. Class weights are learned from the 12,000 fit examples only, not checkpoint labels: O/B/I counts are {pilot['tag_counts']}, and weights are {[round(x,4) for x in pilot['class_weights']]}. The fixed square-root inverse-frequency rule is capped at 8. These weights encourage entity recall but can also increase false positives.

The training optimization loss is class-weighted. For the learning-curve comparison below, both fit-monitor and checkpoint losses are **unweighted** cross-entropy, computed in evaluation mode on real supervised token positions. They are comparable to each other; they should not be directly compared to the weighted training-batch loss.

## 3. Long texts and complete inference

Use windows of at most 128 tokens with overlap 32. Every original token must be covered. During training, overlapped copies of an already supervised token have label -100, so each target token contributes loss once. Inference averages overlapping token probabilities, then decodes BIO **once over the complete original document**. It does not independently decode chunks and then concatenate entity lists.

A controlled test places one entity across a window edge and verifies exact recovery of the full span. Other tests cover 0, 1, 127, 128, 129, 256 and 1,001 token inputs. Empty texts bypass the RNN rather than creating zero-length packed sequences. An orphan predicted `I-LOC` is treated as a new `B-LOC` and the repair count is recorded; this decoding policy is fixed, not tuned on validation.

Actual best-checkpoint inference was run on **all {long['rows']} unlabeled test texts**:

- {long['long_text_rows']} texts exceed 128 tokens; the maximum is {long['maximum_tokens']} lexical/punctuation/newline tokens.
- {long['total_windows']} windows cover all {long['checked_token_positions']:,} token positions.
- All token probabilities are finite and sum to one.
- All predicted spans stay within the original text, are nonempty, ordered and unique.
- No test labels were read and **no test F1 was computed**. Structural correctness of predicted spans does not imply correct locations.

Counts differ from EDA whitespace-token lengths because this tokenizer explicitly separates punctuation, emoji and newlines. No test text is reduced to only its first window in NER inference.

## 4. Tiny-set memorization health check

The model intentionally memorized {pilot['tiny_memorization']['rows']} fit-only texts: {pilot['tiny_memorization']['positive_rows']} positive and {pilot['tiny_memorization']['negative_rows']} negative. Dropout/weight decay were disabled only for this diagnostic, with learning rate 0.01. After {pilot['tiny_memorization']['updates']} updates, official-code F1 and trimmed-span F1 were both **1.0000**. This checks that labels, gradients, model outputs and decoding can work together; it is deliberately overfitted and says nothing about generalization.

## 5. Real training pilot and error comparison

Configuration: learned word embeddings (64), one bidirectional GRU layer (64 hidden units per direction), dropout 0.2; AdamW, learning rate 0.002, weight decay 0.01, batch 64, seed 42. Vocabulary is capped at 30,000 learned words plus PAD/UNK, fitted on pilot fit texts only. No transformers or self-attention are used.

The pilot used **12,000** eligible unique fit groups and **eight fixed epochs**, on {pilot['device']}. A fixed 2,048-text subset of those fit texts monitors training quality in evaluation mode. Every epoch evaluates all 4,000 checkpoint texts. Epoch {pilot['best_epoch']} had the highest published-code checkpoint F1 and was saved. This is development selection, not an unbiased final evaluation or NER cross-validation.

| Best selected epoch | Fit monitor (2,048) | Checkpoint (4,000) |
|---|---:|---:|
{comparisons}

The F1 gap is **{pilot['generalization_gap']:.4f}**, or {100*pilot['generalization_gap']:.2f} percentage points. Checkpoint has TP={val['tp']}, FP={val['fp']}, FN={val['fn']} under official normalization; the gold support differs slightly from raw mention count because the evaluator deduplicates surfaces within a text. Precision is lower than recall, and {100*val['empty_gold_text_fp_rate']:.2f}% of gold-empty checkpoint texts receive at least one prediction. False positives are therefore a concrete issue, not just a small aggregate F1 gap.

**There are signs of limited generalization and emerging overfitting.** Fit-monitor F1 increases from {history.iloc[4].train_monitor_f1:.4f} at epoch 5 to {history.iloc[-1].train_monitor_f1:.4f} at epoch 8, while checkpoint F1 only changes from {history.iloc[4].checkpoint_f1:.4f} to {history.iloc[-1].checkpoint_f1:.4f}. Checkpoint cross-entropy reaches its minimum earlier and later fluctuates. The gap alone cannot identify a single cause: this small training subset has limited vocabulary coverage, checkpoint OOV rate is **{100*pilot['checkpoint_OOV_token_fraction']:.2f}%**, rare names generalize poorly, and entity-weighted loss encourages false positives. One seed and eight epochs do not establish a stable ranking or optimal stopping point.

![Pilot learning curves](../figures/ner_pilot_learning.png)

No thresholds or merging variants were searched, and no calibration or final audit score was observed. Example checkpoint errors are saved in `reports/ner_pilot_error_examples.json`; the first examples are an inspection aid, not a random error-rate estimate. Full checkpoint predictions are saved for reproducibility. The best model and vocabulary are in `outputs/ner_pilot_best.pt`.

Independent rescoring of the saved prediction CSV reproduces F1={independent['f1']:.4f}. A 2,000-replicate group bootstrap gives [{independent['f1_95ci'][0]:.4f}, {independent['f1_95ci'][1]:.4f}]. This interval is conditional on these predictions and does not account for selecting the best epoch, training seeds or target-domain shift; it is not a final performance guarantee.

## 6. What to change before scaling up

1. Use the audited eligibility manifest, preserve original offsets, and retain all checkpoint texts for entity scoring. Do not silently reuse the old O/LOC Parquet as BIO.
2. Scale training data gradually before increasing model size or epochs. This pilot sees about 1.2% of fit rows; more vocabulary/name coverage may help, but improvement is not guaranteed.
3. Use early stopping and checkpoint selection on the fixed checkpoint set. Do not infer from steadily falling training loss that more epochs improve validation.
4. After choosing a checkpoint, tune a **small prespecified** decoding/threshold grid only on calibration. Class-weight strength is a training hyperparameter, selected through development experiments, not by looking at final audit results.
5. Keep dropout/weight decay, compare a few fixed seeds, and measure precision/recall as well as F1. A larger network is not the next automatic step.
6. Keep final audit sealed for model performance, and do not present the source pilot score as expected Public LB performance. Target-domain shift remains substantial.

The shared adversarial tokenization was updated to the same offset-preserving vocabulary-key policy (including underscores/combining marks), and the recurrent adversarial experiment was rerun. Current OOF domain ROC-AUC is **{av['oof_roc_auc']:.4f}**. Its three-fold predictions, result JSON and validation report are updated; it remains a domain-separability metric, not NER F1. The main encoder configuration remains identical to the adversarial encoder.

## Reproduction

```powershell
.venv/Scripts/python.exe -m unittest discover -s assignment3 -p "test_*.py"
.venv/Scripts/python.exe assignment3/audit_ner_preprocessing.py
.venv/Scripts/python.exe assignment3/pilot_ner.py
.venv/Scripts/python.exe assignment3/check_ner_inference.py
.venv/Scripts/python.exe assignment3/build_ner_preflight_report.py
```

Source/partition hashes, eligibility hashes, selected row IDs, model checkpoint, optimization history and predictions are retained. For full adversarial reproduction use `validation.py`, which now imports the same tokenizer. GPU stochastic operations can still differ slightly between environments despite fixed seeds.

AI assistance: Codex implemented and executed these checks and drafted this report from measured outputs. Keep the actual chat link/transcript for the assignment's AI disclosure.
'''
    (BASE/'reports/NER_PREFLIGHT_AND_PILOT.md').write_text(report,encoding='utf-8')
    nb=nbf.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
    nb.cells=[nbf.v4.new_markdown_cell('''# NER preflight and pilot

Read `reports/NER_PREFLIGHT_AND_PILOT.md` for policies and limitations. The pilot uses fit/checkpoint only; calibration and final audit have no NER model scores. Test inference below checks structure without labels or F1.'''),
        nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
BASE = Path.cwd() if (Path.cwd() / 'reports/ner_pilot_results.json').exists() else Path.cwd() / 'assignment3'
BASE = BASE.resolve()
audit = json.loads((BASE / 'reports/ner_preprocessing_audit.json').read_text(encoding='utf-8'))
display(pd.DataFrame(audit['by_partition']).T)"""),
        nbf.v4.new_code_cell("""pilot = json.loads((BASE / 'reports/ner_pilot_results.json').read_text())
print('Tiny-set memorization passed:', pilot['tiny_memorization']['passed'])
display(pd.DataFrame({'Fit monitor': pilot['best_train_monitor'], 'Checkpoint': pilot['best_checkpoint']}))"""),
        nbf.v4.new_code_cell("display(Image(filename=str(BASE / 'figures/ner_pilot_learning.png')))"),
        nbf.v4.new_code_cell("""long_checks = json.loads((BASE / 'reports/ner_long_text_checks.json').read_text())
display(pd.Series(long_checks))
print('No test quality metric was computed.')"""),
        nbf.v4.new_markdown_cell('## Reproduction\n\nRun the audit, pilot, and inference scripts from the report. Training weights are local under `outputs/`; original datasets and frozen splits are unchanged.')]
    nbf.validate(nb);nbf.write(nb,BASE/'ner_preflight.ipynb')
    print('Built NER preflight report and notebook')


if __name__=='__main__':main()
