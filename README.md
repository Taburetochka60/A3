# Assignment 3: EDA and Metric Analysis

This repository contains EDA, metric analysis, a validation strategy, adversarial validation, and a short BiGRU NER pilot. Full-scale training and ensembling have not been completed. The trained pilot uses the shared recurrent encoder and NER head in `recurrent_model.py`.

## Results and deliverables

- [EDA and metric report](reports/EDA_AND_METRIC_ANALYSIS.md): findings, tables, limitations, and recommendations.
- [Results notebook](eda_and_metrics.ipynb): tables, a figure, and reproducible metric examples.
- [Detailed statistics](reports/eda_summary.json): full audit, dataset overlaps, and source-file SHA-256 hashes.
- [Dataset overview](reports/dataset_overview.csv).
- [Metric edge cases](reports/metric_edge_cases.csv).
- [EDA figure](figures/eda_overview.png).
- [Validation strategy](reports/VALIDATION_STRATEGY.md): fixed partitions, leakage prevention, and possible reasons for Public LB discrepancies.
- [Validation notebook](validation.ipynb): measured three-fold adversarial-validation results.
- [Adversarial-validation results](reports/adversarial_results.json) and [figure](figures/adversarial_validation.png).
- [BIO, padding, long-text checks, and NER pilot](reports/NER_PREFLIGHT_AND_PILOT.md), [notebook](ner_preflight.ipynb), and [learning curves](figures/ner_pilot_learning.png).
- [Checkpoint error analysis](reports/CHECKPOINT_ERROR_ANALYSIS.md): seen/unseen names, OOV, boundaries, entity types, and team priorities.
- [Contextual review of 200 cases](reports/CHECKPOINT_REVIEW_200.md), [annotated table](reports/checkpoint_review_200_annotated.csv), and [notebook](checkpoint_error_analysis.ipynb).

The analysis covers the complete Ukrainian raw CSV and processed Parquet, each with 1,010,000 rows, all 477 test texts, and the 26 provided labeled examples. The Russian auxiliary corpus is analyzed separately; see `reports/RUSSIAN_CORPUS_ANALYSIS.md`. Test targets are unknown: empty `locations` lists are placeholders.

## Reproduce the analysis

Run the commands below from this repository's root using its `.venv` environment.

```powershell
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe -r requirements.txt
python analyze.py
python build_deliverables.py
python -m unittest discover -s . -p test_metrics.py
```

## Reproduce validation

Install PyTorch for your CPU/GPU configuration separately. The experiments used the existing `torch 2.11.0+cu130` installation.

```powershell
python validation.py
python build_validation_report.py
python -m unittest discover -s . -p "test_*.py"
```

The split contains 999,948 fit, 4,000 checkpoint, 2,000 calibration, and 4,000 audit texts; 52 overlapping fit rows are excluded. The saved manifest is `data/validation_splits.csv`; its hash and checks are in `reports/validation_splits_summary.json`. Use the same partitions for every subsequent experiment. Evaluate audit only after freezing the model and decoding settings, not while tuning them.

Adversarial validation uses 477 source texts and 477 test texts, three-fold group-aware OOF predictions, and the BiGRU encoder shared by the domain and NER models. After aligning tokenization with the NER pilot, OOF ROC-AUC is 0.9493, indicating distinguishable input distributions. This is not NER F1. If the final main model uses a different recurrent encoder or tokenizer, rerun adversarial validation with it. Actual NER results are recorded in `reports/experiment_ledger.csv`; CV, final audit, and LB results remain unmeasured.

## Checks before full-scale training

```powershell
python audit_ner_preprocessing.py
python pilot_ner.py
python check_ner_inference.py
python build_ner_preflight_report.py
```

The fit partition contains 1,485 rows with boundaries inside words. These are excluded from word-level BIO training; the original data are preserved. The complete checkpoint partition, including incompatible rows, remains in entity evaluation. The pilot uses 12,000 fit texts and eight epochs, reaching F1 0.8131 on the train monitor and 0.5682 on checkpoint. The generalization gap is substantial; these results do not guarantee full-scale training quality. All test tokens are processed through windows. NER calibration and final audit have not been evaluated.

## Error analysis for the team handoff

```powershell
python analyze_checkpoint_errors.py
python review_checkpoint_cases.py
python build_checkpoint_error_report.py
```

The full checkpoint has 917 FP and 431 FN, including 117 FP and 108 FN with overlapping boundaries. F1 is 0.8312 for surfaces previously annotated as LOC in the actual 12k fit subset, and 0.2011 for unseen surfaces. This partitions normalized candidates, rather than evaluating only documents containing new names. Recall is 85.8% for seen surfaces and 30.0% for unseen ones.

The contextual review covers 200 distinct texts: 100 FP and 100 FN focal cases. This diagnostic sample does not estimate the prevalence of annotation noise. Codex notes are AI-assisted review, not independent expert annotations. Findings include suspected type errors, incomplete labels, and inconsistent entity granularity. Gold labels, model weights, and splits are unchanged. Do not copy checkpoint examples into training or adjust answers to match the model; first agree on LOC conventions and version any annotation changes.

Expected files are `data/uk_geo_dataset.csv`, `data/uk_geo_dataset_processed_v1.parquet`, `data/test.csv`, and `data/labeling_sample.csv`. They are downloaded from Kaggle and excluded by the existing `.gitignore` rule for `data`. Official competition pages, including the exact published evaluator, are archived in `sources/`.

To download the files again, configure the Kaggle API and accept the competition rules:

```powershell
kaggle competitions download -c kse-ua-location-extraction-2026 -f uk_geo_dataset.csv -p data
kaggle competitions download -c kse-ua-location-extraction-2026 -f uk_geo_dataset_processed_v1.parquet -p data
kaggle competitions download -c kse-ua-location-extraction-2026 -f test.csv -p data
kaggle competitions download -c kse-ua-location-extraction-2026 -f labeling_sample.csv -p data
```

`download_data.py --file uk_geo_dataset.csv` is an optional resumable HTTP-range downloader using Kaggle authentication. It does not store credentials or signed URLs. The standard commands above are sufficient on a normal connection.

## Important limitations

`metrics.py` deliberately reproduces the published regular expression, including its interpolation defect. Parity was checked against the published function, not Kaggle's hidden server implementation. Corrected normalization is diagnostic only. The proposed alternative metric does not replace the competition target metric.

The EDA stage does not train models or invent CV/LB scores. Codex assistance is disclosed in the reports; attach the actual chat link or transcript as required by the assignment's AI-use policy.

## Russian auxiliary corpus: full audit and experiment

The full audit covers 8,028,840 sentences from 739,190 documents: annotations, duplicates, Ukrainian overlap, and safe splitting by `doc_id`. UA12k, UA12k+RU12k, and UA12k+UA12k were compared across three seeds.

- [Full report](reports/RUSSIAN_CORPUS_ANALYSIS.md).
- [Contextual review of 100 Russian examples](reports/RU_ANNOTATION_REVIEW_100.md).
- [Notebook](russian_corpus_analysis.ipynb).
- [All nine runs](reports/ru_augmentation_runs.csv) and [paired comparisons](reports/ru_augmentation_experiment.json).

### Results and guidance for the next contributor

The Russian corpus contains 525,636 excess rows after normalization, 4,902 groups of identical texts with differing LOC surface sets, and 6,769 rows incompatible with word-level BIO encoding. No exact/canonical matches with Ukrainian texts were found. Semantic review covered 100 contexts; it is not an independent expert estimate of the accuracy of all labels.

After document-level leakage protection and filtering, 5,440,386 RU fit rows and 804,363 RU valid rows remain. Use `data/russian_audit/ru_split_manifest.parquet`, rather than randomly splitting sentences.

Mean results across three seeds:

| Training data | UA checkpoint F1 | RU diagnostic F1 |
|---|---:|---:|
| UA 12k | 0.5771 | 0.1905 |
| UA 12k + RU 12k | 0.6232 | 0.7367 |
| UA 12k + UA 12k | 0.6803 | 0.2453 |

**Conclusion:** Russian augmentation improves the smaller Ukrainian pilot and substantially strengthens Russian NER. However, adding the same number of Ukrainian texts performs better on the Ukrainian checkpoint in all three seeds. This is a language tradeoff for the mixed-language test; do not automatically mix the entire Russian corpus at a 1:1 ratio.

The two 24k arms have the same number of examples and optimizer updates. Comparing the mixture with the 12k baseline alone does not establish an advantage specific to the Russian source. The experiment uses a BiGRU with random word embeddings and a 30k vocabulary. Lower RU ratios, multilingual/subword embeddings, and RU pretraining followed by UA fine-tuning require separate experiments.

With Russian augmentation, the proportion of unknown UA tokens increases from 14.36% to 16.74%. LOC/ORG conventions, addresses, and hyphenated names still require clarification; gold was not modified to match the model. Ukrainian calibration and final audit remain unevaluated, and no Kaggle submissions were made. All 29 tests pass.

```powershell
kaggle competitions download -c kse-ua-location-extraction-2026 -f ru_geo_dataset.csv -p data
python russian_corpus_audit.py
python russian_augmentation_experiment.py
python build_russian_report.py
```

Large metadata files, the DuckDB database, and the manifest are under `data/russian_audit` and excluded from Git; weights are under `outputs/russian_experiment`. Code, small reports, figures, and the notebook can be committed. DuckDB uses a 256 MB memory limit with temporary files on disk.

Russian fit/valid are document-disjoint. Entire fit documents overlapping RU valid are excluded. Ukrainian checkpoint/calibration/audit, test texts, and `labeling_sample` texts are protected. Zero exact/canonical overlap does not establish the absence of translated or semantically similar news. The report documents the filtering cost and limitations.

- [Team handoff update after the Russian analysis](reports/TEAM_HANDOFF_RU_UPDATE.md).
