# Project handoff update: Russian corpus

The Russian corpus has now been fully analyzed. Any earlier handoff statement that it had not been analyzed is outdated.

## Verified findings

- 8,028,840 sentences, 739,190 documents, and 3,708,173 LOC mentions.
- 525,636 excess rows after NFC normalization, case folding, and whitespace collapse.
- 4,902 groups of identical normalized texts with different LOC surface sets.
- 6,769 rows incompatible with word-level BIO encoding; boundaries were not automatically repaired.
- No exact/canonical full-text matches with the Ukrainian corpus were found. This does not check translated or semantically similar news.
- Codex reviewed 100 contexts: 80 positive and 20 negative. This is not an independent expert estimate of global annotation accuracy.

## Training without leakage

Use `data/russian_audit/ru_split_manifest.parquet`, rather than creating a new random sentence split. The base split is by `doc_id`; fit documents containing a duplicate from RU valid are excluded entirely. The retained partitions contain 5,440,386 fit rows and 804,363 valid rows. Active RU partitions have zero document and canonical-text overlap. Ukrainian checkpoint/calibration/audit and external texts are protected.

Manifest SHA-256: `23c17736c799113691b3eddd6bce6eb34c4d971ae1623cc8df87bd60ac535e1b`.

## Measured benefit: three seeds, the same Ukrainian checkpoint

| Training data | Mean UA F1 | Mean RU F1 |
|---|---:|---:|
| UA 12k | 0.5771 | 0.1905 |
| UA 12k + RU 12k | 0.6232 | 0.7367 |
| UA 12k + UA 12k | 0.6803 | 0.2453 |

Russian augmentation helps compared with the smaller pilot, but an equal increase in Ukrainian data performs better on the Ukrainian checkpoint in all three seeds. The mixture is substantially stronger on the Russian diagnostic holdout. This is a language tradeoff, not proof that the entire RU corpus is useful or useless for Kaggle.

The comparison uses random word embeddings, a BiGRU, and a 1:1 mixture. The shared vocabulary is capped at 30k: adding RU increases UA OOV from 14.36% to 16.74%. The two 24k arms match example counts and optimizer updates; sequence lengths and FLOPs differ. Do not draw conclusions solely from comparing the mixture with the 12k baseline, which receives fewer updates.

## Next steps

1. For a strong Ukrainian baseline, first scale UA fit and evaluate better embeddings.
2. For Russian/mixed texts, separately compare a smaller RU fraction, subword/multilingual embeddings, or RU pretraining followed by UA fine-tuning.
3. Agree on LOC/ORG conventions, geographic components of institutions, addresses, and hyphenated names. RU is an automatically annotated source, not error-free gold.
4. Do not use RU diagnostic, UA checkpoint, or reviewed examples as fit data. NER calibration and final audit remain unevaluated.
5. Do not report these development F1 values as CV/LB scores: Public leaderboard performance has not been measured.

## Files

- `reports/RUSSIAN_CORPUS_ANALYSIS.md`: full report.
- `reports/RU_ANNOTATION_REVIEW_100.md`: contextual review.
- `reports/ru_augmentation_runs.csv`: nine runs.
- `reports/ru_augmentation_experiment.json`: protocol and paired comparisons.
- `russian_corpus_analysis.ipynb`: results notebook.
- `outputs/russian_experiment/`: nine selected models and vocabularies; excluded from Git.
- `data/russian_audit/`: large metadata and manifests; excluded from Git.

The original Ukrainian split and pilot are preserved. No gold-label corrections or Kaggle submissions were made.
