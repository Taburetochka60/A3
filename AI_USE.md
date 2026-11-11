# AI use

AI assistance was used for Part 2 code, data checks, the BiLSTM extension, Colab training/resume, experiments, tests and documentation. The final implementation was combined into `part2.py` and `part2_data.py`. The frozen split migration was checked against the original IDs, groups, labels, BIO, offsets and vocabulary.

The team supplied the original EDA, metric, preprocessing, GRU pilot and Russian-data work. These keep their original authorship. Shared changes only add recurrent options, vocabulary keys and inference batch size; default GRU behaviour is preserved.

Real training ran in Colab. The baseline checkpoint and predictions stay unchanged. Local checks used saved weights and small unit fixtures; no real retraining or Kaggle submission was performed. Development comparisons do not use audit or hidden test labels for tuning. Full new GPU training has not been verified after this refactor.

The student must review and understand the code, provide their own contribution statement and retain the chat history. The course disclosure form is not available in this repository; complete it if required.

## FastText experiment

AI assistance supported the preregistered random/FastText 300d control, official-vector cache/provenance checks, training/resume, W&B adaptation, tests, notebook and run instructions. The n-grams representation module is reused unchanged from the existing project. The only shared data-code change makes AST integrity serialization explicit across Python versions; grouping behavior and frozen rows are unchanged. Tiny generated FastText binaries are loader fixtures, not official Ukrainian performance experiments. The experiment later ran in Colab. The current reporting fixes were verified locally; they were not used to retrain models. The student must review and disclose this assistance.

AI also assisted the additional from-scratch subword variant, consistent fit/inference OOV routing, fixed external-vector cache, comparison, memory checks and fixture/CPU preflight tests. This is a separate representation experiment; it does not change the original A/B controls or imply improved measured performance.

AI assisted the inference script, cached-vector extension, offset/schema checks, offline tests, JSON-null fix and provenance review. The final Drive inference revision uses a small absolute probability tolerance while retaining exact entity and offset checks. Historical source and results stay unchanged; the current reproduction manifest has a separate identity. The student must review these changes and complete any required course form.
