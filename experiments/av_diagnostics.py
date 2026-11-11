from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GROUPS = {
    "length_only": ("log_chars", "log_words"),
    "format_only": ("newlines", "nonbmp", "punctuation", "digits", "latin"),
    "all_measured": ("log_chars", "log_words", "newlines", "nonbmp",
                     "punctuation", "digits", "latin"),
}


def describe_text(value: str) -> dict[str, float]:
    n = max(len(value), 1)
    return {"log_chars": math.log1p(len(value)),
            "log_words": math.log1p(len(value.split())),
            "newlines": value.count("\n") / n,
            "nonbmp": sum(ord(c) > 0xFFFF for c in value) / n,
            "punctuation": sum(not c.isalnum() and not c.isspace() for c in value) / n,
            "digits": sum(c.isdigit() for c in value) / n,
            "latin": sum(c.isascii() and c.isalpha() for c in value) / n}


def measure(texts, origin, folds):
    """Three-fold OOF AUC with all preprocessing fit inside training fold."""
    if len(texts) != len(origin) or len(texts) != len(folds):
        raise ValueError("Domain values and frozen folds must align")
    origin = np.asarray(origin, dtype=int)
    folds = np.asarray(folds, dtype=int)
    if set(np.unique(origin)) != {0, 1} or set(np.unique(folds)) != {0, 1, 2}:
        raise ValueError("Expected binary origin and three AV folds")
    if any(set(np.unique(origin[folds != i])) != {0, 1}
           or set(np.unique(origin[folds == i])) != {0, 1} for i in range(3)):
        raise ValueError("Every train/heldout AV fold needs both classes")
    features = pd.DataFrame(describe_text(t) for t in texts)
    results = {}; scores = {}
    for name, columns in GROUPS.items():
        values = features[list(columns)].to_numpy(dtype=np.float64)
        oof = np.empty(len(texts), dtype=np.float64)
        for fold in range(3):
            tr, va = folds != fold, folds == fold
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1., max_iter=1000, random_state=42))
            model.fit(values[tr], origin[tr])
            oof[va] = model.predict_proba(values[va])[:, 1]
        results[name] = {"oof_auc": float(roc_auc_score(origin, oof)),
                         "features": list(columns),
                         "method": "fixed LogisticRegression(C=1), 3-fold out-of-fold"}
        scores[name] = oof
    return results, scores


def run(frames, split, output_dir):
    """Use exactly the prior AV source/test IDs and frozen group folds."""
    from part2_data import save_json

    source = frames["domain_source"]
    test = frames["test"].set_index("text_id").loc[split["av_test"]]
    texts = source.text.tolist() + test.text.tolist()
    folds = split["av_folds"]
    expected_source = len(split["domain_source"])
    expected_test = len(split["av_test"])
    if (len(source) != expected_source or len(test) != expected_test
            or len(texts) != expected_source + expected_test
            or len(folds) != len(texts)):
        raise ValueError("Frozen AV source/test sample or fold count changed")
    labels = [0] * len(source) + [1] * len(test)
    results, oof = measure(texts, labels, folds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "av_ablation.json", dict(results=results,
        source_rows=len(source), unlabeled_test_rows=len(test),
        frozen_split_sha256=split["content_sha256"],
        same_ner_model=False, label_definition="0=source text;1=unlabeled test text",
        never_used_as_ner_selection_metric=True))
    pd.DataFrame(dict(origin=labels, fold=folds,
        **{f"prob_{name}": values for name, values in oof.items()})).to_csv(
        output_dir / "av_ablation_oof.csv", index=False)
    return results
