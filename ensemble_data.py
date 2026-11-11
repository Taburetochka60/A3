"""Part 3 evaluation data: the frozen test-like subset and reference documents.

The test-like subset is chosen once, by an adversarial source/test classifier,
from labelled Ukrainian rows outside every Part 2 partition. Source rows are
single-line and short while most test texts are multi-line posts, so the chosen
rows are packed into posts whose token counts follow the test texts and joined
by line breaks (gold spans shifted accordingly). Posts and their row IDs are
frozen in ensemble_split.json; the rows must be excluded from every member's
training. The classifier only ranks rows for evaluation; it is never an ensemble member.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ner_data import tokenize_offsets
from pilot_ner import documents
from part2_data import fingerprint, read_split, save_json, text_group

PROTECTED = ['fit', 'checkpoint', 'calibration', 'audit']
UK_SPLITS = {'checkpoint', 'calibration', 'audit'}
TESTLIKE_SPLITS = {'testlike_select', 'testlike_report'}
SEPARATORS = ('\n', '\n\n')


def read_uk_rows(data_dir, keep, chunksize=50000):
    chunks = []
    for chunk in pd.read_csv(Path(data_dir)/'uk_geo_dataset.csv', keep_default_na=False,
                             usecols=['text', 'loc_markers', 'is_valid'], chunksize=chunksize):
        chunks.append(chunk.loc[keep(chunk.index)])
    return pd.concat(chunks)


def adversarial_scores(pool_texts, test_texts, folds=5, seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    texts = list(pool_texts)+list(test_texts)
    y = np.r_[np.zeros(len(pool_texts)), np.ones(len(test_texts))]
    x = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=3, max_features=200000,
                        sublinear_tf=True).fit_transform(texts)
    oof = np.zeros(len(texts))
    for train, valid in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
        model = LogisticRegression(C=1., class_weight='balanced', max_iter=2000)
        oof[valid] = model.fit(x[train], y[train]).predict_proba(x[valid])[:, 1]
    return oof[:len(pool_texts)], float(roc_auc_score(y, oof))


def compose_posts(ids, lengths, targets, rng):
    """Pack consecutive rows into posts; each post grows until it reaches a token count drawn from `targets`."""
    posts, i = [], 0
    while i < len(ids):
        target, rows, size = rng.choice(targets), [], 0
        while i < len(ids) and (not rows or size < target):
            rows.append(int(ids[i])); size += lengths[i]; i += 1
        posts.append(dict(rows=rows, separator=str(rng.choice(SEPARATORS))))
    return posts


def compose_frame(rows, posts):
    """One document per post, indexed by its first row ID: texts joined by the separator, spans shifted."""
    index, texts, markers = [], [], []
    for post in posts:
        parts, spans, offset = [], [], 0
        for i in post['rows']:
            text = rows.text.at[i]
            spans += [[a+offset, b+offset] for a, b in ast.literal_eval(rows.loc_markers.at[i])]
            parts.append(text); offset += len(text)+len(post['separator'])
        index.append(post['rows'][0]); texts.append(post['separator'].join(parts)); markers.append(json.dumps(spans))
    return pd.DataFrame(dict(text=texts, loc_markers=markers), index=index)


def testlike_rows(spec):
    """Every source row used by the test-like posts (to exclude from training)."""
    return sorted(i for name in sorted(TESTLIKE_SPLITS) for post in spec[name] for i in post['rows'])


def build_ensemble_split(data_dir, split_path, out_path, pool_rows=200000, testlike_rows=2000, seed=42):
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError('ensemble_split.json is frozen; delete it deliberately to rebuild')
    spec = read_split(split_path)
    data_dir = Path(data_dir)
    protected = set().union(*(spec[name] for name in PROTECTED))
    rng = np.random.default_rng(seed)
    candidates = read_uk_rows(data_dir, lambda index: ~index.isin(protected))
    pool = candidates.loc[rng.choice(candidates.index, min(pool_rows, len(candidates)), replace=False)]
    used = read_uk_rows(data_dir, lambda index: index.isin(protected))
    test = pd.read_csv(data_dir/'test.csv', keep_default_na=False)
    sample = pd.read_csv(data_dir/'labeling_sample.csv', keep_default_na=False)
    blocked = {text_group(t) for t in pd.concat([used.text, test.text, sample.text])}
    pool = pool.assign(group=pool.text.map(text_group))
    pool = pool.loc[~pool.group.isin(blocked)].drop_duplicates('group')
    scores, auc = adversarial_scores(pool.text.tolist(), test.text.tolist(), seed=seed)
    ranked = pool.assign(score=scores).sort_values('score', ascending=False, kind='stable').head(testlike_rows)
    lengths = ranked.text.map(lambda t: len(tokenize_offsets(t)))
    targets = [len(tokenize_offsets(t)) for t in test.text]
    posts = {name: compose_posts(ranked.index[start::2].tolist(), lengths.iloc[start::2].tolist(), targets, rng)
             for name, start in [('testlike_select', 0), ('testlike_report', 1)]}
    multirow = [len(p['rows']) > 1 for v in posts.values() for p in v]
    payload = dict(version=2, part2_split_sha256=spec['content_sha256'], seed=seed, pool_rows=len(pool),
                   classifier='tfidf char_wb 2-4 + logistic regression, 5-fold OOF',
                   adversarial_auc=auc, score_min=float(ranked.score.min()), score_median=float(ranked.score.median()),
                   composition='consecutive ranked rows packed to test token counts, joined by a line break or a blank line',
                   multirow_share=float(np.mean(multirow)), test_newline_share=float(test.text.str.contains('\n').mean()),
                   **posts)
    save_json(out_path, dict(payload, content_sha256=fingerprint(payload)))
    return payload


def read_ensemble_split(path, split_path):
    spec = json.loads(Path(path).read_text())
    payload = {k: v for k, v in spec.items() if k != 'content_sha256'}
    if fingerprint(payload) != spec['content_sha256']:
        raise ValueError('ensemble_split.json content changed')
    if spec['version'] != 2:
        raise ValueError('ensemble_split.json is an old single-row version; rebuild it')
    if spec['part2_split_sha256'] != read_split(split_path)['content_sha256']:
        raise ValueError('ensemble_split.json was built from another Part 2 split; rebuild it')
    return spec


def load_docs(data_dir, names, split_path, ensemble_path=None, allow_audit=False):
    """Reference word-token documents; every member's probabilities are aligned to these tokens."""
    if 'audit' in names and not allow_audit:
        raise ValueError('Audit is evaluated once, after the recipe is frozen (allow_audit=True)')
    data_dir = Path(data_dir)
    spec = read_split(split_path)
    ids = {name: spec[name] for name in names if name in UK_SPLITS}
    posts = {}
    if TESTLIKE_SPLITS & set(names):
        ensemble = read_ensemble_split(ensemble_path, split_path)
        posts = {name: ensemble[name] for name in names if name in TESTLIKE_SPLITS}
    result = {}
    wanted = set().union(*ids.values(), *(p['rows'] for v in posts.values() for p in v))
    if wanted:
        rows = read_uk_rows(data_dir, lambda index: index.isin(wanted))
        result.update({name: documents(rows.loc[values]) for name, values in ids.items()})
        result.update({name: documents(compose_frame(rows, value)) for name, value in posts.items()})
    if 'labeling' in names:
        result['labeling'] = documents(pd.read_csv(data_dir/'labeling_sample.csv', keep_default_na=False))
    if 'test' in names:
        test = pd.read_csv(data_dir/'test.csv', keep_default_na=False)
        if test.text_id.tolist() != spec['test_ids']:
            raise ValueError('Test IDs or order changed')
        docs = documents(test.assign(loc_markers='[]').set_index('text_id'))
        result['test'] = [dict(d, text_id=d['row_id'], bio=None) for d in docs]
    missing = set(names)-set(result)
    if missing:
        raise ValueError(f'Unknown splits: {sorted(missing)}')
    return result
