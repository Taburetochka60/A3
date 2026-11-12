"""Candidate Kaggle submissions built from saved *_spans.csv files (Part 3, roadmap slots S2-S4, S6).

No model runs here: every candidate re-filters spans that a model already predicted on test.csv.

  cutoff         keep spans whose mean B+I confidence is at least t. part2.decode keeps a span when
                 that same score reaches the threshold, so cutting spans decoded at t=0 equals decoding at t.
  clean          rules A+B of postprocess.py, keeping each span's confidence.
  vote           keep a normalised string (metrics.hard_processing) predicted by at least k models;
                 its surface form and span come from the first model in priority order that has it.
  drop_countries rule U: drop exact forms of "Україна"/"Украина". No organiser gold string is a
                 country while 31% of training LOC spans are; "Українка" (a town) stays.

    python3 candidates.py data/test.csv data/submissions data/submissions/candidates
"""
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

from metrics import hard_processing
from postprocess import postprocess, signature_lines

SOURCES = dict(fasttext='submission_fasttext_subword_spans.csv',        # 0.50898, decoded at 0.8/strict
               ngrams='submission_candidate_spans.csv',                 # 0.50370, decoded at 0.8/strict
               greedy='greedy_calibration_t000_strict_spans.csv',       # 0.50211, decoded at 0/strict
               bpe='bpe_t080_strict_spans.csv')                         # 0.47471, decoded at 0.8/strict
COUNTRIES = {'україна', 'україни', 'україні', 'україну', 'україною',
             'украина', 'украины', 'украине', 'украину', 'украиной'}


def load_spans(path, test):
    """Per test text, a list of (start, end, confidence); spans files without confidences get 1.0."""
    table = pd.read_csv(path)
    if table.text_id.tolist() != test.text_id.tolist():
        raise ValueError(f'{path} and test.csv list different text IDs')
    result = []
    for text, raw, conf in zip(test.text, table.spans, table.confidence if 'confidence' in table else [None]*len(table)):
        spans = json.loads(raw)
        scores = json.loads(conf) if conf is not None else [1.]*len(spans)
        if len(scores) != len(spans) or any(not 0 <= a < b <= len(text) for a, b in spans):
            raise ValueError(f'{path}: spans do not fit the test text')
        result.append([(a, b, c) for (a, b), c in zip(spans, scores)])
    return result


def cutoff(spans, t):
    return [[s for s in doc if s[2] >= t] for doc in spans]


def clean(texts, spans, signatures, rules='AB'):
    """postprocess.postprocess on every span, keeping its confidence."""
    result = []
    for text, doc in zip(texts, spans):
        kept = {}
        for a, b, c in doc:
            for span in postprocess(text, [(a, b)], signatures, rules):
                kept.setdefault(span, c)
        result.append([(a, b, c) for (a, b), c in kept.items()])
    return result


def vote(texts, models, k):
    """models is a list of per-text span lists in priority order (the first one supplies surface forms)."""
    result = []
    for i, text in enumerate(texts):
        first, counts = {}, Counter()
        for doc in (m[i] for m in models):
            keys = {}
            for s in doc:
                keys.setdefault(hard_processing(text[s[0]:s[1]]), s)
            counts.update(list(keys))
            for key, s in keys.items():
                first.setdefault(key, s)
        result.append([first[key] for key in first if counts[key] >= k])
    return result


def drop_countries(texts, spans):
    return [[s for s in doc if hard_processing(text[s[0]:s[1]]) not in COUNTRIES] for text, doc in zip(texts, spans)]


def strings(texts, spans):
    """Per text, the set of normalised strings the metric compares."""
    return [{hard_processing(text[a:b]) for a, b, _ in doc} for text, doc in zip(texts, spans)]


def write(path, test, spans):
    """Submission CSV plus the *_spans.csv that part2.export_submission writes next to it."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(text_id=test.text_id, locations=[json.dumps([t[a:b] for a, b, _ in doc], ensure_ascii=False)
                                                       for t, doc in zip(test.text, spans)])).to_csv(path, index=False)
    pd.DataFrame(dict(text_id=test.text_id, spans=[json.dumps([[a, b] for a, b, _ in doc]) for doc in spans],
                      confidence=[json.dumps([c for *_, c in doc]) for doc in spans])).to_csv(
        path.with_name(path.stem+'_spans.csv'), index=False)


def sanity(texts, candidates, base):
    """Per candidate: strings per text (organiser gold: 1.23), strings added/removed vs base, overlap with base."""
    reference = strings(texts, candidates[base])
    rows = []
    for name, spans in candidates.items():
        sets = strings(texts, spans)
        added = sum(len(s-r) for s, r in zip(sets, reference))
        removed = sum(len(r-s) for s, r in zip(sets, reference))
        union = sum(len(s | r) for s, r in zip(sets, reference))
        rows.append(dict(candidate=name, strings=sum(map(len, sets)), per_text=round(sum(map(len, sets))/len(sets), 3),
                         added=added, removed=removed, jaccard_base=round(1-(added+removed)/union, 3) if union else 1.))
    return pd.DataFrame(rows)


def overlap(texts, candidates):
    """Pairwise Jaccard of (text, normalised string) sets."""
    sets = {name: {(i, s) for i, x in enumerate(strings(texts, spans)) for s in x} for name, spans in candidates.items()}
    return pd.DataFrame({a: {b: round(len(sets[a] & sets[b])/max(1, len(sets[a] | sets[b])), 3) for b in sets} for a in sets})


def build_all(test_csv, source_dir, out_dir):
    """Every candidate for S1 (reproduction), S2, S3, S4 (rule U), S6 (cutoff 0.9) and S8 (both); returns the sanity tables."""
    test = pd.read_csv(test_csv, keep_default_na=False)
    texts, signatures = test.text.tolist(), signature_lines(test.text)
    raw = {name: load_spans(Path(source_dir)/file, test) for name, file in SOURCES.items() if (Path(source_dir)/file).exists()}
    missing = sorted(set(SOURCES)-set(raw))
    voters = ['fasttext', 'ngrams', 'greedy' if 'greedy' in raw else 'bpe']   # bpe stands in until the greedy file arrives
    candidates = {}
    for t, tag in [(.8, ''), (.9, '_t090')]:
        members = {name: clean(texts, cutoff(raw[name], t), signatures) for name in voters}
        candidates['S1_ngrams_AB'+tag] = members['ngrams']
        candidates['S2_fasttext_AB'+tag] = members['fasttext']
        candidates[f'S3_vote2_{"_".join(voters)}_AB'+tag] = vote(texts, [members[n] for n in voters], 2)
    for name in list(candidates):
        candidates[name+'_U'] = drop_countries(texts, candidates[name])
    out_dir = Path(out_dir)
    for name, spans in candidates.items():
        write(out_dir/f'{name}.csv', test, spans)
    table, pairs = sanity(texts, candidates, 'S1_ngrams_AB'), overlap(texts, candidates)
    table.to_csv(out_dir/'sanity.csv', index=False); pairs.to_csv(out_dir/'overlap.csv')
    return table, pairs, missing


if __name__ == '__main__':
    table, pairs, missing = build_all(*sys.argv[1:4])
    pd.set_option('display.width', 200)
    print(table.to_string(index=False))
    if missing:
        print('missing spans files (skipped):', ', '.join(SOURCES[m] for m in missing))
