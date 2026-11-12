"""Span post-processing for Telegram-style test posts (Part 3).

Training rows are single news sentences, so the models never learned that a
location ends at a line break, and they tag place names inside channel
signatures ("Київ — пряма трансляція 👈") that the organisers leave unlabelled
(6 of 6 such mentions in labeling_sample). Two rules, applied to any model's spans:

  A. cut a span at its first line break, trim junk at both ends and complete
     hyphenated words ("ТЦ Аркадія\\n\\n📍" -> "ТЦ Аркадія", "-Франківську" -> "Івано-Франківську");
  B. drop spans inside a signature line (a line repeated verbatim in at least
     SIGNATURE_MIN_TEXTS test texts), an @handle or a link.

Rule B reads only the unlabelled test texts, never labels.

    python3 postprocess.py <submission>_spans.csv data/test.csv <output>.csv
"""
from collections import Counter
import json
from pathlib import Path
import re
import sys

import pandas as pd

SIGNATURE_MIN_TEXTS = 4
LINK = re.compile(r'https?://|www\.|t\.me/')
SPAN_BREAK = re.compile(r'\n|\s[-–—]\s')                  # "Чорнобиль - 19", "Київ — пряма трансляція"
JOINERS = "-'’ʼ‘"
CLOSERS = {'»': '«', '"': '"', '”': '“', ')': '(', '“': '„'}


def signature_lines(texts, min_texts=SIGNATURE_MIN_TEXTS):
    """Stripped lines that occur in at least min_texts different texts."""
    counts = Counter(line for t in texts for line in {s.strip() for s in t.split('\n')} if line)
    return {line for line, n in counts.items() if n >= min_texts}


def joined(text, i, j):
    """text[i] is a hyphen or apostrophe between letters text[j] (in the span) and text[2*i-j]."""
    k = 2*i-j
    return 0 <= min(j, k) and max(j, k) < len(text) and text[i] in JOINERS and text[j].isalpha() and text[k].isalpha()


def clean_span(text, a, b):
    """Rule A: trim junk at both ends, cut at a line break or a spaced dash, complete words.

    A closing quote or bracket stays when its opener is inside the span («Центральний»).
    Cut words grow back across at most one hyphen or apostrophe per side ("-Франківську" ->
    "Івано-Франківську"). Spans with fewer than two letters ("№ 38-") are dropped: the metric
    deletes digits, so they can only be false positives.
    """
    while a < b and not text[a].isalnum():
        a += 1
    end = SPAN_BREAK.search(text, a, b)
    if end:
        b = end.start()
    while b > a and not text[b-1].isalnum() and not (text[b-1] in CLOSERS and CLOSERS[text[b-1]] in text[a:b-1]):
        b -= 1
    if a == b:
        return None
    hop = False
    while a > 0 and (text[a-1].isalnum() or not hop and joined(text, a-1, a)):
        hop, a = hop or not text[a-1].isalnum(), a-1
    hop = False
    while b < len(text) and (text[b].isalnum() or not hop and joined(text, b, b-1)):
        hop, b = hop or not text[b].isalnum(), b+1
    if sum(c.isalpha() for c in text[a:b]) < 2:
        return None
    return a, b


def boilerplate(text, a, b, signatures):
    """Rule B: the span sits in a signature line, an @handle or a link."""
    start, end = text.rfind('\n', 0, a)+1, text.find('\n', b)
    if text[start:len(text) if end < 0 else end].strip() in signatures:
        return True
    prefix = re.search(r'\S*$', text[:a]).group()          # the rest of the word the span starts in
    return '@' in prefix or bool(LINK.search(prefix))


def postprocess(text, spans, signatures, rules='AB'):
    kept = []
    for a, b in spans:
        span = clean_span(text, a, b) if 'A' in rules else (a, b)
        if span and not ('B' in rules and boilerplate(text, *span, signatures)) and span not in kept:
            kept.append(span)
    return kept


def postprocess_submission(spans_csv, test_csv, path, rules='AB'):
    """Rewrite a submission from its *_spans.csv (export_submission writes one next to every CSV)."""
    test = pd.read_csv(test_csv, keep_default_na=False)
    table = pd.read_csv(spans_csv)
    if table.text_id.tolist() != test.text_id.tolist():
        raise ValueError('Spans file and test.csv list different text IDs')
    signatures = signature_lines(test.text)
    locations, bounds, removed = [], [], Counter()
    for text, raw in zip(test.text, table.spans):
        before = [tuple(s) for s in json.loads(raw)]
        if any(not 0 <= a < b <= len(text) for a, b in before):
            raise ValueError('Span offsets do not fit the test text')
        after = postprocess(text, before, signatures, rules)
        removed['spans_in'] += len(before)
        removed['spans_out'] += len(after)
        removed['changed'] += sum(s not in after for s in before)
        locations.append([text[a:b] for a, b in after]); bounds.append(after)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(text_id=test.text_id, locations=[json.dumps(v, ensure_ascii=False) for v in locations])).to_csv(path, index=False)
    pd.DataFrame(dict(text_id=test.text_id, spans=[json.dumps([list(s) for s in v]) for v in bounds])).to_csv(
        path.with_name(path.stem+'_spans.csv'), index=False)
    return dict(removed, signatures=len(signatures))


if __name__ == '__main__':
    print(postprocess_submission(*sys.argv[1:4], *sys.argv[4:5]))
