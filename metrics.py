"""Competition-compatible scoring and diagnostic NER metrics.

The published regex is deliberately preserved. It is NOT an f-string.
"""
import re
import string
import unicodedata


def hard_processing(text):
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def intended_processing(text):
    """Diagnostic only: interpolate the punctuation character class."""
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def conservative_processing(text):
    """Alternative metric: preserve address numbers, punctuation and words."""
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def score_details(y_true, y_pred, normalize=hard_processing):
    if len(y_true) != len(y_pred):
        raise ValueError('Gold and prediction must contain the same number of texts')
    tp = fp = fn = 0
    for gold, pred in zip(y_true, y_pred):
        g, p = {normalize(s) for s in gold}, {normalize(s) for s in pred}
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    total_gold = tp + fn
    precision = tp / (tp + fp) if tp + fp else float(total_gold == 0)
    recall = tp / total_gold if total_gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=round(f1, 8))


def comp_metric(y_true, y_pred):
    return score_details(y_true, y_pred)['f1']


def alternative_metric(y_true, y_pred):
    """Proposed conservative surface-set F1, with nonblank-string schema."""
    for texts in [y_true, y_pred]:
        for entities in texts:
            for entity in entities:
                if not isinstance(entity, str) or not conservative_processing(entity):
                    raise ValueError('Alternative metric requires nonblank entity strings')
    return score_details(y_true, y_pred, conservative_processing)['f1']


def exact_span_f1(y_true, y_pred):
    """Spans are (start, end), half-open Python character offsets, per text."""
    return score_details(y_true, y_pred, normalize=tuple)


def document_metrics(y_true, y_pred, normalize=hard_processing):
    if len(y_true) != len(y_pred):
        raise ValueError('Mismatched lengths')
    if not y_true:
        return {'macro_f1': None, 'exact_set_accuracy': None}
    exact = sum({normalize(x) for x in g} == {normalize(x) for x in p}
                for g, p in zip(y_true, y_pred))
    return {'macro_f1': sum(score_details([g], [p], normalize)['f1']
                            for g, p in zip(y_true, y_pred)) / len(y_true),
            'exact_set_accuracy': exact / len(y_true)}
