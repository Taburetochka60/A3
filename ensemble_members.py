"""Ensemble members: checkpoint -> probability bundles on the reference tokens.

A bundle is the only thing the ensemble search reads, so a member can come from
a checkpoint loaded here or from a file exported by whoever trained it. Format
(one .npz per member and split):

    ids      int64 [docs]        row_id (source rows) or text_id (test)
    lengths  int64 [docs]        reference word tokens per document
    starts   int64 [tokens]      reference token character offsets, concatenated
    ends     int64 [tokens]
    probs    float32 [tokens,3]  O/B/I probabilities on the reference tokens
    meta     str                 JSON: member, split, checkpoint_sha256, ...

Reference tokens are ner_data.tokenize_offsets words. Subword members are
projected onto them with the first-piece convention.
"""
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np

from part2_data import file_hash, save_json


def to_reference(reference, tokens, probabilities):
    """Project member-token probabilities onto reference words (first piece of each word)."""
    if len(tokens) != len(probabilities):
        raise ValueError('Member token/probability mismatch')
    starts = [t.start for t in reference]
    for t in tokens:
        i = bisect_right(starts, t.start)-1
        if i < 0 or t.end > reference[i].end:
            raise ValueError('Member tokens must nest inside reference words')
    first = {t.start: i for i, t in enumerate(tokens)}
    missing = [t for t in reference if t.start not in first]
    if missing:
        raise ValueError(f'Reference word without a member token at offset {missing[0].start}')
    return np.asarray(probabilities, dtype=np.float64)[[first[t.start] for t in reference]].reshape(len(reference), 3)


def save_bundle(path, docs, probabilities, meta):
    if len(docs) != len(probabilities):
        raise ValueError('Missing document probabilities')
    for d, p in zip(docs, probabilities):
        if np.shape(p) != (len(d['tokens']), 3):
            raise ValueError('Probabilities must be on the reference tokens')
    tokens = [t for d in docs for t in d['tokens']]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary, ids=np.array([d['row_id'] for d in docs], dtype=np.int64),
                        lengths=np.array([len(d['tokens']) for d in docs], dtype=np.int64),
                        starts=np.array([t.start for t in tokens], dtype=np.int64),
                        ends=np.array([t.end for t in tokens], dtype=np.int64),
                        probs=np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in probabilities]),
                        meta=json.dumps(meta, ensure_ascii=False))
    temporary.replace(path)


def load_bundle(path, docs):
    with np.load(path) as saved:
        ids, lengths, starts, ends, probs = (saved[k] for k in ['ids', 'lengths', 'starts', 'ends', 'probs'])
        meta = json.loads(str(saved['meta']))
    if ids.tolist() != [d['row_id'] for d in docs] or lengths.tolist() != [len(d['tokens']) for d in docs]:
        raise ValueError(f'{path}: documents differ from the evaluation split')
    if (starts.tolist() != [t.start for d in docs for t in d['tokens']]
            or ends.tolist() != [t.end for d in docs for t in d['tokens']]):
        raise ValueError(f'{path}: tokens differ from the reference tokenizer')
    probs = probs.astype(np.float64)
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError(f'{path}: invalid probabilities')
    probs /= probs.sum(1, keepdims=True)
    return np.split(probs, np.cumsum(lengths)[:-1]), meta


def part2_predictor(checkpoint, device='cpu'):
    """Word/BPE BiLSTM checkpoints saved by part2.train (best.pt)."""
    import torch
    from pilot_ner import infer_probabilities
    from part2_data import encode_documents
    from recurrent_model import NERModel
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = saved['encoding']
    model = NERModel(state['size'], **saved['config']['encoder']).to(device).eval()
    model.load_state_dict(saved['model'])

    def predict(docs):
        encoded = encode_documents(docs, state)
        probabilities = infer_probabilities(model, encoded, torch.device(device), batch_size=128)
        return [to_reference(d['tokens'], e['tokens'], p) for d, e, p in zip(docs, encoded, probabilities)]
    info = dict(config=saved['config'], epoch=saved.get('epoch'), split_sha256=saved.get('split_hash'),
                tokenizer=state['kind'])
    return predict, info


def train_member(*args, **kwargs):
    """part2.train for a new member, working around a clash in its epoch record.

    part2.train builds dict(..., precision=<mixed-precision mode>, **metrics), and the
    validation metrics from part2.score also contain 'precision', so every NER epoch
    raises TypeError. Here part2.score reports span precision as 'span_precision' while
    training runs; nothing else changes, and it stays harmless once part2 is fixed.

    W&B is only logging: if wandb.init fails (no access to the entity, a run ID in use),
    the member trains without it and a warning says why.
    """
    import warnings
    import part2
    original, original_start = part2.score, part2.start_run

    def score(*a, **k):
        result = original(*a, **k)
        result['span_precision'] = result.pop('precision')
        return result

    def start_run(*a, **k):
        try:
            return original_start(*a, **k)
        except Exception as error:
            warnings.warn(f'W&B logging off for this member ({type(error).__name__}: {error}); training continues')
            try:
                import wandb
                wandb.finish(exit_code=1)
            except Exception:
                pass
            return None
    part2.score, part2.start_run = score, start_run
    try:
        return part2.train(*args, **kwargs)
    finally:
        part2.score, part2.start_run = original, original_start


# kind -> function(checkpoint, device) returning (predict(docs) -> reference-token probabilities, info).
# The n-gram model needs its own encoder and hashing code before it can be registered here.
PREDICTORS = {'part2': part2_predictor}


def export_member(member, docs_by_split, out_dir, device='cpu'):
    """Write <out_dir>/<name>/<split>.npz; resumable, and refuses a different checkpoint under the same name."""
    folder = Path(out_dir)/member['name']
    sha = file_hash(member['checkpoint'])
    if (folder/'meta.json').exists() and json.loads((folder/'meta.json').read_text())['checkpoint_sha256'] != sha:
        raise ValueError(f"{member['name']}: bundles exist for another checkpoint; use a new member name")
    todo = {s: d for s, d in docs_by_split.items() if not (folder/f'{s}.npz').exists()}
    meta = dict(member=member['name'], kind=member['kind'], checkpoint=str(member['checkpoint']),
                checkpoint_sha256=sha, wandb_run=member.get('wandb_run'))
    if todo:
        predict, info = PREDICTORS[member['kind']](member['checkpoint'], device)
        meta['model'] = info
        for split, docs in todo.items():
            save_bundle(folder/f'{split}.npz', docs, predict(docs), dict(meta, split=split))
    if not (folder/'meta.json').exists():
        save_json(folder/'meta.json', meta)
    return verify_member(member, docs_by_split, out_dir)


def verify_member(member, docs_by_split, out_dir):
    """Reproduce the F1 logged for this checkpoint; a mismatch means a wrong loader or checkpoint."""
    from ensemble import sweep
    expected = member.get('expected')
    if not expected:
        return None
    docs = docs_by_split[expected['split']]
    probabilities, _ = load_bundle(Path(out_dir)/member['name']/f"{expected['split']}.npz", docs)
    f1 = sweep(docs, probabilities, [expected['threshold']], expected['policy'])[0]['f1']
    if abs(f1-expected['f1']) > expected.get('tolerance', 2e-3):
        raise ValueError(f"{member['name']}: {expected['split']} F1 {f1:.4f} != logged {expected['f1']:.4f}")
    return f1


def fetch_wandb_artifact(reference, out_dir):
    """Download a checkpoint logged as a W&B artifact, e.g. 'entity/project/ngrams-main-model:v0'."""
    import wandb
    return Path(wandb.Api().artifact(reference).download(root=str(out_dir)))
