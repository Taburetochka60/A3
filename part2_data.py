import ast
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from ner_data import Token, encode_bio, token_key, tokenize_offsets
from pilot_ner import documents, vocabulary


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def grouping_text(text):
    text = ''.join(c for c in unicodedata.normalize('NFKC', text)
                   if unicodedata.category(c) != 'Cf')
    text = re.sub(r'\b[мМ]\.\s*(?=[А-ЯІЇЄҐA-Z])', 'місті ', text)
    text = re.sub(r'\bвул\.\s*', 'вулиці ', text, flags=re.I)
    text = re.sub(r'\bобл\.\s*', 'області ', text, flags=re.I)
    text = re.sub(r'\bр-н\b', 'район', text, flags=re.I)
    return text


def normalized_words(text):
    text = grouping_text(text)
    words = re.findall(r'[^\W_]+', text.casefold())
    aliases = {'місті': 'місто', 'міста': 'місто', 'області': 'область',
               'району': 'район', 'районі': 'район', 'вулиці': 'вулиця'}
    return tuple(aliases.get(w, w) for w in words)


def separator_signature(text):
    text = grouping_text(text)
    words = list(re.finditer(r'[^\W_]+', text))
    result = []
    dashes = '-‐‑‒–—−'
    for i, (a, b) in enumerate(zip(words, words[1:])):
        separator = text[a.end():b.start()]
        if a.group().isdigit() and b.group().isdigit() and separator.strip() in set(dashes+'.,:/\\'):
            value = separator.strip()
            result.append((i, '-' if value in dashes else value))
        elif a.group().isalpha() and b.group().isalpha() and separator in set(dashes+'/'):
            result.append((i, '-' if separator in dashes else separator))
    return tuple(result)


def text_group(text):
    return fingerprint([normalized_words(text), separator_signature(text)])


def grouping_hash():
    functions = [fingerprint, grouping_text, normalized_words, separator_signature, text_group]
    options = {'show_empty': True} if 'show_empty' in inspect.signature(ast.dump).parameters else {}
    return fingerprint([ast.dump(ast.parse(inspect.getsource(f)), include_attributes=False, **options) for f in functions])


def documents_hash(docs, with_ids=False):
    digest = hashlib.sha256()
    for d in docs:
        value = [d['row_id'], d.get('text_id'), hashlib.sha256(d['text'].encode()).hexdigest(),
                 d['spans'], d['trimmed'], d['bio'].tolist() if d['bio'] is not None else None,
                 d['issues'], d['group'], [(t.value, t.start, t.end) for t in d['tokens']]]
        if with_ids:
            value.append(d['ids'])
        digest.update(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()+b'\n')
    return digest.hexdigest()


def read_split(path):
    spec = json.loads(Path(path).read_text())
    payload = {k: v for k, v in spec.items() if k != 'content_sha256'}
    if fingerprint(payload) != spec['content_sha256']:
        raise ValueError('Frozen split content changed')
    if grouping_hash() != spec['grouping']['implementation_sha256']:
        raise ValueError('Grouping functions changed; verify the migration before using another version')
    return spec


def load_data(data_dir, split_path):
    spec = read_split(split_path)
    data_dir = Path(data_dir)
    for name, expected in spec['files'].items():
        if file_hash(data_dir/name) != expected:
            raise ValueError('Dataset hash mismatch: '+name)
    names = ['fit', 'checkpoint', 'calibration', 'audit', 'domain_source']
    wanted = set(itertools.chain.from_iterable(spec[name] for name in names))
    chunks = []
    for chunk in pd.read_csv(data_dir/'uk_geo_dataset.csv', usecols=['text', 'loc_markers', 'is_valid'],
                             chunksize=50000, keep_default_na=False):
        chunks.append(chunk.loc[chunk.index.isin(wanted)])
    raw = pd.concat(chunks)
    if set(raw.index) != wanted:
        raise ValueError('Missing frozen source rows')
    frames = {name: raw.loc[spec[name]].copy() for name in names}
    frames['test'] = pd.read_csv(data_dir/'test.csv', keep_default_na=False)
    if len(frames['test']) != 477 or not frames['test'].text_id.is_unique:
        raise ValueError('Expected 477 unique test IDs')
    if frames['test'].text_id.tolist() != spec['test_ids']:
        raise ValueError('Test IDs or order changed')
    for name, frame in frames.items():
        keys = [f'test:{int(i)}' for i in frame.text_id] if name == 'test' else [f'uk:{i}' for i in frame.index]
        frame['group'] = [spec['group_overrides'].get(k, text_group(t)) for k, t in zip(keys, frame.text)]
    if not frames['fit'].is_valid.eq(0).all() or not frames['fit'].group.is_unique:
        raise ValueError('Invalid or repeated fit groups')
    for name in ['checkpoint', 'calibration', 'audit']:
        if not frames[name].is_valid.eq(1).all():
            raise ValueError('Official validation rows changed')
    for a, b in itertools.combinations(names[:4], 2):
        if set(frames[a].index) & set(frames[b].index) or set(frames[a].group) & set(frames[b].group):
            raise ValueError('Rows or groups cross protected partitions')
    sample = pd.read_csv(data_dir/'labeling_sample.csv', keep_default_na=False)
    external = set(frames['test'].group) | {text_group(t) for t in sample.text}
    if set(frames['fit'].group) & external:
        raise ValueError('Fit overlaps external groups')
    protected = external | set().union(*(set(frames[n].group) for n in ['checkpoint','calibration','audit']))
    if not frames['domain_source'].is_valid.eq(0).all() or set(frames['domain_source'].group) & protected:
        raise ValueError('AV source overlaps protected data')
    for groups, folds in [(frames['fit'].group.tolist(), spec['cv_folds']),
                          (frames['domain_source'].group.tolist()+frames['test'].set_index('text_id').loc[spec['av_test'], 'group'].tolist(), spec['av_folds'])]:
        seen = {}
        if len(groups) != len(folds) or set(folds) != {0, 1, 2}:
            raise ValueError('Invalid frozen folds')
        for group, fold in zip(groups, folds):
            if group in seen and seen[group] != fold:
                raise ValueError('Group crosses CV/AV folds')
            seen[group] = fold
    return frames, spec


def prepare_data(frames, spec):
    result = {}
    for name, frame in frames.items():
        source = frame.assign(loc_markers='[]') if name == 'test' else frame
        docs = documents(source)
        for i, d in enumerate(docs):
            d['group'] = frame.group.iloc[i]
            if name == 'test':
                d.update(text_id=int(frame.text_id.iloc[i]), bio=None, spans=[], trimmed=[])
        if documents_hash(docs) != spec['documents_sha256'][name]:
            raise ValueError('Text, labels, groups, BIO or offsets changed: '+name)
        result[name] = docs
    if len(result['fit']) != 100000 or any(d['bio'] is None or not d['tokens'] for d in result['fit']):
        raise ValueError('Expected 100000 BIO-compatible fit rows')
    return result


def fit_encoding(train, kind='word'):
    if kind == 'bpe':
        from tokenizers import Tokenizer, models, trainers
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2,
                                       special_tokens=['[PAD]', '[UNK]'], show_progress=False)
        tokenizer.train_from_iterator((t.value for d in train for t in d['tokens']), trainer=trainer)
        return dict(kind=kind, tokenizer_json=tokenizer.to_str(), size=tokenizer.get_vocab_size())
    if kind not in {'word', 'word_raw'}:
        raise ValueError('Unknown tokenizer')
    key = token_key if kind == 'word' else lambda value: value
    words = vocabulary(train, key=key)
    return dict(kind=kind, vocabulary=words, size=len(words)+2)


def encode_documents(docs, state):
    tokenizer = None
    if state['kind'] == 'bpe':
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_str(state['tokenizer_json'])
    result = []
    for d in docs:
        if tokenizer is None:
            key = token_key if state['kind'] == 'word' else lambda value: value
            result.append(dict(d, ids=[state['vocabulary'].get(key(t.value), 1) for t in d['tokens']]))
            continue
        tokens, ids = [], []
        for word in d['tokens']:
            encoded = tokenizer.encode(word.value, add_special_tokens=False)
            end = 0
            for token_id, (a, b) in zip(encoded.ids, encoded.offsets):
                if a != end or b <= a:
                    raise ValueError('Invalid BPE offsets')
                tokens.append(Token(d['text'][word.start+a:word.start+b], word.start+a, word.start+b))
                ids.append(token_id)
                end = b
            if end != len(word.value):
                raise ValueError('BPE lost original characters')
        bio, trimmed, _, issues = encode_bio(d['text'], d['spans'], tokens)
        result.append(dict(d, tokens=tokens, ids=ids, bio=None if 'text_id' in d else bio,
                           trimmed=trimmed, issues=issues))
    return result
