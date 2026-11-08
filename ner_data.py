"""Offset-preserving word/punctuation BIO preparation and complete windows.

Tokenization never consults gold spans. Original strings are never normalized
before offsets are computed. Only vocabulary keys are NFC/case-folded.
"""
from bisect import bisect_left
from dataclasses import dataclass
import re
import unicodedata

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

TOKEN_RE = re.compile(r'\n|[^\W_]+|[^\s]', re.UNICODE)
O, B, I = 0, 1, 2
IGNORE = -100


@dataclass(frozen=True)
class Token:
    value: str
    start: int
    end: int


def tokenize_offsets(text):
    return [Token(m.group(),m.start(),m.end()) for m in TOKEN_RE.finditer(text)]


def token_key(value):
    return unicodedata.normalize('NFC', value).casefold()


def encode_bio(text, spans, tokens=None):
    """Strict alignment; never snap partial-word spans to a token boundary.

    Whitespace at annotation edges is trimmed for BIO encoding only. Return
    original/trimmed span pairs and issues so this policy is always auditable.
    Any incompatibility makes the entire row unsupervised, rather than turning
    an unrepresentable positive entity into an O target.
    """
    tokens=tokenize_offsets(text) if tokens is None else tokens
    starts={t.start:i for i,t in enumerate(tokens)}
    ends={t.end:i for i,t in enumerate(tokens)}
    labels=np.zeros(len(tokens),dtype=np.int64)
    changes=[];issues=[];trimmed=[];last_end=-1
    for raw in sorted(spans):
        if len(raw)!=2 or any(not isinstance(x,(int,np.integer)) for x in raw):
            issues.append('invalid_span');continue
        a,b=map(int,raw)
        if not 0<=a<b<=len(text):issues.append('out_of_bounds');continue
        old=(a,b)
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if old!=(a,b):changes.append((old,(a,b)))
        if a>=b:issues.append('whitespace_only');continue
        if a<last_end:issues.append('overlap');continue
        last_end=b;trimmed.append((a,b))
        if a not in starts or b not in ends:issues.append('partial_token_boundary');continue
        left,right=starts[a],ends[b]
        labels[left]=B;labels[left+1:right+1]=I
    if issues:return None,trimmed,changes,issues
    assert decode_bio(tokens,labels.tolist())[0]==trimmed, 'BIO roundtrip failed'
    return labels,trimmed,changes,issues


def decode_bio(tokens, tags):
    if len(tokens)!=len(tags):raise ValueError('Token/tag length mismatch')
    spans=[];start=None;end=None;orphan_i=0
    for token,tag in zip(tokens,tags):
        if tag not in [O,B,I]:raise ValueError('Invalid BIO tag')
        if tag==B or (tag==I and start is None):
            if start is not None:spans.append((start,end))
            orphan_i+=int(tag==I)
            start,end=token.start,token.end
        elif tag==I:end=token.end
        else:
            if start is not None:spans.append((start,end))
            start=end=None
    if start is not None:spans.append((start,end))
    return spans,orphan_i


def window_ranges(n,max_tokens=128,overlap=32):
    if not 0<=overlap<max_tokens:raise ValueError('Expected 0 <= overlap < max_tokens')
    if n<0:raise ValueError('Negative token count')
    start=0
    while start<n:
        end=min(n,start+max_tokens)
        yield start,end
        if end==n:break
        start+=max_tokens-overlap


def training_windows(ids,labels,max_tokens=128,overlap=32):
    if len(ids)!=len(labels):raise ValueError('Token/label mismatch')
    seen=np.zeros(len(ids),dtype=bool)
    windows=[]
    for start,end in window_ranges(len(ids),max_tokens,overlap):
        targets=np.array(labels[start:end],copy=True)
        targets[seen[start:end]]=IGNORE
        windows.append((ids[start:end],targets.tolist()))
        seen[start:end]=True
    if len(ids):assert seen.all()
    return windows


def collate_ner(rows):
    ids,labels=zip(*rows)
    if any(len(x)==0 for x in ids):raise ValueError('Empty sequences must bypass the RNN')
    if any(len(x)!=len(y) for x,y in rows):raise ValueError('Token/label length mismatch')
    lengths=torch.tensor([len(x) for x in ids],dtype=torch.long)
    return (pad_sequence([torch.tensor(x,dtype=torch.long) for x in ids],batch_first=True,padding_value=0),
            lengths,
            pad_sequence([torch.tensor(x,dtype=torch.long) for x in labels],batch_first=True,padding_value=IGNORE))


def merge_window_probabilities(n,windows,probabilities):
    """Average overlapping token probabilities, then decode BIO once globally."""
    if len(windows)!=len(probabilities):raise ValueError('Window/prediction mismatch')
    sums=np.zeros((n,3),dtype=np.float64);counts=np.zeros(n,dtype=np.int64)
    for (start,end),prob in zip(windows,probabilities):
        prob=np.asarray(prob)
        if prob.shape!=(end-start,3):raise ValueError('Unexpected probability shape')
        sums[start:end]+=prob;counts[start:end]+=1
    if n and (counts==0).any():raise ValueError('A long-text token has no prediction')
    return sums/counts[:,None] if n else sums
