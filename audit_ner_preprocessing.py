"""Full-corpus BIO/offset eligibility census; no model sees final audit texts."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ner_data import tokenize_offsets,encode_bio,window_ranges

BASE=Path(__file__).resolve().parent


def main():
    data=pd.read_csv(BASE/'data/uk_geo_dataset.csv',usecols=['text','loc_markers'])
    split=pd.read_csv(BASE/'data/validation_splits.csv',usecols=['row_id','partition'])
    assert split.row_id.tolist()==list(range(len(data)))
    counters={};examples=[];eligible=np.ones(len(data),dtype=bool)
    for row in data.itertuples():
        partition=split.partition.iloc[row.Index]
        c=counters.setdefault(partition,Counter())
        tokens=tokenize_offsets(row.text);spans=ast.literal_eval(row.loc_markers)
        assert ''.join(t.value for t in tokens if t.value!='\n')==''.join(x for x in row.text if not x.isspace())
        labels,trimmed,changes,issues=encode_bio(row.text,spans,tokens)
        c['rows']+=1;c['gold_spans']+=len(spans);c['trimmed_whitespace_spans']+=len(changes)
        c['tokens']+=len(tokens);c['long_text_rows']+=int(len(tokens)>128)
        c['alignment_incompatible_rows']+=int(bool(issues))
        if issues:
            eligible[row.Index]=False
            c.update(issues)
            if len(examples)<20:examples.append(dict(row_id=row.Index,partition=partition,text=row.text,spans=spans,issues=issues))
        else:
            c['roundtrip_passed_rows']+=1;c['roundtrip_passed_spans']+=len(trimmed)
        if (row.Index+1)%200000==0:print('Audited',row.Index+1,'rows',flush=True)
    eligibility=split.assign(bio_eligible=eligible)
    path=BASE/'data/ner_bio_eligibility.csv';eligibility.to_csv(path,index=False)
    test=pd.read_csv(BASE/'data/test.csv')
    lengths=test.text.map(lambda s:len(tokenize_offsets(s)))
    test_windows=[list(window_ranges(n)) for n in lengths]
    for n,windows in zip(lengths,test_windows):
        coverage=np.zeros(n,dtype=int)
        for start,end in windows:
            assert 0<=start<end<=n and end-start<=128
            coverage[start:end]+=1
        assert (coverage>0).all()
    result=dict(tokenizer='Word/punctuation/newline, including underscores and combining marks; original offsets',
                token_pattern=__import__('ner_data').TOKEN_RE.pattern,
                whitespace_policy='Trim annotation edge whitespace for BIO only; original gold is retained for scoring',
                incompatible_fit_policy='Exclude entire row from pilot fitting; never expand partial-word labels or infer unannotated O targets',
                heldout_policy='Keep every checkpoint text and original gold, including incompatible rows; such rows are excluded only from token CE',
                by_partition={k:dict(v) for k,v in counters.items()},
                eligibility_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                examples=examples,
                test=dict(rows=len(test),longer_than_128_tokens=int((lengths>128).sum()),
                          maximum_tokens=int(lengths.max()),total_windows=sum(map(len,test_windows)),
                          all_token_positions_covered=True),
                final_audit_model_evaluation=False)
    (BASE/'reports/ner_preprocessing_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(result['by_partition']).T.to_csv(BASE/'reports/ner_preprocessing_counts.csv')
    print(json.dumps({k:dict(v) for k,v in counters.items()},indent=2),flush=True)


if __name__=='__main__':main()
