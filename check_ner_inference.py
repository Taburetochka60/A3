"""Forward-pass and long-window checks on unlabeled test; no target scores."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ner_data import tokenize_offsets,token_key,window_ranges,decode_bio
from pilot_ner import infer_probabilities
from recurrent_model import NERModel

BASE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path=BASE/'outputs/ner_pilot_best.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    vocab=saved['vocabulary']
    model=NERModel(len(vocab)+2,**saved['encoder_config']).to(device)
    model.load_state_dict(saved['state_dict']);model.eval()
    test=pd.read_csv(BASE/'data/test.csv',usecols=['text_id','text'])
    docs=[];windows_count=long_texts=covered=0
    for row in test.itertuples():
        tokens=tokenize_offsets(row.text)
        windows=list(window_ranges(len(tokens)))
        coverage=np.zeros(len(tokens),dtype=int)
        for a,b in windows:
            coverage[a:b]+=1
            assert b-a<=128
        assert (coverage>0).all()
        covered+=len(tokens);windows_count+=len(windows);long_texts+=int(len(tokens)>128)
        docs.append(dict(text=row.text,tokens=tokens,ids=[vocab.get(token_key(t.value),1) for t in tokens]))
    probabilities=infer_probabilities(model,docs,device)
    orphan_i=predicted_entities=0
    for doc,prob in zip(docs,probabilities):
        assert prob.shape==(len(doc['tokens']),3)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
        spans,repairs=decode_bio(doc['tokens'],prob.argmax(1));orphan_i+=repairs
        predicted_entities+=len(spans)
        assert all(0<=a<b<=len(doc['text']) and bool(doc['text'][a:b]) for a,b in spans)
        assert spans==sorted(set(spans))
    result=dict(rows=len(docs),long_text_rows=long_texts,total_windows=windows_count,
                checked_token_positions=covered,all_positions_covered=True,
                maximum_tokens=max(len(d['tokens']) for d in docs),
                finite_normalized_probabilities=True,all_prediction_offsets_in_bounds=True,
                predicted_entities=predicted_entities,orphan_I_repaired=orphan_i,
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                no_test_labels_read=True,no_test_quality_metric_computed=True,
                meaning='Structural/inference check only: valid spans do not imply correct locations.')
    (BASE/'reports/ner_long_text_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Make the standalone preprocessing record reflect verified test coverage.
    audit_path=BASE/'reports/ner_preprocessing_audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    audit['test']['all_token_positions_covered']=True
    audit['test']['checked_token_positions']=covered
    audit_path.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
