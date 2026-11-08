"""Short real BiGRU NER training run after strict preprocessing checks.

Uses fit and checkpoint only. Calibration, final audit and target-domain labels
are never evaluated. This is a pipeline pilot, not a full competition solution.
"""
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics import score_details,exact_span_f1
from ner_data import (tokenize_offsets,token_key,encode_bio,training_windows,
                      window_ranges,merge_window_probabilities,decode_bio,collate_ner,IGNORE)
from recurrent_model import NERModel
from validation import ENCODER_CONFIG

BASE=Path(__file__).resolve().parent
SEED=42


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False


def documents(frame):
    docs=[]
    for row in frame.itertuples():
        text=row.text;spans=[tuple(x) for x in ast.literal_eval(row.loc_markers)]
        tokens=tokenize_offsets(text)
        bio,trimmed,changes,issues=encode_bio(text,spans,tokens)
        docs.append(dict(row_id=int(row.Index),text=text,spans=spans,tokens=tokens,
                         bio=bio,trimmed=trimmed,issues=issues))
    return docs


def vocabulary(docs,limit=30000):
    counts=Counter(token_key(t.value) for d in docs for t in d['tokens'])
    return {word:i+2 for i,(word,_) in enumerate(counts.most_common(limit))}


def encode(docs,vocab):
    for doc in docs:doc['ids']=[vocab.get(token_key(t.value),1) for t in doc['tokens']]


def training_loader(docs,seed=SEED):
    examples=[]
    for d in docs:
        if d['bio'] is None:raise ValueError('Incompatible row reached fitting')
        examples.extend(training_windows(d['ids'],d['bio']))
    return DataLoader(examples,batch_size=64,shuffle=True,collate_fn=collate_ner,
                      generator=torch.Generator().manual_seed(seed))


def class_weights(docs,device):
    counts=np.bincount(np.concatenate([d['bio'] for d in docs]),minlength=3)
    weights=np.minimum(np.sqrt(max(1,counts[0])/np.maximum(counts,1)),8.)
    return torch.tensor(weights,dtype=torch.float32,device=device),counts.tolist()


def train_epoch(model,loader,optimizer,weights,device):
    model.train();numerator=denominator=0.;norms=[]
    for step,(ids,lengths,targets) in enumerate(loader):
        ids,lengths,targets=ids.to(device),lengths.to(device),targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(ids,lengths)
        numerator_batch=F.cross_entropy(logits.reshape(-1,3),targets.reshape(-1),weight=weights,ignore_index=IGNORE,reduction='sum')
        denominator_batch=weights[targets[targets!=IGNORE]].sum()
        loss=numerator_batch/denominator_batch
        if not torch.isfinite(loss):raise RuntimeError('Non-finite NER loss')
        loss.backward()
        if step==0 and model.encoder.embedding.weight.grad[0].abs().max().item()!=0:
            raise RuntimeError('Padding embedding received a gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Non-finite NER gradients')
        optimizer.step()
        numerator+=float(numerator_batch.detach());denominator+=float(denominator_batch.detach())
        norms.append(float(norm))
    return dict(train_weighted_ce=numerator/denominator,mean_gradient_norm=float(np.mean(norms)),max_gradient_norm=max(norms))


def infer_probabilities(model,docs,device):
    """Gold-free windowed inference, safe for unlabeled competition texts."""
    rows=[];metadata=[];doc_windows=[]
    for i,d in enumerate(docs):
        windows=list(window_ranges(len(d['ids'])));doc_windows.append(windows)
        for a,b in windows:
            rows.append((d['ids'][a:b],[IGNORE]*(b-a)));metadata.append((i,a,b))
    model.eval();per_doc=[[] for _ in docs];offset=0
    with torch.no_grad():
        for ids,lengths,_ in DataLoader(rows,batch_size=64,collate_fn=collate_ner):
            probs=model(ids.to(device),lengths.to(device)).softmax(-1).cpu().numpy()
            for p,length in zip(probs,lengths.tolist()):
                i,a,b=metadata[offset];offset+=1
                assert length==b-a
                per_doc[i].append(p[:length])
    merged_documents=[]
    for i,d in enumerate(docs):
        merged=merge_window_probabilities(len(d['tokens']),doc_windows[i],per_doc[i])
        if not np.isfinite(merged).all():raise RuntimeError('Non-finite token probabilities')
        merged_documents.append(merged)
    return merged_documents


def predict_documents(model,docs,device):
    merged_documents=infer_probabilities(model,docs,device)
    predictions=[];predicted_spans=[];orphan_count=0;ce_sum=0.;ce_tokens=0
    for d,merged in zip(docs,merged_documents):
        spans,orphans=decode_bio(d['tokens'],merged.argmax(1))
        predicted_spans.append(spans);orphan_count+=orphans
        predictions.append([d['text'][a:b] for a,b in spans])
        if d['bio'] is not None and len(merged):
            ce_sum+=float(-np.log(np.clip(merged[np.arange(len(merged)),d['bio']],1e-12,1)).sum())
            ce_tokens+=len(merged)
    gold=[[d['text'][a:b] for a,b in d['spans']] for d in docs]
    details=score_details(gold,predictions)
    details.update(exact_raw_span_f1=exact_span_f1([d['spans'] for d in docs],predicted_spans)['f1'],
                   exact_trimmed_span_f1=exact_span_f1([d['trimmed'] for d in docs],predicted_spans)['f1'],
                   unweighted_token_ce=ce_sum/ce_tokens if ce_tokens else None,
                   ce_tokens=ce_tokens,ce_excluded_incompatible_rows=sum(d['bio'] is None for d in docs),
                   rows=len(docs),orphan_I_repaired=orphan_count,
                   empty_gold_text_fp_rate=float(np.mean([bool(p) for g,p in zip(gold,predictions) if not g])))
    return details,predictions,predicted_spans


def tiny_memorization(train_docs,device):
    positive=[d for d in train_docs if d['spans']][:16]
    negative=[d for d in train_docs if not d['spans']][:16]
    tiny=copy.deepcopy(positive+negative)
    vocab=vocabulary(tiny);encode(tiny,vocab)
    seed_all(SEED+1)
    model=NERModel(len(vocab)+2,**{**ENCODER_CONFIG,'dropout':0.}).to(device)
    weights,_=class_weights(tiny,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.)
    loader=training_loader(tiny,seed=SEED+1);history=[]
    for step in range(1,251):
        loss=train_epoch(model,loader,optimizer,weights,device)
        if step%20==0 or step==250:
            metrics,_,_=predict_documents(model,tiny,device)
            history.append(dict(update=step,weighted_ce=loss['train_weighted_ce'],f1=metrics['f1'],span_f1=metrics['exact_trimmed_span_f1']))
            if metrics['f1']>=.98 and metrics['exact_trimmed_span_f1']>=.98:break
    result=dict(rows=len(tiny),positive_rows=len(positive),negative_rows=len(negative),
                updates=step,final_metrics=metrics,passed=metrics['f1']>=.98 and metrics['exact_trimmed_span_f1']>=.98,
                meaning='Intentional memorization of fit-only examples tests wiring, not generalization.',history=history)
    print('Tiny memorization:',result['passed'],'F1',metrics['f1'],'updates',step,flush=True)
    if not result['passed']:raise RuntimeError('Tiny-set memorization health check failed')
    return result


def main():
    started=time.time();torch.set_num_threads(4);seed_all(SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expected=json.loads((BASE/'reports/validation_splits_summary.json').read_text())
    manifest_path=BASE/'data/validation_splits.csv'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==expected['manifest_sha256']
    raw=pd.read_csv(BASE/'data/uk_geo_dataset.csv',usecols=['text','loc_markers'])
    assert hashlib.sha256((BASE/'data/uk_geo_dataset.csv').read_bytes()).hexdigest()==expected['dataset_sha256']
    split=pd.read_csv(manifest_path)
    eligible=pd.read_csv(BASE/'data/ner_bio_eligibility.csv')
    preprocessing=json.loads((BASE/'reports/ner_preprocessing_audit.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((BASE/'data/ner_bio_eligibility.csv').read_bytes()).hexdigest()==preprocessing['eligibility_sha256']
    assert eligible.row_id.equals(split.row_id)
    fit=split.loc[split.partition.eq('fit')&eligible.bio_eligible].drop_duplicates('group_id')
    selected=fit.sample(n=12000,random_state=SEED)
    checkpoint=split.loc[split.partition.eq('checkpoint')]
    assert set(selected.group_id).isdisjoint(checkpoint.group_id)
    train_docs=documents(raw.loc[selected.row_id]);valid_docs=documents(raw.loc[checkpoint.row_id])
    assert all(d['bio'] is not None for d in train_docs)
    monitor_indices=np.random.default_rng(SEED).choice(len(train_docs),size=2048,replace=False)
    monitor=[train_docs[i] for i in monitor_indices]
    selection=pd.concat([selected[['row_id','group_id']].assign(role='pilot_fit'),checkpoint[['row_id','group_id']].assign(role='checkpoint')])
    selection['train_monitor']=selection.row_id.isin([d['row_id'] for d in monitor])
    selection.to_csv(BASE/'data/ner_pilot_selection.csv',index=False)
    tiny=tiny_memorization(train_docs,device)
    vocab=vocabulary(train_docs);encode(train_docs,vocab);encode(valid_docs,vocab)
    # Monitor objects reference the already encoded training documents.
    seed_all(SEED)
    model=NERModel(len(vocab)+2,**ENCODER_CONFIG).to(device)
    weights,counts=class_weights(train_docs,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    loader=training_loader(train_docs)
    history=[];best=-1.;best_state=None;best_epoch=None
    for epoch in range(1,9):
        optimization=train_epoch(model,loader,optimizer,weights,device)
        tm,_,_=predict_documents(model,monitor,device)
        vm,_,_=predict_documents(model,valid_docs,device)
        history.append(dict(epoch=epoch,**optimization,
                            **{'train_monitor_'+k:v for k,v in tm.items()},
                            **{'checkpoint_'+k:v for k,v in vm.items()}))
        print('Pilot epoch',epoch,'train F1',round(tm['f1'],4),'checkpoint F1',round(vm['f1'],4),flush=True)
        if vm['f1']>best:
            best=vm['f1'];best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm,train_predictions,_=predict_documents(model,monitor,device)
    vm,valid_predictions,valid_spans=predict_documents(model,valid_docs,device)
    pred_frame=pd.DataFrame(dict(row_id=[d['row_id'] for d in valid_docs],locations=[json.dumps(p,ensure_ascii=False) for p in valid_predictions]))
    pred_frame.to_csv(BASE/'reports/ner_pilot_checkpoint_predictions.csv',index=False)
    errors=[]
    for d,p in zip(valid_docs,valid_predictions):
        g=[d['text'][a:b] for a,b in d['spans']]
        local=score_details([g],[p])
        if local['fp'] or local['fn']:
            errors.append(dict(row_id=d['row_id'],text=d['text'],gold=g,pred=p,
                               alignment_issues=d['issues'],tp=local['tp'],fp=local['fp'],fn=local['fn']))
    (BASE/'reports/ner_pilot_error_examples.json').write_text(json.dumps(errors[:30],ensure_ascii=False,indent=2),encoding='utf-8')
    outputs=BASE/'outputs';outputs.mkdir(exist_ok=True)
    torch.save(dict(state_dict=best_state,vocabulary=vocab,encoder_config=ENCODER_CONFIG,
                    tags={'O':0,'B-LOC':1,'I-LOC':2},epoch=best_epoch,
                    split_sha256=expected['manifest_sha256'],max_tokens=128,overlap=32),outputs/'ner_pilot_best.pt')
    result=dict(seed=SEED,device=str(device),torch_version=torch.__version__,train_rows=len(train_docs),
                train_monitor_rows=len(monitor),checkpoint_rows=len(valid_docs),epochs=8,best_epoch=best_epoch,
                encoder_config=ENCODER_CONFIG,vocabulary_size=len(vocab)+2,vocabulary_fit='Pilot fit texts only',
                tag_counts=counts,class_weights=weights.cpu().tolist(),optimizer='AdamW',lr=.002,weight_decay=.01,
                max_tokens=128,window_overlap=32,padding_label=IGNORE,
                best_train_monitor=tm,best_checkpoint=vm,tiny_memorization=tiny,
                generalization_gap=tm['f1']-vm['f1'],
                checkpoint_OOV_token_fraction=sum(i==1 for d in valid_docs for i in d['ids'])/sum(len(d['ids']) for d in valid_docs),
                elapsed_seconds=time.time()-started,split_sha256=expected['manifest_sha256'],
                evaluation_policy='Argmax BIO, orphan I becomes B, mean overlapping probabilities before global decoding; no threshold tuning',
                calibration_evaluated=False,final_audit_evaluated=False,target_labels_evaluated=False,
                limitations='One seed, 12k subset, eight epochs. Checkpoint selects epoch and is a development score. Gap does not alone establish overfitting.')
    (BASE/'reports/ner_pilot_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    h=pd.DataFrame(history);h.to_csv(BASE/'reports/ner_pilot_history.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for prefix,label in [('train_monitor','Fit monitor (2048)'),('checkpoint','Checkpoint (4000)')]:
        axes[0].plot(h.epoch,h[prefix+'_unweighted_token_ce'],label=label)
        axes[1].plot(h.epoch,h[prefix+'_f1'],label=label)
        axes[2].plot(h.epoch,h[prefix+'_exact_raw_span_f1'],label=label)
    for ax in axes:ax.axvline(best_epoch,ls='--',color='gray',alpha=.6);ax.legend(fontsize=8);ax.set_xlabel('Epoch')
    axes[0].set(title='Comparable inference loss',ylabel='Unweighted token CE')
    axes[1].set(title='Published-code entity F1',ylabel='Micro F1',ylim=(0,1))
    axes[2].set(title='Exact original span F1',ylabel='Micro F1',ylim=(0,1))
    fig.savefig(BASE/'figures/ner_pilot_learning.png',dpi=170);plt.close(fig)
    print('Pilot finished; best epoch',best_epoch,'checkpoint F1',vm['f1'],flush=True)


if __name__=='__main__':main()
