"""Frozen group-aware holdouts and recurrent adversarial validation.

No NER test labels are observed or created. Domain labels indicate file origin.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from recurrent_model import DomainModel, NERModel
from ner_data import TOKEN_RE, tokenize_offsets, token_key

BASE = Path(__file__).resolve().parent
SEED = 42
ENCODER_CONFIG = dict(embedding_dim=64, hidden_size=64, dropout=0.2)


def canonical(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def group_id(text):
    return hashlib.sha256(canonical(text).encode('utf-8')).hexdigest()


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_splits():
    train = pd.read_csv(BASE/'data/uk_geo_dataset.csv', usecols=['text','loc_markers','is_valid'])
    test = pd.read_csv(BASE/'data/test.csv')
    sample = pd.read_csv(BASE/'data/labeling_sample.csv')
    groups = train.text.map(group_id)
    reserved = set(groups[train.is_valid.eq(1)])
    in_domain = set(sample.text.map(group_id))
    test_groups = set(test.text.map(group_id))
    manifest = pd.DataFrame(dict(row_id=np.arange(len(train)),group_id=groups,
                                 official_is_valid=train.is_valid.to_numpy(),partition='fit'))
    excluded = train.is_valid.eq(0) & groups.isin(reserved|in_domain|test_groups)
    manifest.loc[excluded,'partition'] = 'excluded_overlap'
    vi = train.index[train.is_valid.eq(1)].to_numpy()
    count = train.loc[vi,'loc_markers'].map(lambda x:len(ast.literal_eval(x)))
    length_bin = pd.cut(train.loc[vi,'text'].str.split().str.len(),[-1,15,30,float('inf')],labels=['short','medium','long'])
    strata = count.clip(upper=2).astype(str)+'_'+length_bin.astype(str)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold,(_,hold) in enumerate(cv.split(vi,strata,groups.iloc[vi])):
        partition = 'checkpoint' if fold in [0,1] else ('calibration' if fold==2 else 'audit')
        manifest.loc[vi[hold],'partition'] = partition
    active=manifest.loc[manifest.partition.ne('excluded_overlap')]
    assert active.groupby('group_id').partition.nunique().max()==1
    assert len(manifest)==len(train) and manifest.row_id.is_unique
    assert set(manifest.loc[manifest.partition.eq('fit'),'group_id']).isdisjoint(test_groups|in_domain|reserved)
    rows=[]
    for name,part in manifest.groupby('partition'):
        counts=train.loc[part.row_id,'loc_markers'].map(lambda x:len(ast.literal_eval(x)))
        rows.append(dict(partition=name,rows=len(part),groups=part.group_id.nunique(),
                         loc_mentions=int(counts.sum()),positive_rows=int(counts.gt(0).sum()),
                         no_location_fraction=float(counts.eq(0).mean())))
    path=BASE/'data/validation_splits.csv'
    manifest.to_csv(path,index=False)
    summary=dict(seed=SEED,grouping='sha256(NFC + casefold + whitespace collapse)',
                 partitions=rows,excluded_fit_rows=int(excluded.sum()),
                 cross_partition_duplicate_groups=0,
                 manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 dataset_sha256=hashlib.sha256((BASE/'data/uk_geo_dataset.csv').read_bytes()).hexdigest(),
                 rule='Official is_valid=1 only: SGKF(5), folds 0/1 checkpoint, 2 calibration, 3/4 audit. Fit excludes reserved/sample/test canonical overlaps.')
    (BASE/'reports/validation_splits_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(BASE/'reports/validation_partitions.csv',index=False)
    print('Frozen partitions:',rows,flush=True)
    return train,manifest


def tokenize(text):
    # Retain newlines and punctuation: these also occur in NER input.
    return [token_key(t.value) for t in tokenize_offsets(text)][:128] or ['<EMPTY>']


def collate(rows):
    sequences, labels = zip(*rows)
    lengths=torch.tensor([len(s) for s in sequences],dtype=torch.long)
    return pad_sequence([torch.tensor(s,dtype=torch.long) for s in sequences],batch_first=True), lengths, torch.tensor(labels,dtype=torch.float32)


def predict(model,loader,device):
    model.eval();probabilities=[]
    with torch.no_grad():
        for ids,lengths,_ in loader:
            logits=model(ids.to(device),lengths.to(device))
            probabilities.extend(logits.sigmoid().cpu().numpy().tolist())
    return np.asarray(probabilities)


def run_adversarial(train,manifest,epochs=10):
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fit=train.loc[manifest.partition.eq('fit'),['text']].copy()
    fit['group_id']=fit.text.map(group_id)
    fit=fit.drop_duplicates('group_id')
    target=pd.read_csv(BASE/'data/test.csv')
    target['group_id']=target.text.map(group_id)
    target=target.drop_duplicates('group_id')
    source=fit.sample(n=len(target),random_state=SEED).copy()
    source['domain']=0;target['domain']=1
    source['source_row_id']=source.index
    target['source_row_id']=target.text_id
    data=pd.concat([source[['text','group_id','domain','source_row_id']],target[['text','group_id','domain','source_row_id']]],ignore_index=True)
    tokens=[tokenize(t) for t in data.text]
    y=data.domain.to_numpy();oof=np.full(len(data),np.nan);fold_ids=np.full(len(data),-1)
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    histories=[];fold_results=[]
    for fold,(train_idx,val_idx) in enumerate(cv.split(data,y,data.group_id)):
        seed_everything(SEED+fold)
        assert set(data.group_id.iloc[train_idx]).isdisjoint(data.group_id.iloc[val_idx])
        counts=Counter(t for i in train_idx for t in tokens[i])
        vocab={word:i+2 for i,(word,_) in enumerate(counts.most_common(12000))}
        encoded=[[vocab.get(t,1) for t in row] for row in tokens]
        generator=torch.Generator().manual_seed(SEED+fold)
        tr=DataLoader([(encoded[i],y[i]) for i in train_idx],batch_size=64,shuffle=True,collate_fn=collate,generator=generator)
        va=DataLoader([(encoded[i],y[i]) for i in val_idx],batch_size=64,shuffle=False,collate_fn=collate)
        model=DomainModel(len(vocab)+2,**ENCODER_CONFIG).to(device)
        # Both tasks use exactly this encoder class and config; only their heads differ.
        reference=NERModel(len(vocab)+2,**ENCODER_CONFIG)
        assert [(n,tuple(p.shape)) for n,p in model.encoder.named_parameters()]==[(n,tuple(p.shape)) for n,p in reference.encoder.named_parameters()]
        del reference
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.01)
        loss_fn=torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            model.train();total=0;seen=0;norms=[]
            for ids,lengths,labels in tr:
                ids,lengths,labels=ids.to(device),lengths.to(device),labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(ids,lengths),labels)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite domain loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if not torch.isfinite(norm):raise RuntimeError('Non-finite domain gradients')
                optimizer.step();total+=float(loss.item())*len(labels);seen+=len(labels);norms.append(float(norm))
            p=predict(model,va,device)
            histories.append(dict(fold=fold,epoch=epoch+1,train_bce=total/seen,heldout_auc=roc_auc_score(y[val_idx],p),heldout_bce=log_loss(y[val_idx],p,labels=[0,1]),mean_gradient_norm=float(np.mean(norms))))
        # Fixed final epoch. Held-out AUC never selects an epoch or configuration.
        oof[val_idx]=p;fold_ids[val_idx]=fold
        unknown=sum(t not in vocab for i in val_idx for t in tokens[i])/sum(len(tokens[i]) for i in val_idx)
        fold_results.append(dict(fold=fold,train_rows=len(train_idx),heldout_rows=len(val_idx),auc=roc_auc_score(y[val_idx],p),oov_token_fraction=unknown,vocab_size=len(vocab)))
        print('Adversarial fold',fold,'AUC',fold_results[-1]['auc'],flush=True)
    assert np.isfinite(oof).all() and (fold_ids>=0).all()
    rng=np.random.default_rng(SEED)
    # Stratified text bootstrap: domain labels are balanced, no duplicate groups.
    neg,pos=np.flatnonzero(y==0),np.flatnonzero(y==1)
    boot=[]
    for _ in range(2000):
        ix=np.r_[rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)]
        boot.append(roc_auc_score(y[ix],oof[ix]))
    results=dict(seed=SEED,device=str(device),source_rows=len(source),test_rows=len(target),
                 encoder='RecurrentEncoder: embedding + 1-layer bidirectional GRU; no attention',
                 encoder_config=ENCODER_CONFIG,epochs=epochs,batch_size=64,optimizer='AdamW',lr=.002,weight_decay=.01,
                 maximum_tokens=128,tokenizer=TOKEN_RE.pattern,vocabulary_cap=12000,
                 vocabulary_fit='Each domain-training fold only; unknown index=1, padding index=0',
                 oof_roc_auc=float(roc_auc_score(y,oof)),oof_accuracy=float(accuracy_score(y,oof>=.5)),
                 oof_bce=float(log_loss(y,oof,labels=[0,1])),bootstrap_auc_95ci=[float(x) for x in np.quantile(boot,[.025,.975])],
                 bootstrap_caveat='Conditional text bootstrap of fixed OOF predictions; does not include retraining/seed uncertainty.',
                 folds=fold_results,domain_label_definition='0=uniform sampled Ukrainian fit text; 1=unlabeled competition test text',
                 main_task_status='Shared BiGRU encoder/tokenizer; short NER pilot is documented separately when available. Final full-scale model is not fixed. Rerun AV if final encoder/preprocessing changes.')
    data[['group_id','domain','source_row_id']].assign(fold=fold_ids,p_test_domain=oof).to_csv(BASE/'reports/adversarial_oof.csv',index=False)
    pd.DataFrame(histories).to_csv(BASE/'reports/adversarial_history.csv',index=False)
    (BASE/'reports/adversarial_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    fpr,tpr,_=roc_curve(y,oof)
    axes[0].plot(fpr,tpr,label=f'OOF AUC={results["oof_roc_auc"]:.3f}')
    axes[0].plot([0,1],[0,1],ls='--',color='gray');axes[0].legend()
    axes[0].set(title='Recurrent domain classifier: ROC',xlabel='False-positive rate',ylabel='True-positive rate')
    for domain,label in [(0,'Ukrainian source'),(1,'Unlabeled test')]:
        axes[1].hist(oof[y==domain],bins=np.linspace(0,1,21),alpha=.6,label=label)
    axes[1].legend();axes[1].set(title='OOF domain scores',xlabel='P(test domain)',ylabel='Texts')
    for fold in range(3):
        h=pd.DataFrame(histories).query('fold == @fold')
        axes[2].plot(h.epoch,h.train_bce,label=f'Fold {fold}')
    axes[2].legend();axes[2].set(title='Optimization health',xlabel='Epoch',ylabel='Training binary cross-entropy')
    fig.savefig(BASE/'figures/adversarial_validation.png',dpi=170);plt.close(fig)
    print('OOF AUC',results['oof_roc_auc'],'95% conditional CI',results['bootstrap_auc_95ci'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--splits-only',action='store_true');args=parser.parse_args()
    train,manifest=make_splits()
    if not args.splits_only:run_adversarial(train,manifest)
