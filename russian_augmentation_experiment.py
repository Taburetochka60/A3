"""Three-seed matched-update auxiliary-corpus experiment; no audit or LB use."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate_validation import counts_per_text,f1_counts
from metrics import score_details
from pilot_ner import (documents,vocabulary,encode,class_weights,training_loader,
                       train_epoch,predict_documents,seed_all)
from recurrent_model import NERModel
from validation import ENCODER_CONFIG


def read_selected(path,indices):
    wanted=set(map(int,indices));parts=[];start=0
    for chunk in pd.read_csv(path,usecols=['text','loc_markers'],chunksize=40000):
        chunk.index=np.arange(start,start+len(chunk));start+=len(chunk)
        keep=[i for i in chunk.index if i in wanted]
        if keep:parts.append(chunk.loc[keep])
    result=pd.concat(parts)
    assert set(result.index)==wanted
    return result


def metrics_and_counts(model,docs,device):
    metrics,pred,spans=predict_documents(model,docs,device)
    gold=[[d['text'][a:b] for a,b in d['spans']] for d in docs]
    return metrics,pred,counts_per_text(gold,pred)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.project_root.resolve()
    started=time.time();torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original=pd.read_csv(root/'data/ner_pilot_selection.csv')
    chosen=pd.read_csv(root/'data/russian_audit/experiment_selection.csv')
    uk_ids=original.loc[original.role.eq('pilot_fit'),'row_id'].tolist()
    valid_ids=original.loc[original.role.eq('checkpoint'),'row_id'].tolist()
    extra_ids=chosen.loc[chosen.role.eq('extra_uk'),'row_id'].tolist()
    ru_ids=chosen.loc[chosen.role.eq('aux_ru'),'row_id'].tolist()
    rv_ids=chosen.loc[chosen.role.eq('ru_diagnostic'),'row_id'].tolist()
    assert set(uk_ids).isdisjoint(extra_ids) and set(uk_ids+extra_ids).isdisjoint(valid_ids)
    uk=read_selected(root/'data/uk_geo_dataset.csv',uk_ids+extra_ids+valid_ids)
    ru=read_selected(root/'data/ru_geo_dataset.csv',ru_ids+rv_ids)
    base_docs=documents(uk.loc[uk_ids]);extra_docs=documents(uk.loc[extra_ids])
    valid_docs=documents(uk.loc[valid_ids]);ru_docs=documents(ru.loc[ru_ids]);rv_docs=documents(ru.loc[rv_ids])
    del uk,ru
    for docs,corpus in [(base_docs,'uk'),(extra_docs,'uk'),(valid_docs,'uk'),(ru_docs,'ru'),(rv_docs,'ru')]:
        for d in docs:d['corpus']=corpus
    assert all(d['bio'] is not None and len(d['tokens'])<=128 for d in base_docs+extra_docs+ru_docs)
    monitor_ix=np.random.default_rng(42).choice(len(base_docs),2048,replace=False)
    monitor=[base_docs[i] for i in monitor_ix]
    arms={'UA_12k':base_docs,'UA_12k_plus_RU_12k':base_docs+ru_docs,'UA_12k_plus_UA_12k':base_docs+extra_docs}
    fixed_weights,base_counts=class_weights(base_docs,device)
    result_rows=[];history=[];all_counts={};prediction_outputs=root/'reports/russian_experiment'
    prediction_outputs.mkdir(exist_ok=True)
    for seed in [42,43,44]:
        for name,fit_docs in arms.items():
            run_started=time.time();seed_all(seed)
            vocab=vocabulary(fit_docs,limit=30000)
            for group in [fit_docs,valid_docs,rv_docs]:encode(group,vocab)
            model=NERModel(len(vocab)+2,**ENCODER_CONFIG).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            loader=training_loader(fit_docs,seed=seed)
            best=-1;best_state=None;best_epoch=None
            for epoch in range(1,9):
                optimization=train_epoch(model,loader,optimizer,fixed_weights,device)
                metrics,_,_=predict_documents(model,valid_docs,device)
                history.append(dict(seed=seed,arm=name,epoch=epoch,**optimization,
                                    checkpoint_f1=metrics['f1'],checkpoint_precision=metrics['precision'],checkpoint_recall=metrics['recall']))
                if metrics['f1']>best:
                    best=metrics['f1'];best_epoch=epoch
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_state)
            metrics,pred,counts=metrics_and_counts(model,valid_docs,device)
            ru_metrics,ru_pred,_=metrics_and_counts(model,rv_docs,device)
            monitor_metrics,_,_=predict_documents(model,monitor,device)
            oov=sum(i==1 for d in valid_docs for i in d['ids'])/sum(len(d['ids']) for d in valid_docs)
            row=dict(seed=seed,arm=name,fit_rows=len(fit_docs),vocabulary_size=len(vocab)+2,
                     optimizer_updates=8*len(loader),best_epoch=best_epoch,checkpoint_f1=metrics['f1'],
                     checkpoint_precision=metrics['precision'],checkpoint_recall=metrics['recall'],
                     checkpoint_span_f1=metrics['exact_raw_span_f1'],checkpoint_OOV=oov,
                     ru_diagnostic_f1=ru_metrics['f1'],ru_diagnostic_precision=ru_metrics['precision'],ru_diagnostic_recall=ru_metrics['recall'],
                     train_monitor_f1=monitor_metrics['f1'],seconds=time.time()-run_started)
            row['fit_positive_rows']=sum(bool(d['spans']) for d in fit_docs)
            row['fit_LOC_mentions']=sum(len(d['spans']) for d in fit_docs)
            row['fit_tokens']=sum(len(d['tokens']) for d in fit_docs)
            result_rows.append(row);all_counts[(seed,name)]=counts
            for corpus,ids,predictions in [('uk',valid_ids,pred),('ru',rv_ids,ru_pred)]:
                pd.DataFrame(dict(corpus=corpus,row_id=ids,locations=[json.dumps(p,ensure_ascii=False) for p in predictions])).to_csv(prediction_outputs/f'{name}_seed{seed}_{corpus}.csv',index=False)
            np.save(prediction_outputs/f'{name}_seed{seed}_uk_counts.npy',counts)
            weight_dir=root/'outputs/russian_experiment';weight_dir.mkdir(exist_ok=True,parents=True)
            torch.save(dict(state_dict=best_state,vocabulary=vocab,encoder_config=ENCODER_CONFIG,
                            epoch=best_epoch,seed=seed,arm=name,max_tokens=128,overlap=32),
                       weight_dir/f'{name}_seed{seed}.pt')
            print('Finished',name,'seed',seed,'UA F1',round(metrics['f1'],4),'RU F1',round(ru_metrics['f1'],4),flush=True)
            pd.DataFrame(result_rows).to_csv(root/'reports/ru_augmentation_runs.csv',index=False)
            pd.DataFrame(history).to_csv(root/'reports/ru_augmentation_history.csv',index=False)
            del model,optimizer,loader,best_state
            if torch.cuda.is_available():torch.cuda.empty_cache()
    runs=pd.DataFrame(result_rows)
    comparisons=[];rng=np.random.default_rng(42)
    contrasts=[('UA_12k_plus_RU_12k','UA_12k'),('UA_12k_plus_UA_12k','UA_12k'),('UA_12k_plus_RU_12k','UA_12k_plus_UA_12k')]
    for a,b in contrasts:
        deltas=[]
        for seed in [42,43,44]:
            av=runs.loc[(runs.seed==seed)&(runs.arm==a),'checkpoint_f1'].iloc[0]
            bv=runs.loc[(runs.seed==seed)&(runs.arm==b),'checkpoint_f1'].iloc[0]
            deltas.append(float(av-bv))
        boots=[]
        for _ in range(2000):
            ix=rng.integers(0,len(valid_docs),len(valid_docs))
            boots.append(np.mean([float(f1_counts(all_counts[(seed,a)][ix].sum(0)))-float(f1_counts(all_counts[(seed,b)][ix].sum(0))) for seed in [42,43,44]]))
        comparisons.append(dict(treatment=a,reference=b,seed_deltas=deltas,
                                mean_delta=float(np.mean(deltas)),conditional_paired_document_95ci=np.quantile(boots,[.025,.975]).tolist()))
    summary=dict(seeds=[42,43,44],runs=result_rows,comparisons=comparisons,
                 fixed_loss_weights=fixed_weights.cpu().tolist(),base_UA_tag_counts=base_counts,
                 encoder_config=ENCODER_CONFIG,epochs=8,batch_size=64,lr=.002,weight_decay=.01,
                 vocabulary_cap=30000,checkpoint_rows=4000,ru_diagnostic_rows=2000,
                 selection_policy='Same 12k base, fixed additional 12k RU or UA; selected group-safe pools; same eight epochs. Mixed RU/UA controls match optimizer updates and example counts, not exact FLOPs/token counts.',
                 checkpoint_policy='Best epoch on same development checkpoint; never calibration or final audit',
                 interval_policy='Paired resampling of 4000 unique checkpoint groups, same resample across three fixed seeds; conditional on fitted/selected models, excludes unseen-seed/selection/domain uncertainty.',
                 source_selection_sha256=hashlib.sha256((root/'data/russian_audit/experiment_selection.csv').read_bytes()).hexdigest(),
                 no_final_audit=True,no_calibration=True,no_target_labels=True,elapsed_seconds=time.time()-started,
                 limitations='Small sampled pilot, random embeddings and shared capped vocabulary; not a verdict on all Russian data, pretrained multilingual embeddings, larger training, pretraining/fine-tuning, or Public LB.')
    (root/'reports/ru_augmentation_experiment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    grouped=runs.groupby('arm').agg(checkpoint_mean=('checkpoint_f1','mean'),checkpoint_sd=('checkpoint_f1','std'),
                                   ru_mean=('ru_diagnostic_f1','mean'),oov_mean=('checkpoint_OOV','mean')).reset_index()
    grouped.to_csv(root/'reports/ru_augmentation_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    order=list(arms);x=np.arange(3)
    for i,seed in enumerate([42,43,44]):
        for ax,column in [(axes[0],'checkpoint_f1'),(axes[1],'ru_diagnostic_f1')]:
            vals=[runs.loc[(runs.seed==seed)&(runs.arm==arm),column].iloc[0] for arm in order]
            ax.plot(x,vals,marker='o',label=f'Seed {seed}')
    for ax in axes:ax.set(xticks=x,xticklabels=['UA 12k','UA 12k + RU 12k','UA 12k + UA 12k'],ylabel='Official-code entity F1',ylim=(0,1));ax.tick_params(axis='x',rotation=12);ax.legend()
    axes[0].set_title('Same Ukrainian checkpoint (4000)');axes[1].set_title('Separate Russian diagnostic holdout (2000)')
    fig.savefig(root/'figures/ru_augmentation_experiment.png',dpi=170);plt.close(fig)
    print('Three-seed experiment and paired comparisons complete',flush=True)


if __name__=='__main__':main()
