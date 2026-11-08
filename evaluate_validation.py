"""Evaluate saved NER strings on a frozen partition with group bootstrap CIs.

Prediction CSV: row_id,locations (a JSON or Python-literal list of strings).
This never creates predictions and never reads test targets.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import hard_processing, score_details, document_metrics

BASE=Path(__file__).resolve().parent


def counts_per_text(gold,pred):
    if len(gold)!=len(pred):raise ValueError('Length mismatch')
    counts=[]
    for g,p in zip(gold,pred):
        gg,pp=set(map(hard_processing,g)),set(map(hard_processing,p))
        counts.append([len(gg&pp),len(pp-gg),len(gg-pp)])
    return np.asarray(counts,dtype=np.int64).reshape(-1,3)


def f1_counts(counts):
    tp,fp,fn=np.asarray(counts).T
    denominator=2*tp+fp+fn
    return np.divide(2*tp,denominator,out=np.ones_like(denominator,dtype=float),where=denominator!=0)


def group_bootstrap(gold,pred,groups,repetitions=2000,seed=42):
    if len(gold)!=len(groups):raise ValueError('Group count mismatch')
    if len(gold)==0:raise ValueError('Cannot bootstrap an empty partition')
    codes,unique=pd.factorize(np.asarray(groups,dtype=object))
    if (codes<0).any():raise ValueError('Missing group identifiers')
    grouped=np.zeros((len(unique),3),dtype=np.int64)
    np.add.at(grouped,codes,counts_per_text(gold,pred))
    rng=np.random.default_rng(seed);scores=[]
    for _ in range(repetitions):
        scores.append(float(f1_counts(grouped[rng.integers(0,len(unique),len(unique))].sum(0))))
    return dict(f1_95ci=np.quantile(scores,[.025,.975]).tolist(),bootstrap_groups=len(unique),
                repetitions=repetitions,seed=seed,
                limitation='Conditional on one fitted model and this source-domain partition; excludes training-seed and domain-shift uncertainty.')


def parse_predictions(value):
    result=ast.literal_eval(value)
    if not isinstance(result,list) or any(not isinstance(x,str) for x in result):
        raise ValueError('locations must be a list of strings')
    return result


def evaluate(path,partition,final_audit=False):
    if partition=='audit' and not final_audit:
        raise ValueError('Audit is reserved: use --final-audit only after freezing model and thresholds')
    manifest=pd.read_csv(BASE/'data/validation_splits.csv')
    hold=manifest.loc[manifest.partition.eq(partition)].copy()
    predictions=pd.read_csv(path)
    required={'row_id','locations'}
    if not required.issubset(predictions):raise ValueError('Expected row_id and locations')
    predictions=predictions[['row_id','locations']]
    if predictions.row_id.isna().any() or predictions.row_id.duplicated().any():raise ValueError('Prediction IDs must be present and unique')
    if set(predictions.row_id)!=set(hold.row_id):raise ValueError('Predictions must cover exactly the selected partition; no missing or extra IDs')
    predictions=hold[['row_id','group_id']].merge(predictions,on='row_id',how='left',validate='one_to_one')
    raw=pd.read_csv(BASE/'data/uk_geo_dataset.csv',usecols=['text','loc_markers'])
    gold=[]
    for row_id in predictions.row_id:
        row=raw.iloc[int(row_id)]
        gold.append([row.text[int(a):int(b)] for a,b in ast.literal_eval(row.loc_markers)])
    pred=predictions.locations.map(parse_predictions).tolist()
    details=score_details(gold,pred)
    positive=[i for i,g in enumerate(gold) if len(g)>0]
    negative=[i for i,g in enumerate(gold) if len(g)==0]
    output=dict(partition=partition,rows=len(gold),**details,**document_metrics(gold,pred),
                **group_bootstrap(gold,pred,predictions.group_id),
                positive_text_macro_f1=document_metrics([gold[i] for i in positive],[pred[i] for i in positive])['macro_f1'],
                empty_gold_text_false_positive_rate=float(np.mean([len(pred[i])>0 for i in negative])) if negative else None)
    print(json.dumps(output,indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('predictions');parser.add_argument('--partition',choices=['checkpoint','calibration','audit'],required=True)
    parser.add_argument('--final-audit',action='store_true');args=parser.parse_args()
    evaluate(args.predictions,args.partition,args.final_audit)
