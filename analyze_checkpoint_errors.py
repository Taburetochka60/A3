"""Frozen-checkpoint error census, novelty slices, and 200 review cases.

No training, threshold search, test labels, calibration or final audit scoring.
Official normalized surface-set errors and occurrence/span errors are kept
separate. Location novelty is measured against the actual 12k pilot fit only.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch

from metrics import hard_processing, score_details, conservative_processing, exact_span_f1
from ner_data import tokenize_offsets,token_key,decode_bio
from pilot_ner import infer_probabilities
from recurrent_model import NERModel

BASE=Path(__file__).resolve().parent


def overlap(a,b):
    return max(0,min(a[1],b[1])-max(a[0],b[0]))


def boundary_kind(text,gold,pred):
    """Observed character-boundary relation, not a causal explanation."""
    if gold==pred:return 'exact'
    if not overlap(gold,pred):return 'no_overlap'
    g,p=text[gold[0]:gold[1]],text[pred[0]:pred[1]]
    def strip_edges(s):
        def edge(c):return c.isspace() or unicodedata.category(c).startswith('P')
        while s and edge(s[0]):s=s[1:]
        while s and edge(s[-1]):s=s[:-1]
        return s
    if strip_edges(g)==strip_edges(p):return 'punctuation_or_whitespace'
    if gold[0]==pred[0]:
        delta=text[min(gold[1],pred[1]):max(gold[1],pred[1])]
        if delta and all(unicodedata.category(c)[0] in ['L','M'] for c in delta):
            return 'word_ending_fragment'
    if pred[0]<=gold[0] and pred[1]>=gold[1]:return 'extra_neighbor_material'
    if gold[0]<=pred[0] and gold[1]>=pred[1]:return 'missing_name_material'
    return 'shifted_partial_overlap'


def span_diagnostics(text,gold,pred):
    """Exact occurrence matches first, then maximum-cardinality overlap pairs."""
    gold=list(map(tuple,gold));pred=list(map(tuple,pred))
    exact=set(gold)&set(pred)
    g=[x for x in gold if x not in exact];p=[x for x in pred if x not in exact]
    pairs=[]
    if g and p:
        scores=np.zeros((len(g),len(p)))
        for i,a in enumerate(g):
            for j,b in enumerate(p):
                inter=overlap(a,b)
                if inter:scores[i,j]=1000+inter/(max(a[1],b[1])-min(a[0],b[0]))
        ii,jj=linear_sum_assignment(-scores)
        for i,j in zip(ii,jj):
            if scores[i,j]>0:
                pairs.append(dict(gold=list(g[i]),pred=list(p[j]),
                                  gold_text=text[g[i][0]:g[i][1]],pred_text=text[p[j][0]:p[j][1]],
                                  kind=boundary_kind(text,g[i],p[j])))
    # Component cardinalities distinguish merge/split topology, not just pairs.
    edges={(i,j) for i,a in enumerate(g) for j,b in enumerate(p) if overlap(a,b)}
    components=[]
    while edges:
        i,j=next(iter(edges));gg={i};pp={j}
        changed=True
        while changed:
            before=(len(gg),len(pp))
            for a,b in edges:
                if a in gg or b in pp:gg.add(a);pp.add(b)
            changed=before!=(len(gg),len(pp))
        edges={e for e in edges if not(e[0] in gg and e[1] in pp)}
        name=('one_to_one_boundary' if len(gg)==len(pp)==1 else
              'merged_locations' if len(gg)>1 and len(pp)==1 else
              'split_location' if len(gg)==1 and len(pp)>1 else 'complex_overlap')
        components.append(dict(kind=name,gold_count=len(gg),pred_count=len(pp)))
    return dict(exact_matches=len(exact),unmatched_gold=len(g),unmatched_pred=len(p),pairs=pairs,components=components,
                gold_with_no_pred_overlap=sum(not any(overlap(a,b) for b in pred) for a in g),
                pred_with_no_gold_overlap=sum(not any(overlap(a,b) for a in gold) for b in p))


def count_scores(tp,fp,fn):
    precision=tp/(tp+fp) if tp+fp else float(tp+fn==0)
    recall=tp/(tp+fn) if tp+fn else 1.
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=precision,recall=recall,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)


def frequency_bin(n):
    return 'unseen_0' if n==0 else ('seen_1' if n==1 else 'seen_2_to_4' if n<5 else 'seen_5_plus')


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw=pd.read_csv(BASE/'data/uk_geo_dataset.csv')
    chosen=pd.read_csv(BASE/'data/ner_pilot_selection.csv')
    fitting=chosen.loc[chosen.role.eq('pilot_fit')]
    frozen=pd.read_csv(BASE/'data/validation_splits.csv')
    saved_predictions=pd.read_csv(BASE/'reports/ner_pilot_checkpoint_predictions.csv')
    assert set(saved_predictions.row_id)==set(frozen.loc[frozen.partition.eq('checkpoint'),'row_id'])
    assert set(fitting.row_id).isdisjoint(saved_predictions.row_id)
    assert len(fitting)==12000 and len(saved_predictions)==4000
    model_path=BASE/'outputs/ner_pilot_best.pt'
    saved=torch.load(model_path,map_location='cpu',weights_only=True)
    vocab=saved['vocabulary']
    model=NERModel(len(vocab)+2,**saved['encoder_config']).to(device)
    model.load_state_dict(saved['state_dict'])
    known=Counter();conservative_known=Counter()
    for row in raw.loc[fitting.row_id].itertuples():
        for a,b in ast.literal_eval(row.loc_markers):
            known[hard_processing(row.text[a:b])]+=1
            conservative_known[conservative_processing(row.text[a:b])]+=1
    docs=[]
    for row_id in saved_predictions.row_id:
        row=raw.loc[row_id];tokens=tokenize_offsets(row.text)
        docs.append(dict(row_id=int(row_id),text=row.text,tokens=tokens,
                         ids=[vocab.get(token_key(t.value),1) for t in tokens],
                         gold=list(map(tuple,ast.literal_eval(row.loc_markers))),
                         org=list(map(tuple,ast.literal_eval(row.org_markers))),
                         per=list(map(tuple,ast.literal_eval(row.per_markers)))))
    probabilities=infer_probabilities(model,docs,device)
    all_gold=[];all_pred=[];all_pred_spans=[];events=[];doc_records=[];pairs=[]
    novelty={name:np.zeros((len(docs),3),dtype=np.int64) for name in ['seen','unseen']}
    frequency=Counter();gold_oov=Counter();strict_novel=Counter();semantic=Counter();topology=Counter()
    for di,(doc,prob,csv_row) in enumerate(zip(docs,probabilities,saved_predictions.itertuples())):
        text=doc['text'];pred_spans,_=decode_bio(doc['tokens'],prob.argmax(1))
        predicted=[text[a:b] for a,b in pred_spans]
        assert predicted==ast.literal_eval(csv_row.locations),f'Frozen predictions changed at {doc["row_id"]}'
        gold=[text[a:b] for a,b in doc['gold']]
        all_gold.append(gold);all_pred.append(predicted);all_pred_spans.append(pred_spans)
        gg={hard_processing(x) for x in gold};pp={hard_processing(x) for x in predicted}
        for norm in gg|pp:
            seen='seen' if known[norm]>0 else 'unseen'
            status=0 if norm in gg&pp else 1 if norm in pp else 2
            novelty[seen][di,status]+=1
            frequency[(frequency_bin(known[norm]),['tp','fp','fn'][status])]+=1
        diag=span_diagnostics(text,doc['gold'],pred_spans)
        for pair in diag['pairs']:pairs.append(dict(row_id=doc['row_id'],**pair))
        topology.update(x['kind'] for x in diag['components'])
        for a,b in doc['gold']:
            seen='seen' if known[hard_processing(text[a:b])]>0 else 'unseen'
            strict_novel[(seen,'gold')]+=1
            strict_novel[(seen,'exact_tp')]+=int((a,b) in pred_spans)
        for norm in gg:
            spans=[s for s in doc['gold'] if hard_processing(text[s[0]:s[1]])==norm]
            contains_unk=any(doc['ids'][j]==1 for a,b in spans for j,t in enumerate(doc['tokens']) if overlap((a,b),(t.start,t.end)))
            tag='contains_UNK' if contains_unk else 'all_tokens_in_vocab'
            gold_oov[(tag,'gold')]+=1;gold_oov[(tag,'tp')]+=int(norm in pp)
        for kind,norms,spans in [('FP',pp-gg,pred_spans),('FN',gg-pp,doc['gold'])]:
            for norm in sorted(norms):
                occurrences=[s for s in spans if hard_processing(text[s[0]:s[1]])==norm]
                other=doc['gold'] if kind=='FP' else pred_spans
                overlapping=[(s,t) for s in occurrences for t in other if overlap(s,t)]
                contains_unk=any(doc['ids'][j]==1 for a,b in occurrences for j,t in enumerate(doc['tokens']) if overlap((a,b),(t.start,t.end)))
                aux_per=any(overlap(s,t) for s in occurrences for t in doc['per'])
                aux_org=any(overlap(s,t) for s in occurrences for t in doc['org'])
                if overlapping:category='boundary_overlap'
                elif kind=='FP':category='PER_annotation_overlap' if aux_per else 'ORG_annotation_overlap' if aux_org else 'other_unannotated_FP'
                else:category='missed_with_UNK' if contains_unk else 'missed_unseen_surface_known_tokens' if not known[norm] else 'missed_seen_surface'
                semantic[(kind,category)]+=1
                best=max(overlapping,key=lambda pair:overlap(*pair)) if overlapping else None
                relation=boundary_kind(text,best[1],best[0]) if best and kind=='FP' else boundary_kind(text,*best) if best else None
                events.append(dict(event_id=len(events)+1,row_id=doc['row_id'],kind=kind,category=category,
                                   surface=text[occurrences[0][0]:occurrences[0][1]],normalized=norm,
                                   spans=[list(s) for s in occurrences],seen_in_pilot=bool(known[norm]),
                                   pilot_training_mentions=int(known[norm]),contains_UNK=bool(contains_unk),
                                   boundary_relation=relation,
                                   counterpart=text[best[1][0]:best[1][1]] if best else None,
                                   auxiliary_PER_overlap=bool(aux_per),auxiliary_ORG_overlap=bool(aux_org)))
        details=score_details([gold],[predicted])
        doc_records.append(dict(row_id=doc['row_id'],text=text,gold=gold,pred=predicted,
                                **details,strict=diag,org=[text[a:b] for a,b in doc['org']],
                                per=[text[a:b] for a,b in doc['per']]))
    overall=score_details(all_gold,all_pred)
    expected=json.loads((BASE/'reports/ner_pilot_results.json').read_text())['best_checkpoint']
    assert all(overall[k]==expected[k] for k in ['tp','fp','fn','f1'])
    assert sum(e['kind']=='FP' for e in events)==917 and sum(e['kind']=='FN' for e in events)==431
    slices={}
    rng=np.random.default_rng(42)
    boots={key:[] for key in novelty}
    for _ in range(2000):
        ix=rng.integers(0,len(docs),len(docs))
        for key,counts in novelty.items():
            metric=count_scores(*counts[ix].sum(0));boots[key].append((metric['recall'],metric['f1']))
    for key,counts in novelty.items():
        metric=count_scores(*counts.sum(0));boot=np.asarray(boots[key])
        metric.update(recall_95ci=np.quantile(boot[:,0],[.025,.975]).tolist(),
                      f1_95ci=np.quantile(boot[:,1],[.025,.975]).tolist(),
                      raw_gold_mentions=int(strict_novel[(key,'gold')]),
                      strict_exact_mention_recall=strict_novel[(key,'exact_tp')]/max(1,strict_novel[(key,'gold')]))
        slices[key]=metric
    assert all(sum(slices[key][field] for key in slices)==overall[field] for field in ['tp','fp','fn'])
    frequency_rows=[]
    for name in ['unseen_0','seen_1','seen_2_to_4','seen_5_plus']:
        frequency_rows.append(dict(frequency_bin=name,**count_scores(*(frequency[(name,status)] for status in ['tp','fp','fn']))))
    oov_rows=[]
    for tag in ['all_tokens_in_vocab','contains_UNK']:
        count=gold_oov[(tag,'gold')];tp=gold_oov[(tag,'tp')]
        oov_rows.append(dict(gold_slice=tag,gold=count,tp=tp,recall=tp/count if count else None))
    # 200 unique erroneous texts, with one focal event each. Deliberate balance:
    # 100 FP / 100 FN, covering boundary/non-overlap when available.
    selected=[];used=set();rng=np.random.default_rng(2026)
    for kind in ['FN','FP']:
        for boundary in [True,False]:
            pool=[e for e in events if e['kind']==kind and (e['category']=='boundary_overlap')==boundary]
            for idx in rng.permutation(len(pool)):
                e=pool[idx]
                if e['row_id'] in used:continue
                selected.append(dict(e));used.add(e['row_id'])
                if sum(x['kind']==kind and (x['category']=='boundary_overlap')==boundary for x in selected)>=50:break
        pool=[e for e in events if e['kind']==kind]
        for idx in rng.permutation(len(pool)):
            if sum(x['kind']==kind for x in selected)>=100:break
            e=pool[idx]
            if e['row_id'] not in used:selected.append(dict(e));used.add(e['row_id'])
    assert len(selected)==200 and len(used)==200
    by_row={d['row_id']:d for d in doc_records}
    for i,e in enumerate(selected,1):
        e['case_id']=i
        d=by_row[e['row_id']]
        e.update(text=d['text'],gold=d['gold'],pred=d['pred'],ORG_annotations=d['org'],PER_annotations=d['per'])
    result=dict(overall=overall,novelty=slices,frequency_slices=frequency_rows,gold_token_OOV_slices=oov_rows,
                strict_span_score=exact_span_f1([d['gold'] for d in docs],all_pred_spans),
                strict_span_totals={key:sum(d['strict'][key] for d in doc_records) for key in ['exact_matches','unmatched_gold','unmatched_pred','gold_with_no_pred_overlap','pred_with_no_gold_overlap']},
                overlap_pair_relations=dict(Counter(p['kind'] for p in pairs)),overlap_component_topology=dict(topology),
                official_error_categories=[dict(kind=kind,category=category,count=count) for (kind,category),count in semantic.items()],
                erroneous_texts=sum(d['fp']+d['fn']>0 for d in doc_records),
                review_cases=len(selected),review_FP=sum(e['kind']=='FP' for e in selected),review_FN=sum(e['kind']=='FN' for e in selected),
                selection_policy='Seed 2026; unique texts; 100 FP and 100 FN focal events, up to 50 overlap and 50 non-overlap per kind. Diagnostic sample, not prevalence estimator.',
                annotation_policy='Automated overlap/frequency categories are evidence, not proof of semantic cause. LOC absence is not proof of hallucination.',
                interval_policy='Paired document bootstrap (checkpoint groups unique), 2000 repetitions, seed 42; conditional on selected pilot.',
                model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                pilot_selection_sha256=hashlib.sha256((BASE/'data/ner_pilot_selection.csv').read_bytes()).hexdigest(),
                final_audit_evaluated=False,calibration_evaluated=False,test_targets_read=False,
                conservative_distinct_training_surfaces=len(conservative_known),official_distinct_training_surfaces=len(known))
    reports=BASE/'reports'
    for name,content in [('checkpoint_error_analysis.json',result),('checkpoint_all_error_events.json',events),
                         ('checkpoint_review_200.json',selected),('checkpoint_boundary_pairs.json',pairs)]:
        (reports/name).write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([dict(slice=k,**v) for k,v in slices.items()]).to_csv(reports/'checkpoint_novelty_metrics.csv',index=False)
    pd.DataFrame(frequency_rows).to_csv(reports/'checkpoint_frequency_metrics.csv',index=False)
    pd.DataFrame(oov_rows).to_csv(reports/'checkpoint_gold_OOV_recall.csv',index=False)
    flat=[]
    for e in selected:
        row=dict(e)
        for key,value in row.items():
            if isinstance(value,list):row[key]=json.dumps(value,ensure_ascii=False)
        flat.append(row)
    pd.DataFrame(flat).to_csv(reports/'checkpoint_review_200.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    labels=['Seen surface','Unseen surface'];x=np.arange(2)
    for j,key in enumerate(['precision','recall','f1']):axes[0].bar(x+(j-1)*.24,[slices[k][key] for k in ['seen','unseen']],.24,label=key)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1),title='Actual 12k-fit surface novelty');axes[0].legend()
    cats=result['official_error_categories'];names=[x['kind']+': '+x['category'].replace('_',' ') for x in cats]
    axes[1].barh(names[::-1],[x['count'] for x in cats][::-1]);axes[1].set(title='Official error events',xlabel='Normalized per-text set errors')
    rel=result['overlap_pair_relations']
    axes[2].barh([k.replace('_',' ') for k in rel][::-1],list(rel.values())[::-1],color='#4a9b78')
    axes[2].set(title='Strict unmatched overlap pairs',xlabel='One-to-one pairs')
    fig.savefig(BASE/'figures/checkpoint_error_analysis.png',dpi=170);plt.close(fig)
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
