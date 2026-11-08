"""Reproducible, label-safe EDA. Run from any directory with the project venv."""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json
import re
import sys
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from metrics import hard_processing, intended_processing, score_details

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
REPORTS = BASE / 'reports'
FIGURES = BASE / 'figures'
for folder in [REPORTS, FIGURES]:
    folder.mkdir(exist_ok=True)


def parse_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError('Expected a list')
        return list(parsed)
    raise ValueError(f'Unsupported list representation: {type(value)}')


def canonical(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def features(frame):
    text = frame['text'].fillna('').astype(str)
    result = pd.DataFrame(index=frame.index)
    result['characters'] = text.str.len()
    result['whitespace_tokens'] = text.str.split().str.len()
    result['has_url'] = text.str.contains(r'https?://|www\.', regex=True)
    result['has_newline'] = text.str.contains('\n', regex=False)
    result['has_latin'] = text.str.contains(r'[A-Za-z]', regex=True)
    result['has_ua_specific'] = text.str.contains(r'[іїєґІЇЄҐ]', regex=True)
    result['has_ru_specific'] = text.str.contains(r'[ыэёъЫЭЁЪ]', regex=True)
    result['has_astral_unicode'] = text.map(lambda s: any(ord(c) > 0xFFFF for c in s))
    return result


def audit(frame, name, labeled=True):
    texts = frame['text'].fillna('').astype(str)
    feat = features(frame)
    norm = texts.map(canonical)
    summary = dict(rows=len(frame), columns=list(frame.columns),
                   missing_text=int(frame['text'].isna().sum()),
                   empty_text=int(texts.str.strip().eq('').sum()),
                   exact_duplicate_excess=int(texts.duplicated().sum()),
                   canonical_duplicate_excess=int(norm.duplicated().sum()),
                   chars=feat['characters'].describe(percentiles=[.5, .9, .95, .99]).to_dict(),
                   tokens=feat['whitespace_tokens'].describe(percentiles=[.5, .9, .95, .99]).to_dict(),
                   whitespace_token_overflow_rates={str(limit):float((feat.whitespace_tokens>limit).mean()) for limit in [128,256,512]},
                   feature_rates={c: float(feat[c].mean()) for c in feat if c.startswith('has_')})
    if 'text_id' in frame:
        summary['missing_ids'] = int(frame.text_id.isna().sum())
        summary['duplicate_ids'] = int(frame.text_id.duplicated().sum())
    if not labeled:
        summary['locations_is_placeholder'] = bool(frame.locations.map(parse_list).map(len).eq(0).all())
        return summary, feat, []
    if 'loc_markers' not in frame:
        raise ValueError(f'{name}: missing loc_markers')
    counts, entity_lengths, entity_token_lengths = [], [], []
    raw_freq, norm_freq = Counter(), Counter()
    surface_map = {}
    issues = Counter()
    examples = []
    all_gold = []
    positive_chars = 0
    for idx, (text, raw) in enumerate(zip(texts, frame.loc_markers)):
        spans = parse_list(raw)
        valid_spans, entities = [], []
        for span in spans:
            if isinstance(span, dict):
                start, end = span.get('start'), span.get('end')
            else:
                start, end = span
            if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
                issues['noninteger_offsets'] += 1
                continue
            start, end = int(start), int(end)
            if not 0 <= start < end <= len(text):
                issues['out_of_bounds_or_empty'] += 1
                if len(examples) < 10: examples.append(dict(row=idx, span=[start, end], text_length=len(text)))
                continue
            valid_spans.append((start, end))
            entity = text[start:end]
            entities.append(entity)
            entity_lengths.append(end-start)
            entity_token_lengths.append(len(entity.split()))
            raw_freq[entity] += 1
            normalized = hard_processing(entity)
            norm_freq[normalized] += 1
            surface_map.setdefault(normalized, set()).add(entity)
            issues['whitespace_boundaries'] += int(entity != entity.strip())
            issues['empty_after_normalization'] += int(normalized == '')
        ordered = sorted(set(valid_spans))
        issues['duplicate_spans'] += len(valid_spans)-len(ordered)
        covered, right = 0, -1
        for start,end in ordered:
            issues['overlapping_spans'] += int(start < right)
            covered += max(0, end-max(start, right))
            right = max(right,end)
        positive_chars += covered
        counts.append(len(spans))
        issues['per_text_normalization_collapses'] += len(entities)-len(set(map(hard_processing,entities)))
        if 'locations' in frame:
            listed = parse_list(frame.iloc[idx]['locations'])
            issues['locations_span_set_mismatch'] += int(set(listed) != set(entities))
        all_gold.append(entities)
    summary.update(loc_mentions=int(sum(counts)), positive_rows=int(sum(c>0 for c in counts)),
                   valid_loc_mentions=sum(map(len, all_gold)),
                   missing_loc_markers=int(frame.loc_markers.isna().sum()),
                   no_location_rows=int(sum(c==0 for c in counts)),
                   no_location_fraction=float(np.mean(np.array(counts)==0)),
                   mentions_per_text=pd.Series(counts).describe(percentiles=[.5,.9,.99]).to_dict(),
                   loc_count_distribution=dict(sorted(Counter(counts).items())),
                   entity_char_lengths=pd.Series(entity_lengths,dtype=float).describe().to_dict(),
                   entity_token_lengths=pd.Series(entity_token_lengths,dtype=float).describe().to_dict(),
                   multiword_entity_fraction=float(np.mean(np.array(entity_token_lengths)>1)) if entity_token_lengths else None,
                   location_character_fraction=positive_chars/max(1,int(feat.characters.sum())),
                   distinct_raw_entities=len(raw_freq), distinct_normalized_entities=len(norm_freq),
                   normalization_collision_groups=sum(len(v)>1 for v in surface_map.values()),
                   normalization_collision_examples=[dict(normalized=k,surfaces=sorted(v)[:6]) for k,v in surface_map.items() if len(v)>1][:10],
                   top10_mention_fraction=sum(v for _,v in norm_freq.most_common(10))/max(1,sum(norm_freq.values())),
                   singleton_surface_fraction=sum(v==1 for v in norm_freq.values())/max(1,len(norm_freq)),
                   top_locations=norm_freq.most_common(15), issues=dict(issues), invalid_examples=examples)
    feat['loc_count'] = counts
    return summary, feat, all_gold


def main():
    source = (BASE/'sources'/'kaggle_pages.json').read_text(encoding='utf-8-sig')
    pages = json.loads(source[source.index('['):])
    for page in pages:
        (BASE/'sources'/f"{page['name']}.md").write_text(page['content'], encoding='utf-8')
    # Compile the actual published code in an isolated namespace to check parity.
    evaluation = next(p['content'] for p in pages if p['name']=='Evaluation')
    code = evaluation.split('```')[1]
    import string
    namespace = {'re':re,'string':string,'List':list}
    exec(code,namespace)
    test = pd.read_csv(DATA/'test.csv')
    sample = pd.read_csv(DATA/'labeling_sample.csv')
    parquet = DATA/'uk_geo_dataset_processed_v1.parquet'
    train = pd.read_csv(DATA/'uk_geo_dataset.csv')
    processed = pd.read_parquet(parquet)
    tag_counts = Counter()
    token_label_mismatch = unknown_tag_rows = 0
    token_count_list, io_run_counts = [], []
    for tokens_str, labels_str in zip(processed.tokens, processed.labels):
        tokens, labels = json.loads(tokens_str), json.loads(labels_str)
        token_label_mismatch += int(len(tokens) != len(labels))
        unknown_tag_rows += int(any(x not in ['O','LOC'] for x in labels))
        tag_counts.update(labels)
        token_count_list.append(len(tokens))
        io_run_counts.append(sum(x=='LOC' and (i==0 or labels[i-1]!='LOC') for i,x in enumerate(labels)))
    processed_stats = dict(rows=len(processed), columns=list(processed.columns),
                           split_counts={str(k):int(v) for k,v in processed.is_valid.value_counts().items()},
                           tag_counts=dict(tag_counts), token_label_length_mismatch=token_label_mismatch,
                           unknown_tag_rows=unknown_tag_rows,
                           all_O_token_accuracy=tag_counts['O']/max(1,sum(tag_counts.values())),
                           tokens_per_sequence=pd.Series(token_count_list).describe(percentiles=[.5,.95,.99]).to_dict(),
                           raw_processed_row_count_equal=len(processed)==len(train),
                           raw_processed_split_equal=bool(processed.is_valid.equals(train.is_valid)) if len(processed)==len(train) else False)
    if len(processed)==len(train):
        raw_counts=train.loc_markers.map(lambda x:len(parse_list(x)))
        processed_stats['io_run_count_differs_from_raw_mentions_rows']=int(np.sum(raw_counts.to_numpy()!=np.array(io_run_counts)))
    del processed, token_count_list, io_run_counts
    print('TRAIN',train.shape,list(train.columns),flush=True)
    frames = {'uk_train_all':train, 'labeling_sample':sample, 'test':test}
    if 'is_valid' in train:
        frames['uk_fit'] = train.loc[train.is_valid.eq(0)]
        frames['uk_valid'] = train.loc[train.is_valid.eq(1)]
    summaries, feats, gold = {}, {}, {}
    for name, frame in frames.items():
        summaries[name], feats[name], gold[name] = audit(frame,name,labeled=name!='test')
        print('Audited',name,len(frame),flush=True)
    duplicate_rows=train.loc[train.text.duplicated(keep=False)]
    duplicate_label_sets=duplicate_rows.assign(span_key=duplicate_rows.loc_markers.map(lambda x:repr(sorted(parse_list(x)))))
    summaries['uk_train_all']['exact_duplicate_groups_conflicting_loc_labels']=int((duplicate_label_sets.groupby('text').span_key.nunique()>1).sum())
    overlap = {}
    pairs = [('uk_train_all','test'),('uk_train_all','labeling_sample'),('labeling_sample','test')]
    if 'uk_fit' in frames:pairs.append(('uk_fit','uk_valid'))
    for a,b in pairs:
        aa,bb=frames[a].text.fillna('').astype(str),frames[b].text.fillna('').astype(str)
        exact=set(aa)&set(bb); norm_a=set(aa.map(canonical)); norm_b=bb.map(canonical)
        overlap[f'{a}__{b}']=dict(exact_shared_unique=len(exact),exact_rows_in_second=int(bb.isin(exact).sum()),
                                  canonical_rows_in_second=int(norm_b.isin(norm_a).sum()))
    shifts={}
    for col in ['characters','whitespace_tokens']:
        result=ks_2samp(feats['uk_train_all'][col],feats['test'][col],method='asymp')
        shifts[col]=dict(ks_statistic=float(result.statistic),pvalue=float(result.pvalue))
    if 'uk_fit' in frames:
        known={hard_processing(e) for row in gold['uk_fit'] for e in row}
        for name in ['uk_valid','labeling_sample']:
            ents=[hard_processing(e) for row in gold[name] for e in row]
            summaries[name]['unseen_entity_mention_fraction']=sum(e not in known for e in ents)/max(1,len(ents))
    cases=[('case',[['Київ']],[['київ']]),('duplicate',[['Київ','Київ']],[['Київ']]),
           ('address_number',[['вул. Миру 12']],[['вул. Миру 99']]),
           ('punctuation',[['Київ']],[['Київ!']]),('empty_normalized',[['12']],[['99']]),
           ('empty_prediction',[['Київ']],[[]]),('all_empty',[[]],[[]]),
           ('partial_name',[['Біла Церква']],[['Церква']]),('morphology',[['Київ']],[['Києві']]),
           ('latin_regex',[['Paris']],[['Cairo']])]
    case_rows=[]
    for name,g,p in cases:
        details=score_details(g,p)
        assert details['f1']==namespace['comp_metric'](g,p),(name,details)
        case_rows.append(dict(case=name,gold=g,pred=p,**details,
                              intended_f1=score_details(g,p,intended_processing)['f1']))
    pd.DataFrame(case_rows).to_csv(REPORTS/'metric_edge_cases.csv',index=False)
    # Validate against the published evaluator on randomized sequences too.
    rng=np.random.default_rng(42)
    pool=['Київ','Львів','12','Paris','Rome','Київ!','вул. Миру 12','']
    for _ in range(500):
        g=[rng.choice(pool,int(rng.integers(0,5))).tolist() for _ in range(5)]
        p=[rng.choice(pool,int(rng.integers(0,5))).tolist() for _ in range(5)]
        assert score_details(g,p)['f1']==namespace['comp_metric'](g,p)
    manifest={}
    for f in [DATA/'uk_geo_dataset.csv',parquet,DATA/'test.csv',DATA/'labeling_sample.csv',BASE/'sources'/'kaggle_pages.json']:
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        manifest[f.name]=dict(bytes=f.stat().st_size,sha256=h.hexdigest())
    output=dict(summary=summaries,processed=processed_stats,overlap=overlap,shift=shifts,files=manifest,
                metric_parity_random_cases=500,
                scope='Full Ukrainian raw CSV and processed parquet + full test + labeling sample. Russian auxiliary corpus not analyzed.')
    (REPORTS/'eda_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for name,s in summaries.items():
        rows.append(dict(dataset=name,rows=s['rows'],missing_text=s['missing_text'],
                         median_chars=s['chars']['50%'],p95_chars=s['chars']['95%'],
                         median_tokens=s['tokens']['50%'],duplicates=s['exact_duplicate_excess'],
                         loc_mentions=s.get('loc_mentions'),empty_loc_fraction=s.get('no_location_fraction')))
    pd.DataFrame(rows).to_csv(REPORTS/'dataset_overview.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,label in [('uk_train_all','Ukrainian corpus'),('test','Test'),('labeling_sample','Labeled sample')]:
        values=np.sort(feats[name].characters.to_numpy())
        axes[0,0].plot(values,np.arange(1,len(values)+1)/len(values),label=label)
    axes[0,0].set(xscale='symlog',xlabel='Characters per text (symlog scale)',ylabel='Cumulative fraction',title='Text length distribution')
    axes[0,0].legend()
    dist=Counter(feats['uk_train_all'].loc_count.clip(upper=5))
    axes[0,1].bar(range(6),[dist.get(i,0)/len(train) for i in range(6)],color='#387eac')
    axes[0,1].set(xticks=range(6),xticklabels=['0','1','2','3','4','5+'],xlabel='LOC mentions per text',ylabel='Fraction of texts',title='Target sparsity: Ukrainian corpus')
    cols=['has_url','has_newline','has_latin','has_ua_specific','has_ru_specific','has_astral_unicode']
    x=np.arange(len(cols));width=.25
    for i,name in enumerate(['uk_train_all','labeling_sample','test']):
        axes[1,0].bar(x+(i-1)*width,[feats[name][c].mean() for c in cols],width,label=name)
    axes[1,0].set(xticks=x,xticklabels=['URL','Newline','Latin','UA letters','RU letters','Astral'],ylabel='Fraction of texts',title='Domain indicators (not language classification)')
    axes[1,0].tick_params(axis='x',rotation=25);axes[1,0].legend(fontsize=8)
    top=summaries['uk_train_all']['top_locations'][:10]
    axes[1,1].barh([x[0] or '<empty>' for x in top][::-1],[x[1] for x in top][::-1],color='#4a9b78')
    axes[1,1].set(xlabel='Mention count',title='Most frequent normalized LOC surfaces')
    fig.savefig(FIGURES/'eda_overview.png',dpi=170);plt.close(fig)
    print(json.dumps(output,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
