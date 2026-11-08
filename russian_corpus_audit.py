"""Bounded-memory census of both corpora and leakage-safe Russian splits.

Run with --project-root pointing to the repository containing data/. Large
metadata/duckdb/manifest files stay under ignored data/russian_audit/.
"""
import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import time
import unicodedata

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TOKEN_PATTERN=r'\n|[^\W_]+|[^\s]'
TOKEN_RE=re.compile(TOKEN_PATTERN,re.UNICODE)


def canonical(text):
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())


def digest(value):return hashlib.sha256(value.encode('utf-8')).digest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=16384)
def markers(value):
    parsed=ast.literal_eval(value)
    if not isinstance(parsed,(list,tuple)):raise ValueError('Expected span list')
    return tuple(tuple(s) for s in parsed)


@lru_cache(maxsize=65536)
def doc_role(doc_id):
    key=hashlib.blake2b(f'42:ru:{doc_id}'.encode(),digest_size=8).digest()
    return 'ru_valid' if int.from_bytes(key,'big')%10==0 else 'ru_fit'


def check_row(text,loc,org='[]',per='[]'):
    """Same whole-word boundary rule as ner_data, without importing torch."""
    issues=Counter();surfaces=[];valid=[];aux={}
    tokens=[(m.start(),m.end()) for m in TOKEN_RE.finditer(text)]
    starts={a for a,b in tokens};ends={b for a,b in tokens}
    for label,value in [('LOC',loc),('ORG',org),('PER',per)]:
        try:spans=markers(value)
        except (ValueError,TypeError,SyntaxError):issues[label+'_parse_failure']+=1;spans=[]
        accepted=[]
        for raw in spans:
            if len(raw)!=2 or any(not isinstance(x,int) for x in raw):
                issues[label+'_noninteger_or_shape']+=1;continue
            a,b=raw
            if not 0<=a<b<=len(text):issues[label+'_out_of_bounds']+=1;continue
            accepted.append((a,b))
            if label=='LOC':
                surfaces.append(text[a:b]);valid.append((a,b))
                aa,bb=a,b
                while aa<bb and text[aa].isspace():aa+=1
                while bb>aa and text[bb-1].isspace():bb-=1
                issues['LOC_edge_whitespace']+=int((a,b)!=(aa,bb))
                if aa>=bb:issues['LOC_whitespace_only']+=1
                elif aa not in starts or bb not in ends:issues['LOC_partial_token']+=1
        aux[label]=accepted
    right=-1
    for a,b in sorted(valid):
        issues['LOC_overlap']+=int(a<right);right=max(right,b)
    issues['LOC_duplicate_spans']+=len(valid)-len(set(valid))
    for key in ['ORG','PER']:
        issues['LOC_'+key+'_overlaps']+=sum(any(max(a,c)<min(b,d) for c,d in aux[key]) for a,b in valid)
    fatal=['LOC_parse_failure','LOC_noninteger_or_shape','LOC_out_of_bounds','LOC_whitespace_only','LOC_partial_token','LOC_overlap']
    eligible=bool(text.strip()) and not any(issues[k] for k in fatal)
    signature=digest(json.dumps(sorted(set(map(canonical,surfaces))),ensure_ascii=False))
    return dict(eligible=eligible,issues=issues,surfaces=surfaces,n_tokens=len(tokens),
                n_loc=len(aux['LOC']),signature=signature,aux=aux)


def scan(root,corpus,roles=None):
    folder=root/'data/russian_audit';folder.mkdir(parents=True,exist_ok=True)
    path=root/'data'/f'{corpus}_geo_dataset.csv';meta_path=folder/f'{corpus}_metadata.parquet'
    entity_path=folder/f'{corpus}_entities.parquet'
    counts=Counter();char_hist=Counter();token_hist=Counter();loc_hist=Counter()
    writer=entity_writer=None;row_id=0
    sampled={'positive':[],'negative':[]};sample_seen=Counter();rng=random.Random(42)
    kwargs=dict(chunksize=40000,keep_default_na=False)
    if corpus=='ru':kwargs['dtype']={'doc_id':'Int64','sent_id':'Int64'}
    for chunk in pd.read_csv(path,**kwargs):
        records=[];entities=[]
        for row in chunk.itertuples(index=False):
            text=row.text if isinstance(row.text,str) else ''
            checked=check_row(text,row.loc_markers,row.org_markers,row.per_markers)
            counts.update(checked['issues']);counts['rows']+=1;counts['LOC_mentions']+=checked['n_loc']
            counts['positive_rows']+=int(checked['n_loc']>0);counts['bio_eligible_rows']+=int(checked['eligible'])
            counts['empty_text_rows']+=int(not text.strip())
            counts['has_newline']+=int('\n' in text);counts['has_latin']+=int(bool(re.search('[A-Za-z]',text)))
            counts['has_ua_specific']+=int(bool(re.search('[іїєґІЇЄҐ]',text)))
            counts['has_ru_specific']+=int(bool(re.search('[ыэёъЫЭЁЪ]',text)))
            counts['has_astral']+=int(any(ord(c)>0xFFFF for c in text))
            char_hist[len(text)]+=1;token_hist[checked['n_tokens']]+=1;loc_hist[checked['n_loc']]+=1
            if corpus=='ru':
                missing_doc=pd.isna(row.doc_id);missing_sent=pd.isna(row.sent_id)
                doc_id=-1 if missing_doc else int(row.doc_id)
                sent_id=-1 if missing_sent else int(row.sent_id)
                counts['missing_doc_id']+=int(missing_doc);counts['missing_sent_id']+=int(missing_sent)
                base=doc_role(doc_id)
            else:
                doc_id=-1;sent_id=-1;base=str(roles[row_id]);missing_doc=missing_sent=False
            records.append(dict(row_id=row_id,doc_id=doc_id,sent_id=sent_id,base_role=base,
                                exact_hash=digest(text),canon_hash=digest(canonical(text)),label_sig=checked['signature'],
                                n_chars=len(text),n_tokens=checked['n_tokens'],n_loc=checked['n_loc'],
                                bio_eligible=checked['eligible'] and not missing_doc and not missing_sent))
            for surface in checked['surfaces']:entities.append(dict(surface=canonical(surface),raw_hash=digest(surface)))
            if corpus=='ru':
                key='positive' if checked['n_loc'] else 'negative'
                sample_seen[key]+=1;capacity=80 if key=='positive' else 20
                slot=len(sampled[key]) if len(sampled[key])<capacity else rng.randrange(sample_seen[key])
                if slot<capacity:
                    item=dict(row_id=row_id,doc_id=doc_id,sent_id=sent_id,text=text,
                              LOC=checked['surfaces'],ORG=[text[a:b] for a,b in checked['aux']['ORG']],
                              PER=[text[a:b] for a,b in checked['aux']['PER']],issues=dict(checked['issues']))
                    if slot==len(sampled[key]):sampled[key].append(item)
                    else:sampled[key][slot]=item
            row_id+=1
        table=pa.Table.from_pylist(records)
        if writer is None:writer=pq.ParquetWriter(meta_path,table.schema,compression='zstd')
        writer.write_table(table)
        if entities:
            table=pa.Table.from_pylist(entities)
            if entity_writer is None:entity_writer=pq.ParquetWriter(entity_path,table.schema,compression='zstd')
            entity_writer.write_table(table)
        if row_id%200000==0:print(corpus,'scanned',row_id,flush=True)
        del records,entities,table,chunk
    writer.close()
    if entity_writer:entity_writer.close()
    result=dict(counts=dict(counts),character_hist=dict(char_hist),token_hist=dict(token_hist),loc_count_hist=dict(loc_hist),
                source_bytes=path.stat().st_size,source_sha256=file_hash(path))
    (root/'reports'/f'{corpus}_full_structural_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if corpus=='ru':
        cases=sampled['positive']+sampled['negative']
        (root/'reports/ru_annotation_review_sample.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def sql_path(path):return "'"+str(path).replace('\\','/').replace("'","''")+"'"


def database(root):
    work=root/'data/russian_audit';work.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(work/'corpus_audit.duckdb'))
    con.execute("SET memory_limit='256MB'");con.execute('SET threads=2')
    con.execute("SET preserve_insertion_order=false")
    con.execute('SET temp_directory='+sql_path(work/'spill'))
    # Refresh paths when an ignored metadata bundle is moved to another host.
    for name in ['uk','ru']:
        meta=work/(name+'_metadata.parquet')
        if meta.exists():con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({sql_path(meta)})')
    manifest=work/'ru_split_manifest.parquet'
    if manifest.exists():con.execute(f'CREATE OR REPLACE VIEW ru_split AS SELECT * FROM read_parquet({sql_path(manifest)})')
    con.fetchall()
    return con


def summarize_and_split(root):
    con=database(root);work=root/'data/russian_audit'
    # Views keep large row-level data out of the persistent DB unless needed.
    for name in ['uk','ru']:
        con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({sql_path(work/(name+"_metadata.parquet"))})')
    for name in ['uk','ru']:
        for kind in ['canon','exact']:
            con.execute(f'CREATE OR REPLACE TABLE {name}_{kind}_duplicates AS SELECT {kind}_hash AS key,count(*) AS n FROM {name} GROUP BY {kind}_hash HAVING count(*)>1')
            print('Grouped duplicates',name,kind,flush=True)
    con.execute('CREATE OR REPLACE TABLE ru_conflicts AS SELECT r.canon_hash FROM ru r JOIN ru_canon_duplicates d ON r.canon_hash=d.key GROUP BY r.canon_hash HAVING count(DISTINCT r.label_sig)>1')
    protected=[]
    for file in ['test.csv','labeling_sample.csv']:
        for text in pd.read_csv(root/'data'/file,usecols=['text']).text:protected.append(dict(canon_hash=digest(canonical(text))))
    protected_frame=pd.DataFrame(protected);con.register('external_protected',protected_frame)
    con.execute("CREATE OR REPLACE TABLE protected_keys AS SELECT DISTINCT canon_hash FROM uk WHERE base_role IN ('checkpoint','calibration','audit') UNION SELECT canon_hash FROM external_protected")
    con.execute('CREATE OR REPLACE TABLE ru_protected_docs AS SELECT DISTINCT r.doc_id FROM ru r JOIN protected_keys p USING(canon_hash)')
    con.execute("CREATE OR REPLACE TABLE ru_valid_keys AS SELECT DISTINCT r.canon_hash FROM ru r ANTI JOIN ru_protected_docs p USING(doc_id) WHERE r.base_role='ru_valid'")
    con.execute("CREATE OR REPLACE TABLE ru_cross_valid_docs AS SELECT DISTINCT r.doc_id FROM ru r JOIN ru_valid_keys v USING(canon_hash) WHERE r.base_role='ru_fit'")
    con.execute("CREATE OR REPLACE TABLE uk_fit_keys AS SELECT DISTINCT canon_hash FROM uk WHERE base_role='fit'")
    con.execute("""CREATE OR REPLACE TABLE ru_first_fit AS
      SELECT r.canon_hash,min(r.row_id) AS first_row
      FROM ru r
      ANTI JOIN ru_protected_docs p USING(doc_id)
      ANTI JOIN ru_cross_valid_docs v USING(doc_id)
      ANTI JOIN ru_conflicts c USING(canon_hash)
      ANTI JOIN uk_fit_keys u USING(canon_hash)
      WHERE r.base_role='ru_fit' AND r.bio_eligible GROUP BY r.canon_hash""")
    manifest=work/'ru_split_manifest.parquet'
    con.execute('COPY ('+"""SELECT r.row_id,r.doc_id,r.sent_id,
      CASE WHEN p.doc_id IS NOT NULL THEN 'excluded_protected_UA_doc'
           WHEN v.doc_id IS NOT NULL THEN 'excluded_RU_valid_overlap_doc'
           WHEN r.doc_id<0 OR r.sent_id<0 THEN 'excluded_schema'
           WHEN r.base_role='ru_valid' THEN 'ru_valid'
           WHEN NOT r.bio_eligible THEN 'excluded_alignment'
           WHEN c.canon_hash IS NOT NULL THEN 'excluded_label_conflict'
           WHEN u.canon_hash IS NOT NULL THEN 'excluded_UA_fit_duplicate'
           WHEN r.row_id<>f.first_row THEN 'excluded_RU_duplicate'
           ELSE 'ru_fit' END AS partition
      FROM ru r
      LEFT JOIN ru_protected_docs p USING(doc_id)
      LEFT JOIN ru_cross_valid_docs v USING(doc_id)
      LEFT JOIN ru_conflicts c USING(canon_hash)
      LEFT JOIN uk_fit_keys u USING(canon_hash)
      LEFT JOIN ru_first_fit f USING(canon_hash)
      """+') TO '+sql_path(manifest)+" (FORMAT PARQUET,COMPRESSION ZSTD)")
    con.execute(f'CREATE OR REPLACE VIEW ru_split AS SELECT * FROM read_parquet({sql_path(manifest)})')
    bad_docs=con.execute("SELECT count(*) FROM (SELECT doc_id FROM ru_split WHERE partition IN ('ru_fit','ru_valid') GROUP BY doc_id HAVING count(DISTINCT partition)>1)").fetchone()[0]
    bad_keys=con.execute("SELECT count(*) FROM (SELECT r.canon_hash FROM ru r JOIN ru_split s USING(row_id) WHERE s.partition IN ('ru_fit','ru_valid') GROUP BY r.canon_hash HAVING count(DISTINCT s.partition)>1)").fetchone()[0]
    protected_leaks=con.execute("SELECT count(*) FROM ru r JOIN ru_split s USING(row_id) JOIN protected_keys p USING(canon_hash) WHERE s.partition IN ('ru_fit','ru_valid')").fetchone()[0]
    assert bad_docs==bad_keys==protected_leaks==0
    stats={}
    for name in ['uk','ru']:
        stats[name]={}
        for kind in ['canon','exact']:
            groups,excess=con.execute(f'SELECT count(*),coalesce(sum(n-1),0) FROM {name}_{kind}_duplicates').fetchone()
            stats[name][kind+'_duplicate_groups']=groups;stats[name][kind+'_duplicate_excess']=excess
    cross=con.execute("SELECT u.base_role,count(*) AS matched_RU_rows,count(DISTINCT r.canon_hash) AS shared_keys,count(DISTINCT r.doc_id) AS RU_docs FROM ru r JOIN (SELECT DISTINCT canon_hash,base_role FROM uk) u USING(canon_hash) GROUP BY u.base_role").fetchdf()
    partitions=con.execute("SELECT s.partition,count(*) AS rows,count(DISTINCT s.doc_id) AS documents,sum(r.n_loc) AS LOC_mentions,avg(CASE WHEN r.n_loc=0 THEN 1. ELSE 0. END) AS empty_LOC_fraction FROM ru_split s JOIN ru r USING(row_id) GROUP BY s.partition ORDER BY s.partition").fetchdf()
    docstats=con.execute('SELECT count(DISTINCT doc_id),count(*) FROM ru').fetchone()
    dupl_sent=con.execute('SELECT count(*),coalesce(sum(n-1),0) FROM (SELECT doc_id,sent_id,count(*) n FROM ru GROUP BY doc_id,sent_id HAVING count(*)>1)').fetchone()
    conflict_count=con.execute('SELECT count(*) FROM ru_conflicts').fetchone()[0]
    stats.update(cross_corpus=cross.to_dict(orient='records'),partitions=partitions.to_dict(orient='records'),
                 RU_documents=docstats[0],RU_rows=docstats[1],duplicate_doc_sent_groups=dupl_sent[0],duplicate_doc_sent_excess=dupl_sent[1],
                 canonical_duplicate_groups_with_LOC_surface_conflicts=conflict_count,
                 protected_UA_document_exclusions=con.execute('SELECT count(*) FROM ru_protected_docs').fetchone()[0],
                 RU_validation_overlap_document_exclusions=con.execute('SELECT count(*) FROM ru_cross_valid_docs').fetchone()[0],
                 active_cross_partition_docs=bad_docs,active_cross_partition_canonical_texts=bad_keys,protected_UA_matches_in_active_RU=protected_leaks,
                 split_sha256=file_hash(manifest),duckdb_version=duckdb.__version__,
                 split_rule='BLAKE2b(42:ru:doc_id) mod 10: 0=RU valid. Whole-doc exclusions for protected UA/external texts and RU fit/valid canonical overlap. Fit-only row-level alignment/conflict/dedup exclusions.',
                 limitations='No fuzzy/semantic/translation-equivalence matching. LOC surface-set conflict is a flag, not independent semantic adjudication.')
    entities={}
    for name in ['uk','ru']:
        con.execute(f'CREATE OR REPLACE TABLE {name}_entity_counts AS SELECT surface,count(*) AS n FROM read_parquet({sql_path(work/(name+"_entities.parquet"))}) GROUP BY surface')
        unique,total,single=con.execute(f'SELECT count(*),sum(n),sum(CASE WHEN n=1 THEN 1 ELSE 0 END) FROM {name}_entity_counts').fetchone()
        entities[name]=dict(distinct_surfaces=unique,mentions=total,singleton_surfaces=single,
                            top20=con.execute(f'SELECT * FROM {name}_entity_counts ORDER BY n DESC,surface LIMIT 20').fetchdf().to_dict(orient='records'))
    shared=con.execute('SELECT count(*) FROM uk_entity_counts u JOIN ru_entity_counts r USING(surface)').fetchone()[0]
    stats['entity_surfaces']=entities;stats['shared_conservative_LOC_surfaces']=shared
    partitions.to_csv(root/'reports/ru_validation_partitions.csv',index=False)
    cross.to_csv(root/'reports/ru_uk_text_overlap.csv',index=False)
    (root/'reports/ru_corpus_split_audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    con.close()
    freeze_samples(root)
    print('Leakage-safe RU split and fixed experiment selections complete',flush=True)
    return stats


def freeze_samples(root):
    con=database(root);work=root/'data/russian_audit'
    stats=json.loads((root/'reports/ru_corpus_split_audit.json').read_text(encoding='utf-8'))
    # Frozen auxiliary sample IDs and matched-compute Ukrainian control.
    original=pd.read_csv(root/'data/ner_pilot_selection.csv')
    frame=original.loc[original.role.eq('pilot_fit'),['row_id']]
    con.fetchall()  # DuckDB 2 requires previous results drained before register.
    con.register('pilot_ids',frame)
    con.execute('CREATE OR REPLACE TABLE pilot_keys AS SELECT canon_hash FROM uk JOIN pilot_ids USING(row_id)')
    extra=con.execute("""SELECT row_id FROM (
      SELECT row_id,canon_hash,row_number() OVER(PARTITION BY canon_hash ORDER BY row_id) rn
      FROM uk ANTI JOIN pilot_keys USING(canon_hash) WHERE base_role='fit' AND bio_eligible
      ) WHERE rn=1 ORDER BY md5('extra-UA-42:'||CAST(row_id AS VARCHAR)) LIMIT 12000""").fetchdf().assign(corpus='uk',role='extra_uk')
    ru=con.execute("SELECT s.row_id,s.doc_id FROM ru_split s JOIN ru r USING(row_id) WHERE s.partition='ru_fit' AND r.n_tokens BETWEEN 1 AND 128 ORDER BY md5('aux-RU-42:'||CAST(s.row_id AS VARCHAR)) LIMIT 12000").fetchdf().assign(corpus='ru',role='aux_ru')
    rv=con.execute("SELECT row_id,doc_id FROM ru_split WHERE partition='ru_valid' ORDER BY md5('eval-RU-42:'||CAST(row_id AS VARCHAR)) LIMIT 2000").fetchdf().assign(corpus='ru',role='ru_diagnostic')
    assert len(extra)==len(ru)==12000 and len(rv)==2000
    assert set(ru.doc_id).isdisjoint(rv.doc_id)
    selection=pd.concat([extra,ru,rv],ignore_index=True)
    selection.to_csv(work/'experiment_selection.csv',index=False)
    stats['experiment_selection_sha256']=file_hash(work/'experiment_selection.csv')
    (root/'reports/ru_corpus_split_audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    con.close();print('Fixed experiment selections complete',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--stage',choices=['all','scan','split','samples'],default='all');args=parser.parse_args()
    root=args.project_root.resolve();started=time.time()
    if args.stage in ['all','scan']:
        roles=pd.read_csv(root/'data/validation_splits.csv',usecols=['partition'],dtype={'partition':'category'}).partition
        expected=json.loads((root/'reports/validation_splits_summary.json').read_text())
        assert file_hash(root/'data/uk_geo_dataset.csv')==expected['dataset_sha256']
        scan(root,'uk',roles);del roles
        scan(root,'ru')
    if args.stage in ['all','split']:summarize_and_split(root)
    if args.stage=='samples':freeze_samples(root)
    print('Audit elapsed seconds',round(time.time()-started,1),flush=True)


if __name__=='__main__':main()
