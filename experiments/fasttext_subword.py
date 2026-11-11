"""Additional fresh BiLSTM with fixed FastText vectors for every NER OOV."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ner_data import token_key
import part2_data as source
from experiments import fasttext_experiment as experiment
from experiments.fasttext_embeddings import (URL, checked_vector, immutable_json,
    official_archive, unpack_model, load_fasttext, available_memory)
from experiments.tracking import Tracker

PRIMARY_SOURCE_ID = '23a407119034d6dda0891c0400c0de4dd56d925c747684698fc64836cd829808'


def oov_keys(words, docs):
    return sorted({token_key(t.value) for d in docs for t in d['tokens']} - set(words))


def export_vectors(words, fit, evaluation, provider, provenance, cache):
    """No labels or corpus statistics enter the external vectors."""
    fit_keys=oov_keys(words,fit);keys=sorted(set(fit_keys) | set(oov_keys(words,evaluation)))
    identity=dict(binary_sha256=provenance['sha256'],vocabulary=source.fingerprint(words),
        fit_keys=source.fingerprint(fit_keys),keys=source.fingerprint(keys),normalization='unit L2',version=1)
    folder=Path(cache)/'subword'/source.fingerprint(identity);meta=folder/'vectors.json'
    if meta.exists():
        record=json.loads(meta.read_text())
        experiment.require(record['identity']==identity,'Subword cache input changed')
        verify_vectors(folder,record,words)
        return folder,record
    if provider is None:return None,None
    extra=np.empty((len(keys),300),dtype=np.float32);dictionary_hits=zero=0
    for i,key in enumerate(keys):
        value=checked_vector(provider,key);norm=np.linalg.norm(value)
        extra[i]=value/norm if norm else value
        zero+=norm==0;dictionary_hits+=provider.get_word_id(key)>=0
    folder.mkdir(parents=True,exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        local=Path(tmp);np.save(local/'oov.npy',extra);source.save_json(local/'keys.json',keys)
        for name in ['oov.npy','keys.json']:
            target=folder/name
            if target.exists():experiment.require(source.file_hash(target)==source.file_hash(local/name),'Partial OOV cache differs')
            else:experiment.base.copy_file(local/name,target)
    record=dict(identity=identity,fit_keys=len(fit_keys),keys=len(keys),dictionary_hits=dictionary_hits,
        subword_generated=len(keys)-dictionary_hits,zero_vectors=int(zero),buffer_bytes=max(1,len(keys))*300*4,
        files={name:source.file_hash(folder/name) for name in ['oov.npy','keys.json']})
    immutable_json(meta,record)
    return folder,record


def verify_vectors(folder, record, words):
    folder=Path(folder)
    experiment.require(record['identity']['vocabulary']==source.fingerprint(words),'Subword vocabulary changed')
    for name,digest in record['files'].items():
        experiment.require(source.file_hash(folder/name)==digest,'Subword cache checksum changed: '+name)
    keys=json.loads((folder/'keys.json').read_text());values=np.load(folder/'oov.npy',mmap_mode='r')
    experiment.require(keys==sorted(set(keys)) and not set(keys)&set(words),'Invalid OOV routing keys')
    experiment.require(source.fingerprint(keys)==record['identity']['keys'],'OOV key identity changed')
    experiment.require(values.shape==(len(keys),300) and np.isfinite(values).all(),'Invalid OOV vector table')
    return keys,values


def prepare_subwords(words,fit,evaluation,vector_info,cache,work):
    experiment.require(vector_info['source']['url']==URL and vector_info['verification']['header']['words']==2000000,
                       'Official pretrained model verification missing')
    binary_sha=vector_info['binary']['sha256']
    provenance=dict(sha256=binary_sha)
    folder,record=export_vectors(words,fit,evaluation,None,provenance,cache)
    if folder is not None:return folder,record
    archive,origin=official_archive(cache)
    experiment.require(origin==vector_info['source'],'Primary and subword FastText archives differ')
    binary,metadata=unpack_model(archive,origin,work)
    experiment.require(metadata['sha256']==binary_sha,'Primary and subword FastText binaries differ')
    provider,verified=load_fasttext(binary,binary_sha)
    try:return export_vectors(words,fit,evaluation,provider,provenance,cache)
    finally:del provider;gc.collect()


class SubwordEmbedding(nn.Module):
    def __init__(self,word_embedding,vectors):
        super().__init__()
        self.weight=word_embedding.weight
        values=np.asarray(vectors) if len(vectors) else np.zeros((1,300),dtype=np.float32)
        self.register_buffer('oov',torch.from_numpy(np.array(values,copy=True)),persistent=False)

    def forward(self,ids):
        size=len(self.weight);unknown=ids>=size
        known=F.embedding(ids.masked_fill(unknown,0),self.weight,padding_idx=0)
        extra=F.embedding((ids-size).clamp(min=0),self.oov)
        return torch.where(unknown.unsqueeze(-1),extra,known)


class Representation:
    def __init__(self,words,folder,record):
        keys,self.vectors=verify_vectors(folder,record,words)
        self.words=words;self.lookup={key:i+len(words)+2 for i,key in enumerate(keys)}
        self.record=record

    def encode(self,docs):
        result=[]
        for doc in docs:
            ids=[]
            for token in doc['tokens']:
                key=token_key(token.value)
                index=self.words.get(key,self.lookup.get(key))
                if index is None:raise ValueError('Missing FastText vector; prepare a verified cache for token: '+repr(key))
                ids.append(index)
            result.append(dict(doc,ids=ids))
        return result

    def build(self,matrix,config=experiment.CONFIG):
        model=experiment.build_model(matrix,config)
        model.encoder.embedding=SubwordEmbedding(model.encoder.embedding,self.vectors)
        return model


def primary_source(output,current_source):
    path=Path(output)/'preflight.json'
    if not path.exists():return current_source
    previous=json.loads(path.read_text())['source_id']
    experiment.require(previous in [current_source,PRIMARY_SOURCE_ID],
                       'Unknown previous primary source; use its original pinned notebook/source')
    return previous


def run_primary(docs,spec,words,prepared,vector_info,output,work,source_id,minutes=45):
    path=Path(output)/'experiment_identity.json'
    if not path.exists() or json.loads(path.read_text())['source_id']==source_id:
        return experiment.run_training(docs,spec,words,prepared,vector_info,output,work,source_id,minutes)
    common=json.loads(path.read_text())
    experiment.require(common['source_id']==PRIMARY_SOURCE_ID,'Unknown primary source identity')
    expected=dict(config=experiment.CONFIG,source_id=PRIMARY_SOURCE_ID,split=spec['content_sha256'],
                  vocabulary=source.fingerprint(words),vectors=vector_info)
    experiment.require(common==expected,'Previous A/B experiment configuration changed')
    experiment.require((Path(output)/'selection_locked.json').exists(),
        'Previous-source A/B training is incomplete: resume with its original pinned notebook/source')
    results={}
    lock=json.loads((Path(output)/'selection_locked.json').read_text())
    experiment.require(lock['source_id']==PRIMARY_SOURCE_ID and lock['split']==spec['content_sha256']
                       and lock['threshold']==.8 and lock['prediction_filter']=='strict','Primary lock changed')
    for arm in ['random','fasttext']:
        model,result=experiment.load_arm(Path(output)/arm,words)
        experiment.require(result['identity']==source.fingerprint(dict(common,arm=arm)),'Primary model provenance changed')
        experiment.require(result['checkpoint_sha256']==lock['checkpoints'][arm]['sha256'],'Primary checkpoint lock changed')
        results[arm]=result;del model
    return results


def run_subword(docs,spec,words,prepared,vector_info,table,record,output,work,source_id,minutes=45):
    hardware=experiment.base.gpu()
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    for name,digest in vector_info['files'].items():
        experiment.require(source.file_hash(Path(prepared)/name)==digest,'Primary vector cache changed')
    experiment.require(vector_info['source']['url']==URL and vector_info['verification']['header']['words']==2000000,'Official vectors required')
    experiment.require(record['identity']['binary_sha256']==vector_info['binary']['sha256'],'Subword binary differs')
    for part in ['fit','checkpoint','calibration']:
        experiment.require([d['row_id'] for d in docs[part]]==spec[part],'Frozen IDs changed: '+part)
    experiment.require(record['identity']['fit_keys']==source.fingerprint(oov_keys(words,docs['fit'])),'Fit OOV identity changed')
    experiment.require(available_memory()>record['buffer_bytes']*2+512*1024**2,'Insufficient host memory for OOV table and training')
    experiment.require(torch.cuda.mem_get_info()[0]>record['buffer_bytes']+1024**3,'Insufficient GPU memory for OOV table plus training')
    representation=Representation(words,table,record)
    for part in ['fit','checkpoint','calibration']:representation.encode(docs[part])
    matrix=np.load(Path(prepared)/'fasttext.npy',mmap_mode='r')
    model=representation.build(matrix);control=experiment.build_model(matrix)
    initial=experiment.recurrent_hash(model)
    experiment.require(initial==experiment.recurrent_hash(control),'Subword recurrent initialization differs from B')
    parameters=sum(p.numel() for p in model.parameters());buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers())
    del model,control
    provenance=dict(source_id=source_id,split=spec['content_sha256'],data_files=spec['files'],vectors=vector_info,
        subwords=record,parameters=parameters,buffer_bytes=buffer_bytes,
        initial_recurrent_hash=initial,fit_ids=source.fingerprint(spec['fit']),vocabulary=source.fingerprint(words))
    folder=Path(output)/'subword';identity=source.fingerprint(dict(config=experiment.CONFIG,provenance=provenance))
    immutable_json(folder/'experiment_identity.json',dict(identity=identity,provenance=provenance,config=experiment.CONFIG))
    immutable_json(folder/f'environment_{time.time_ns()}.json',hardware)
    try:
        result=experiment.fit_arm('fasttext_subword',docs,words,matrix,folder,identity,provenance,work,'cuda',
            time.monotonic()+minutes*60,encode=representation.encode,build=representation.build)
    except experiment.BudgetEnd as error:
        source.save_json(folder/'session_status.json',dict(complete=False,message=str(error)));return None
    immutable_json(folder/'selection_locked.json',dict(checkpoint_sha256=result['checkpoint_sha256'],
        identity=identity,split=spec['content_sha256'],threshold=.8,prediction_filter='strict',
        policy='Raw checkpoint F1; no calibration or audit selection'))
    return result


def evaluate_subword(docs,spec,words,prepared,table,record,output,reference,device='cpu'):
    output=Path(output);folder=output/'subword'
    lock=json.loads((folder/'selection_locked.json').read_text())
    experiment.require(lock['threshold']==.8 and lock['prediction_filter']=='strict','Subword evaluation policy differs')
    common=json.loads((folder/'experiment_identity.json').read_text())
    experiment.require(common['config']==experiment.CONFIG,'Subword training protocol differs')
    experiment.require(lock['identity']==common['identity'] and lock['split']==spec['content_sha256'],'Subword lock changed')
    experiment.require(common['provenance']['subwords']==record,'Evaluation OOV cache differs from training')
    experiment.require(common['identity']==source.fingerprint(dict(config=common['config'],provenance=common['provenance'])),'Subword identity changed')
    result=json.loads((folder/'result.json').read_text());path=folder/result['checkpoint']
    experiment.require(result['identity']==lock['identity'],'Subword result identity changed')
    experiment.require(source.file_hash(path)==result['checkpoint_sha256']==lock['checkpoint_sha256'],'Selected subword checkpoint changed')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    experiment.require(saved['identity']==lock['identity'] and saved['vocabulary']==words and saved['encoder']==experiment.CONFIG['encoder'],
                       'Subword checkpoint configuration differs')
    experiment.require(saved['arm']=='fasttext_subword','Wrong subword checkpoint arm')
    vectors=common['provenance']['vectors']
    experiment.require(vectors['binary']['sha256']==record['identity']['binary_sha256'],'External vector source differs')
    for name,digest in vectors['files'].items():
        experiment.require(source.file_hash(Path(prepared)/name)==digest,'Primary embedding cache changed')
    representation=Representation(words,table,record)
    matrix=np.load(Path(prepared)/'fasttext.npy',mmap_mode='r');model=representation.build(matrix)
    model.load_state_dict(saved['model']);model.to(device).eval()
    primary=pd.read_csv(output/'comparison.csv')
    rows=primary.astype(object).where(pd.notna(primary),None).to_dict('records');intervals=[]
    primary_common=json.loads((output/'experiment_identity.json').read_text())
    primary_lock=json.loads((output/'selection_locked.json').read_text())
    experiment.require(primary_common['config']==experiment.CONFIG and primary_common['split']==spec['content_sha256']
        and primary_common['vocabulary']==source.fingerprint(words) and primary_common['vectors']==vectors,
        'Primary controls use a different protocol or embedding source')
    experiment.require(primary_lock['split']==spec['content_sha256'] and primary_lock['threshold']==.8
        and primary_lock['prediction_filter']=='strict','Primary evaluation policy differs')
    for part in ['checkpoint','calibration']:
        encoded=representation.encode(docs[part])
        p=experiment.prediction_cache(folder/(part+'.npz'),model,encoded,result['checkpoint_sha256'],'trained_fasttext_oov_'+source.fingerprint(record))
        summary,counts=experiment.export_metrics(folder,part,docs[part],p,words)
        rows.append(dict(model='fasttext_subword',partition=part,**summary,
            **{k:result[k] for k in ['parameters','parameter_bytes','checkpoint_bytes','seconds']},buffer_bytes=record['buffer_bytes']))
        for arm in ['random','fasttext','reference']:
            if arm=='reference':
                previous,state=experiment.restore_model(reference,experiment.REFERENCE_SHA,spec['content_sha256'],device)
                old_encoded=experiment.feature_documents(docs[part],state,previous)
                experiment.require((output/arm/(part+'.npz')).exists(),'Run primary comparison before subword comparison')
                old=experiment.prediction_cache(output/arm/(part+'.npz'),previous,old_encoded,experiment.REFERENCE_SHA,'reference_ngrams')
                del previous
            else:
                previous,prior=experiment.load_arm(output/arm,words,device=device)
                experiment.require(prior['identity']==source.fingerprint(dict(primary_common,arm=arm)) and
                    prior['checkpoint_sha256']==primary_lock['checkpoints'][arm]['sha256'],'Primary checkpoint provenance differs')
                old_encoded=source.encode_documents(docs[part],dict(kind='word',vocabulary=words))
                experiment.require((output/arm/(part+'.npz')).exists(),'Run primary comparison before subword comparison')
                old=experiment.prediction_cache(output/arm/(part+'.npz'),previous,old_encoded,prior['checkpoint_sha256'],'primary_word_unk')
                del previous
            old_counts=experiment.metric_counts(docs[part],old,words)
            old_summary=experiment.summarize(docs[part],old,words)
            stored=primary[(primary.model==arm)&(primary.partition==part)]
            experiment.require(len(stored)==1 and all(abs(float(stored.iloc[0][k])-old_summary[k])<1e-10
                for k in ['f1','precision','recall','oov_recall','iv_recall']),'Primary comparison metrics changed')
            interval=experiment.paired_bootstrap(old_counts,counts,[d['group'] for d in docs[part]],experiment.CONFIG['bootstrap'])
            intervals.append(dict(partition=part,comparison='fasttext_subword - '+arm,**interval))
    previous_results=[json.loads((output/arm/'result.json').read_text()) for arm in ['random','fasttext']]
    matched=all(r['amp']==result['amp'] and not r['fallbacks'] for r in previous_results) and not result['fallbacks']
    report=dict(metrics=rows,intervals=intervals,buffer_bytes=record['buffer_bytes'],reference_replaced=False,
        protocol_precision_matches=bool(matched),
        caveats=['Representation comparison; A/B initialization-only controls unchanged','Single seed, reused calibration holdout',
                 'External OOV rows frozen; known vocabulary rows trainable','No audit, new CV or Kaggle evaluation'])
    immutable_json(folder/'comparison.json',report)
    text=pd.DataFrame(rows).to_csv(index=False);path=folder/'comparison.csv'
    if path.exists():experiment.require(path.read_text()==text,'Subword comparison changed')
    else:path.write_text(text)
    tracker=Tracker(folder/'comparison_tracking',dict(identity=source.fingerprint(lock),name='fasttext_subword_comparison',protocol=experiment.CONFIG))
    tracker.finish({r['model']+'/'+r['partition']+'/'+k:r[k] for r in rows for k in ['f1','precision','recall','oov_recall','iv_recall']})
    return report
