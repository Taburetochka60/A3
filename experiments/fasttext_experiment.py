"""Controlled random/FastText initialization; GPU training is an explicit step."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import part2 as base
import part2_data as source
from metrics import hard_processing
from ner_data import token_key, collate_ner, training_windows
from pilot_ner import vocabulary, class_weights, infer_probabilities, seed_all
from recurrent_model import NERModel
from experiments.fasttext_embeddings import URL, immutable_json, prepare_vectors
from experiments.representation import restore_model, feature_documents
from experiments.tracking import Tracker

REFERENCE_SHA = 'ed3c9e14bba969a129d3f70e670e05b968815bb93e4586fb1161cc0f7afeaa29'
CONFIG = dict(seed=42, train_rows=100000, batch_size=128, epochs=15, patience=6,
    encoder=dict(rnn_type='lstm', num_layers=2, embedding_dim=300, hidden_size=64, dropout=.2, bidirectional=True),
    optimizer='AdamW', lr=.002, weight_decay=.01, scheduler='none', threshold=.8, prediction_filter='strict',
    gradient_clip=1., windows=128, overlap=32, bootstrap=2000)
RUNTIME_FILES = ['part2.py','part2_data.py','part2_split.json','ner_data.py','pilot_ner.py',
    'recurrent_model.py','metrics.py','validation.py','experiments/__init__.py',
    'experiments/representation.py','experiments/tracking.py','experiments/fasttext_embeddings.py',
    'experiments/fasttext_experiment.py','experiments/fasttext_subword.py','test_fasttext.py',
    'test_fasttext_subword.py','requirements-fasttext.txt','docs/FASTTEXT_PROTOCOL.md']


class BudgetEnd(RuntimeError):
    pass


def require(condition, message):
    if not condition:raise ValueError(message)


def make_source(source_root, destination):
    source_root=Path(source_root);destination=Path(destination)
    files={name:source.file_hash(source_root/name) for name in RUNTIME_FILES}
    manifest=dict(files=files,source_id=source.fingerprint(files))
    for name in RUNTIME_FILES:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(source.file_hash(target)==files[name],f'Existing source differs: {target}')
        else:shutil.copy2(source_root/name,target)
    immutable_json(destination/'source_manifest.json',manifest)
    return manifest


def verify_source(root, manifest_path):
    root=Path(root);manifest=json.loads(Path(manifest_path).read_text())
    require(manifest['source_id']==source.fingerprint(manifest['files']),'Source manifest identity changed')
    require(set(manifest['files'])==set(RUNTIME_FILES),'Unexpected source file set')
    for name,expected in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe source path')
        require(source.file_hash(root/name)==expected,'Source hash mismatch: '+name)
    return manifest['source_id']


def preflight(data_dir, split_path, reference, output, source_id):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    require(source.file_hash(reference)==REFERENCE_SHA,'Wrong reference checkpoint; rejected fine-tune is not allowed')
    frames,spec=source.load_data(data_dir,split_path)
    docs=source.prepare_data({k:frames[k] for k in ['fit','checkpoint','calibration']},spec)
    require([len(docs[k]) for k in ['fit','checkpoint','calibration']]==[100000,4000,2000],'Wrong frozen partition sizes')
    words=vocabulary(docs['fit'])
    model,state=restore_model(reference,REFERENCE_SHA,spec['content_sha256'],'cpu')
    require(words==state['vocabulary'],'Reference and fit vocabularies differ')
    require(state['train_ids']==source.fingerprint(spec['fit']),'Reference fit IDs differ')
    encoded=feature_documents(docs['checkpoint'],state,model)
    p=prediction_cache(output/'reference/checkpoint.npz',model,encoded,REFERENCE_SHA,'reference_ngrams')
    reference_metric=summarize(docs['checkpoint'],p,words)
    require(reference_metric['f1']==.83198146,'Reference checkpoint F1 failed preflight reproduction')
    del model
    incompatible={k:[d['row_id'] for d in docs[k] if d['bio'] is None] for k in docs}
    require(not incompatible['fit'],'Training contains BIO-incompatible rows')
    report=dict(source_id=source_id,split_sha256=source.file_hash(split_path),split_identity=spec['content_sha256'],
        partitions={k:len(spec[k]) for k in ['fit','checkpoint','calibration','audit','domain_source','av_test']},
        test_rows=len(frames['test']),fit_domain_source_overlap=len(set(spec['fit']) & set(spec['domain_source'])),cv_fold_counts={str(f):spec['cv_folds'].count(f) for f in range(3)},
        incompatible_rows=incompatible,fit_ids_sha256=source.fingerprint(spec['fit']),
        checkpoint_ids_sha256=source.fingerprint(spec['checkpoint']),calibration_ids_sha256=source.fingerprint(spec['calibration']),
        vocabulary_sha256=source.fingerprint(words),reference_sha256=REFERENCE_SHA,reference_checkpoint_f1=reference_metric['f1'],
        note='All evaluation rows retained for surface-set metrics; incompatible BIO excluded only from token loss')
    immutable_json(output/'preflight.json',report)
    immutable_json(output/'vocabulary.json',dict(kind='word',vocabulary=words,size=len(words)+2))
    return docs,spec,words,report


def recurrent_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name!='encoder.embedding.weight':digest.update(name.encode()+value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(matrix, config=CONFIG):
    require(matrix.ndim==2 and matrix.shape[1]==300 and np.isfinite(matrix).all(),'Invalid 300d initialization')
    seed_all(config['seed']);model=NERModel(len(matrix),**config['encoder'])
    with torch.no_grad():model.encoder.embedding.weight.copy_(torch.from_numpy(np.array(matrix,copy=True)))
    require(torch.count_nonzero(model.encoder.embedding.weight[0]).item()==0,'PAD embedding must be zero')
    return model


def metric_counts(docs, probabilities, words):
    locations,bounds,_=base.decode(docs,probabilities,.8,'strict')
    rows=[]
    for doc,predicted,spans in zip(docs,locations,bounds):
        require(all(doc['text'][a:b]==text for (a,b),text in zip(spans,predicted)),'Decoded original offset changed')
        gold={}
        for a,b in doc['spans']:
            key=hard_processing(doc['text'][a:b])
            unknown=any(token_key(t.value) not in words for t in doc['tokens'] if a<t.end and t.start<b)
            gold[key]=gold.get(key,False) or unknown
        predicted={hard_processing(s) for s in predicted};keys=set(gold)
        rows.append([len(keys&predicted),len(predicted-keys),len(keys-predicted),
            sum(v and k in predicted for k,v in gold.items()),sum(gold.values()),
            sum(not v and k in predicted for k,v in gold.items()),sum(not v for v in gold.values())])
    return np.array(rows,dtype=np.int64)


def summarize(docs, probabilities, words):
    metrics=base.score(docs,probabilities,.8,'strict');rows=metric_counts(docs,probabilities,words)
    totals=rows.sum(0)
    metrics.update(oov_recall=float(totals[3]/totals[4]) if totals[4] else None,
                   iv_recall=float(totals[5]/totals[6]) if totals[6] else None,
                   oov_entities=int(totals[4]),iv_entities=int(totals[6]),
                   raw_f1=base.score(docs,probabilities)['f1'],rows=len(docs),
                   bio_incompatible=sum(d['bio'] is None for d in docs),threshold=.8,prediction_filter='strict')
    return metrics


def check_time(deadline):
    if time.monotonic()>deadline-20:raise BudgetEnd('GPU session budget exhausted; saved state can be resumed')


def save_new_torch(path,payload,work):
    path=Path(path);require(not path.exists(),'Checkpoint already exists: '+str(path))
    Path(work).mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=work,suffix='.pt') as temporary:
        torch.save(payload,temporary.name);base.copy_file(temporary.name,path)


def append_history(path,records):
    path=Path(path)
    existing=pd.read_csv(path) if path.exists() else pd.DataFrame()
    require(len(existing)<=len(records),'CSV is ahead of the selected resume checkpoint')
    if len(existing):require(existing.epoch.tolist()==[r['epoch'] for r in records[:len(existing)]],'History epochs differ')
    for row in records[len(existing):]:
        pd.DataFrame([row]).to_csv(path,mode='a',header=not path.exists(),index=False)


def fit_arm(arm,docs,words,matrix,folder,identity,provenance,work,device,deadline,config=CONFIG,
            encode=None,build=None):
    """Internal loop; run_training enforces the hosted CUDA guard."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    run_config=dict(config,name='fasttext_v1_'+arm,arm=arm,identity=identity,provenance=provenance,
        fit_ids_sha256=source.fingerprint([d['row_id'] for d in docs['fit']]),vocabulary_sha256=source.fingerprint(words))
    immutable_json(folder/'identity.json',run_config)
    result_path=folder/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());require(result['identity']==identity,'Completed run identity changed')
        require(source.file_hash(folder/result['checkpoint'])==result['checkpoint_sha256'],'Completed checkpoint changed')
        tracker=Tracker(folder,run_config)
        for row in pd.read_csv(folder/'history.csv').to_dict('records'):tracker.log(row)
        history=pd.read_csv(folder/'history.csv').to_dict('records');plots(folder,history)
        tracker.finish(result,[folder/'learning_curves.png',folder/'gradient_flow.png'])
        return result
    require({d['row_id'] for d in docs['fit']}.isdisjoint(d['row_id'] for d in docs['checkpoint']),'Training row leakage')
    require({d['group'] for d in docs['fit']}.isdisjoint(d['group'] for d in docs['checkpoint']),'Training group leakage')
    encoded={k:encode(docs[k]) if encode else source.encode_documents(docs[k],dict(kind='word',vocabulary=words)) for k in ['fit','checkpoint']}
    require(len(matrix)==len(words)+2,'Embedding/vocabulary row mismatch')
    model=(build(matrix,config) if build else build_model(matrix,config)).to(device);initial=recurrent_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],betas=(.9,.999),eps=1e-8)
    weights,counts=class_weights(docs['fit'],torch.device(device))
    mode='bf16' if device=='cuda' and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0]>=8 else 'fp16' if device=='cuda' else 'fp32'
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.,enabled=mode=='fp16')
    progress=dict(epoch=1,batch=0,best=-1.,bad=0,history=[],totals=[0.,0.,0.,0.,0,0.],seconds=0.,best_file=None,fallbacks=[])
    pointer=folder/'resume.json'
    if pointer.exists():
        item=json.loads(pointer.read_text());path=folder/item['file']
        require(path.resolve().is_relative_to(folder.resolve()),'Unsafe resume path')
        require(source.file_hash(path)==item['sha256'],'Resume checksum mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['identity']==identity and saved['initial_hash']==initial,'Resume identity or initial model changed')
        require(saved['mode']==mode or saved['mode']=='fp32','Resume GPU precision differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        mode=saved['mode'];scaler=torch.amp.GradScaler('cuda',init_scale=1024.,enabled=mode=='fp16');scaler.load_state_dict(saved['scaler'])
        progress=saved['progress'];base.restore_rng(saved['rng'])
    append_history(folder/'history.csv',progress['history'])
    tracker=Tracker(folder,dict(run_config,initial_recurrent_hash=initial,parameters=sum(p.numel() for p in model.parameters())))
    for row in progress['history']:tracker.log(row)
    examples=[row for d in encoded['fit'] for row in training_windows(d['ids'],d['bio'])]
    segment=last_save=time.monotonic()

    def persist():
        nonlocal segment,last_save
        progress['seconds']+=time.monotonic()-segment;segment=time.monotonic()
        name=f"state_e{progress['epoch']:02}_b{progress['batch']:05}_{time.time_ns()}.pt"
        save_new_torch(folder/name,dict(identity=identity,initial_hash=initial,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),mode=mode,progress=progress,rng=base.rng_state()),work)
        source.save_json(pointer,dict(file=name,sha256=source.file_hash(folder/name)))
        append_history(folder/'history.csv',progress['history']);last_save=time.monotonic()

    try:
        for epoch in range(progress['epoch'],config['epochs']+1):
            if progress['bad']>=config['patience']:break
            model.train()
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            loader=DataLoader(examples,batch_sampler=base.batches(examples,config['batch_size'],config['seed'],epoch)[progress['batch']:],
                collate_fn=collate_ner,num_workers=2 if device=='cuda' else 0,pin_memory=device=='cuda',
                generator=torch.Generator().manual_seed(config['seed']+epoch))
            for number,batch in enumerate(loader,start=progress['batch']):
                check_time(deadline);tick=time.monotonic();before=base.rng_state()
                try:numerator,denominator,norm=base.update(model,batch,optimizer,scaler,weights,mode)
                except FloatingPointError as error:
                    if mode=='fp32':raise
                    print(f'{arm}: nonfinite AMP update, retrying in FP32: {error}',flush=True)
                    base.restore_rng(before);optimizer.zero_grad(set_to_none=True)
                    progress['fallbacks'].append(dict(epoch=epoch,batch=number,previous=mode));mode='fp32'
                    scaler=torch.amp.GradScaler('cuda',enabled=False)
                    numerator,denominator,norm=base.update(model,batch,optimizer,scaler,weights,mode)
                totals=progress['totals'];totals[0]+=numerator;totals[1]+=denominator;totals[2]+=norm
                totals[3]=max(totals[3],norm);totals[4]+=1;totals[5]+=time.monotonic()-tick;progress['batch']=number+1
                if time.monotonic()-last_save>120:persist()
            check_time(deadline);tick=time.monotonic()
            p=infer_probabilities(model,encoded['checkpoint'],torch.device(device),batch_size=config['batch_size'])
            metrics=summarize(docs['checkpoint'],p,words);raw=base.score(encoded['checkpoint'],p);totals=progress['totals']
            row=dict(epoch=epoch,train_loss=totals[0]/totals[1],val_loss=raw['val_loss'],f1=raw['f1'],precision=raw['precision'],recall=raw['recall'],
                strict_f1=metrics['f1'],oov_recall=metrics['oov_recall'],iv_recall=metrics['iv_recall'],learning_rate=config['lr'],
                gradient_norm=totals[2]/totals[4],max_gradient_norm=totals[3],train_seconds=totals[5],
                validation_seconds=time.monotonic()-tick,peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0.,amp=mode,
                gradient_flow=json.dumps({n:float(v.grad.detach().norm()) for n,v in model.named_parameters() if v.grad is not None}))
            progress['history'].append(row)
            if raw['f1']>progress['best']:
                name=f'model_e{epoch:02}_{time.time_ns()}.pt'
                save_new_torch(folder/name,dict(model=model.state_dict(),encoder=config['encoder'],vocabulary=words,
                    identity=identity,arm=arm,epoch=epoch,initial_hash=initial,metrics=metrics),work)
                progress.update(best=raw['f1'],bad=0,best_file=name)
            else:progress['bad']+=1
            progress.update(epoch=epoch+1,batch=0,totals=[0.,0.,0.,0.,0,0.]);persist();tracker.log(row)
            print(arm,'epoch',epoch,'raw F1',row['f1'],'strict F1',row['strict_f1'],flush=True)
    except Exception:
        persist();tracker.finish(dict(complete=False,epochs=len(progress['history'])));raise
    best=torch.load(folder/progress['best_file'],map_location='cpu',weights_only=True)
    result=dict(identity=identity,name=arm,complete=True,checkpoint=progress['best_file'],checkpoint_sha256=source.file_hash(folder/progress['best_file']),
        epochs=len(progress['history']),best_epoch=best['epoch'],seconds=progress['seconds'],initial_recurrent_hash=initial,
        parameters=sum(v.numel() for v in model.parameters()),parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters()),
        checkpoint_bytes=(folder/progress['best_file']).stat().st_size,amp=mode,fallbacks=progress['fallbacks'],class_counts=counts)
    immutable_json(result_path,result);plots(folder,progress['history']);tracker.finish(result,[folder/'learning_curves.png',folder/'gradient_flow.png'])
    return result


def plots(folder,history):
    folder=Path(folder)
    if (folder/'learning_curves.png').exists():return
    table=pd.DataFrame(history);fig,axes=plt.subplots(1,3,figsize=(12,3))
    axes[0].plot(table.epoch,table.train_loss,label='train');axes[0].plot(table.epoch,table.val_loss,label='validation');axes[0].legend()
    for name in ['f1','precision','recall']:axes[1].plot(table.epoch,table[name],label=name)
    axes[1].legend();axes[2].plot(table.epoch,table.gradient_norm,label='gradient norm');axes[2].legend()
    for axis in axes:axis.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(folder/'learning_curves.png',dpi=120);plt.close(fig)
    flow=json.loads(history[-1]['gradient_flow']);fig,axis=plt.subplots(figsize=(10,3))
    axis.bar(range(len(flow)),[max(x,1e-12) for x in flow.values()]);axis.set_yscale('log')
    axis.set_xticks(range(len(flow)),flow.keys(),rotation=90,fontsize=6);axis.set_ylabel('Last-batch gradient L2 after clipping')
    fig.tight_layout();fig.savefig(folder/'gradient_flow.png',dpi=120);plt.close(fig)


def run_training(docs,spec,words,prepared,vector_info,output,work,source_id,minutes=45):
    hardware=base.gpu();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    require(vector_info['source']['url']==URL,'Wrong pretrained source')
    require(vector_info['verification']['header']['words']==2000000,'Official model was not verified')
    require(source.fingerprint([d['row_id'] for d in docs['fit']])==source.fingerprint(spec['fit']),'Training IDs changed')
    require([d['row_id'] for d in docs['checkpoint']]==spec['checkpoint'],'Checkpoint IDs changed')
    require([d['row_id'] for d in docs['calibration']]==spec['calibration'],'Calibration IDs changed')
    for name,expected in vector_info['files'].items():require(source.file_hash(Path(prepared)/name)==expected,'Embedding cache changed: '+name)
    matrices=[np.load(Path(prepared)/(arm+'.npy'),mmap_mode='r') for arm in ['random','fasttext']]
    a,b=[build_model(matrix) for matrix in matrices]
    require(recurrent_hash(a)==recurrent_hash(b),'A/B recurrent initialization differs before training')
    require(np.array_equal(matrices[0][1],matrices[1][1]),'Primary UNK initialization differs')
    del a,b
    output=Path(output);deadline=time.monotonic()+minutes*60;results={}
    common=dict(config=CONFIG,source_id=source_id,split=spec['content_sha256'],vocabulary=source.fingerprint(words),vectors=vector_info)
    immutable_json(output/'experiment_identity.json',common)
    import importlib.metadata
    environment=dict(hardware=hardware,torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),deterministic=True,
        packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','fasttext','matplotlib','scikit-learn']})
    immutable_json(output/f'environment_{time.time_ns()}.json',environment)
    for arm in ['random','fasttext']:
        identity=source.fingerprint(dict(common,arm=arm))
        matrix=np.load(Path(prepared)/(arm+'.npy'),mmap_mode='r')
        try:
            results[arm]=fit_arm(arm,docs,words,matrix,output/arm,identity,dict(vector_info,source_id=source_id,split_identity=spec['content_sha256'],data_files=spec['files']),work,'cuda',deadline)
        except BudgetEnd as error:
            source.save_json(output/'session_status.json',dict(complete=False,interrupted_arm=arm,message=str(error)))
            return None
    require(results['random']['initial_recurrent_hash']==results['fasttext']['initial_recurrent_hash'],'A/B recurrent initialization differs')
    immutable_json(output/'selection_locked.json',dict(policy='Raw checkpoint F1 only; calibration/audit did not select epochs',
        checkpoints={arm:dict(sha256=r['checkpoint_sha256'],epoch=r['best_epoch']) for arm,r in results.items()},
        threshold=.8,prediction_filter='strict',source_id=source_id,split=spec['content_sha256']))
    source.save_json(output/'session_status.json',dict(complete=True,training_complete=True,evaluation_pending=True))
    return results


def load_arm(folder,words,config=CONFIG,device='cpu'):
    folder=Path(folder);result=json.loads((folder/'result.json').read_text())
    path=folder/result['checkpoint'];require(source.file_hash(path)==result['checkpoint_sha256'],'Selected checkpoint changed')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    require(saved['arm'] in ['random','fasttext'],'Subword checkpoint requires its verified OOV representation')
    require(saved['vocabulary']==words and saved['encoder']==config['encoder'],'Checkpoint vocabulary or architecture differs')
    require(saved['identity']==result['identity'],'Checkpoint identity differs')
    model=NERModel(len(words)+2,**saved['encoder']);model.load_state_dict(saved['model']);model.to(device).eval()
    return model,result


def prediction_cache(path,model,encoded,model_hash,policy):
    path=Path(path);meta=path.with_suffix('.json')
    identity=source.fingerprint(dict(model=model_hash,documents=source.documents_hash(encoded,with_ids=True),policy=policy))
    if path.exists():
        saved=json.loads(meta.read_text());require(saved['identity']==identity,'Prediction cache identity differs')
        require(source.file_hash(path)==saved['sha256'],'Prediction cache changed')
        with np.load(path) as data:values=data['values'];lengths=data['lengths']
        require(lengths.tolist()==[len(d['tokens']) for d in encoded],'Cached prediction lengths changed')
        return list(np.split(values,np.cumsum(lengths)[:-1]))
    p=infer_probabilities(model,encoded,next(model.parameters()).device,batch_size=128)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz') as temporary:
        np.savez_compressed(temporary.name,values=np.concatenate(p),lengths=[len(x) for x in p])
        base.copy_file(temporary.name,path)
    immutable_json(meta,dict(identity=identity,sha256=source.file_hash(path)))
    return p


def paired_bootstrap(old,new,groups,repeats=2000):
    require(old.shape==new.shape==(len(groups),7),'Unpaired bootstrap rows')
    bygroup={};a=[];b=[]
    for group,left,right in zip(groups,old,new):
        if group not in bygroup:bygroup[group]=len(a);a.append(np.zeros(7,dtype=np.int64));b.append(np.zeros(7,dtype=np.int64))
        a[bygroup[group]]+=left;b[bygroup[group]]+=right
    a=np.array(a);b=np.array(b);rng=np.random.default_rng(42);deltas=[]
    for _ in range(repeats):
        ix=rng.integers(0,len(a),len(a));x=a[ix,:3].sum(0);y=b[ix,:3].sum(0)
        score=lambda r:2*r[0]/max(1,2*r[0]+r[1]+r[2])
        deltas.append(score(y)-score(x))
    return dict(low=float(np.quantile(deltas,.025)),high=float(np.quantile(deltas,.975)),repeats=repeats,groups=len(a))


def export_metrics(folder,partition,docs,p,words):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    summary=summarize(docs,p,words);immutable_json(folder/(partition+'_metrics.json'),summary)
    lengths=[]
    for lo,hi in [(1,16),(17,32),(33,64),(65,128),(129,10**9)]:
        indices=[i for i,d in enumerate(docs) if lo<=len(d['tokens'])<=hi]
        if indices:lengths.append(dict(length=f'{lo}-{hi}',**summarize([docs[i] for i in indices],[p[i] for i in indices],words)))
    immutable_json(folder/(partition+'_lengths.json'),lengths)
    return summary,metric_counts(docs,p,words)


def evaluate(docs,spec,words,reference,prepared,output,device='cpu'):
    output=Path(output);lock=json.loads((output/'selection_locked.json').read_text())
    common=json.loads((output/'experiment_identity.json').read_text())
    for name,digest in common['vectors']['files'].items():
        require(source.file_hash(Path(prepared)/name)==digest,'Embedding cache changed before evaluation: '+name)
    require(lock['split']==spec['content_sha256'],'Locked split differs')
    all_metrics=[];counts={};modelsizes={};selected={}
    for arm in ['random','fasttext']:
        model,result=load_arm(output/arm,words,device=device)
        require(result['checkpoint_sha256']==lock['checkpoints'][arm]['sha256'],'Epoch choice changed after locking')
        require(result['identity']==source.fingerprint(dict(common,arm=arm)),'Checkpoint experiment provenance differs')
        selected[arm]=result
        modelsizes[arm]={k:result[k] for k in ['parameters','parameter_bytes','checkpoint_bytes','seconds']}
        modelsizes[arm]['fasttext_dictionary_coverage']=common['vectors']['coverage']['dictionary_coverage'] if arm=='fasttext' else None
        for part in ['checkpoint','calibration']:
            encoded=source.encode_documents(docs[part],dict(kind='word',vocabulary=words))
            p=prediction_cache(output/arm/(part+'.npz'),model,encoded,result['checkpoint_sha256'],'primary_word_unk')
            summary,counts[arm,part]=export_metrics(output/arm,part,docs[part],p,words)
            all_metrics.append(dict(model=arm,partition=part,**summary,**modelsizes[arm]))
        del model
    model,state=restore_model(reference,REFERENCE_SHA,spec['content_sha256'],device)
    modelsizes['reference']=dict(parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),checkpoint_bytes=Path(reference).stat().st_size,seconds=177.69991729399862)
    for part,expected in [('checkpoint',.83198146),('calibration',.80399693)]:
        encoded=feature_documents(docs[part],state,model)
        p=prediction_cache(output/'reference'/(part+'.npz'),model,encoded,REFERENCE_SHA,'reference_ngrams')
        summary,counts['reference',part]=export_metrics(output/'reference',part,docs[part],p,words)
        require(summary['f1']==expected,'Frozen reference metric does not reproduce: '+part)
        all_metrics.append(dict(model='reference',partition=part,**summary,**modelsizes['reference']))
    del model
    intervals=[]
    for part in ['checkpoint','calibration']:
        for old,new in [('random','fasttext'),('reference','random'),('reference','fasttext')]:
            interval=paired_bootstrap(counts[old,part],counts[new,part],[d['group'] for d in docs[part]],CONFIG['bootstrap'])
            intervals.append(dict(partition=part,comparison=new+' - '+old,**interval))
    immutable_json(output/'bootstrap.json',intervals)
    path=output/'comparison.csv'
    text=pd.DataFrame(all_metrics).to_csv(index=False)
    if path.exists():require(path.read_text()==text,'Existing comparison differs')
    else:path.write_text(text)
    diagnostic=fasttext_oov_diagnostic(docs,words,prepared,output,device)
    effect=next(x for x in intervals if x['partition']=='calibration' and x['comparison']=='fasttext - random')
    scores={(r['model'],r['partition']):r['f1'] for r in all_metrics}
    precision_matches=selected['random']['amp']==selected['fasttext']['amp'] and not any(r['fallbacks'] for r in selected.values())
    evidence=scores['fasttext','calibration']-scores['random','calibration']>=.01 and effect['low']>0 and precision_matches
    report=dict(primary='Initialization-only, same shared UNK',source_domain_initialization_benefit=bool(evidence),
        candidate_metrics=all_metrics,intervals=intervals,subword_diagnostic=diagnostic,embedding_provenance=common['vectors'],
        caveats=['Single seed; bootstrap conditions on selected models','Calibration is a reused source-domain holdout',
                 'Reference has a different input representation and parameter count','No new CV or Kaggle results','Audit not evaluated'],
        protocol_precision_matches=precision_matches,
        reference_replaced=False,kaggle_submission_generated=False)
    immutable_json(output/'comparison.json',report)
    tracker=Tracker(output/'comparison_tracking',dict(identity=source.fingerprint(lock),name='fasttext_v1_comparison',protocol=CONFIG))
    tracker.finish({'initialization_benefit':bool(evidence),**{r['model']+'/'+r['partition']+'/f1':r['f1'] for r in all_metrics}})
    return report


def fasttext_oov_diagnostic(docs,words,prepared,output,device):
    model,result=load_arm(Path(output)/'fasttext',words,device=device)
    keys=json.loads((Path(prepared)/'oov_keys.json').read_text());vectors=np.load(Path(prepared)/'oov.npy')
    require(vectors.shape==(len(keys),300) and np.isfinite(vectors).all(),'Invalid diagnostic OOV vectors')
    lookup={word:i+len(words)+2 for i,word in enumerate(keys)}
    # Only checkpoint/calibration documents are requested by this diagnostic.
    requested={token_key(t.value) for part in ['checkpoint','calibration'] for d in docs[part] for t in d['tokens'] if token_key(t.value) not in words}
    require(requested.issubset(lookup),'Diagnostic OOV cache is incomplete')
    weights=torch.cat([model.encoder.embedding.weight.detach().cpu(),torch.from_numpy(vectors.copy())])
    model.encoder.embedding=torch.nn.Embedding.from_pretrained(weights,freeze=True,padding_idx=0).to(device)
    result_metrics={}
    for part in ['checkpoint','calibration']:
        encoded=[dict(d,ids=[words[token_key(t.value)] if token_key(t.value) in words else lookup[token_key(t.value)] for t in d['tokens']]) for d in docs[part]]
        p=prediction_cache(Path(output)/'fasttext_oov_diagnostic'/(part+'.npz'),model,encoded,result['checkpoint_sha256'],'fixed_fasttext_unseen_vectors')
        result_metrics[part],_=export_metrics(Path(output)/'fasttext_oov_diagnostic',part,docs[part],p,words)
    return result_metrics
