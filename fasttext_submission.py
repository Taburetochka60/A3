"""Inference only. AI-assisted; uses the locked FastText Subword checkpoint."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

MANIFEST_SHA = '56490f12c8bb53e37431bd6141772e9612cc2edc4012d4da52c76d0f2e580230'


def verified_source(root, work):
    folder=Path(root)/'fasttext_subword_source'
    raw=(folder/'source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise ValueError('Training source manifest changed')
    manifest=json.loads(raw);local=Path(work)/'source'
    for name,digest in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe source path')
        data=(folder/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Training source changed: '+name)
        target=local/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (local/'source_manifest.json').write_bytes(raw)
    sys.path.insert(0,str(local))
    return local,manifest['source_id']


def locked_checkpoint(folder, spec, source_id):
    import torch
    from experiments import fasttext_experiment as experiment
    from part2_data import file_hash, fingerprint
    common=json.loads((folder/'experiment_identity.json').read_text())
    result=json.loads((folder/'result.json').read_text());lock=json.loads((folder/'selection_locked.json').read_text())
    provenance=common['provenance']
    if common['config']!=experiment.CONFIG or provenance['source_id']!=source_id:
        raise ValueError('Training config or source identity differs')
    if common['identity']!=fingerprint(dict(config=common['config'],provenance=provenance)):
        raise ValueError('Experiment identity changed')
    if not result['complete'] or result['name']!='fasttext_subword':raise ValueError('Model training is incomplete')
    if not common['identity']==result['identity']==lock['identity']:raise ValueError('Checkpoint lock identity differs')
    if not provenance['split']==lock['split']==spec['content_sha256']:raise ValueError('Frozen split differs')
    if provenance['data_files']!=spec['files'] or provenance['fit_ids']!=fingerprint(spec['fit']):
        raise ValueError('Frozen training data differs')
    if lock['threshold']!=.8 or lock['prediction_filter']!='strict':raise ValueError('Prediction policy changed')
    path=folder/result['checkpoint']
    if not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe checkpoint path')
    if file_hash(path)!=result['checkpoint_sha256'] or result['checkpoint_sha256']!=lock['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from result/selection lock')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['identity']!=lock['identity'] or saved['arm']!='fasttext_subword' or saved['epoch']!=result['best_epoch']:
        raise ValueError('Wrong selected checkpoint')
    if saved['encoder']!=common['config']['encoder'] or fingerprint(saved['vocabulary'])!=provenance['vocabulary']:
        raise ValueError('Checkpoint architecture or vocabulary differs')
    if sorted(saved['vocabulary'].values())!=list(range(2,len(saved['vocabulary'])+2)):
        raise ValueError('Invalid word vocabulary IDs')
    return saved,result,provenance


def verified_provider(info, cache, work):
    from experiments.fasttext_embeddings import URL, official_archive, unpack_model, load_fasttext
    if info['source']['url']!=URL or info['verification']['sha256']!=info['binary']['sha256']:
        raise ValueError('Official FastText provenance differs')
    for folder in [Path('/content/a3_fasttext_v1/fasttext'),Path(work)/'fasttext',cache]:
        binary=folder/'cc.uk.300.bin'
        if binary.exists():break
    else:
        if (cache/'cc.uk.300.bin.gz').exists():
            print('Reusing the Drive archive; extraction is needed in this runtime.',flush=True)
        else:
            print('No cached binary/archive: the official 4.50 GB download is required.',flush=True)
        archive,origin=official_archive(cache)
        if origin!=info['source']:raise ValueError('FastText archive provenance differs')
        binary,metadata=unpack_model(archive,origin,Path(work)/'fasttext')
        if metadata!=info['binary']:raise ValueError('FastText binary provenance differs')
    if importlib.util.find_spec('fasttext') is None or importlib.metadata.version('fasttext')!='0.9.3':
        subprocess.check_call([sys.executable,'-m','pip','install','-q','fasttext==0.9.3','pybind11>=2.13'])
    provider,_=load_fasttext(binary,info['binary']['sha256'])
    return provider


def write_once(source, target):
    from part2_data import file_hash
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if file_hash(source)!=file_hash(target):raise FileExistsError('Existing file differs; nothing overwritten: '+str(target))
        return
    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)


def extend_vectors(words, test, original, info, cache, work):
    import numpy as np
    from experiments import fasttext_subword as subword
    from experiments.fasttext_embeddings import checked_vector, immutable_json
    from part2_data import fingerprint, save_json
    cache=Path(cache);old_folder=cache/'subword'/fingerprint(original['identity'])
    old_keys,old_vectors=subword.verify_vectors(old_folder,original,words)
    if original['identity']['binary_sha256']!=info['binary']['sha256']:raise ValueError('OOV binary differs')
    keys=sorted(set(old_keys)|set(subword.oov_keys(words,test)))
    missing=sorted(set(keys)-set(old_keys))
    if not missing:return old_folder,original,0
    identity=dict(original['identity'],keys=fingerprint(keys));folder=cache/'subword'/fingerprint(identity)
    if (folder/'vectors.json').exists():
        record=json.loads((folder/'vectors.json').read_text())
        if record['identity']!=identity:raise ValueError('Extended cache identity changed')
        new_keys,values=subword.verify_vectors(folder,record,words)
    else:
        print('New test OOV keys requiring official FastText vectors:',len(missing),flush=True)
        provider=verified_provider(info,cache,work)
        values=np.empty((len(keys),300),dtype=np.float32);old_lookup={k:i for i,k in enumerate(old_keys)}
        hits=zero=0
        try:
            for i,key in enumerate(keys):
                if key in old_lookup:values[i]=old_vectors[old_lookup[key]]
                else:
                    value=checked_vector(provider,key);norm=np.linalg.norm(value)
                    values[i]=value/norm if norm else value
                    hits+=provider.get_word_id(key)>=0;zero+=norm==0
        finally:del provider;gc.collect()
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temporary:
            local=Path(temporary);np.save(local/'oov.npy',values);save_json(local/'keys.json',keys)
            for name in ['oov.npy','keys.json']:write_once(local/name,folder/name)
        from part2_data import file_hash
        record=dict(original,identity=identity,keys=len(keys),dictionary_hits=original['dictionary_hits']+hits,
            subword_generated=original['subword_generated']+len(missing)-hits,zero_vectors=original['zero_vectors']+int(zero),
            buffer_bytes=len(keys)*1200,files={name:file_hash(folder/name) for name in ['oov.npy','keys.json']})
        immutable_json(folder/'vectors.json',record)
        new_keys=keys
    lookup={key:i for i,key in enumerate(new_keys)}
    for begin in range(0,len(old_keys),1000):
        indices=[lookup[key] for key in old_keys[begin:begin+1000]]
        if not np.array_equal(values[indices],old_vectors[begin:begin+1000]):raise ValueError('Old OOV vectors changed')
    return folder,record,len(missing)


def check_old_predictions(docs, original_rep, expanded_rep, saved, result, folder, device):
    import numpy as np
    from experiments import fasttext_experiment as experiment
    from part2 import decode
    from part2_data import fingerprint, documents_hash, file_hash
    matrix=saved['model']['encoder.embedding.weight'].numpy()
    before=original_rep.build(matrix);after=expanded_rep.build(matrix)
    if experiment.recurrent_hash(before)!=result['initial_recurrent_hash']:
        raise ValueError('Fresh recurrent configuration does not reproduce training initialization')
    for model in [before,after]:model.load_state_dict(saved['model'],strict=True);model.to(device).eval()
    errors={}
    for part in ['checkpoint','calibration']:
        encoded=original_rep.encode(docs[part]);path=folder/(part+'.npz')
        meta=json.loads(path.with_suffix('.json').read_text())
        identity=fingerprint(dict(model=result['checkpoint_sha256'],documents=documents_hash(encoded,with_ids=True),
            policy='trained_fasttext_oov_'+fingerprint(original_rep.record)))
        if meta['identity']!=identity or file_hash(path)!=meta['sha256']:raise ValueError('Old prediction cache changed: '+part)
        with np.load(path) as data:
            if data['lengths'].tolist()!=[len(d['tokens']) for d in docs[part]]:raise ValueError('Old prediction rows differ')
            cached=list(np.split(data['values'],np.cumsum(data['lengths'])[:-1]))[:32]
        sample=docs[part][:32]
        left=experiment.infer_probabilities(before,original_rep.encode(sample),device,batch_size=128)
        right=experiment.infer_probabilities(after,expanded_rep.encode(sample),device,batch_size=128)
        for a,b,c in zip(left,right,cached):
            np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(a,c,rtol=0,atol=5e-4)
        if decode(sample,right,.8,'strict')[:2]!=decode(sample,cached,.8,'strict')[:2]:
            raise ValueError('Old decoded entities or offsets differ: '+part)
        errors[part]=dict(rows=len(sample),max_cached_difference=max(float(np.max(np.abs(a-c))) if a.size else 0. for a,c in zip(left,cached)),
            expanded_predictions_bit_identical=True)
    del before
    return after,errors


def run(root,work):
    root,work=Path(root),Path(work);work.mkdir(parents=True,exist_ok=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ.setdefault('MPLCONFIGDIR',str(work/'matplotlib'))
    if any(name in sys.modules for name in ['part2','part2_data','experiments.fasttext_subword']):
        raise RuntimeError('Run this script in a fresh Python subprocess to use the pinned training source')
    local,source_id=verified_source(root,work)
    import numpy as np
    import torch
    from experiments import fasttext_experiment as experiment
    from experiments import fasttext_subword as subword
    import part2
    import part2_data as source
    from pilot_ner import vocabulary,seed_all
    experiment.verify_source(local,local/'source_manifest.json')
    seed_all(42);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('Inference device:',device,flush=True)
    folder=root/'results/fasttext_v1/subword';spec=source.read_split(local/'part2_split.json')
    saved,result,provenance=locked_checkpoint(folder,spec,source_id)
    frames,spec=source.load_data(root/'data',local/'part2_split.json')
    docs=source.prepare_data({k:frames[k] for k in ['fit','checkpoint','calibration','test']},spec)
    words=saved['vocabulary'];fit=docs.pop('fit')
    if vocabulary(fit)!=words:raise ValueError('Fit-only vocabulary does not reproduce the checkpoint')
    original=provenance['subwords']
    if source.fingerprint(subword.oov_keys(words,fit))!=original['identity']['fit_keys']:
        raise ValueError('Frozen fit OOV keys differ')
    del fit,frames;gc.collect()
    cache=root/'cache/fasttext';old_table=cache/'subword'/source.fingerprint(original['identity'])
    table,record,missing=extend_vectors(words,docs['test'],original,provenance['vectors'],cache,work)
    old_rep=subword.Representation(words,old_table,original);new_rep=subword.Representation(words,table,record)
    model,checks=check_old_predictions(docs,old_rep,new_rep,saved,result,folder,device)
    test=docs['test'];encoded=new_rep.encode(test)
    if [d['text_id'] for d in test]!=spec['test_ids'] or len(test)!=477:raise ValueError('Test IDs or order changed')
    for before,after in zip(test,encoded):
        if before['text']!=after['text'] or before['tokens']!=after['tokens'] or after['spans'] or after['bio'] is not None:
            raise ValueError('Test text, offsets or gold-free encoding changed')
    probabilities=experiment.infer_probabilities(model,encoded,device,batch_size=128)
    output=root/'submissions/submission_fasttext_subword.csv'
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        local_csv=Path(temporary)/output.name
        part2.export_submission(test,probabilities,local_csv,threshold=.8,policy='strict')
        details=dict(checkpoint_sha256=result['checkpoint_sha256'],source_id=source_id,
            split_sha256=source.file_hash(local/'part2_split.json'),test_sha256=spec['files']['test.csv'],
            vocabulary_sha256=source.fingerprint(words),binary_sha256=provenance['vectors']['binary']['sha256'],
            original_vectors=original,extended_vectors=record,new_oov_keys=missing,old_document_checks=checks,
            threshold=.8,prediction_filter='strict',test_rows=477,device=str(device),torch=str(torch.__version__),
            submission_sha256=source.file_hash(local_csv),script_sha256=source.file_hash(__file__))
        source.save_json(Path(temporary)/'submission_fasttext_subword.json',details)
        # Check every existing destination before publishing any new output.
        for item in Path(temporary).iterdir():
            target=output.parent/item.name
            if target.exists() and source.file_hash(target)!=source.file_hash(item):raise FileExistsError('Existing submission artifact differs: '+str(target))
        for item in Path(temporary).iterdir():write_once(item,output.parent/item.name)
    print('Verified 477 unique IDs, original order, JSON lists and original offsets.',flush=True)
    print('Submission:',output,'SHA256:',details['submission_sha256'],flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Locked FastText Subword inference only')
    parser.add_argument('--root',type=Path,default=Path('/content/drive/MyDrive/KSE_AI301_A3'))
    parser.add_argument('--work',type=Path,default=Path('/content/a3_fasttext_submission'))
    args=parser.parse_args();run(args.root,args.work)
