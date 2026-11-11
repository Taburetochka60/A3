"""Official Ukrainian vectors and matched initialization matrices."""
import gc
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import struct
import urllib.request

import numpy as np

from ner_data import token_key
from part2_data import file_hash, fingerprint, save_json

URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.bin.gz'
ARCHIVE_BYTES = 4504123670
DIMENSION = 300
LICENSE = 'https://creativecommons.org/licenses/by-sa/3.0/'


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing metadata differs: {path}')
    else:
        save_json(path, value)


def binary_header(path):
    with Path(path).open('rb') as stream:
        data = stream.read(92)
    if len(data) < 92:
        raise ValueError('Incomplete FastText binary header')
    magic, version = struct.unpack_from('<ii', data)
    args = struct.unpack_from('<12id', data, 8)
    size, words, labels, tokens, prune = struct.unpack_from('<iiiqq', data, 64)
    if magic != 793712314 or version != 12 or args[0] != DIMENSION:
        raise ValueError('Expected FastText v12 binary with 300 dimensions')
    return dict(magic=magic, version=version, dimensions=args[0], bucket=args[8],
                minn=args[9], maxn=args[10], words=words, labels=labels,
                matrix_bytes=(2*words+args[8])*args[0]*4)


def official_archive(cache):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache/'cc.uk.300.bin.gz'; meta = cache/'archive.json'; part = cache/'cc.uk.300.bin.gz.part'
    if path.exists():
        if not meta.exists():
            raise ValueError('Existing FastText archive lacks its provenance; inspect it before reuse')
        record = json.loads(meta.read_text())
        if record['url'] != URL or record['bytes'] != ARCHIVE_BYTES or file_hash(path) != record['sha256']:
            raise ValueError('Cached official archive changed')
        return path, record
    start = part.stat().st_size if part.exists() else 0
    if start > ARCHIVE_BYTES:
        raise ValueError('Partial archive is larger than the official file')
    headers = {'Range': f'bytes={start}-'} if start else {}
    server = {}
    if start < ARCHIVE_BYTES:
        with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=90) as response:
            if start and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):
                raise ValueError('Server did not honor archive resume range')
            server = {k:response.headers.get(k) for k in ['ETag','Last-Modified']}
            expected = int(response.headers.get('Content-Length', ARCHIVE_BYTES-start))
            if start+expected != ARCHIVE_BYTES:
                raise ValueError('Official archive size changed; review provenance before downloading')
            with part.open('ab' if start else 'wb') as stream:
                total = start; last = total
                while chunk := response.read(8*1024**2):
                    stream.write(chunk); total += len(chunk)
                    if total-last >= 256*1024**2:
                        print(f'FastText download: {total/1024**3:.2f} GiB', flush=True); last=total
    if part.stat().st_size != ARCHIVE_BYTES:
        raise ValueError('Official FastText download is incomplete')
    record = dict(url=URL, bytes=ARCHIVE_BYTES, sha256=file_hash(part), model='cc.uk.300',
                  license=LICENSE, checksum_policy='First complete HTTPS download pins SHA256; no vendor SHA256 published')
    # Server metadata is informational; an interrupted final rename can be resumed.
    if meta.exists():
        old=json.loads(meta.read_text())
        if any(old.get(k)!=v for k,v in record.items()):raise ValueError('Archive provenance mismatch')
        record=old
    else:
        record['server']=server; save_json(meta,record)
    part.replace(path)
    return path, record


def unpack_model(archive, provenance, work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    path=work/'cc.uk.300.bin';meta=work/'binary.json'
    if path.exists():
        record=json.loads(meta.read_text()) if meta.exists() else {}
        if record.get('archive_sha256')!=provenance['sha256'] or file_hash(path)!=record.get('sha256'):
            raise ValueError('Existing local FastText binary differs or lacks provenance')
        return path,record
    if shutil.disk_usage(work).free < 9*1024**3:
        raise RuntimeError('FastText extraction needs at least 9 GiB free local disk')
    part=work/'cc.uk.300.bin.part';digest=hashlib.sha256();size=0
    with gzip.open(archive,'rb') as src, part.open('wb') as dst:
        while chunk:=src.read(8*1024**2):
            dst.write(chunk);digest.update(chunk);size+=len(chunk)
    header=binary_header(part)
    if (header['words'],header['bucket'],header['minn'],header['maxn'])!=(2000000,2000000,5,5):
        raise ValueError('Unexpected official Ukrainian FastText architecture')
    record=dict(archive_sha256=provenance['sha256'],sha256=digest.hexdigest(),bytes=size,header=header)
    immutable_json(meta,record);part.replace(path)
    return path,record


def available_memory():
    path=Path('/proc/meminfo')
    if not path.exists():raise RuntimeError('Cannot verify host RAM for the full FastText model')
    available=int(next(line.split()[1] for line in path.read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    for limit_path,usage_path in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                                  ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        limit_file,usage_file=Path(limit_path),Path(usage_path)
        if limit_file.exists() and usage_file.exists():
            limit=limit_file.read_text().strip()
            if limit!='max':available=min(available,max(0,int(limit)-int(usage_file.read_text())))
    return available


def load_fasttext(path, expected_sha256, *, official=True):
    if file_hash(path)!=expected_sha256:raise ValueError('FastText binary checksum mismatch')
    header=binary_header(path)
    if official:
        if (header['words'],header['bucket'],header['minn'],header['maxn'])!=(2000000,2000000,5,5):
            raise ValueError('Not the expected official crawl Ukrainian binary')
        if available_memory()<9*1024**3:
            raise RuntimeError('Full FastText loading needs about 9 GiB available RAM; prepare vectors in a High-RAM CPU runtime')
    import fasttext
    if importlib.metadata.version('fasttext')!='0.9.3':raise RuntimeError('Install official fasttext==0.9.3')
    model=fasttext.load_model(str(path))
    if model.get_dimension()!=DIMENSION:raise ValueError('Loaded model is not 300-dimensional')
    probes={}
    for word in ['київ','львів','україна']:
        vector=checked_vector(model,word)
        if np.linalg.norm(vector)==0:raise ValueError(f'FastText known-word probe is zero: {word}')
        probes[word]=dict(dictionary_id=model.get_word_id(word),norm=float(np.linalg.norm(vector)))
    if official and not any(p['dictionary_id']>=0 for p in probes.values()):
        raise ValueError('Ukrainian dictionary probes failed')
    return model,dict(header=header,probes=probes,sha256=expected_sha256,sdk='fasttext==0.9.3')


def checked_vector(model, word):
    value=np.asarray(model.get_word_vector(word),dtype=np.float32)
    if value.shape!=(DIMENSION,) or not np.isfinite(value).all():
        raise ValueError(f'Invalid pretrained vector for token {word!r}')
    return value


def unit_rows(matrix):
    result=np.array(matrix,dtype=np.float32,copy=True)
    norms=np.linalg.norm(result,axis=1,keepdims=True)
    np.divide(result,norms,out=result,where=norms!=0)
    return result


def matrices(vocabulary, provider, seed=42):
    if sorted(vocabulary.values())!=list(range(2,len(vocabulary)+2)):
        raise ValueError('Vocabulary IDs must be contiguous after PAD/UNK')
    random=unit_rows(np.random.default_rng(seed).standard_normal((len(vocabulary)+2,DIMENSION),dtype=np.float32))
    random[0]=0;pretrained=np.zeros_like(random);pretrained[1]=random[1]
    hits=zero=0
    for word,index in vocabulary.items():
        vector=checked_vector(provider,word);pretrained[index]=vector
        hits+=provider.get_word_id(word)>=0;zero+=int(np.linalg.norm(vector)==0)
    pretrained=unit_rows(pretrained);pretrained[1]=random[1]
    return random,pretrained,dict(vocabulary_words=len(vocabulary),dictionary_hits=hits,
        dictionary_coverage=hits/max(1,len(vocabulary)),subword_generated=len(vocabulary)-hits,zero_vectors=zero,
        dimensions=DIMENSION,normalization='unit L2 for nonzero rows',trainable=True,
        primary_oov='Shared seeded UNK; subword OOV inference is a separate diagnostic')


def prepare_vectors(vocabulary, fit, evaluation, cache, work):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    identity=fingerprint(dict(vocabulary=vocabulary,source=URL,seed=42,version=1))
    folder=cache/'matrices'/identity;meta=folder/'vectors.json'
    keys=sorted({token_key(t.value) for d in evaluation for t in d['tokens']} - set(vocabulary))
    if meta.exists():
        record=json.loads(meta.read_text())
        if record['identity']!=identity or record['diagnostic_keys_sha256']!=fingerprint(keys):
            raise ValueError('Prepared vector cache belongs to another input')
        for name,digest in record['files'].items():
            if file_hash(folder/name)!=digest:raise ValueError('Prepared embedding cache checksum changed')
        return folder,record
    archive,origin=official_archive(cache)
    binary,binary_meta=unpack_model(archive,origin,work)
    provider,verified=load_fasttext(binary,binary_meta['sha256'])
    random,pretrained,coverage=matrices(vocabulary,provider)
    extra=unit_rows(np.array([checked_vector(provider,k) for k in keys],dtype=np.float32).reshape(-1,DIMENSION))
    from collections import Counter
    counts=Counter(token_key(t.value) for d in fit for t in d['tokens'])
    known=sum(count for word,count in counts.items() if provider.get_word_id(word)>=0)
    coverage.update(fit_tokens=sum(counts.values()),fit_dictionary_token_hits=known,
                    fit_dictionary_token_coverage=known/max(1,sum(counts.values())),
                    diagnostic_oov_keys=len(keys),diagnostic_nonzero=int(np.count_nonzero(np.linalg.norm(extra,axis=1))))
    del provider;gc.collect()
    folder.mkdir(parents=True,exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(dir=work,prefix='vector_export_') as temporary:
        local=Path(temporary)
        for name,value in [('random.npy',random),('fasttext.npy',pretrained),('oov.npy',extra)]:np.save(local/name,value)
        save_json(local/'oov_keys.json',keys)
        for name in ['random.npy','fasttext.npy','oov.npy','oov_keys.json']:
            target=folder/name
            if target.exists():
                if file_hash(target)!=file_hash(local/name):raise ValueError('Partial vector cache differs; inspect before rerun')
            else:
                pending=folder/(name+'.part');shutil.copy2(local/name,pending);pending.replace(target)
    record=dict(identity=identity,source=origin,binary=binary_meta,verification=verified,coverage=coverage,
                diagnostic_keys_sha256=fingerprint(keys),files={name:file_hash(folder/name) for name in ['random.npy','fasttext.npy','oov.npy','oov_keys.json']})
    save_json(meta,record)
    return folder,record
