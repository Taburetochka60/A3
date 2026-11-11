import importlib.util
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss

from metrics import score_details
from ner_data import IGNORE, collate_ner, decode_bio, training_windows
from pilot_ner import class_weights, infer_probabilities, seed_all
from recurrent_model import NERModel, DomainModel
from validation import collate as collate_domain
from part2_data import file_hash, fingerprint, save_json, fit_encoding, encode_documents
from wandb_logger import start_run, log_epoch, finish_run

CONFIG = dict(name='CLEAN_LSTM2_100k', seed=42, train_rows=100000, batch_size=128,
    encoder=dict(rnn_type='lstm', num_layers=2, embedding_dim=64, hidden_size=64,
                 dropout=.2, bidirectional=True),
    tokenizer='word', optimizer='AdamW', lr=.002, weight_decay=.01, scheduler='none',
    epochs=15, patience=6, threshold=.8, prediction_filter='strict')

EXPERIMENTS = [
    dict(name='GRU24k', rnn_type='gru', num_layers=1, train_rows=24000),
    dict(name='LSTM1_24k', num_layers=1, train_rows=24000),
    dict(name='LSTM2_24k', train_rows=24000),
    dict(name='AdamW_low_lr', train_rows=24000, lr=.0005),
    dict(name='Adam_low_lr', train_rows=24000, optimizer='Adam', lr=.0005),
    dict(name='Adam_no_decay', train_rows=24000, optimizer='Adam', lr=.0005, weight_decay=0.),
    dict(name='cosine24k', train_rows=24000, scheduler='cosine'),
    dict(name='plateau24k', train_rows=24000, scheduler='plateau'),
    dict(name='word_raw24k', train_rows=24000, tokenizer='word_raw'),
    dict(name='BPE24k', train_rows=24000, tokenizer='bpe'),
]


def configuration(**changes):
    result = dict(CONFIG, encoder=CONFIG['encoder'].copy())
    for key in ['rnn_type', 'num_layers']:
        if key in changes:
            result['encoder'][key] = changes.pop(key)
    result.update(changes)
    return result


def gpu():
    if not torch.cuda.is_available() or not Path('/content').exists() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Training requires a hosted Google Colab CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    return dict(name=properties.name, vram_gb=properties.total_memory/2**30,
                bf16=torch.cuda.is_bf16_supported() and properties.major >= 8)


def keep_span(text, a, b, policy):
    if policy == 'original':
        return True
    if policy not in {'strict', 'context'}:
        raise ValueError('Unknown prediction filter')
    if any(c.isalnum() for c in text[a:b]):
        return True
    return (policy == 'context' and text[a:b] in {'-', '‐', '‑'} and 0 < a < b < len(text)
            and text[a-1].isalpha() and text[b].isalpha())


def decode(docs, probabilities, threshold=0., policy='original', decoder=decode_bio):
    if len(docs) != len(probabilities):
        raise ValueError('Missing document probabilities')
    locations, bounds, confidence = [], [], []
    for d, p in zip(docs, probabilities):
        if p.shape != (len(d['tokens']), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5):
            raise ValueError('Invalid BIO probabilities')
        spans, _ = decoder(d['tokens'], p.argmax(1))
        kept, scores = [], []
        for a, b in spans:
            if not keep_span(d['text'], a, b, policy):
                continue
            indices = [i for i, t in enumerate(d['tokens']) if a <= t.start < t.end <= b]
            score = float(p[indices, 1:].sum(1).mean())
            if score >= threshold:
                kept.append((a, b)); scores.append(score)
        locations.append([d['text'][a:b] for a, b in kept])
        bounds.append(kept); confidence.append(scores)
    return locations, bounds, confidence


def score(docs, probabilities, threshold=0., policy='original', decoder=decode_bio):
    predicted, _, _ = decode(docs, probabilities, threshold, policy, decoder)
    gold = [[d['text'][a:b] for a, b in d['spans']] for d in docs]
    loss, tokens = 0., 0
    for d, p in zip(docs, probabilities):
        if d['bio'] is not None and len(p):
            loss -= float(np.log(np.clip(p[np.arange(len(p)), d['bio']], 1e-12, 1.)).sum())
            tokens += len(p)
    return dict(score_details(gold, predicted), val_loss=loss/tokens if tokens else None)


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def batches(examples, size, seed, epoch):
    rng = np.random.default_rng(seed+epoch)
    indices = rng.permutation(len(examples)).tolist()
    result = []
    for a in range(0, len(indices), size*20):
        pool = sorted(indices[a:a+size*20], key=lambda i: len(examples[i][0]))
        result.extend(pool[b:b+size] for b in range(0, len(pool), size))
    rng.shuffle(result)
    return result


def update(model, batch, optimizer, scaler, weights, precision, domain=False):
    ids, lengths, targets = batch
    device = next(model.parameters()).device
    ids, targets = ids.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype={'bf16':torch.bfloat16, 'fp16':torch.float16}.get(precision),
                        enabled=device.type == 'cuda' and precision != 'fp32'):
        logits = model(ids, lengths)
    if domain:
        numerator = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='sum')
        denominator = torch.tensor(len(targets), device=device)
    else:
        numerator = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 3), targets.reshape(-1),
                         weight=weights, ignore_index=IGNORE, reduction='sum')
        denominator = weights[targets[targets != IGNORE]].sum()
    loss = numerator/denominator
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradients')
    scaler.step(optimizer); scaler.update()
    return float(numerator.detach()), float(denominator.detach()), float(norm)


def train(config, training, validation, spec, folder, work_dir='/content/a3_runs',
          domain=False, fixed_epochs=None, rates=None):
    hardware = gpu()
    split_hash = spec['content_sha256']
    if config['scheduler'] not in {'none', 'cosine', 'plateau'}:
        raise ValueError('Unknown scheduler')
    if domain and fixed_epochs is None:
        raise ValueError('Domain validation uses fixed epochs')
    training = training[:config['train_rows']]
    if {d['group'] for d in training} & {d['group'] for d in validation}:
        raise ValueError('Training and validation groups overlap')
    if not domain and (not {d['row_id'] for d in training} <= set(spec['fit'])
                       or len({d['row_id'] for d in training}) != len(training)):
        raise ValueError('NER training rows must be a unique subset of frozen fit')
    if not domain:
        allowed = set(spec['fit']) if fixed_epochs is not None else set(spec['checkpoint'])
        if any('text_id' in d or d['row_id'] not in allowed for d in validation):
            raise ValueError('Epoch selection uses checkpoint only; CV uses its frozen heldout fold')
    identity = fingerprint(dict(config=config, split=split_hash, domain=domain, fixed_epochs=fixed_epochs,
        rates=rates, train=[d['row_id'] for d in training], validation=[d['row_id'] for d in validation]))
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists():
        result = json.loads((folder/'result.json').read_text())
        if result['identity'] != identity or file_hash(folder/'best.pt') != result['checkpoint_sha256']:
            raise ValueError('Existing run differs from this configuration')
        return result
    local = Path(work_dir)/identity[:12]; local.mkdir(parents=True, exist_ok=True)
    seed_all(config['seed'])
    state = fit_encoding(training, config['tokenizer'])
    if not domain and len(training) == 100000 and config['tokenizer'] == 'word':
        if fingerprint(state['vocabulary']) != spec['reference']['vocabulary_sha256']:
            raise ValueError('Final fit vocabulary changed')
    tr, va = encode_documents(training, state), encode_documents(validation, state)
    model = (DomainModel if domain else NERModel)(state['size'], **config['encoder']).cuda()
    weights, counts = (None, None) if domain else class_weights(training, torch.device('cuda'))
    examples = ([(d['ids'][:128] or [1], float(d['domain'])) for d in tr] if domain else
                [row for d in tr for row in training_windows(d['ids'], d['bio'])])
    if not domain and any(d['bio'] is None or not d['ids'] for d in tr):
        raise ValueError('Invalid training BIO')
    optimizer = {'AdamW':torch.optim.AdamW, 'Adam':torch.optim.Adam}[config['optimizer']](
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], betas=(.9, .999), eps=1e-8)
    scheduler = None
    if rates is None and config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    elif rates is None and config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5,
            patience=2, threshold=0., threshold_mode='abs', min_lr=1e-5)
    precision = 'bf16' if hardware['bf16'] else 'fp16'
    scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=precision == 'fp16')
    progress = dict(epoch=1, batch=0, best=-1., bad=0, history=[], seconds=0., totals=[0.,0.,0.,0.,0,0.])
    latest = local/'latest.pt'
    resuming = (folder/'latest.pt').exists()
    if resuming:
        shutil.copy2(folder/'latest.pt', latest)
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved['identity'] != identity or saved['encoding'] != state:
            raise ValueError('Interrupted run has another configuration or vocabulary')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if scheduler:
            scheduler.load_state_dict(saved['scheduler'])
        progress, precision = saved['progress'], saved['precision']
        scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=precision == 'fp16')
        scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
        if precision == 'bf16' and not hardware['bf16']:
            precision = 'fp16'; scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    logger = start_run(folder, identity, config, hardware, domain=domain)
    last_save = segment_start = time.monotonic()

    def save_progress():
        nonlocal last_save, segment_start
        progress['seconds'] += time.monotonic()-segment_start
        segment_start = time.monotonic()
        payload = dict(identity=identity, encoding=state, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None, scaler=scaler.state_dict(),
            progress=progress, precision=precision, rng=rng_state())
        temporary = latest.with_suffix('.tmp'); torch.save(payload, temporary); temporary.replace(latest)
        copy_file(latest, folder/'latest.pt'); last_save = time.monotonic()

    if not resuming:
        save_progress()
    for epoch in range(progress['epoch'], (fixed_epochs or config['epochs'])+1):
        if fixed_epochs is None and progress['bad'] >= config['patience']:
            break
        if rates is not None:
            for group in optimizer.param_groups:
                group['lr'] = rates[min(epoch-1, len(rates)-1)]
        model.train(); torch.cuda.reset_peak_memory_stats()
        loader = DataLoader(examples, batch_sampler=batches(examples, config['batch_size'], config['seed'], epoch)[progress['batch']:],
            collate_fn=collate_domain if domain else collate_ner, pin_memory=True, num_workers=2,
            generator=torch.Generator().manual_seed(config['seed']+epoch))
        for number, batch in enumerate(loader, start=progress['batch']):
            before = rng_state(); tick = time.monotonic()
            try:
                numerator, denominator, norm = update(model, batch, optimizer, scaler, weights, precision, domain)
            except (FloatingPointError, RuntimeError) as error:
                unsupported = any(w in str(error).lower() for w in ['bfloat16','not implemented for','cudnn_status_not_supported','cudnn_status_bad_param','expected scalar type','non-finite'])
                if precision == 'fp32' or not (isinstance(error, FloatingPointError) or unsupported):
                    raise
                restore_rng(before); precision = 'fp32'
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                numerator, denominator, norm = update(model, batch, optimizer, scaler, weights, precision, domain)
            totals = progress['totals']
            totals[0] += numerator; totals[1] += denominator; totals[2] += norm
            totals[3] = max(totals[3], norm); totals[4] += 1; totals[5] += time.monotonic()-tick
            progress['batch'] = number+1
            if time.monotonic()-last_save >= 300:
                save_progress()
        tick = time.monotonic()
        if domain:
            p = domain_probabilities(model, va)
            labels = [d['domain'] for d in va]
            metrics = dict(auc=float(roc_auc_score(labels, p)), val_loss=float(log_loss(labels, p, labels=[0,1])))
        else:
            metrics = score(va, infer_probabilities(model, va, torch.device('cuda'), batch_size=128))
        totals = progress['totals']
        record = dict(epoch=epoch, train_loss=totals[0]/totals[1], learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm=totals[2]/totals[4], max_gradient_norm=totals[3], train_seconds=totals[5],
            validation_seconds=time.monotonic()-tick, epoch_seconds=totals[5]+time.monotonic()-tick,
            peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30,
            precision=precision, **metrics)
        progress['history'].append(record)
        log_epoch(logger, record)
        if fixed_epochs is not None or metrics['f1'] > progress['best']:
            progress['best'] = metrics.get('f1', -1.); progress['bad'] = 0
            torch.save(dict(model=model.state_dict(), config=config, encoding=state, epoch=epoch,
                            identity=identity, split_hash=split_hash, metrics=metrics), local/'best.pt')
            copy_file(local/'best.pt', folder/'best.pt')
        else:
            progress['bad'] += 1
        if scheduler:
            scheduler.step(metrics['f1']) if config['scheduler'] == 'plateau' else scheduler.step()
        pd.DataFrame(progress['history']).to_csv(folder/'history.csv', index=False)
        progress.update(epoch=epoch+1, batch=0, totals=[0.,0.,0.,0.,0,0.]); save_progress()
    saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    result = dict(identity=identity, checkpoint=str(folder/'best.pt'), checkpoint_sha256=file_hash(folder/'best.pt'),
        encoding_sha256=fingerprint(state), config=config, best_epoch=saved['epoch'], metrics=saved['metrics'],
        train_rows=len(training), validation_rows=len(validation), split_sha256=split_hash,
        train_ids=[d['row_id'] for d in training], validation_ids=[d['row_id'] for d in validation],
        epochs=len(progress['history']), seconds=progress['seconds'], class_counts=counts, hardware=hardware,
        history=progress['history'])
    save_json(folder/'result.json', result)
    finish_run(logger, result)
    del model; torch.cuda.empty_cache()
    return result


def copy_file(source, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.tmp')
    shutil.copy2(source, temporary); temporary.replace(target)


def load_model(path, spec, device='cpu', reference=False):
    if reference and file_hash(path) != spec['reference']['checkpoint_sha256']:
        raise ValueError('Reference checkpoint hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['encoding']
    if reference:
        expected = spec['reference']
        if (CONFIG['threshold'], CONFIG['prediction_filter']) != (expected['threshold'], expected['prediction_filter']):
            raise ValueError('Reference threshold or filter changed')
        if (fingerprint(state['vocabulary']) != expected['vocabulary_sha256']
                or fingerprint(state) != expected['encoding_sha256']):
            raise ValueError('Reference vocabulary hash mismatch')
        config = saved['config']
        for key in ['encoder', 'seed', 'train_rows', 'batch_size', 'optimizer', 'lr', 'weight_decay', 'epochs', 'patience', 'tokenizer']:
            if config[key] != CONFIG[key]:
                raise ValueError('Reference model configuration changed: '+key)
        if (config['split_sha256'] != expected['source_split_sha256']
                or config['train_ids_sha256'] != fingerprint(spec['fit'])
                or config['validation_ids_sha256'] != fingerprint(spec['checkpoint'])):
            raise ValueError('Reference model used another split')
    elif saved['split_hash'] != spec['content_sha256']:
        raise ValueError('Checkpoint used another frozen split')
    model = NERModel(state['size'], **saved['config']['encoder']).to(device).eval()
    model.load_state_dict(saved['model'])
    return model, state


def domain_probabilities(model, docs):
    rows = [(d['ids'][:128] or [1], float(d['domain'])) for d in docs]
    values = []; model.eval()
    with torch.inference_mode():
        for ids, lengths, _ in DataLoader(rows, batch_size=128, collate_fn=collate_domain, pin_memory=True):
            values.extend(model(ids.cuda(non_blocking=True), lengths).float().sigmoid().cpu().tolist())
    return np.array(values)


def cross_validate(data, spec, config, result, folder, domain=False):
    if domain:
        test = {d['text_id']: d for d in data['test']}
        source = data['domain_source']+[test[i] for i in spec['av_test']]
        pool = [dict(d, row_id=i, domain=int(i >= len(data['domain_source'])),
                     spans=[], bio=np.zeros(len(d['tokens']), dtype=np.int64), trimmed=[], issues=[]) for i, d in enumerate(source)]
        for d in pool:
            d.pop('text_id', None)
        folds, epochs = spec['av_folds'], 10
    else:
        pool, folds, epochs = data['fit'], spec['cv_folds'], result['best_epoch']
    rates = [r['learning_rate'] for r in result['history'][:result['best_epoch']]]
    rows = []; folder = Path(folder)
    for fold in range(3):
        tr = [d for d, f in zip(pool, folds) if f != fold]
        va = [d for d, f in zip(pool, folds) if f == fold]
        if {d['group'] for d in tr} & {d['group'] for d in va}:
            raise ValueError('Group crosses folds')
        cfg = dict(config, seed=42+fold, train_rows=len(tr))
        run = train(cfg, tr, va, spec, folder/f'fold{fold}', domain=domain, fixed_epochs=epochs, rates=rates)
        saved = torch.load(run['checkpoint'], map_location='cpu', weights_only=True)
        model = (DomainModel if domain else NERModel)(saved['encoding']['size'], **cfg['encoder']).cuda()
        model.load_state_dict(saved['model']); model.eval()
        encoded = encode_documents(va, saved['encoding'])
        if domain:
            rows.extend(dict(row_id=d['row_id'], fold=fold, domain=d['domain'], probability=float(p))
                        for d, p in zip(va, domain_probabilities(model, encoded)))
        else:
            p = infer_probabilities(model, encoded, torch.device('cuda'), batch_size=128)
            locations, _, _ = decode(encoded, p)
            rows.extend(dict(row_id=d['row_id'], fold=fold, locations=json.dumps(v, ensure_ascii=False)) for d, v in zip(va, locations))
        del model; torch.cuda.empty_cache()
    table = pd.DataFrame(rows).set_index('row_id').loc[[d['row_id'] for d in pool]]
    if not table.index.is_unique:
        raise ValueError('Duplicate OOF row IDs')
    table.to_csv(folder/'oof.csv', index_label='row_id')
    if domain:
        summary = dict(auc=float(roc_auc_score(table.domain, table.probability)), rows=len(pool), folds=3)
    else:
        gold = [[d['text'][a:b] for a, b in d['spans']] for d in pool]
        summary = dict(score_details(gold, table.locations.map(json.loads).tolist()), rows=len(pool), folds=3)
    save_json(folder/'metrics.json', summary)
    return summary


def select_postprocessing(validation, vp, calibration, cp):
    comparison = []
    for name, docs, probabilities in [('checkpoint', validation, vp), ('calibration', calibration, cp)]:
        for policy in ['original', 'strict', 'context']:
            original = decode(docs, probabilities)[1]
            retained = decode(docs, probabilities, policy=policy)[1]
            comparison.append(dict(partition=name, filter=policy, threshold=0.,
                removed=sum(map(len, original))-sum(map(len, retained)), **score(docs, probabilities, policy=policy)))
    validation_scores = {r['filter']:r for r in comparison if r['partition'] == 'checkpoint'}
    a, b = validation_scores['strict'], validation_scores['context']
    policy = 'context' if b['f1'] >= a['f1']+.001 and b['fp'] <= a['fp'] else 'strict'
    grid = [dict(threshold=t, **score(calibration, cp, t, policy)) for t in [0.,.5,.6,.7,.8]]
    chosen = max(grid, key=lambda r:(r['f1'], -r['threshold']))
    return dict(filter=policy, threshold=chosen['threshold'], grid=grid, comparison=comparison)


def evaluate(model, state, data, spec, folder, checkpoint, reference=False, final_audit=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    encoded = {name:encode_documents(data[name], state) for name in ['checkpoint', 'calibration']}
    probabilities = {name:infer_probabilities(model, ds, device, batch_size=128) for name, ds in encoded.items()}
    selection = dict(filter=CONFIG['prediction_filter'], threshold=CONFIG['threshold'],
                     checkpoint_sha256=file_hash(checkpoint), split_sha256=spec['content_sha256'])
    if (folder/'selection.json').exists() and json.loads((folder/'selection.json').read_text()) != selection:
        raise ValueError('Selection changed after evaluation')
    save_json(folder/'selection.json', selection)
    metrics = {name:score(encoded[name], probabilities[name], .8, 'strict') for name in encoded}
    metrics['checkpoint_original'] = score(encoded['checkpoint'], probabilities['checkpoint'])
    if not reference:
        development = select_postprocessing(encoded['checkpoint'], probabilities['checkpoint'], encoded['calibration'], probabilities['calibration'])
        save_json(folder/'development.json', development)
    if final_audit:
        audit = encode_documents(data['audit'], state)
        ap = infer_probabilities(model, audit, device, batch_size=128)
        metrics['audit'] = score(audit, ap, .8, 'strict')
    if reference:
        for name in ['checkpoint_original', 'calibration']+(['audit'] if final_audit else []):
            if metrics[name]['f1'] != spec['reference']['metrics'][name]['f1']:
                raise ValueError('Reference metric differs: '+name)
    save_json(folder/'metrics.json', metrics)
    return metrics


def export_submission(docs, probabilities, path, threshold=.8, policy='strict', expected_hash=None):
    locations, spans, confidence = decode(docs, probabilities, threshold, policy)
    ids = [d['text_id'] for d in docs]
    if len(ids) != 477 or len(set(ids)) != 477:
        raise ValueError('Expected 477 unique text IDs')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    pd.DataFrame(dict(text_id=ids, locations=[json.dumps(v, ensure_ascii=False) for v in locations])).to_csv(temporary, index=False)
    table = pd.read_csv(temporary, keep_default_na=False)
    if table.columns.tolist() != ['text_id','locations'] or table.text_id.tolist() != ids:
        raise ValueError('Submission schema or order changed')
    for d, raw, bounds in zip(docs, table.locations, spans):
        values = json.loads(raw)
        if not isinstance(values, list) or len(values) != len(bounds):
            raise ValueError('Invalid JSON locations')
        for value, (a, b) in zip(values, bounds):
            if not isinstance(value, str) or not 0 <= a < b <= len(d['text']) or d['text'][a:b] != value or not keep_span(d['text'],a,b,policy):
                raise ValueError('Invalid entity or original offset')
    if expected_hash and file_hash(temporary) != expected_hash:
        raise ValueError('Submission differs from the saved reference; output was not replaced')
    if path.exists() and path.read_bytes() != temporary.read_bytes():
        raise ValueError('Existing submission differs; choose another output path')
    temporary.replace(path)
    pd.DataFrame(dict(text_id=ids, spans=[json.dumps(v) for v in spans], confidence=[json.dumps(v) for v in confidence])).to_csv(path.with_name(path.stem+'_spans.csv'), index=False)
    return path


def predict(model, state, test, spec, path, reference=False):
    docs = encode_documents(test, state)
    if reference:
        from part2_data import documents_hash
        if documents_hash(docs, with_ids=True) != spec['encoded_sha256']['test']:
            raise ValueError('Reference test encoding changed')
    probabilities = infer_probabilities(model, docs, next(model.parameters()).device, batch_size=128)
    expected = spec['reference']['submission_sha256'] if reference else None
    return export_submission(docs, probabilities, path, CONFIG['threshold'], CONFIG['prediction_filter'], expected)
