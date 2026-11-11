"""Part 3 ensemble: average member BIO probabilities and choose the recipe on test-like data.

Selection uses only the testlike_select split. testlike_report, labeling (the 26
organiser examples), checkpoint and calibration are reported for every candidate
but never steer the choice; audit is scored once after the recipe is frozen.
No member is retrained here: every combination reuses saved probability bundles.
"""
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import score_details
from part2 import decode, export_submission
from part2_data import fingerprint, save_json
from ensemble_members import load_bundle

THRESHOLDS = (0., .5, .6, .7, .8, .9)
REPORT = ('testlike_report', 'labeling', 'checkpoint', 'calibration')


def sweep(docs, probabilities, thresholds=THRESHOLDS, policy='strict'):
    """Metrics for every threshold from one decode (threshold only drops low-confidence spans)."""
    locations, _, confidence = decode(docs, probabilities, 0., policy)
    gold = [[d['text'][a:b] for a, b in d['spans']] for d in docs]
    return [dict(threshold=t, **score_details(gold, [[s for s, c in zip(l, cs) if c >= t]
                                                     for l, cs in zip(locations, confidence)]))
            for t in thresholds]


def best(rows):
    return max(rows, key=lambda r: (r['f1'], -r['threshold']))


def document_counts(docs, probabilities, threshold, policy='strict'):
    """Per-document tp/fp/fn; their column sums are exactly the competition counts."""
    locations, _, _ = decode(docs, probabilities, threshold, policy)
    rows = [score_details([[d['text'][a:b] for a, b in d['spans']]], [l]) for d, l in zip(docs, locations)]
    return np.array([[r['tp'], r['fp'], r['fn']] for r in rows])


def bootstrap_delta(a, b, samples=2000, seed=42, chunk=250):
    """Paired bootstrap over documents of F1(b) - F1(a), given document_counts of both."""
    def f1(c):
        return 2*c[..., 0]/np.maximum(2*c[..., 0]+c[..., 1]+c[..., 2], 1)
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, samples, chunk):
        w = rng.multinomial(len(a), np.full(len(a), 1/len(a)), size=min(chunk, samples-start))
        deltas.append(f1(w@b)-f1(w@a))
    deltas = np.concatenate(deltas)
    low, high = np.quantile(deltas, [.025, .975])
    return dict(delta=float(f1(b.sum(0))-f1(a.sum(0))), low=float(low), high=float(high),
                share_positive=float((deltas > 0).mean()))


def blend(members, weights):
    weights = {n: w for n, w in weights.items() if w > 0}
    if not weights:
        raise ValueError('At least one member needs a positive weight')
    total = sum(weights.values())
    count = len(members[next(iter(weights))])
    return [sum(w*members[n][i] for n, w in weights.items())/total for i in range(count)]


def load_members(bundle_dir, names, split, docs):
    return {n: load_bundle(Path(bundle_dir)/n/f'{split}.npz', docs)[0] for n in names}


def subset_search(members, docs, thresholds=THRESHOLDS, policy='strict', max_size=4):
    rows = []
    for size in range(1, min(max_size, len(members))+1):
        for combo in itertools.combinations(sorted(members), size):
            r = best(sweep(docs, blend(members, dict.fromkeys(combo, 1.)), thresholds, policy))
            rows.append(dict(members='+'.join(combo), size=size, **r))
    return pd.DataFrame(rows).sort_values(['f1', 'size'], ascending=[False, True], kind='stable')


def greedy_search(members, docs, rounds=10, thresholds=THRESHOLDS, policy='strict'):
    """Forward selection with replacement (Caruana et al., 2004); weights are selection counts."""
    counts = dict.fromkeys(sorted(members), 0)
    trace = []
    for step in range(1, rounds+1):
        trials = [(best(sweep(docs, blend(members, dict(counts, **{n: counts[n]+1})), thresholds, policy)), n)
                  for n in counts]
        r, name = max(trials, key=lambda x: (x[0]['f1'], -x[0]['threshold']))
        counts[name] += 1
        trace.append(dict(step=step, added=name, weights=json.dumps(counts), **r))
    return pd.DataFrame(trace)


def recipe(name, weights, threshold, policy, selected_on, f1):
    total = sum(weights.values())
    weights = {n: w/total for n, w in sorted(weights.items()) if w > 0}
    return dict(name=name, weights=weights, threshold=threshold, policy=policy, selected_on=selected_on,
                select_f1=f1, recipe_sha256=fingerprint([weights, threshold, policy]))


def evaluate_recipe(r, members_by_split, docs_by_split):
    rows = []
    for split, members in members_by_split.items():
        probabilities = blend(members, r['weights'])
        rows.append(dict(recipe=r['name'], split=split, **sweep(docs_by_split[split], probabilities,
                                                               [r['threshold']], r['policy'])[0]))
    return rows


def run_search(bundle_dir, names, docs_by_split, out_dir, select='testlike_select', report=REPORT,
               thresholds=THRESHOLDS, policy='strict', rounds=10, max_size=4):
    """Choose a recipe on `select`, report every candidate on the other splits, save everything to out_dir."""
    out_dir = Path(out_dir)
    if 'audit' in report:
        raise ValueError('Audit is scored with score_audit after the recipe is frozen')
    splits = [select]+[s for s in report if s in docs_by_split]
    members = {s: load_members(bundle_dir, names, s, docs_by_split[s]) for s in splits}
    chosen = members[select]; docs = docs_by_split[select]
    subsets = subset_search(chosen, docs, thresholds, policy, max_size)
    greedy = greedy_search(chosen, docs, rounds, thresholds, policy)
    top = greedy.sort_values(['f1', 'step'], ascending=[False, True], kind='stable').iloc[0]
    candidates = [recipe('greedy', json.loads(top.weights), float(top.threshold), policy, select, float(top.f1))]
    first = subsets.iloc[0]
    candidates.append(recipe('best_subset', dict.fromkeys(first.members.split('+'), 1.), float(first.threshold),
                             policy, select, float(first.f1)))
    for row in subsets.loc[subsets['size'].eq(1)].itertuples():
        candidates.append(recipe('single_'+row.members, {row.members: 1.}, float(row.threshold), policy, select, float(row.f1)))
    final = max(candidates, key=lambda r: (r['select_f1'], -len(r['weights'])))
    table = pd.DataFrame([row for r in candidates for row in evaluate_recipe(r, members, docs_by_split)])
    result = dict(final=final, candidates=candidates, members=list(names), select_rows=len(docs),
                  created=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    if (out_dir/'recipe.json').exists() and json.loads((out_dir/'recipe.json').read_text())['final'] != final:
        raise ValueError('A different recipe is already saved here; use a new search folder')
    out_dir.mkdir(parents=True, exist_ok=True)
    subsets.to_csv(out_dir/'subsets.csv', index=False)
    greedy.to_csv(out_dir/'greedy.csv', index=False)
    table.to_csv(out_dir/'report.csv', index=False)
    save_json(out_dir/'recipe.json', result)
    return result, table


def score_audit(r, bundle_dir, audit_docs, out_dir):
    """One-time audit score of a frozen recipe; refuses to run twice in the same folder."""
    path = Path(out_dir)/'audit.json'
    if path.exists():
        raise ValueError('Audit was already scored for this recipe')
    members = {'audit': load_members(bundle_dir, list(r['weights']), 'audit', audit_docs)}
    row = evaluate_recipe(r, members, {'audit': audit_docs})[0]
    save_json(path, dict(row, recipe_sha256=r['recipe_sha256']))
    return row


def export_test(r, bundle_dir, test_docs, path):
    members = load_members(bundle_dir, list(r['weights']), 'test', test_docs)
    return export_submission(test_docs, blend(members, r['weights']), path, r['threshold'], r['policy'])


def record_submission(ledger, r, submission, public_f1, note=''):
    """Append an actual Kaggle public score; never enter a score that was not submitted."""
    ledger = Path(ledger)
    row = pd.DataFrame([dict(date=datetime.now(timezone.utc).date().isoformat(), recipe=r['name'],
                             recipe_sha256=r['recipe_sha256'], weights=json.dumps(r['weights']),
                             threshold=r['threshold'], policy=r['policy'], select_f1=r['select_f1'],
                             submission=str(submission), public_f1=public_f1, note=note)])
    row.to_csv(ledger, mode='a', header=not ledger.exists(), index=False)
    return row


def log_wandb(name, out_dir):
    """Log a finished search to W&B (job_type='ensemble'); off unless WANDB_PROJECT is set."""
    project = os.environ.get('WANDB_PROJECT')
    if not project:
        return None
    import wandb
    out_dir = Path(out_dir)
    result = json.loads((out_dir/'recipe.json').read_text())
    run = wandb.init(project=project, entity=os.environ.get('WANDB_ENTITY') or None, job_type='ensemble',
                     name=name, config=dict(final=result['final'], members=result['members']),
                     reinit='finish_previous')
    for table in ['subsets', 'greedy', 'report']:
        run.log({table: wandb.Table(dataframe=pd.read_csv(out_dir/f'{table}.csv'))})
    report = pd.read_csv(out_dir/'report.csv')
    final = report.loc[report.recipe.eq(result['final']['name'])]
    run.summary.update({f'{r.split}/f1': r.f1 for r in final.itertuples()})
    run.finish()
    return run
