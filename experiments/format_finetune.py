from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import part2 as base
import part2_data as source
from ner_data import IGNORE, collate_ner, encode_bio, tokenize_offsets, training_windows
from pilot_ner import class_weights, infer_probabilities, seed_all
from metrics import hard_processing
from ner_data import token_key
from experiments.domain_shift_data import case_manifest, eligible_stress_rows, stress_cases
from experiments.representation import feature_documents, restore_model


POLICY = dict(seed=42, max_epochs=2, learning_rate=0.0001, weight_decay=0.01,
              batch_size=128, synthetic_source_rows=30000,
              group_sizes=(3, 3, 4, 10), threshold=0.8, prediction_filter="strict",
              source_f1_tolerance=-0.005, calibration_f1_tolerance=-0.005,
              stress_f1_gain=0.02, bootstrap_repeats=2000)


def make_stress_docs(rows, *, seed, require_all_compatible):
    compatible, selection = eligible_stress_rows(
        rows, fail_on_incompatible=require_all_compatible)
    cases = stress_cases(compatible, seed=seed, group_sizes=POLICY["group_sizes"])
    assembled = []
    for index, case in enumerate(cases):
        tokens = tokenize_offsets(case.text)
        bio, trimmed, _, issues = encode_bio(case.text, case.spans, tokens)
        if issues or bio is None:
            raise ValueError(f"Synthetic document {index} lost BIO alignment: {issues}")
        assembled.append(dict(row_id=-(index + 1), text=case.text,
                              spans=list(case.spans), trimmed=trimmed,
                              tokens=tokens, bio=bio, issues=[],
                              source_ids=case.source_ids, separator=case.separator))
    return assembled, dict(case_manifest(cases), **selection)


def matched_scores(docs, probabilities):
    pred, _, _ = base.decode(docs, probabilities, POLICY["threshold"], POLICY["prediction_filter"])
    gold = [[d["text"][a:b] for a, b in d["spans"]] for d in docs]
    metrics = base.score(docs, probabilities, POLICY["threshold"], POLICY["prediction_filter"])
    # The challenge's normalizer lives in metrics.score_details, not here.
    return metrics, gold, pred


def novelty_recall(docs, predicted, vocabulary):
    """Gold-entity recall by frozen fit vocabulary membership (diagnostic)."""
    counts = {"oov": [0, 0], "iv": [0, 0]}
    for doc, surfaces in zip(docs, predicted):
        normalized = {hard_processing(surface) for surface in surfaces}
        for start, end in doc["spans"]:
            tokens = [token for token in doc["tokens"]
                      if start <= token.start < token.end <= end]
            kind = "oov" if any(token_key(token.value) not in vocabulary for token in tokens) else "iv"
            counts[kind][1] += 1
            counts[kind][0] += hard_processing(doc["text"][start:end]) in normalized
    return {f"{kind}_recall": positive / total if total else None
            for kind, (positive, total) in counts.items()}


def paired_interval(gold, before, after, repeats=2000, seed=42):
    """Paired document resampling; input compositions contain disjoint rows."""
    from metrics import score_details

    if not (len(gold) == len(before) == len(after)):
        raise ValueError("Unpaired diagnostic predictions")
    if len(gold) < 2:
        raise ValueError("Too few independent synthetic posts for bootstrap")
    base_counts = np.array([[score_details([g], [p])[k] for k in ("tp", "fp", "fn")]
                            for g, p in zip(gold, before)], dtype=np.int64)
    next_counts = np.array([[score_details([g], [p])[k] for k in ("tp", "fp", "fn")]
                            for g, p in zip(gold, after)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        selected = rng.integers(0, len(gold), len(gold))
        old = base_counts[selected].sum(axis=0)
        new = next_counts[selected].sum(axis=0)
        f1 = lambda row: (2 * row[0] / (2 * row[0] + row[1] + row[2])) if row.sum() else 1.
        deltas.append(f1(new) - f1(old))
    return [float(x) for x in np.quantile(deltas, [0.025, 0.975])]


def checked_source_fit(all_docs, split, checkpoint_docs):
    fit_ids = [d["row_id"] for d in all_docs]
    if fit_ids != split["fit"]:
        raise ValueError("Unfrozen or reordered source fit")
    if len(set(fit_ids)) != len(fit_ids):
        raise ValueError("Repeated fit IDs")
    if {d["row_id"] for d in all_docs} & {d["row_id"] for d in checkpoint_docs}:
        raise ValueError("Development labels overlap train")
    if {d["group"] for d in all_docs} & {d["group"] for d in checkpoint_docs}:
        raise ValueError("Canonical train/development leakage")


def tracking(project, entity, config):
    import wandb

    # Require online tracking for the course's W&B evidence.
    if not wandb.login():
        raise RuntimeError("W&B login unsuccessful; do not start GPU training")
    kwargs = dict(project=project, job_type="target_format_adaptation",
                  config=config, name="ngrams_multiline_finetune", reinit="finish_previous",
                  id=config["identity"][:8], resume="allow")
    if entity:
        kwargs["entity"] = entity
    return wandb.init(**kwargs)


def save_checkpoint(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(content, tmp)
    tmp.replace(path)


def run(data_root, split_path, reference_checkpoint, reference_sha256, output_root,
        *, project="ai301-a3", entity="dtkach-kse"):
    def stage(message):
        print(f"[A3 fine-tune] {message}", flush=True)

    stage("1/7: verifying frozen source split and labeled documents")
    hardware = base.gpu()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frames, split = source.load_data(data_root, split_path)
    docs = source.prepare_data({k: frames[k] for k in ("fit", "checkpoint", "calibration")}, split)
    checked_source_fit(docs["fit"], split, docs["checkpoint"] + docs["calibration"])
    stage("2/7: verifying the exact original n-grams checkpoint")
    model, representation = restore_model(reference_checkpoint, reference_sha256,
                                           split["content_sha256"])

    # The augmented documents are constructed from fit-only rows; the
    # checkpoint/calibration compositions are a stress *measurement* only.
    rng = random.Random(POLICY["seed"])
    selected = rng.sample(docs["fit"], POLICY["synthetic_source_rows"])
    train_stress, train_manifest = make_stress_docs(
        selected, seed=POLICY["seed"], require_all_compatible=True)
    checkpoint_stress, checkpoint_manifest = make_stress_docs(
        docs["checkpoint"], seed=1337, require_all_compatible=False)
    calibration_stress, calibration_manifest = make_stress_docs(
        docs["calibration"], seed=1338, require_all_compatible=False)
    stage("3/7: deterministic fit-only multiline augmentation complete; "
          f"stress eligibility: checkpoint {checkpoint_manifest['eligible_rows']}/"
          f"{checkpoint_manifest['input_rows']}, calibration "
          f"{calibration_manifest['eligible_rows']}/{calibration_manifest['input_rows']}")
    all_train_ids = {d["row_id"] for d in docs["fit"]}
    if any(any(i not in all_train_ids for i in d["source_ids"]) for d in train_stress):
        raise ValueError("Synthetic training imported a protected row")

    # One deterministic, non-persistent word-routing table.  Evaluation keys
    # do not fit an embedding or subword vocabulary.
    counts = (len(docs["fit"]), len(train_stress), len(docs["checkpoint"]),
              len(checkpoint_stress), len(docs["calibration"]), len(calibration_stress))
    encoded = feature_documents(docs["fit"] + train_stress + docs["checkpoint"]
                                + checkpoint_stress + docs["calibration"]
                                + calibration_stress, representation, model)
    parts = []; cursor = 0
    for length in counts:
        parts.append(encoded[cursor:cursor + length])
        cursor += length
    fit, synthetic, checkpoint, checkpoint_synthetic, calibration, calibration_synthetic = parts
    del encoded
    stage("4/7: source and synthetic token features encoded")

    checkpoint_sha = source.file_hash(reference_checkpoint)
    module_root = Path(__file__).resolve().parent
    code_sha256 = source.fingerprint({name: source.file_hash(module_root / name)
        for name in ("domain_shift_data.py", "representation.py", "format_finetune.py")})
    source_commit = os.environ.get("A3_SOURCE_SHA", "local-uncommitted")
    identity = source.fingerprint(dict(config=POLICY, split=split["content_sha256"],
                                       base=checkpoint_sha, manifests=[train_manifest,
                                                                        checkpoint_manifest, calibration_manifest],
                                       source_commit=source_commit, code=code_sha256))
    state_path = output_root / "run_state.pt"
    report_path = output_root / "decision.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing["identity"] != identity:
            raise ValueError("Existing result belongs to another experiment")
        if source.file_hash(output_root / "candidate.pt") != existing["checkpoint_sha256"]:
            raise ValueError("Existing completed candidate checkpoint changed")
        return existing
    source.save_json(output_root / "identity.json", dict(identity=identity, policy=POLICY,
        source_commit=source_commit, code_sha256=code_sha256,
        split_sha256=split["content_sha256"], reference_checkpoint_sha256=checkpoint_sha,
        source_manifests=[train_manifest, checkpoint_manifest, calibration_manifest],
        synthetic_format_not_target_labels=True, hardware=hardware))

    seed_all(POLICY["seed"])
    stage("5/7: reproducing unchanged full checkpoint/calibration baseline")
    original_check_probs = infer_probabilities(model, checkpoint, torch.device("cuda"), batch_size=128)
    original_stress_probs = infer_probabilities(model, checkpoint_synthetic, torch.device("cuda"), batch_size=128)
    original_cal_probs = infer_probabilities(model, calibration, torch.device("cuda"), batch_size=128)
    original_cal_stress_probs = infer_probabilities(model, calibration_synthetic, torch.device("cuda"), batch_size=128)
    original_checkpoint, _, original_check_pred = matched_scores(checkpoint, original_check_probs)
    original_stress, stress_gold, stress_preds = matched_scores(checkpoint_synthetic, original_stress_probs)
    original_cal, _, original_cal_pred = matched_scores(calibration, original_cal_probs)
    original_cal_stress, _, _ = matched_scores(calibration_synthetic, original_cal_stress_probs)
    # Reference metric is a hard guard against changing the old model's math.
    if abs(original_checkpoint["f1"] - 0.831981460023175) > 1e-8:
        raise ValueError("Failed to reproduce the recorded ngrams checkpoint F1")
    if abs(original_cal["f1"] - 0.8039969254419678) > 1e-8:
        raise ValueError("Failed to reproduce the recorded ngrams calibration F1")
    stage(f"Baseline reproduced: checkpoint F1={original_checkpoint['f1']:.8f}, "
          f"calibration F1={original_cal['f1']:.8f}")
    baseline = {"checkpoint": dict(original_checkpoint,
        **novelty_recall(checkpoint, original_check_pred, representation["vocabulary"])),
        "stress": original_stress,
        "calibration": dict(original_cal,
            **novelty_recall(calibration, original_cal_pred, representation["vocabulary"])),
        "calibration_stress": original_cal_stress}
    source.save_json(output_root / "baseline_metrics.json", baseline)

    tracker = tracking(project, entity, dict(POLICY, identity=identity,
        reference_checkpoint_sha256=checkpoint_sha,
        synthetic_rows=len(synthetic), stress_rows=len(checkpoint_synthetic)))
    tracker.log({f"baseline/{k}/{name}": value for k, metrics in baseline.items()
                 for name, value in metrics.items() if isinstance(value, (float, int))}, step=0)
    training = fit + synthetic
    windows = [w for d in training for w in training_windows(d["ids"], d["bio"])]
    weights, _ = class_weights(docs["fit"], torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=POLICY["learning_rate"],
                                  weight_decay=POLICY["weight_decay"])
    precision = "bf16" if hardware["bf16"] else "fp16"
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=precision == "fp16")
    progress = dict(epoch=1, next_batch=0, completed_epochs=[], elapsed=0.,
                    totals=[0., 0., 0., 0., 0.])
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Checkpoint resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        base.restore_rng(state["rng"])
        progress = state["progress"]
    last_save = time.monotonic()

    def persist():
        nonlocal last_save
        save_checkpoint(state_path, dict(identity=identity, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            rng=base.rng_state(), progress=progress))
        last_save = time.monotonic()

    try:
        stage("6/7: W&B-connected fine-tuning; two fixed epochs, "
              "progress saved to Drive every ~2 minutes")
        for epoch in range(progress["epoch"], POLICY["max_epochs"] + 1):
            stage(f"Starting epoch {epoch}/{POLICY['max_epochs']} "
                  f"(resuming from batch {progress['next_batch']})")
            model.train()
            torch.cuda.reset_peak_memory_stats()
            batches = base.batches(windows, POLICY["batch_size"], POLICY["seed"], epoch)
            loader = DataLoader(windows, batch_sampler=batches[progress["next_batch"]:],
                collate_fn=collate_ner, pin_memory=True, num_workers=2,
                generator=torch.Generator().manual_seed(POLICY["seed"] + epoch))
            started = time.monotonic()
            for number, batch in enumerate(loader, start=progress["next_batch"]):
                numer, denom, norm = base.update(model, batch, optimizer, scaler, weights, precision)
                totals = progress["totals"]
                totals[0] += numer; totals[1] += denom
                totals[2] += norm; totals[3] += 1; totals[4] = max(totals[4], norm)
                progress["next_batch"] = number + 1
                if time.monotonic() - last_save >= 120:
                    persist()
            model.eval()
            p = infer_probabilities(model, checkpoint, torch.device("cuda"), batch_size=128)
            check, _, _ = matched_scores(checkpoint, p)
            p = infer_probabilities(model, checkpoint_synthetic, torch.device("cuda"), batch_size=128)
            stress, _, _ = matched_scores(checkpoint_synthetic, p)
            elapsed = time.monotonic() - started
            totals = progress["totals"]
            flow = {f"grad/{name}": float(param.grad.detach().norm())
                    for name, param in model.named_parameters() if param.grad is not None}
            record = dict(epoch=epoch, train_loss=totals[0] / totals[1],
                mean_gradient_norm=totals[2] / totals[3], max_gradient_norm=totals[4],
                epoch_seconds=elapsed,
                checkpoint_f1=check["f1"], synthetic_stress_f1=stress["f1"],
                checkpoint_precision=check["precision"], checkpoint_recall=check["recall"],
                learning_rate=POLICY["learning_rate"], gpu=hardware["name"],
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30)
            progress["completed_epochs"].append(record)
            progress.update(epoch=epoch + 1, next_batch=0,
                            elapsed=progress["elapsed"] + elapsed, totals=[0., 0., 0., 0., 0.])
            pd.DataFrame(progress["completed_epochs"]).to_csv(output_root / "history.csv", index=False)
            persist()
            tracker.log(dict(record, **flow), step=epoch)
            stage(f"Finished epoch {epoch}: checkpoint F1={check['f1']:.5f}, "
                  f"multiline stress F1={stress['f1']:.5f}")
        # Fixed epoch budget: no selection based on calibration/audit/Kaggle.
        model.eval()
        new_checkpoint_probs = infer_probabilities(model, checkpoint, torch.device("cuda"), batch_size=128)
        new_stress_probs = infer_probabilities(model, checkpoint_synthetic, torch.device("cuda"), batch_size=128)
        new_cal_probs = infer_probabilities(model, calibration, torch.device("cuda"), batch_size=128)
        new_cal_stress_probs = infer_probabilities(model, calibration_synthetic, torch.device("cuda"), batch_size=128)
        now_checkpoint, _, new_check_pred = matched_scores(checkpoint, new_checkpoint_probs)
        now_stress, _, new_stress_preds = matched_scores(checkpoint_synthetic, new_stress_probs)
        now_cal, _, new_cal_pred = matched_scores(calibration, new_cal_probs)
        now_cal_stress, _, _ = matched_scores(calibration_synthetic, new_cal_stress_probs)
        ci = paired_interval(stress_gold, stress_preds, new_stress_preds,
                             repeats=POLICY["bootstrap_repeats"])
        gains = dict(checkpoint=now_checkpoint["f1"] - original_checkpoint["f1"],
            stress=now_stress["f1"] - original_stress["f1"],
            calibration=now_cal["f1"] - original_cal["f1"])
        gates = dict(checkpoint=gains["checkpoint"] >= POLICY["source_f1_tolerance"],
            calibration=gains["calibration"] >= POLICY["calibration_f1_tolerance"],
            stress=gains["stress"] >= POLICY["stress_f1_gain"],
            bootstrap=ci[0] > 0)
        result = dict(identity=identity, status="complete", accepted=all(gates.values()),
            architecture="2-layer BiLSTM + word and char n-grams",
            method="fit-only label-preserving multiline source concatenation, two-epoch finetuning",
            first_model_unchanged=checkpoint_sha == reference_sha256,
            baseline=baseline, candidate={"checkpoint": dict(now_checkpoint,
                    **novelty_recall(checkpoint, new_check_pred, representation["vocabulary"])),
                                         "stress": now_stress,
                                         "calibration": dict(now_cal,
                    **novelty_recall(calibration, new_cal_pred, representation["vocabulary"])),
                                         "calibration_stress": now_cal_stress},
            paired_stress_delta_f1_95ci=ci, gains=gains, gates=gates,
            skipped_audit=True, no_test_labels=True, no_kaggle_submission=True,
            stress_eligibility={"checkpoint": checkpoint_manifest,
                                "calibration": calibration_manifest},
            timestamp=datetime.now(timezone.utc).isoformat())
        save_checkpoint(output_root / "candidate.pt", dict(model=model.state_dict(),
            representation=representation, split=split["content_sha256"],
            identity=identity, policy=POLICY))
        result["checkpoint_sha256"] = source.file_hash(output_root / "candidate.pt")
        source.save_json(report_path, result)
        # Report figures are derived from saved epoch rows, not hidden notebook
        # output. They can be recreated without GPU training.
        import matplotlib.pyplot as plt
        history = pd.DataFrame(progress["completed_epochs"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for key in ("train_loss",):
            axes[0].plot(history.epoch, history[key], marker="o", label=key)
        axes[0].set(title="Optimization", xlabel="Epoch", ylabel="Train loss")
        axes[1].plot(history.epoch, history.checkpoint_f1, marker="o", label="Source checkpoint F1")
        axes[1].plot(history.epoch, history.synthetic_stress_f1, marker="o", label="Multiline stress F1")
        axes[1].set(title="Validation", xlabel="Epoch", ylabel="F1")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_root / "learning_curves.png", dpi=140)
        tracker.log({"learning_curves": __import__("wandb").Image(fig)})
        plt.close(fig)
        tracker.log({"accepted": int(result["accepted"]),
                     "delta_stress_f1": gains["stress"],
                     "delta_source_f1": gains["checkpoint"],
                     "delta_calibration_f1": gains["calibration"]})
        tracker.summary.update(result)
        stage("7/7: saved candidate, decision, plots and W&B metrics; "
              f"accepted={result['accepted']}")
        return result
    finally:
        tracker.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", default="ai301-a3")
    parser.add_argument("--entity", default="dtkach-kse")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.split, args.checkpoint,
                         args.checkpoint_sha256, args.output_dir,
                         project=args.project, entity=args.entity), indent=2))


if __name__ == "__main__":
    main()
