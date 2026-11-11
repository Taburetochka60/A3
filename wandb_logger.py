"""Optional shared Weights & Biases logging for A3 training and CV runs.

Set WANDB_PROJECT (and WANDB_API_KEY for online mode) to enable it.
No import, network call, or authentication occurs when tracking is disabled.
"""

import os


def start_run(folder, identity, config, hardware, domain=False):
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    mode = os.environ.get("WANDB_MODE", "online").lower()
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("W&B needs WANDB_API_KEY; alternatively set WANDB_MODE=offline")
    import wandb

    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY") or None,
        mode=mode,
        id=identity[:8],
        resume="allow",
        job_type="adversarial_validation" if domain else "ner_training",
        name=config["name"] + ("_AV" if domain else ""),
        config={"training": config, "hardware": hardware, "identity": identity,
                "domain": domain},
        dir=str(folder),
        reinit="finish_previous",
    )


def log_epoch(run, record):
    if run is not None:
        run.log(record, step=int(record["epoch"]))


def finish_run(run, result):
    if run is not None:
        run.summary.update({
            "best_epoch": result["best_epoch"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "train_rows": result["train_rows"],
            "validation_rows": result["validation_rows"],
            "epochs": result["epochs"],
            "runtime_seconds": result["seconds"],
            **{f"best_{key}": value for key, value in result["metrics"].items()
               if isinstance(value, (int, float))},
        })
        run.finish()
