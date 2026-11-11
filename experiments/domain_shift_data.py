from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Iterable, Mapping, Sequence


SEPARATORS = ("\n", "\n\n", "\n• ")


@dataclass(frozen=True)
class ComposedText:
    text: str
    spans: tuple[tuple[int, int], ...]
    source_ids: tuple[int, ...]
    separator: str


def _check_row(row: Mapping) -> None:
    text = row["text"]
    if not isinstance(text, str) or not text:
        raise ValueError("Rows must have nonempty original text")
    if not isinstance(row["row_id"], int):
        raise ValueError("Rows require unique integer source row_ids")
    previous_end = -1
    for start, end in sorted(row["spans"]):
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError("Invalid original character offsets")
        if start < previous_end:
            raise ValueError("Overlapping source entity spans")
        previous_end = end


def concatenate(rows: Sequence[Mapping], separator: str = "\n") -> ComposedText:
    """Stitch 2-12 source texts; shift, never infer, labelled LOC offsets."""
    if not 2 <= len(rows) <= 12:
        raise ValueError("Expected two to twelve source texts")
    if separator not in SEPARATORS:
        raise ValueError("Use only preregistered harmless layout separators")
    ids = tuple(int(row["row_id"]) for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated source row in one synthetic text")
    for row in rows:
        _check_row(row)
    combined = []
    spans = []
    position = 0
    for index, row in enumerate(rows):
        if index:
            combined.append(separator)
            position += len(separator)
        value = row["text"]
        combined.append(value)
        for start, end in row["spans"]:
            spans.append((position + int(start), position + int(end)))
        position += len(value)
    result = ComposedText("".join(combined), tuple(sorted(spans)), ids, separator)
    for row, start_index in zip(rows, result.source_ids):
        assert row["row_id"] == start_index
    return result


def stress_cases(rows: Sequence[Mapping], seed: int = 42,
                 group_sizes: Iterable[int] = (3, 3, 4, 10)) -> list[ComposedText]:
    """Deterministic, disjoint composition of a *single* source partition.

    No Kaggle text, organizer labels, checkpoint rows (when training), or audit
    rows are imported.  The caller must enforce the frozen fit/fold boundary.
    """
    sizes = tuple(group_sizes)
    if not sizes or any(not 2 <= size <= 12 for size in sizes):
        raise ValueError("Supported group sizes: 2..12")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if len({rows[i]["row_id"] for i in indices}) != len(indices):
        raise ValueError("Duplicate source row IDs")
    output = []
    index = 0
    case = 0
    while index + 1 < len(indices):
        size = sizes[case % len(sizes)]
        if len(indices) - index < size:
            size = len(indices) - index
        if size < 2:
            break
        selection = [rows[i] for i in indices[index:index + size]]
        output.append(concatenate(selection, SEPARATORS[case % len(SEPARATORS)]))
        case += 1
        index += size
    return output


def eligible_stress_rows(rows: Sequence[Mapping], *, fail_on_incompatible: bool):
    """Exclude incompatible BIO labels from *synthetic* evaluation only.

    Original checkpoint/calibration rows remain in the unchanged main metrics.
    The fit partition must be fully compatible; it must never be filtered.
    """
    eligible, excluded_ids = [], []
    for row in rows:
        if row.get("bio") is None or row.get("issues"):
            excluded_ids.append(int(row["row_id"]))
        else:
            eligible.append(row)
    if fail_on_incompatible and excluded_ids:
        raise ValueError("Incompatible BIO annotations in the frozen fit partition")
    if len(eligible) < 2:
        raise ValueError("Not enough compatible labeled rows for the stress test")
    fingerprint = sha256(json.dumps(excluded_ids, separators=(",", ":")).encode()).hexdigest()
    return eligible, {"input_rows": len(rows), "eligible_rows": len(eligible),
                      "excluded_incompatible_rows": len(excluded_ids),
                      "excluded_ids_sha256": fingerprint}


def case_manifest(cases: Sequence[ComposedText]) -> dict:
    payload = [{"ids": x.source_ids, "sep": x.separator,
                "text_sha256": sha256(x.text.encode()).hexdigest(), "spans": x.spans}
               for x in cases]
    digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"cases": len(cases), "unique_source_rows": len(set(i for c in cases for i in c.source_ids)),
            "sha256": digest, "synthetic_not_target_labels": True}
