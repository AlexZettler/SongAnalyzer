"""Append-only JSONL history of proposed parameter vectors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1


@dataclass
class RunRecord:
    step: int
    preset: str
    mode: str
    params: dict[str, Any]
    vector: list[float]
    metric: float | None
    min_dist_to_history: float | None
    weights: list[float] | None = None

    def to_json_obj(self) -> dict[str, Any]:
        md = self.min_dist_to_history
        if md is not None and not math.isfinite(md):
            md = None
        met = self.metric
        if met is not None and not math.isfinite(met):
            met = None
        return {
            "schema": SCHEMA_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
            "step": self.step,
            "preset": self.preset,
            "mode": self.mode,
            "params": self.params,
            "vector": self.vector,
            "metric": met,
            "min_dist_to_history": md,
            "weights": self.weights,
        }


def load_history(path: Path) -> list[RunRecord]:
    if not path.is_file():
        return []
    rows: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rows.append(
            RunRecord(
                step=int(o["step"]),
                preset=str(o["preset"]),
                mode=str(o["mode"]),
                params=dict(o["params"]),
                vector=[float(x) for x in o["vector"]],
                metric=None if o.get("metric") is None else float(o["metric"]),
                min_dist_to_history=(
                    None
                    if o.get("min_dist_to_history") is None
                    else float(o["min_dist_to_history"])
                ),
                weights=(
                    None
                    if o.get("weights") is None
                    else [float(x) for x in o["weights"]]
                ),
            )
        )
    return rows


def append_record(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_json_obj(), sort_keys=True) + "\n")


def iter_vectors(path: Path) -> Iterator[list[float]]:
    for r in load_history(path):
        yield r.vector
