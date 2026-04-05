"""Named parameter dimensions, normalization to [0, 1]^d, and project presets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Kind = Literal["log_float", "linear_float", "int", "bool", "categorical_float"]


@dataclass(frozen=True)
class ParamDim:
    """One searchable dimension with decode rules from a unit interval sample."""

    name: str
    kind: Kind
    low: float | None = None
    high: float | None = None
    # For categorical_float: ordered numeric choices; u in [0,1) maps by bucket.
    choices: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ParamSpace:
    dims: tuple[ParamDim, ...]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        out = np.zeros(self.ndim, dtype=np.float64)
        for i, d in enumerate(self.dims):
            v = params[d.name]
            if d.kind == "bool":
                out[i] = 1.0 if bool(v) else 0.0
            elif d.kind == "categorical_float":
                assert d.choices is not None
                fv = float(v)
                try:
                    j = d.choices.index(fv)
                except ValueError:
                    nearest = min(range(len(d.choices)), key=lambda k: abs(d.choices[k] - fv))
                    j = nearest
                n = len(d.choices)
                out[i] = (j + 0.5) / n
            elif d.kind == "int":
                assert d.low is not None and d.high is not None
                lo, hi = int(d.low), int(d.high)
                x = int(round(float(v)))
                x = max(lo, min(hi, x))
                if hi == lo:
                    out[i] = 0.5
                else:
                    out[i] = (x - lo) / (hi - lo)
            elif d.kind == "linear_float":
                assert d.low is not None and d.high is not None
                lo, hi = float(d.low), float(d.high)
                x = float(v)
                x = max(lo, min(hi, x))
                if abs(hi - lo) < 1e-15:
                    out[i] = 0.5
                else:
                    out[i] = (x - lo) / (hi - lo)
            elif d.kind == "log_float":
                assert d.low is not None and d.high is not None
                lo, hi = float(d.low), float(d.high)
                x = max(lo, min(hi, float(v)))
                ln_lo = math.log(lo)
                ln_hi = math.log(hi)
                out[i] = (math.log(x) - ln_lo) / (ln_hi - ln_lo)
            else:
                raise RuntimeError(f"unknown kind {d.kind}")
        return np.clip(out, 0.0, 1.0)

    def decode(self, u: np.ndarray) -> dict[str, Any]:
        if u.shape != (self.ndim,):
            raise ValueError(f"expected vector shape ({self.ndim},), got {u.shape}")
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        out: dict[str, Any] = {}
        for i, d in enumerate(self.dims):
            t = float(u[i])
            if d.kind == "bool":
                out[d.name] = t >= 0.5
            elif d.kind == "categorical_float":
                assert d.choices is not None
                n = len(d.choices)
                j = int(np.floor(t * n))
                j = max(0, min(n - 1, j))
                out[d.name] = d.choices[j]
            elif d.kind == "int":
                assert d.low is not None and d.high is not None
                lo, hi = int(d.low), int(d.high)
                if hi == lo:
                    out[d.name] = lo
                else:
                    val = int(round(lo + t * (hi - lo)))
                    out[d.name] = max(lo, min(hi, val))
            elif d.kind == "linear_float":
                assert d.low is not None and d.high is not None
                lo, hi = float(d.low), float(d.high)
                out[d.name] = lo + t * (hi - lo)
            elif d.kind == "log_float":
                assert d.low is not None and d.high is not None
                lo, hi = float(d.low), float(d.high)
                ln_lo = math.log(lo)
                ln_hi = math.log(hi)
                out[d.name] = math.exp(ln_lo + t * (ln_hi - ln_lo))
            else:
                raise RuntimeError(f"unknown kind {d.kind}")
        return out


def preset_nsynth_tune() -> ParamSpace:
    """Same knobs as ``tune_nsynth`` Optuna search (training hyperparameters)."""
    return ParamSpace(
        dims=(
            ParamDim("lr", "log_float", 1e-5, 1e-2),
            ParamDim("batch_size", "categorical_float", choices=(16.0, 32.0, 64.0)),
            ParamDim("epochs", "int", 1, 5),
            ParamDim("max_steps_per_epoch", "int", 100, 800),
            ParamDim("weight_decay", "log_float", 1e-6, 0.2),
        )
    )


def preset_dense_eval() -> ParamSpace:
    """Dense synthetic clip controls (see ``eval.sidecar.DensityParams``)."""
    return ParamSpace(
        dims=(
            ParamDim("n_notes", "int", 2, 32),
            ParamDim("clip_duration_s", "linear_float", 5.0, 45.0),
            ParamDim("same_family_stack", "bool"),
            ParamDim("detune_max_cents", "linear_float", 0.0, 80.0),
            ParamDim("level_jitter_db", "linear_float", 0.0, 18.0),
            ParamDim("note_audio_duration_s", "linear_float", 3.5, 4.0),
        )
    )
