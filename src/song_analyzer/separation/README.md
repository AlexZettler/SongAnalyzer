# Demucs integration

[`demucs_sep.py`](demucs_sep.py) loads a pretrained Demucs model by name and returns a dictionary keyed by **`model.sources`** for that checkpoint. Examples:

| Model (typical) | Stems |
|-----------------|--------|
| `htdemucs` | `drums`, `bass`, `other`, `vocals` |
| `htdemucs_6s` | adds `guitar`, `piano` |

Callers should iterate `stems.keys()` rather than hardcoding four stems. Audio is processed at `model.samplerate`; the original mix sample rate is only used for resampling into the model.
