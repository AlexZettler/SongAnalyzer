# Pub/Sub and Apache Beam streaming (local-first)

This document describes the **optional** messaging layer: **Google Pub/Sub** (including the **emulator**) and **Apache Beam** streaming pipelines that connect corpus expansion, NSynth hyperparameter search, and training. For **topic names, JSON field reference, and request/completion contracts**, see [MESSAGING.md](MESSAGING.md). For corpus schema and training details, see [CORPUS_AND_DATA_PIPELINES.md](CORPUS_AND_DATA_PIPELINES.md) and [PIPELINE_AND_TRAINING.md](PIPELINE_AND_TRAINING.md).

## Install

```bash
pip install -e ".[stream]"
```

NSynth HPO and training handlers also require **`[train]`** (TensorFlow, Optuna, etc.):

```bash
pip install -e ".[stream,train]"
```

HTTP analyze API (separate from Pub/Sub):

```bash
pip install -e ".[api]"
```

## Environment

| Variable | Purpose |
|----------|---------|
| `PUBSUB_EMULATOR_HOST` | `host:port` of the Pub/Sub emulator (e.g. `localhost:8085`) |
| `PUBSUB_PROJECT_ID` | Project id string (any value with the emulator; used by CLI `--project` default) |

With the emulator running, point clients at it by setting `PUBSUB_EMULATOR_HOST`. The official [Pub/Sub emulator](https://cloud.google.com/pubsub/docs/emulator) creates an empty project; topics and subscriptions must exist before Beam consumes messages.

## One-time setup (topics + subscriptions)

```bash
set PUBSUB_EMULATOR_HOST=localhost:8085
song-analyzer messaging setup-pubsub --project dev-local
```

This creates topics `song.requests`, `song.complete`, `hpo.requests`, `hpo.complete`, `train.requests`, `train.complete`, and pull subscriptions `song-requests-worker`, `hpo-requests-worker`, `train-requests-worker`.

## Workflows

### 1) Song corpus expansion

- **Publish** JSON to `song.requests` (CLI: `song-analyzer messaging publish-song-request`).
- **Worker** (`song-analyzer messaging beam-song-worker`): Beam reads the subscription, runs [`process_song_request`](../src/song_analyzer/corpus/song_request.py), publishes `song.complete`.
- **Debug without Pub/Sub:** `song-analyzer messaging run-local-song -f payload.json`.

Payload models live in [`song_analyzer/messaging/payloads.py`](../src/song_analyzer/messaging/payloads.py). Lyrics use the configured connector (the in-tree Genius entry is still a stub).

### 2) Hyperparameter exploration (Optuna)

- **Publish** to `hpo.requests` (`publish-hpo-request` or JSON with `HpoRequestPayload`).
- **Worker:** `song-analyzer messaging beam-hpo-worker`.
- **Persistence:** Each `exploration_id` gets a distinct Optuna **study name** under the same SQLite file (`nsynth_tune.db`). Optional **archive** copies the DB into `tune_archive/` before a run; trials are exported to `exports/<id>_trials.csv` under the tune cache root.
- **Debug:** `song-analyzer messaging run-local-hpo -f payload.json` (requires `[train]`).

Default HPO message uses `skip_final_train=true` so **full training** is triggered separately.

### 3) Training from best trial

- **Publish** to `train.requests` with `study_name` from `hpo.complete` (and matching `tune_cache_dir`).
- **Worker:** `song-analyzer messaging beam-train-worker`.
- **Debug:** `song-analyzer messaging run-local-train -f payload.json`.

### 4) HTTP API

With `[api]` installed:

```bash
set SONGANALYZER_NSYNTH_CHECKPOINT=path\to\nsynth.pt
uvicorn song_analyzer.api.app:app --host 0.0.0.0 --port 8000
```

`POST /analyze` accepts multipart file upload; response is the same structure as `analysis.json` from the CLI (`AnalysisResult`).

Environment knobs: `SONGANALYZER_API_DEVICE`, `SONGANALYZER_API_DEMUCS_MODEL`, `SONGANALYZER_NSYNTH_CHECKPOINT`, `SONGANALYZER_API_STAGED` (`true` / `1` for staged pipeline).

## Sample payloads

```bash
song-analyzer messaging print-sample-payloads
```

## GCP migration

The same topic names and JSON payloads work against real Google Cloud Pub/Sub and Dataflow by changing the Beam **runner** and credentials; replace local SQLite with a single-writer database or object store before scaling workers beyond one process.

## Integration tests (optional)

With the emulator listening and topics created, you can run workers and publish messages manually. Automated tests use mocks and do not require the emulator unless you add a dedicated `RUN_PUBSUB_EMULATOR=1` suite later.
