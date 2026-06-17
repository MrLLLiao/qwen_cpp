# Datasets Guide

This project does not currently implement a real data pipeline. The `python/data_pipeline/prepare_dataset.py` entry point is a scaffold for later supervised fine-tuning or evaluation data preparation.

The rules below define how dataset work should be added when the training path becomes active.

## Goals

Dataset processing must make training inputs reproducible, auditable, and legally reviewable. A dataset should not be introduced only as a path in a config file; it should include source, license, transformation, split, and quality information.

## Recommended Directory Layout

```text
data/
  raw/
    <dataset-name>/<version>/
  interim/
    <dataset-name>/<version>/
  processed/
    <dataset-name>/<version>/
      train.jsonl
      validation.jsonl
      test.jsonl
      DATASET_CARD.md
      quality_report.json
```

`data/` is intentionally not part of the current repository layout. Large raw or processed datasets should normally stay outside Git and be referenced through manifests or reproducible download scripts.

## Minimum Dataset Card

Every dataset used for training or evaluation should document:

| Field | Meaning |
| --- | --- |
| `name` | Human-readable dataset name. |
| `version` | Source version, date, commit, release tag, or snapshot id. |
| `source` | URL, local origin, or generation process. |
| `license` | License identifier and any redistribution restrictions. |
| `intended_use` | Training, validation, regression test, benchmark, or debugging. |
| `schema` | Required fields and field types. |
| `filters` | Cleaning, deduplication, language filtering, or safety filters applied. |
| `splits` | Train/validation/test construction rule and sample counts. |
| `quality_checks` | Empty-field rate, duplicate rate, token-length distribution, and rejected sample count. |

## JSONL Schema Recommendation

For instruction-style supervised fine-tuning, use a simple JSONL schema:

```json
{"id":"sample-000001","instruction":"...","input":"","output":"...","metadata":{"source":"...","license":"..."}}
```

For chat-style data, use:

```json
{"id":"sample-000001","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"source":"...","license":"..."}}
```

Do not mix incompatible schemas in the same processed split. If multiple schemas are needed, record the converter and target schema in the dataset card.

## Processing Contract

`python/data_pipeline/prepare_dataset.py` should eventually provide:

- input source resolution from config or CLI arguments;
- schema validation before writing processed files;
- deterministic split generation through an explicit random seed;
- duplicate and near-duplicate reporting;
- quality report generation;
- license/provenance metadata preservation.

Until that script is implemented, dataset docs should describe intended behavior and must not claim that training data preparation is available.
