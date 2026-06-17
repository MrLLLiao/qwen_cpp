# Training Guide

The repository contains early scaffolding for local fine-tuning, but it does not yet provide a working training implementation. `python/training/train_lora.py` and `configs/training/lora_sft.yaml` are placeholders for a future LoRA/QLoRA workflow.

This guide defines the expected shape of the training path so later implementation can remain reproducible.

## Planned Workflow

```text
raw dataset
  -> dataset preparation
  -> validated JSONL splits
  -> LoRA/QLoRA training
  -> adapter checkpoint
  -> evaluation report
  -> optional merge/export
  -> model manifest update
```

Training should remain Python-first. The C++ side should consume exported inference artifacts rather than becoming a training framework.

## Configuration Contract

The draft config at [../../configs/training/lora_sft.yaml](../../configs/training/lora_sft.yaml) should evolve around these groups:

| Section | Purpose |
| --- | --- |
| `run_name` | Stable run id used in checkpoints, logs, and reports. |
| `base_model` | Local path or registry id for the base model. |
| `dataset` | Processed dataset path, split names, schema, and max sequence length. |
| `training` | Epochs, learning rate, batch size, gradient accumulation, precision, seed, save interval. |
| `lora` | Rank, alpha, dropout, target modules, bias policy. |
| `evaluation` | Validation split, metrics, generation samples, and thresholds. |
| `output` | Adapter directory, logs, report path, and manifest update path. |

## Artifact Layout

Recommended local output layout:

```text
models/adapters/
  <base-model-id>/
    <run-name>/
      adapter_config.json
      adapter_model.safetensors
      training_config.yaml
      eval_report.json
      TRAINING_CARD.md
```

Each adapter must record the base model identity, base model license, dataset versions, training config, code commit, and evaluation summary.

## Evaluation Before Export

A training run should not be treated as usable until it has a minimum evaluation report. For the first implementation, use small but deterministic checks:

- validation loss or perplexity on a fixed split;
- several fixed instruction prompts with captured outputs;
- basic latency and memory notes if the adapter is merged and exported;
- regression notes against the previous accepted adapter.

## Current Limitations

- No training dependencies are declared.
- No LoRA/QLoRA code is implemented.
- No checkpoint, resume, metric logging, or adapter export exists.
- No training artifacts are consumed by the C++ runtime.

The near-term priority is still GGUF inference and runtime integration. Training should progress after a usable local inference path exists.
