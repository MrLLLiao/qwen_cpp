# Adapters

This directory is reserved for LoRA/QLoRA adapter artifacts produced by the future training pipeline.

## Recommended Layout

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

## Required Metadata

Each adapter should document:

- base model id, revision, and license;
- dataset versions and licenses;
- training config and random seed;
- code commit or tool version;
- validation metrics and sample outputs;
- export compatibility notes.

## Current Status

No adapter training, merge, or export workflow is implemented. This directory only defines the artifact convention for future work.
