# Hugging Face Artifacts

This directory is reserved for local Hugging Face-style artifacts such as tokenizer files, base model metadata, safetensors checkpoints, and intermediate training outputs.

The intended role of this directory is training and export support. The C++ runtime should normally consume a manifest and GGUF file rather than directly loading a Hugging Face checkpoint.

## Recommended Layout

```text
models/hf/
  <model-id>/
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    README.md
```

Large weight files should normally stay outside Git. If a local artifact is required for reproducibility, document its source and hash in a manifest or model card.

## Relationship to Export

The future export path should read from Hugging Face-compatible artifacts, optionally merge adapters, produce GGUF files, and update a manifest under `models/manifests/`.
