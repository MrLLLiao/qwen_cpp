# Model Manifests

Model manifests describe how an inference artifact is identified, loaded, validated, and traced back to its source. The current runtime does not load manifests yet, but the schema below defines the expected direction for GGUF and future adapter workflows.

See [model.manifest.example.json](model.manifest.example.json) for the current minimal example.

## Purpose

A manifest should answer five questions:

- Which model family and version is this artifact?
- Where are the model, tokenizer, and optional adapter files?
- What license and source restrictions apply?
- How can the artifact be verified?
- Which training/export process produced it?

## Recommended Schema

```json
{
  "manifest_version": "0.1.0",
  "model_id": "qwen2.5-0.5b-instruct-q4_k_m",
  "family": "qwen",
  "format": "gguf",
  "quantization": "Q4_K_M",
  "artifacts": {
    "gguf": "../gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "tokenizer": "../hf/qwen2.5-0.5b-instruct/tokenizer.json",
    "adapter": ""
  },
  "runtime": {
    "n_ctx": 4096,
    "n_gpu_layers": 0
  },
  "metadata": {
    "license": "Apache-2.0",
    "sha256": "<file-sha256>",
    "source": "<download-or-export-source>",
    "created_at": "2026-06-17",
    "created_by": "<tool-or-person>",
    "base_model": "<base-model-id>",
    "training_run": ""
  }
}
```

The current example is intentionally smaller because manifest parsing has not been implemented. Add fields only when the runtime or tooling consumes them.

## Versioning Rules

- `manifest_version` describes the manifest schema, not the model.
- Manifest changes that add optional fields can keep the same minor version.
- Manifest changes that remove or rename fields should bump the minor version and include a migration note.
- Runtime code must reject unsupported manifest versions with a clear error.

## Validation Requirements

Before a manifest is accepted by runtime code, it should validate:

- required fields are present;
- referenced artifact paths exist;
- SHA-256 hashes match when provided;
- model format is supported by the selected backend;
- license metadata is not empty;
- tokenizer path is present when backend does not provide tokenization.

## Relationship to Training and Export

Training outputs should write adapter metadata under `models/adapters/`. Export tools should generate or update a manifest when producing a GGUF or merged artifact. The manifest is the bridge between training-side artifacts and C++ inference-side loading.
