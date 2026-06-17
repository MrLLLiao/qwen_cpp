# GGUF Models

This directory is reserved for local GGUF model files. The repository does not include model weights.

## Recommended Layout

```text
models/gguf/
  <model-family>/
    <model-id>/
      model.gguf
      SHA256SUMS
      SOURCE.md
```

For small local experiments, a flat file is acceptable, but every GGUF artifact should still have a matching manifest under `models/manifests/`.

## Required Metadata

Each GGUF artifact should record:

- source URL or export command;
- base model family and version;
- quantization type;
- license;
- SHA-256 hash;
- export tool version;
- tokenizer compatibility note.

## Current Status

The C++ backend scaffold points toward GGUF support through `GgufLlamaCppBackend`, but real GGUF loading is not implemented. Files placed here are not consumed by the current build unless future runtime/backend code is added.
