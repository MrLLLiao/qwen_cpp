# Architecture Index

This directory collects architecture notes that complement the main architecture contract in [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Current Architecture Boundary

The stable learning path is:

```text
tensor -> ops -> cache -> engine -> model
```

The project also contains tokenizer, runtime, backend, CLI, service, Python training, and artifact-management scaffolding. Those modules define the future direction, but they are not yet part of the stable runtime surface.

## Module Responsibilities

| Module | Responsibility | Current maturity |
| --- | --- | --- |
| `tensor` | Row-major tensor storage, shape/stride metadata, accessors, views, and basic mutation. | Stable and tested. |
| `ops` | Stateless numerical kernels such as `matmul`, `softmax`, and attention. | Stable and tested for current scope. |
| `cache` | KV cache state, capacity, append lifecycle, and cache allocation/management. | Stable and tested. |
| `engine` | Prefill/decode KV orchestration through `CacheManager`. | Stable for KV append orchestration. |
| `model` | Qwen-style layer composition and minimal forward skeleton. | Compiled and tested, still learning-stage. |
| `tokenizer` | Text/token conversion. | Interface scaffold only. |
| `runtime` | Session/model runner facade. | Interface scaffold only. |
| `backend` | Future model backend adapter, especially GGUF/llama.cpp. | Interface scaffold only. |
| `cli` / `service` | User-facing command/API entry points. | Scaffold only and not wired into the main build. |

## Dependency Rules

Allowed high-level direction:

```text
engine -> model
engine -> cache
engine -> ops
model  -> ops
model  -> cache when teaching examples need explicit cache structures
cache  -> tensor
ops    -> tensor
```

Disallowed direction:

```text
ops    -> cache / engine / model / runtime
cache  -> engine / runtime / service
model  -> engine
backend/runtime -> low-level internals except through documented interfaces
```

These rules keep low-level learning modules reusable and prevent future runtime work from leaking into core math and cache code.

## Documentation Map

- [../ARCHITECTURE.md](../ARCHITECTURE.md) is the source of truth for current architecture contracts.
- [../zh/README.md](../zh/README.md) provides the Chinese documentation entry point.
- [../roadmap.md](../roadmap.md) records staged work toward GGUF inference and local training.
- [../../README.md](../../README.md) gives the public project overview and quick-start commands.

When implementation changes module ownership, update the architecture contract and the tests in the same change.
