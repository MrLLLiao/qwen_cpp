# Deployment Guide

qwen_cpp is not ready for production deployment. The repository currently builds core C++ libraries, tests, and a benchmark; CLI and HTTP service code exists only as scaffolding.

This guide describes current local validation and the deployment model expected once runtime/backend work is implemented.

## Current Local Validation

Use these commands as the supported local verification path:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional benchmark:

```powershell
.\build\benchmarks\qwen_decode_bench.exe
```

This validates the core learning baseline. It does not start a model server or load a real GGUF model.

## Planned Runtime Modes

| Mode | Intended use | Current status |
| --- | --- | --- |
| C++ library | Reusable core for tensor, ops, cache, engine, and model experiments. | Available for current targets. |
| CLI | Local prompt-to-text testing and smoke runs. | Scaffold only. |
| HTTP service | Local API for `/health`, `/models`, and `/generate`. | Scaffold only. |
| Benchmark | Decode-path microbenchmark for attention/cache behavior. | Available. |

## Planned CLI Contract

A future CLI should follow this shape:

```powershell
qwen_cpp generate --config configs/inference/default.yaml --prompt "Explain KV cache."
qwen_cpp serve --config configs/inference/default.yaml --host 127.0.0.1 --port 8080
```

The CLI should load a model manifest, initialize the backend, run tokenizer/runtime generation, and produce clear errors for missing artifacts or unsupported configs.

## Planned Service Contract

The local service should start with a minimal API:

| Route | Purpose |
| --- | --- |
| `GET /health` | Process and backend readiness. |
| `GET /models` | Loaded model metadata from manifests. |
| `POST /generate` | Prompt or chat-style generation request. |

Service responses should include structured error codes from `include/service/api_error.h` once service implementation exists.

## Operational Requirements Before Real Deployment

Before this project is used as a deployable local inference service, it needs:

- real tokenizer implementation or backend-delegated tokenization;
- GGUF/llama.cpp backend integration;
- manifest/config validation;
- CLI build target and smoke test;
- HTTP server implementation and e2e tests;
- structured logging;
- explicit model, dataset, and dependency licenses;
- memory/thread configuration docs for CPU and optional GPU paths.

Until those are implemented, use this repository as a learning and experimentation project rather than an application runtime.
