# qwen_cpp

qwen_cpp is a C++ learning project for understanding the implementation path of Transformer inference systems. The repository focuses on the engineering layers behind a Qwen-style local inference stack: tensor storage, stateless operators, KV cache management, prefill/decode orchestration, and a minimal model forward skeleton.

The project is not yet a production inference runtime. The stable learning baseline is `tensor -> ops -> cache -> engine -> model`. Higher-level modules such as tokenizer, runtime, backend, CLI, HTTP service, data pipeline, training, and GGUF export are present as scaffolding for later milestones.

## Project Status

| Area | Status | Notes |
| --- | --- | --- |
| Tensor | Available | N-dimensional row-major tensor, matrix-compatible accessors, resize/reshape, append, views, and validation. |
| Operators | Available | `matmul`, `softmax`, scaled dot-product attention, GQA attention, RoPE, and causal/additive masking. |
| KV cache | Available | Multi-layer key/value storage, append lifecycle, zero-copy row views, capacity limits, and allocator/manager helpers. |
| Engine | Available for KV orchestration | Prefill and single-step decode append KV tensors into managed caches. It does not yet run a full model-generated token loop. |
| Model | Minimal forward skeleton | Qwen-style `RMSNorm`, `SelfAttention`, `MLP`, `TransformerBlock`, `QwenModel`, and `ModelWeights` are compiled and tested. |
| Benchmark | Available | `qwen_decode_bench` covers the core GQA + RoPE + KV append path. |
| Tokenizer/runtime/backend/CLI/service | Scaffolding | Interfaces exist, but the full prompt-to-text inference loop is not implemented. |
| Training/data/export | Scaffolding | Python entry points and configs exist, but still raise TODO/placeholder behavior. |

## Repository Layout

```text
include/                 Public C++ headers
src/                     C++ library implementations
tests/                   CTest-based unit and integration tests
benchmarks/              Local benchmark executables
cmake/                   Project options and compiler warning settings
configs/                 Draft inference, training, and evaluation configs
docs/                    Architecture, roadmap, data, training, and deployment docs
models/                  Local model artifact conventions and manifest examples
python/                  Future data, training, and export utilities
third_party/simdjson/    Vendored simdjson dependency
```

Current CMake targets cover the core libraries and tests:

| Target | Purpose |
| --- | --- |
| `ops_core` | `Tensor`, `matmul`, `softmax`, and attention operators. |
| `cache_core` | `KVCache`, `CacheAllocator`, and `CacheManager`. |
| `engine_core` | Prefill and decode KV orchestration. |
| `model_core` | Minimal Qwen-style model components and model tests. |
| `qwen_decode_bench` | Decode-path benchmark executable. |

## Requirements

- CMake 3.20 or newer
- C++20-capable compiler
- A build tool supported by CMake, such as Ninja, Visual Studio, MinGW, or Make
- PowerShell examples below assume Windows. Equivalent shell commands work on Linux/macOS.

The project uses the bundled `third_party/simdjson` by default. Set `QWEN_CPP_USE_BUNDLED_SIMDJSON=OFF` if you want to provide `simdjson` through a system CMake package.

## Quick Start

Configure and build:

```powershell
cmake -S . -B build
cmake --build build
```

Run the complete CTest suite:

```powershell
ctest --test-dir build --output-on-failure
```

Run a subset by label:

```powershell
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -L ops --output-on-failure
ctest --test-dir build -L cache --output-on-failure
ctest --test-dir build -L model --output-on-failure
```

Run the decode-path benchmark:

```powershell
.\build\benchmarks\qwen_decode_bench.exe
.\build\benchmarks\qwen_decode_bench.exe --prompt 128 --decode 64 --layers 4 --q-heads 8 --kv-heads 2 --head-dim 16
```

Disable optional benchmark builds:

```powershell
cmake -S . -B build -DQWEN_CPP_BUILD_BENCHMARKS=OFF
```

## Architecture

The repository keeps a strict separation between mathematical kernels, stateful cache ownership, request-level orchestration, and model semantics.

```text
engine
  -> model
  -> cache
  -> ops

model
  -> ops
  -> cache only where the teaching implementation needs explicit cache structures

cache
  -> tensor

ops
  -> tensor
```

Core rules:

- `ops` is stateless and deterministic. It validates shapes and computes results.
- `cache` owns KV state and lifecycle. It does not decide inference flow.
- `engine` coordinates prefill/decode phases. It does not implement low-level math.
- `model` describes Qwen-style layers and forward semantics. It does not own global request/session lifecycle.
- `tokenizer`, `runtime`, `backend`, `cli`, and `service` are future-facing integration layers and must not be treated as stable product APIs yet.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed module contracts and dependency rules.

## Current Learning Path

Recommended study order:

1. Start with `include/tensor.h` and `src/tensor.cpp` to understand storage, shape, strides, and bounds behavior.
2. Read `include/ops/*` and `src/ops/*` to see how matrix multiplication, softmax, and attention are expressed on top of `Tensor`.
3. Read `include/cache/*` and `src/cache/*` to understand KV cache growth, views, capacity, and lifecycle.
4. Read `include/engine/prefill.h`, `include/engine/decode.h`, and matching `src/engine/*` files to understand phase-level KV orchestration.
5. Read `include/model/*` and `src/model/*` to understand how Qwen-style blocks are being assembled from the lower layers.
6. Use `tests/*_test.cpp` as executable documentation. They are the most accurate contract for current behavior.

## Documentation

- [中文文档](docs/zh/README.md): Chinese project overview, developer guide, module guide, testing guide, and glossary.
- [Architecture](docs/ARCHITECTURE.md): layer boundaries, dependency direction, and interface contracts.
- [Architecture Index](docs/architecture/README.md): quick navigation for the architecture docs.
- [Roadmap](docs/roadmap.md): staged plan toward GGUF inference and local training/fine-tuning.
- [Datasets](docs/datasets/README.md): future data governance and dataset processing rules.
- [Training](docs/training/README.md): planned LoRA/QLoRA training workflow.
- [Deployment](docs/deployment/README.md): planned local CLI/API deployment model.
- [Model Manifests](models/manifests/README.md): model artifact metadata format.
- [GGUF Models](models/gguf/README.md), [HF Models](models/hf/README.md), and [Adapters](models/adapters/README.md): local artifact directory conventions.
- [Unit Tests](tests/unit/README.md), [Integration Tests](tests/integration/README.md), and [E2E Tests](tests/e2e/README.md): test taxonomy.

## Development Standards

- Keep changes scoped to the active learning layer.
- Add or update tests with any behavioral change.
- Prefer explicit shape validation and clear exceptions over silent truncation or implicit broadcasting.
- Do not make `ops` depend on cache, engine, runtime, or model.
- Do not make `cache` perform attention math or request-flow decisions.
- Treat placeholder modules as scaffolding unless they are added to CMake and covered by tests.
- Keep third-party code isolated under `third_party/`.

## Roadmap Summary

The next useful engineering milestones are:

1. Finish the stage-0 baseline cleanup: clarify executable entry points, remove or quarantine unrelated tests, and add CI.
2. Turn tokenizer/runtime/backend scaffolding into a minimal compilable interface layer.
3. Integrate a GGUF-capable backend, likely through `llama.cpp`, and expose the first prompt-to-text CLI path.
4. Connect manifest/config loading to runtime and backend selection.
5. Add smoke inference, tokenizer, backend, and e2e tests.
6. Build Python data, LoRA/QLoRA training, artifact export, and evaluation workflows after inference is usable.

The full plan is maintained in [docs/roadmap.md](docs/roadmap.md).

## License

Project licensing is not fully documented yet. Before publishing binary builds, trained adapters, datasets, or model artifacts, document licenses for the project code, vendored dependencies, datasets, base models, and generated artifacts.
