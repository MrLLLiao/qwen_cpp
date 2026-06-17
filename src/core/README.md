# Core Layer

The current core learning layer is implemented across `tensor`, `ops`, `cache`, `engine`, and `model` directories rather than a single `src/core` target.

This file exists to document the intended boundary.

## Stable Core Targets

| CMake target | Implementation files | Responsibility |
| --- | --- | --- |
| `ops_core` | `src/tensor.cpp`, `src/ops/*` | Tensor storage and stateless compute kernels. |
| `cache_core` | `src/cache/*` | KV cache, cache allocator, and cache manager. |
| `engine_core` | `src/engine/*` | Prefill/decode KV orchestration. |
| `model_core` | `src/model/*` | Minimal Qwen-style model components. |

## Boundary

Core code should be usable without HTTP service, CLI, Python training scripts, or a real GGUF backend. New runtime features should depend on these core targets through documented headers rather than pushing runtime concepts back into low-level math or cache code.

## Future Cleanup

If the project later introduces a physical `core` source directory, move code gradually and preserve CMake target compatibility during the transition.
