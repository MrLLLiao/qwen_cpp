# Unit Tests

Unit tests validate one module at a time without requiring model files, network access, external services, or Python training dependencies.

## Current Coverage

| Label | Executables | Scope |
| --- | --- | --- |
| `unit;tensor` | `tensor-test` | Shape, stride, access, append, reshape, transpose, bounds, and empty tensor contract. |
| `unit;ops` | `matmul-test`, `softmax-test`, `attention-test` | Stateless math kernels, masking, GQA, RoPE, and validation behavior. |
| `unit;cache` | `kvcache-test`, `cache-allocator-test`, `cache-manager-test` | KV append lifecycle, views, capacity, allocator reuse, and manager boundaries. |
| `unit;model` | `embedding-test`, `model-test` | Embedding vocabulary behavior and minimal Qwen-style model components. |

## Commands

```powershell
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L tensor --output-on-failure
ctest --test-dir build -L ops --output-on-failure
ctest --test-dir build -L cache --output-on-failure
ctest --test-dir build -L model --output-on-failure
```

## Expectations

New unit tests should be deterministic, fast, and explicit about invalid inputs. Shape and lifecycle errors should be asserted through the exception type where the contract defines one.
