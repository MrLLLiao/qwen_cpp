# Integration Tests

Integration tests validate collaboration across stable modules. They may cross module boundaries but should not require real model files, a running service, or network access.

## Current Coverage

| Label | Executables | Scope |
| --- | --- | --- |
| `integration;engine;cache` | `prefill-test`, `decode-test` | `PrefillEngine` and `DecodeEngine` writing multi-layer KV tensors through `CacheManager` into `KVCache`. |

## Commands

```powershell
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -L engine --output-on-failure
```

## Planned Coverage

- model block forward plus KV cache interaction when a stable interface exists;
- manifest parsing plus backend selection after a real loader is implemented;
- tokenizer plus runtime input normalization once tokenizer behavior is available.

Integration tests should document real collaboration contracts. Avoid using them as broad smoke tests for placeholder modules.
