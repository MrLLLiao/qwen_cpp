# Unit Tests

Unit tests validate individual modules without requiring model files or external services.

Current labels:
- `unit;tensor`: `tensor-test`
- `unit;ops`: `matmul-test`, `softmax-test`, `attention-test`
- `unit;cache`: `kvcache-test`, `cache-allocator-test`, `cache-manager-test`
- `unit;model`: `embedding-test`, `model-test`

Run:

```powershell
ctest --test-dir build -L unit --output-on-failure
```
