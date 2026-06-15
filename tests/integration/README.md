# Integration Tests

Integration tests validate collaboration across stable modules.

Current labels:
- `integration;engine;cache`: `prefill-test`, `decode-test`

Run:

```powershell
ctest --test-dir build -L integration --output-on-failure
```

Planned next coverage:
- model block forward plus KV write orchestration
- manifest parsing plus backend selection once a real loader exists
