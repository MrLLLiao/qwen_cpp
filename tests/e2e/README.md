# E2E Tests

No end-to-end executable is wired yet.

E2E coverage should start after a real runtime entry point exists:
- CLI smoke: prompt input -> generated text
- local service smoke: `/health`, `/models`, `/generate`
- artifact loop: exported model manifest -> load -> generate

Target command once available:

```powershell
ctest --test-dir build -L e2e --output-on-failure
```
