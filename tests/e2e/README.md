# E2E Tests

No end-to-end executable is wired into CTest yet.

E2E tests should start only after the project has a real runtime entry point. A useful e2e test should exercise a user-observable workflow rather than only internal classes.

## Planned Coverage

| Workflow | Expected contract |
| --- | --- |
| CLI smoke | Prompt input produces generated text or a structured, actionable error. |
| Local service smoke | `/health`, `/models`, and `/generate` behave consistently. |
| Artifact loop | Manifest path resolves model/tokenizer files and loads through the selected backend. |
| Export loop | Training/export artifact can be loaded by the inference side after conversion. |

## Future Command

```powershell
ctest --test-dir build -L e2e --output-on-failure
```

Until these tests exist, `ctest --test-dir build --output-on-failure` validates the core learning baseline but not a complete local LLM application.
