# Contributing

qwen_cpp is a learning-oriented C++ project. Contributions should make the implementation easier to understand, test, and evolve toward a local GGUF inference path.

## Development Principles

- Keep module boundaries clear. `ops` must stay stateless; `cache` owns KV state; `engine` orchestrates phases; `model` expresses layer semantics.
- Add tests for behavior changes. Current tests are the most reliable documentation of supported behavior.
- Prefer explicit validation and clear exceptions for invalid shapes, invalid configs, or lifecycle errors.
- Avoid widening scope in the same change. Refactors, feature work, and documentation updates should be easy to review independently.
- Do not modify vendored code under `third_party/` unless the change is explicitly about updating that dependency.

## Local Workflow

Configure, build, and test before submitting changes:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run focused labels while developing:

```powershell
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L integration --output-on-failure
ctest --test-dir build -L model --output-on-failure
```

## Code Style

- Use C++20 features only where they simplify code or improve safety.
- Keep public headers small and explicit.
- Prefer value objects and configuration structs for operator/model settings.
- Keep comments useful. Explain contracts, invariants, and non-obvious behavior rather than restating code.
- Preserve existing naming conventions in the file you edit. The repository currently has both global core types and newer `mini_llm::*` scaffolding; avoid mixing styles unnecessarily.

## Documentation Style

- Document what works today separately from planned functionality.
- Link to tests when a behavior is executable.
- Keep TODOs concrete and tied to modules or milestones.
- Update `README.md`, `docs/ARCHITECTURE.md`, or `docs/roadmap.md` when a change alters project status or module ownership.

## Pull Request Checklist

- Build succeeds.
- Relevant CTest labels pass.
- New behavior has tests.
- Public API changes are documented.
- Runtime/backend/training claims distinguish implemented behavior from scaffolding.
- Model, dataset, and third-party license implications are recorded when artifacts are added.
