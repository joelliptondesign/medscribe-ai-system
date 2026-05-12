# Environment Verification 2026-05-11

## Purpose

Record environment-aware dependency verification for MedScribe using `python3` as the canonical Python executable, without installing dependencies or changing runtime behavior.

## Python Executable Finding

- `which python3` resolved to `/usr/bin/python3`.
- `python3 --version` reported `Python 3.9.6`.
- `.venv` exists.
- `.venv/bin/python --version` reported `Python 3.9.6`.

## System Python3 Dependency Availability

Checked with `python3 -B`:

- `dotenv`: unavailable
- `langgraph`: unavailable
- `langsmith`: unavailable
- `langchain_core`: unavailable
- `openai`: unavailable
- `pytest`: unavailable

## Virtualenv Dependency Availability

Checked with `.venv/bin/python -B`:

- `dotenv`: available
- `langgraph`: available
- `langsmith`: available
- `langchain_core`: available
- `openai`: available
- `pytest`: unavailable

## Missing Dependency Observations

- Previous system `python3` import smoke checks failed for existing runtime modules because `langgraph` and `dotenv` were unavailable under system `python3`.
- `pytest` is unavailable under both system `python3` and `.venv/bin/python`.

## Requirements Comparison

- `dotenv` corresponds to `python-dotenv`, which is declared in `requirements.txt`.
- `langgraph` is declared in `requirements.txt`.
- `pytest` is now declared in `requirements.txt`.
- The existing `.venv` was updated with `pytest` from the repo dependency declaration.

## Verification Commands Run

```bash
which python3 || true
python3 --version
test -d .venv && echo ".venv exists" || echo ".venv missing"
test -x .venv/bin/python && .venv/bin/python --version || echo ".venv python unavailable"
python3 -B -c "import importlib.util; mods=['dotenv','langgraph','langsmith','langchain_core','openai','pytest']; print({m: bool(importlib.util.find_spec(m)) for m in mods})"
.venv/bin/python -B -c "import importlib.util; mods=['dotenv','langgraph','langsmith','langchain_core','openai','pytest']; print({m: bool(importlib.util.find_spec(m)) for m in mods})"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/test_denial_graph.py tests/test_execution_mode.py tests/test_hybrid_mode_opt_in.py tests/test_run_lifecycle.py
.venv/bin/python evaluation/langsmith_experiment_runner.py --workflow denial --experiment-label baseline --limit 2 --local-only
.venv/bin/python evaluation/langsmith_experiment_runner.py --workflow denial --pairwise --limit 2 --local-only
.venv/bin/python evaluation/run_synthetic_incidents.py --workflow denial --output /private/tmp/medscribe_denial_regression_incidents.json
.venv/bin/python evaluation/langsmith_experiment_runner.py --experiment-label baseline --limit 2 --local-only
.venv/bin/python evaluation/run_synthetic_incidents.py --output /private/tmp/medscribe_cdi_regression_incidents.json
```

## Limitations

- System `python3` can compile pure Python files but cannot import project runtime dependencies that are not installed in the system interpreter.
- `.venv/bin/python` is the canonical executable for repo runtime and regression verification.
- `.venv/bin/python -m pytest` is available after installing declared requirements.
- No hosted LangSmith experiment was run during the grouped regression pass.
- Local runs emitted an urllib3/OpenSSL compatibility warning from the virtualenv.
- No secrets or environment values were printed.

## Next Operational Implication

Use `.venv/bin/python` for runtime import checks, grouped pytest regression, and local CDI/denial smoke verification. Use system `python3` only for checks that do not require repo runtime dependencies, unless the system interpreter is prepared with the declared requirements.
