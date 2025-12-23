# Architecture

This project is a long-horizon LLM harness implemented as a full-stack resume builder. The UI and data pipeline are a realistic workload for long-running agents, while the test suite stresses the system under both happy-path and failure scenarios.

## System overview

```
[User/Agent]
    |
    v
[Reflex UI + State Machine] ---> [Playwright UI automation]
    |
    v
[Harness Orchestrator (harness.py)]
    |        |        |
    |        |        +--> [Typst PDF pipeline + fonts/assets]
    |        +--> [Any-LLM provider abstraction]
    +--> [Neo4j persistence]
```

## Core components

- **Reflex UI and state**: The interface and state machine live in `harness.py`. This is the surface area for long-horizon behavior and UI tests.
- **Orchestrator**: `harness.py` also provides the CLI entry points, test modes, and coverage instrumentation.
- **Data layer**: Neo4j persists candidate data, job requisitions, and history. Defaults are defined in `docker-compose.yml`.
- **LLM integration**: Any-LLM abstractions route calls to OpenAI or Gemini providers. Keys are loaded from env or local key files.
- **PDF generation**: Typst renders the resume pipeline with bundled assets and fonts.
- **Coverage and UI automation**: The `--maximum-coverage` path and Playwright checks validate the system end-to-end.

## Long-horizon design intent

The system forces agents to deal with:

- Durable state transitions and database consistency
- Data extraction and parsing from `req.txt` and `prompt.yaml`
- UI and PDF consistency under long-run workloads
- Failure injection to validate recovery paths

See `docs/TESTING.md` and `docs/LONG_RUN.md` for the operational protocol.
