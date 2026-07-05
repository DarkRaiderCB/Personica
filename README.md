# Personica

[![CI](https://github.com/DarkRaiderCB/Personica/actions/workflows/ci.yml/badge.svg)](https://github.com/DarkRaiderCB/Personica/actions/workflows/ci.yml)

A CLI personal assistant that remembers you across sessions — with principled retrieval, write-time memory consolidation, and an eval harness that proves the memory actually works.

## How memory works

**Per turn** (`pipeline.py`):

1. **Query rewriting** — a utility model expands the user message plus recent context into a keyword query with synonyms, detecting topic changes.
2. **Multi-query retrieval** — the store is searched with *both* the rewritten query and the raw message, and the results are union-merged. A single keyword query biases toward one sub-topic of a multi-part question ("what's my name *and where do I work*?"); the raw phrasing balances differently.
3. **Hybrid ranking** — candidates are ranked by `w · similarity + (1−w) · 0.5^(age/half-life)`: cosine similarity blended with exponential recency decay (the retrieval formula from Park et al., *Generative Agents*, 2023). Fresh memories win among comparably relevant ones; a much more relevant old memory still outranks a barely relevant new one.
4. **Relevance gatekeeping** — a second LLM check confirms the retrieved memories actually help before they are injected; the newest injected memory is tagged `[LATEST]` for conflict resolution.
5. **Sliding-window short-term memory** — the last N user/assistant pairs stay verbatim; older turns fold into a rolling LLM-generated summary so the prompt stays bounded.

**At session end** (`consolidation.py`):

6. **Atomic fact extraction** — an LLM extracts the user-stated facts as a JSON list of self-contained statements (never facts the assistant merely recalled).
7. **Write-time consolidation** — each fact is checked against its most similar existing memories and an LLM decides: **add** (new), **skip** (duplicate), or **replace** (supersedes stale memories, which are deleted and merged). Append-only memory degrades; this is the consolidation approach used by systems like MemGPT and mem0. If the decision step fails, the fact is added anyway — losing a memory is worse than a temporary duplicate.

## Evals

`evals/run_evals.py` runs scripted multi-session scenarios end-to-end against the real pipeline (real LLM, real vector store in a temp dir) and scores them:

| Scenario | Verifies |
| --- | --- |
| cross-session recall | facts stated in session 1 are answered in session 2 |
| fact update (supersession) | an updated fact overrides the stale one |
| multi-fact recall | several facts from one session are all retrievable |
| no fabrication | with empty memory, the assistant admits it doesn't know |

```bash
uv run python evals/run_evals.py   # requires OPENROUTER_API_KEY, costs a few cents
```

Current result: **4/4**. The harness earned its keep during development — it caught the extraction model fabricating `[UPDATED] (was: ...)` values, over-splitting facts ("works as a data scientist **in Berlin**" losing its location), and single-query retrieval missing half of a multi-part question. Each fix is regression-guarded by the scenarios above.

## Project layout

```
src/personica/
├── cli.py             REPL, session lifecycle, /remember & /forget, graceful shutdown
├── pipeline.py        per-turn flow: rewrite → multi-query retrieve → gate → respond
├── consolidation.py   write-time add/skip/replace integration of new facts
├── config.py          all settings (frozen dataclass, loaded from env/.env)
├── llm.py             LiteLLM/OpenRouter client: retries, timing, token+cost tracking
├── prompts.py         every prompt template in one place
├── memory/
│   ├── short_term.py  sliding window + rolling summary
│   └── long_term.py   ChromaDB wrapper with hybrid similarity+recency ranking
├── tracing.py         JSONL trace events + transcript records
├── logging_setup.py   console + rotating file logs, session-id stamped
├── jsonutil.py        tolerant JSON extraction from LLM output
└── inspect_memory.py  dump all stored memories
tests/                 79 offline unit tests (fakes/mocks — no API calls)
evals/                 live end-to-end memory evals
```

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your OpenRouter API key
```

## Usage

```bash
uv run personica            # start a chat session
uv run personica --reset    # wipe long-term memory first
uv run personica-inspect    # inspect what has been remembered
```

In-chat commands:

- `/mem` — show the memories retrieved on the last turn and the current short-term summary
- `/remember <fact>` — store a fact explicitly
- `/forget <query>` — find matching memories and delete the ones you pick
- `/exit` (or Ctrl+C / Ctrl+D) — end the session; the transcript is saved, facts are extracted and consolidated, and LLM token/cost totals are printed

Try it: tell Personica a few facts about yourself, exit, start a new session, and ask what it knows.

## Traceability & records

Everything Personica does is recorded under `PERSONICA_DATA_DIR` (default `./personica_data`):

| Path | Contents |
| --- | --- |
| `traces/<session>.jsonl` | One JSON event per pipeline step: query rewrites, retrieval hits with similarity + hybrid rank scores, gatekeeper verdicts, every LLM call with model/latency/tokens/cost, consolidation decisions |
| `transcripts/<session>.json` | The full conversation, saved on any exit path |
| `logs/personica.log` | Rotating DEBUG log (5 MB × 3), every record stamped with the session id |
| `chroma/` | The persistent long-term memory vector store |

## Testing & CI

```bash
uv run pytest              # 79 offline tests, no API key needed
uv run ruff check src tests evals
```

The unit suite covers the turn pipeline with scripted LLM/store doubles (memory injection, `[LATEST]` tagging, gatekeeper blocking, multi-query union/dedup), consolidation decisions (add/skip/replace, malformed-output fallbacks), hybrid-score properties (half-life decay, relevance-vs-recency trade-off, clock-skew clamping), LLM retry/cost-tracking semantics against a monkeypatched litellm, real ChromaDB persistence with a deterministic fake embedding function, sliding-window guarantees, and config precedence. GitHub Actions runs lint + tests on every push (`.github/workflows/ci.yml`).

## Configuration

All settings are optional except the API key (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | **Required.** OpenRouter API key |
| `OPENROUTER_MODEL` | `openrouter/openai/gpt-4o` | Main chat + fact-extraction model |
| `OPENROUTER_UTILITY_MODEL` | `openrouter/openai/gpt-4o-mini` | Query rewriting, relevance checks, consolidation, rolling summaries |
| `PERSONICA_DATA_DIR` | `./personica_data` | Where memories, transcripts, traces, and logs live |
| `PERSONICA_TIMEZONE` | `Asia/Kolkata` | Timezone for the time context |
| `PERSONICA_KEEP_LAST_TURNS` | `5` | Sliding-window size (user/assistant pairs) |
| `PERSONICA_RETRIEVAL_TOP_K` | `5` | Max memories retrieved per turn |
| `PERSONICA_RETRIEVAL_MIN_SCORE` | `0.20` | Minimum cosine similarity to consider a memory |
| `PERSONICA_RELEVANCE_WEIGHT` | `0.7` | Hybrid rank: weight on similarity vs recency |
| `PERSONICA_RECENCY_HALF_LIFE_DAYS` | `30` | Hybrid rank: recency decay half-life |
| `PERSONICA_LOG_LEVEL` | `INFO` | Console verbosity (`WARNING` hides the pipeline trace) |

Legacy `ASSISTANT_*` variable names are still accepted as fallbacks.

## License

Copyright © Sanyog Mishra. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may use, modify, and share this software for **noncommercial purposes** (personal projects, research, education) provided you keep the copyright notice. **Any commercial use requires the author's written permission.**

> Required Notice: Copyright Sanyog Mishra (https://github.com/DarkRaiderCB)
