# ReiL — Claude Code Repo Instructions

This is the **RL framework** repo. It's domain-agnostic; warfarin-specific logic lives in `reil/healthcare/` and is consumed by the sibling repo `warfarin_dosing/`.

## Environment

- Poetry env: `reil-l-k_YBAA-py3.13`
- Env path: `C:\Users\sj_an\AppData\Local\pypoetry\Cache\virtualenvs\reil-l-k_YBAA-py3.13`
- Run all commands from this repo root via `poetry run …`.
- **Never assume** the sibling `warfarin_dosing` Poetry env is active — it isn't.

## Read before editing

In order:

1. [`copilot-instructions.md`](copilot-instructions.md) — high-level RL abstractions and the agent/subject/learner decision guide.
2. [`reil/.instructions.md`](reil/.instructions.md) — core architecture (ReilBase → Stateful → Agent / Subject / Environment), Feature/FeatureSet, learner contract.
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design rationale.
4. [`.skills/YAML_CONFIG_MASTERY.md`](.skills/YAML_CONFIG_MASTERY.md) and [`.skills/EXPERIMENT_ORCHESTRATION.md`](.skills/EXPERIMENT_ORCHESTRATION.md) — when configs or experiment runners are involved.

## Project context

The work in `ReiL` ultimately serves two papers being prepared in:

- `C:\Users\sj_an\Documents\Claude\Projects\Dissertation papers\` — canonical reference docs.

When a change to `ReiL` is motivated by a paper / dissertation requirement, link the change to the relevant doc (e.g. "supports `[EXP-C3-002]` per `60_paper2_chapter3_canonical.md` §Tandem Conditional").

## Rules specific to this repo

- **Backwards compatibility matters.** Older serialized agents (`.pkl`) need to load. If you rename or move a class, update `reil/serialization.py` with the mapping rather than breaking deserialization.
- **No new abstractions without a justified need.** Three similar lines is better than a premature abstraction (per the project's standing rules).
- **Tests live in `tests/`.** Run them under the Poetry env before declaring a change done.
- **Don't edit `warfarin_dosing/` from this repo.** If a change requires coordination, propose it and let the user run it in the sibling repo.

## Pointer back

For everything that crosses the repo boundary, see [`C:\Users\sj_an\Documents\Claude\Projects\Dissertation papers\CLAUDE.md`](../../Claude/Projects/Dissertation papers/CLAUDE.md).
