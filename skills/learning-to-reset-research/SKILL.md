---
name: learning-to-reset-research
description: Use when working in the Learning to Reset repository to keep code, tests, and docs aligned with the project references while prioritizing future extensions.
---

# Learning to Reset Research Skill

## When to use

- Any coding, planning, evaluation, or documentation work in this repository
- Any task involving `<clean>` behavior, context management, SFT trace curation, or modified RLOO
- Any extension planning for multi-step cleaning, selective retention, or memory-aware recall

## Workflow

1. Use `main.pdf` as the primary project reference.
2. Use `rl_proposal.pdf` to prioritize the extension roadmap.
3. Before coding, identify which reference section the change belongs to:
   - Section 3.2.1 for trace curation
   - Section 3.2.2 for the context manager
   - Section 3.2.3 for modified RLOO
   - Section 4 for evaluation hooks
   - Section 5 and 6 for extension motivation
4. Prefer tests first for any behavioral change.
5. Keep baseline code separate from extension code when possible.

## Read These Files First

- `docs/method-overview.md`
- `docs/project-plan.md`
- `docs/extension-roadmap.md`

## Working Rules

- If a behavior is ambiguous, follow `main.pdf`.
- If a change goes beyond single-use cleaning, label it explicitly as an extension.
- Do not add generic RL abstractions unless they serve a defined mechanism or a named extension.
