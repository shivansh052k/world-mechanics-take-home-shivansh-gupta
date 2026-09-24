# PROGRESS.md — where we are right now

Last updated: 2026-09-24.

This file is the live status tracker. Update it at the end of every phase (and any time work
pauses mid-phase), so this chat, Claude Code, and any future session can pick up instantly
without re-reading everything else.

---

## Current status

**Planning: complete** (including two review rounds). `CLAUDE.md`, `DECISIONS.md`, and
`EXECUTION_PLAN.md` are finalized and in place. `CLAUDE.md` (file) is at the repo root; `docs/` and `data/` (folders) exist; `src/`,
`scripts/`, `artifacts/`, `results/`, `slides/` are not yet created.

**Execution: not started.** Next step is **0.1 (Project setup)** in `docs/EXECUTION_PLAN.md`.

---

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Environment and model setup | ⬜ Not started | |
| 1 — Data audit | ⬜ Not started | |
| 2 — Splits and activation extraction | ⬜ Not started | |
| 3 — Part 1a: Layer-wise probing | ⬜ Not started | |
| 4 — Part 1b: Iterative nullspace probing | ⬜ Not started | |
| 5 — Part 1c: Multi-probe subspace steering | ⬜ Not started | |
| 6 — Part 2: Spline steering | ⬜ Not started | |
| 7 — Confounds and robustness | ⬜ Not started | |
| 8 — Presentation and final delivery | ⬜ Not started | |

Status legend: ⬜ Not started · 🟨 In progress · ✅ Passed gate · ⚠️ Blocked (see Open issues below)

---

## Key numbers so far

*(Populate as phases complete. Example rows shown for format — remove once real ones are added.)*

| Metric | Value | Source |
|---|---|---|
| — | — | — |

---

## Open issues / blockers

*(Anything stopping progress goes here, with enough context to resume without re-deriving it.)*

No blockers. Known decision points the plan cannot remove in advance (each has a planned fallback):
- **O-18** — which encoder weights HF ships (target vs context); check in step 0.15.
- **Phase 1 may overturn scouting facts** (F-21–F-36); D-14's split counts are provisional until step 1.12.
- **0.15 may fail** → step 5.2 falls back to a later-layer readout, and Part 2's behavior manifold (D-17) moves with it.
- **Phase 3 may show no clean transition** → O-15/O-16 become documented judgment calls.
- **Downstream steering effects may wash out** → a finding, not a failure; the same-layer control keeps it interpretable.
- **MPS operator gaps and the compute budget** are unknown until steps 0.6 and 0.16.

---

## Log

Newest entry on top. One entry per work session: what was done, what passed, what didn't,
what's next.

### 2026-09-24 — Plan review, round 2 (researched fixes)
- Verified the round-1 D-07 fix; found remaining leak and underspecification points plus blockers.
- Researched fixes against primary sources (physics paper App. B/C.11/C.12, Goodfire §5, V-JEPA 2
  training code, sklearn and PyTorch source), then a self-review corrected 8 flaws in the first draft.
- `DECISIONS.md`: added F-37–F-40, D-14–D-17, O-18; narrowed O-01/O-05/O-10/O-11; clarified P-06.
- `EXECUTION_PLAN.md`: reworded 28 steps; still **107 steps**, none added or removed.
- **Next:** Step 0.1 (Project setup).

### 2026-09-24 — Planning complete, then corrected
- Finalized `CLAUDE.md`, `DECISIONS.md`, `EXECUTION_PLAN.md` (**107 steps** across Phases 0–8;
  originally miscounted as 117 — corrected).
- Files placed in the repo: `CLAUDE.md` at root, `DECISIONS.md` / `EXECUTION_PLAN.md` / this file in `docs/`.
- Repo folders still needed: `src/`, `scripts/`, `artifacts/`, `results/`, `slides/`.
- A fresh review caught a real methodology bug (Phase 3/4 wording had layer-selection and
  nullspace stopping using test data, contradicting D-07) and some stale cross-references;
  both fixed. See `DECISIONS.md` change log for details.
- **Next:** Step 0.1 (Project setup).
