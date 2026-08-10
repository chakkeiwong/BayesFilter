# BayesFilter Clean Reset Memo

Date: 2026-07-23  
Branch: `main`  
Snapshot base: `9303ed7` (`Correct PP-UKF coverage promotion semantics`)  

## Snapshot Boundary

This reset captures the current trusted-local lane before the next research
restart. The snapshot includes the pending source and test changes for the
NeuTra/HMC sequential controller, batch-native target/value-score execution,
HMC convergence and uncertainty handling, q=20 CPU diagnostics, and the
associated benchmark harnesses and Tier-2 plans/results. The modified LGSSM
fixture is retained because its hash and floating-point payload are a tracked
evidence artifact, not disposable run state.

## Artifact Policy

- Source, tests, benchmark harnesses, TeX, plans, result notes, and compact
  promotion/claim receipts are tracked.
- Checkpoints, progress snapshots, private payloads, per-arm tuning traces,
  TensorFlow tensors, logs, caches, build products, and transient campaign
  locks are ignored.
- Compact receipts explicitly re-included by `.gitignore` are the only generated
  artifacts in this snapshot intended to support a decision or claim.
- No result in this memo promotes posterior correctness, HMC convergence,
  transport quality, architecture superiority, GPU readiness, or a new default;
  those claims remain governed by their respective plan gates.

## Validation Contract

- Run `python -m py_compile` over staged Python files.
- Run `git diff --cached --check`; intentional Markdown hard-break spaces are
  documented if reported.
- Confirm no unresolved merge entries, no non-ignored untracked files, and no
  staged checkpoint/private/lock output.
- Merge the current remote `main` after this snapshot commit, resolve only
  actual conflicts, and push the resulting merge tip.

## Restart State

The next agent should begin from the pushed commit, read the q=20 result notes
and their compact receipts, and choose the smallest discriminating repair. The
CPU batch-native performance stop is an engineering blocker, not evidence
against the NeuTra direction. The q=20 HMC tuning result requires a fresh
scope-specific tuning repair before any retained HMC run. GPU runs must set and
verify `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow initialization.
