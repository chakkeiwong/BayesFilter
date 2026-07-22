# NeuTra HMC Core Consolidation Phase C1 Subplan

Date: 2026-07-15  
Status: `EXECUTED_AND_CLOSED`

## Objective And Entry Conditions

Implement the shared TensorFlow/TFP sequential NeuTra HMC controller. C0 froze
the canonical policy, route ledger, migration boundary, and enforcement guard.

## Required API And Artifacts

Create `bayesfilter/inference/neutra_hmc.py` with generic batched fixed-kernel
HMC, sequential warm-up and retained control, model-coordinate transformation,
diagnostic callback, archive callback, health vetoes, separated private tensors,
and public-safe summaries. Export it lazily through `bayesfilter.inference`.

The module must contain no model names, repository paths, JSON schemas tied to
LGSSM, NumPy, or host callbacks. Chain count and dimension come from the initial
state. The ledger must exclude the core implementation explicitly in the same
change that creates it.

## Checks

- configuration/default/cap validation;
- generic chain and dimension behavior;
- warm-up retention and posterior exclusion;
- recent-window warm-up modern R-hat;
- cumulative retained R-hat and full-diagnostic extension;
- health veto before posterior sampling;
- deterministic disjoint chunk seeds;
- archive callback and no-archive operation;
- no NumPy/host callbacks/model-specific tokens;
- Gaussian CPU/XLA integration smoke;
- route ledger remains complete.

## Evidence, Nonclaims, And Handoff

C1 establishes engineering semantics only, not posterior correctness or
robustness. Stop for wrong HMC state handoff, diagnostic mismatch, warm-up
leakage, nonfinite handling, XLA failure, or policy-guard drift. If checks pass,
write C1 result and refresh C2 migration subplan.

Suitability verdict: `PASS`; the API boundary is model-agnostic and directly
implements the reviewed C0 contract.
