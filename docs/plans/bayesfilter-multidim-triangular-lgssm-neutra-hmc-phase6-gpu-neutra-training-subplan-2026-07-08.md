# Phase 6 Subplan: GPU NeuTra Training

Date: 2026-07-08

## Phase Objective

Train NeuTra for the multidimensional triangular LGSSM target on GPU only with
`jit_compile=True`.

## Entry Conditions Inherited From Previous Phase

- Phase 5 reference/comparator standard passed review.
- Explicit human approval for GPU training has been granted.

## Required Artifacts

- Training config, logs, JSON training state, and Phase 6 result.

## Required Checks/Tests/Reviews

- Trusted GPU probe before training.
- `jit_compile=True` only.
- Training manifest with seed, target signature, GPU status, wall time.
- No CPU training fallback.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter train a frozen NeuTra transport candidate for this target under GPU/JIT policy? |
| Baseline/comparator | Phase 4 target and Phase 5 comparator standard. |
| Primary criterion | Finite training artifact with hashes, no policy violations, and adequate training diagnostics for Phase 7 packaging. |
| Veto diagnostics | CPU training, `jit_compile=false`, nonfinite loss/weights, target mismatch, GPU unavailable. |
| Explanatory diagnostics | Loss curves, gradient norms if available, wall time. |
| Not concluded | HMC readiness or posterior correctness. |
| Artifact | Training state/log/result. |

## Forbidden Claims/Actions

- Do not sample HMC in this phase.
- Do not claim convergence from training loss alone.

## Exact Next-Phase Handoff Conditions

Phase 7 may begin only if the training artifact is finite, hashed, and bound to
the target signature.

## Stop Conditions

Stop for GPU/JIT/training failure or missing approval.
