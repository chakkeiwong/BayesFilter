# Phase 12 Subplan: CPU Multicore External Sample Boundary

Date: 2026-07-07

## Phase Objective

Define and smoke-check the BayesFilter boundary that keeps external sample
generation separate from GPU NeuTra training.  The phase should establish how
post-training sample generation can be requested as a multicore CPU job with
explicit provenance, without running HMC sampling/tuning and without treating
sample generation as transport training.

This phase is a boundary and mechanics phase. It must not train NeuTra, run GPU
training, run HMC sampling or tuning, repair XLA, use DSGE/c603, rank routes,
change defaults, or make posterior/product/scientific claims.

## Entry Conditions Inherited From Previous Phase

- Phase 11 has either passed frozen GPU-trained affine payload packaging or
  written a blocker that still leaves the CPU sample-generation boundary
  meaningful.
- The Phase 10 GPU training artifact remains the only source of learned affine
  parameters for this LGSSM route.
- NeuTra training policy remains GPU-only; CPU NeuTra training fallback is
  forbidden.
- External sample generation is intentionally CPU multicore and separate from
  GPU training.
- Phase 9 XLA/JIT blocker remains open unless Phase 13 later repairs it.
- No HMC sampling or tuning is authorized by this subplan.

## Required Artifacts

- A CPU sample-generation boundary helper or documented adapter decision,
  scoped to `bayesfilter/testing/` or `bayesfilter/inference/` according to the
  existing code ownership pattern.
  If no new helper is added, the focused tests must bind to an existing
  boundary surface and identify that surface in the Phase 12 result.
- Focused tests proving:
  - CPU-only execution is explicit through `CUDA_VISIBLE_DEVICES=-1`;
  - worker-count and seed/provenance fields are recorded;
  - the boundary refuses to run when configured as GPU training, NeuTra
    training, HMC sampling, or HMC tuning;
  - generated outputs are marked diagnostic/sample-boundary artifacts only.
- Optional tiny diagnostic JSON under `docs/plans/artifacts/` if a smoke is
  run and the file is below repository policy.
- Phase 12 result or blocker:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase12-cpu-multicore-sample-generation-result-2026-07-07.md`
- Refreshed Phase 13 subplan before advancing.

## Required Checks/Tests/Reviews

- CPU-only checks must set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import.
- Run focused pytest for any Phase 12 helper/tests.
- Run `python -m py_compile` on any new Phase 12 helper/test modules.
- Run `python -m json.tool` on any Phase 12 JSON artifacts.
- Run `git diff --check` on Phase 12 code, result, and subplan artifacts.
- Bounded read-only review must inspect the Phase 12 result/blocker and Phase
  13 subplan before entering XLA repair, HMC sampling/tuning, or additional
  training.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter express post-training external sample generation as a CPU multicore boundary that is separate from GPU NeuTra training and from HMC sampling/tuning? |
| Baseline/comparator | Phase 10 GPU-only training policy, Phase 11 frozen payload boundary, and existing fixed-transport mechanics nonclaims. |
| Primary criterion | A reviewed helper/design and focused tests record CPU-only multicore provenance and forbid GPU training, CPU NeuTra training, HMC sampling/tuning, and XLA dependence. |
| Veto diagnostics | Hidden NeuTra training, CPU training fallback, hidden GPU training, hidden HMC sampling/tuning, unrecorded worker/seed/provenance, nonfinite diagnostic outputs, XLA/JIT requirement, or unsupported readiness/scientific claim. |
| Explanatory diagnostics | Worker count, seed policy, sample shape, finite diagnostic checks, artifact hashes, and route signatures. |
| Not concluded | Posterior correctness, HMC convergence, sampler quality, transport quality, route superiority, production readiness, default-policy change, XLA readiness, or scientific validity. |
| Artifact | Phase 12 helper/design, tests, optional tiny diagnostic JSON, result/blocker, and refreshed Phase 13 subplan. |

## Forbidden Claims/Actions

- Do not run or authorize NeuTra training.
- Do not run CPU NeuTra training.
- Do not run HMC sampling or tuning.
- Do not use external sample generation as evidence of posterior correctness.
- Do not use GPU for sample generation in this phase.
- Do not repair or claim XLA/JIT readiness.
- Do not use DSGE/c603.
- Do not change BayesFilter default policy.
- Do not rank routes or samplers.

## Exact Next-Phase Handoff Conditions

Phase 13 may begin only if:

- Phase 12 records pass/blocker with a result artifact;
- any helper/tests preserve explicit CPU-only sample-generation provenance;
- no NeuTra training, GPU training, HMC sampling/tuning, or XLA repair occurred;
- any generated JSON artifacts parse and comply with repository size policy;
- the Phase 13 subplan restates the inherited XLA/JIT blocker and exact repair
  gate;
- bounded read-only review agrees that Phase 13 remains an XLA/JIT repair gate
  rather than an HMC-readiness or production-readiness claim.

## Stop Conditions

Stop if:

- implementing the boundary requires HMC sampling/tuning;
- implementing the boundary requires CPU NeuTra training;
- CPU-only execution cannot be enforced or recorded;
- worker/seed/provenance cannot be preserved;
- any diagnostic output is nonfinite or malformed;
- artifact sizes violate repository policy;
- review does not converge after five rounds.

## Phase Close Duties

At close:

1. run required local checks;
2. write Phase 12 result or blocker;
3. refresh the Phase 13 subplan;
4. review the Phase 13 subplan for consistency, correctness, feasibility,
   artifact coverage, GPU/CPU boundary safety, and claim discipline.
