# Phase 14 Subplan: Superseded XLA TensorList Boundary Repair

Date: 2026-07-08

## Supersession Notice

This subplan is superseded by:

- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase14a-no-gradienttape-policy-subplan-2026-07-08.md`
- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase14a-no-gradienttape-policy-result-2026-07-08.md`
- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-gpu-rerun-subplan-2026-07-08.md`

Reason: the user clarified that BayesFilter-owned target/HMC code should not
use `GradientTape` except for diagnostics. Therefore the correct repair is not
to make the taped diagnostic XLA-ready, but to remove taped score authority from
the admitted LGSSM target and regenerate NeuTra artifacts under the manual-score
target signatures.

## Phase Objective

Repair or preserve the remaining Phase 13 XLA blocker:

```text
Support for TensorList crossing the XLA/TF boundary is not implemented
```

The phase should investigate whether the LGSSM NeuTra value/gradient diagnostic
can be made TensorList-boundary-free under XLA without changing the target,
training policy, frozen payload, CPU sample-generation boundary, or HMC
boundary.

This phase is an XLA repair phase only. It must not run NeuTra training, HMC
sampling/tuning, external sample generation, DSGE/c603, route ranking,
default-policy changes, or scientific/product/readiness claims.

## Entry Conditions Inherited From Previous Phase

- Phase 13 added `maximum_iterations=n_timesteps` to the QR while-loop
  likelihood kernels, which moved the trusted GPU/XLA failure past the original
  fixed tensor-list-size complaint.
- Phase 13 still blocked under trusted GPU/XLA execution with:
  `Support for TensorList crossing the XLA/TF boundary is not implemented`.
- Phase 13 wrote a parseable blocker artifact:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-diagnostic-2026-07-07.json`.
- Until Phase 14 passes, all NeuTra training and fixed-transport mechanics must
  keep `jit_compile=false`.
- GPU/CUDA commands require trusted/escalated execution per local policy.

## Required Artifacts

- A minimal TensorList-boundary investigation or repair helper, preferably
  scoped to `bayesfilter/testing/` unless production QR code must be patched.
- Focused tests proving that any production QR patch preserves existing
  non-XLA QR likelihood behavior.
- A trusted GPU/XLA diagnostic JSON under `docs/plans/` or
  `docs/plans/artifacts/`, below repository size policy.
- Phase 14 result or blocker:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase14-xla-tensorlist-boundary-repair-result-2026-07-08.md`
- Draft next subplan after Phase 14, which may only propose a non-XLA
  fixed-transport HMC mechanics/smoke gate or another bounded XLA repair if the
  evidence permits.

## Required Checks/Tests/Reviews

- Trusted GPU probe before GPU/XLA execution.
- Any GPU TensorFlow/XLA command must run with escalated/trusted permissions.
- CPU-only support checks must set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow
  import and must not be interpreted as GPU/XLA evidence.
- Run focused pytest for affected QR/XLA helper tests.
- Run `python -m py_compile` on any changed helper/test modules.
- Run `python -m json.tool` on diagnostic JSON artifacts.
- Run `git diff --check` on Phase 14 code and artifacts.
- Bounded read-only review must inspect the Phase 14 result/blocker and next
  subplan before any HMC sampling/tuning, additional training, or default-policy
  change.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the remaining TensorList crossing-boundary XLA blocker be removed for the LGSSM NeuTra value/gradient diagnostic without changing the target or overclaiming readiness? |
| Baseline/comparator | Phase 13 blocker artifact, Phase 10 non-XLA GPU training, Phase 11 non-XLA frozen-payload mechanics, and Phase 12 CPU sample-generation boundary. |
| Primary criterion | Either a trusted GPU/XLA diagnostic compiles and records finite value/gradient checks for the same target boundary, or the blocker is preserved with exact provenance and `jit_compile=false` remains policy. |
| Veto diagnostics | Target/signature change, hidden training, hidden HMC sampling/tuning, hidden sample generation, CPU-only result treated as GPU/XLA evidence, nonfinite diagnostics, TensorList/XLA abort without parseable artifact, or unsupported readiness/product/scientific claim. |
| Explanatory diagnostics | TensorFlow/XLA versions, GPU manifest, compile/abort error class, target signature, finite value/gradient checks, and runtime. |
| Not concluded | HMC convergence, posterior correctness, sampler quality, transport quality, route superiority, production readiness, default-policy change, or broad XLA readiness beyond the exact bounded gate. |
| Artifact | Phase 14 diagnostic JSON/log, result/blocker, helper/tests, and next subplan. |

## Forbidden Claims/Actions

- Do not run NeuTra training.
- Do not run HMC sampling or tuning.
- Do not generate external samples.
- Do not run optimizer updates, parameter updates, or HMC leapfrog steps inside
  the value/gradient diagnostic.
- Do not use DSGE/c603.
- Do not treat CPU-only XLA checks as GPU/XLA evidence.
- Do not claim broad XLA, HMC, production, or scientific readiness.
- Do not change default execution policy.
- Do not rank routes or samplers.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 14 records pass/blocker with a result artifact;
- Phase 14 records a run manifest with git commit, environment, CPU/GPU
  status, seed, wall time, output artifact path, and command;
- any XLA/JIT pass records trusted GPU execution, exact target signature,
  finite diagnostics, and the repaired TensorList boundary;
- any XLA/JIT blocker records exact error/provenance and keeps
  `jit_compile=false` as the inherited policy;
- no training, HMC sampling/tuning, or external sample generation occurred;
- the next subplan states whether it is a non-XLA fixed-transport HMC
  mechanics/smoke gate, another XLA repair, or a different bounded non-DSGE
  gate;
- bounded read-only review agrees with the boundary.

## Stop Conditions

Stop if:

- trusted GPU/XLA access is unavailable;
- the target signature changes unexpectedly;
- the repair would require changing the scientific target;
- HMC sampling/tuning, training, or sample generation becomes necessary;
- the XLA process aborts without a parseable artifact and no smaller
  diagnostic is available;
- diagnostics are nonfinite or malformed;
- artifact sizes violate repository policy;
- review does not converge after five rounds.

## Phase Close Duties

At close:

1. run required local checks;
2. write Phase 14 result or blocker;
3. draft or refresh the next subplan;
4. review the next subplan for consistency, correctness, feasibility,
   artifact coverage, runtime boundary safety, and claim discipline.
