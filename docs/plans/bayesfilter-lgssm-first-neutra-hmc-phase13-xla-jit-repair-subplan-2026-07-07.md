# Phase 13 Subplan: XLA/JIT Repair Gate

Date: 2026-07-07

## Phase Objective

Investigate and either repair or preserve the inherited Phase 9 TensorFlow
XLA/JIT blocker for the LGSSM-first NeuTra path under a bounded, reviewed gate.
The phase should decide whether the fixed tensor-list-size compile failure can
be eliminated for the admitted LGSSM route without changing the target,
training policy, or sample-generation boundary.

This phase is an XLA/JIT repair gate only. It must not run HMC sampling/tuning,
train NeuTra, generate external samples, use DSGE/c603, rank routes, change
defaults, or claim HMC/posterior/product/scientific readiness.

## Entry Conditions Inherited From Previous Phase

- Phase 11 frozen GPU-trained affine payload packaging has passed or written a
  blocker that does not invalidate XLA investigation.
- Phase 12 has separated external sample generation as a CPU multicore boundary
  or written a blocker that does not require XLA to continue governance work.
- Phase 9 XLA/JIT blocker is inherited:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase9-gpu-neutra-training-preflight-xla-blocker-2026-07-07.json`.
- Until this phase passes, all NeuTra training and fixed-transport mechanics
  must keep `jit_compile=false`.
- GPU/CUDA commands require trusted/escalated execution per local policy.

## Required Artifacts

- A minimal XLA/JIT reproduction or repair helper, preferably scoped to
  `bayesfilter/testing/` unless production code must be patched.
- Focused tests or diagnostics proving one of:
  - the LGSSM route now compiles under the reviewed JIT boundary; or
  - the XLA/JIT blocker is still present and accurately recorded.
- XLA diagnostic JSON under `docs/plans/` or `docs/plans/artifacts/`, below
  repository size policy.
- Phase 13 result or blocker:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-result-2026-07-07.md`
- Draft next subplan after Phase 13, which may only propose fixed-transport
  HMC mechanics/smoke or another bounded repair if Phase 13 evidence permits.

## Required Checks/Tests/Reviews

- Trusted GPU probe before GPU/XLA execution.
- Any GPU TensorFlow/XLA command must run with escalated/trusted permissions.
- CPU-only support checks must set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow
  import and must not be interpreted as GPU/XLA evidence.
- Run focused pytest or a bounded diagnostic command named in the result.
- Run `python -m py_compile` on any new helper/test modules.
- Run `python -m json.tool` on any diagnostic JSON artifacts.
- Run `git diff --check` on Phase 13 code and artifacts.
- Bounded read-only review must inspect the Phase 13 result/blocker and next
  subplan before any HMC sampling/tuning, additional training, or default-policy
  change.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the inherited TensorFlow fixed tensor-list-size XLA/JIT blocker be repaired for the admitted LGSSM NeuTra route without changing the target or overclaiming readiness? |
| Baseline/comparator | Phase 9 XLA blocker artifact, Phase 10 successful non-XLA GPU training, and Phase 11 non-XLA frozen-payload mechanics. |
| Primary criterion | Either a bounded trusted GPU/XLA diagnostic compiles and records finite value/gradient checks for the same target boundary, or the blocker is preserved with exact error/provenance and no readiness claim. |
| Veto diagnostics | Target/signature change, hidden training, hidden HMC sampling/tuning, hidden sample generation, CPU-only result treated as GPU/XLA evidence, nonfinite diagnostics, unbounded runtime, or unsupported readiness/product/scientific claim. |
| Explanatory diagnostics | TensorFlow/XLA versions, GPU manifest, compile error class, finite value/gradient checks, target/adapter signatures, and runtime. |
| Not concluded | HMC convergence, posterior correctness, sampler quality, transport quality, route superiority, production readiness, default-policy change, or broad XLA readiness beyond the exact bounded gate. |
| Artifact | Phase 13 diagnostic JSON/log, result/blocker, helper/tests, and next subplan. |

## Forbidden Claims/Actions

- Do not run NeuTra training.
- Do not run HMC sampling or tuning.
- Do not generate external samples.
- Do not use DSGE/c603.
- Do not treat CPU-only XLA checks as GPU/XLA evidence.
- Do not claim broad XLA readiness unless the exact reviewed gate passes, and
  even then do not claim HMC or production readiness.
- Do not change default execution policy.
- Do not rank routes or samplers.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 13 records pass/blocker with a result artifact;
- any XLA/JIT pass records trusted GPU execution, exact target/adapter
  signatures, finite diagnostics, and the repaired blocker boundary;
- any XLA/JIT blocker records exact error/provenance and keeps
  `jit_compile=false` as the inherited policy;
- no training, HMC sampling/tuning, or external sample generation occurred;
- the next subplan states whether it is fixed-transport HMC mechanics/smoke,
  another XLA repair, or a different bounded non-DSGE gate;
- bounded read-only review agrees with the boundary.

## Stop Conditions

Stop if:

- trusted GPU/XLA access is unavailable;
- the target or adapter signature changes unexpectedly;
- the XLA repair would require changing the scientific target;
- HMC sampling/tuning, training, or sample generation becomes necessary;
- diagnostics are nonfinite or malformed;
- artifact sizes violate repository policy;
- review does not converge after five rounds.

## Phase Close Duties

At close:

1. run required local checks;
2. write Phase 13 result or blocker;
3. draft or refresh the next subplan;
4. review the next subplan for consistency, correctness, feasibility,
   artifact coverage, runtime boundary safety, and claim discipline.
