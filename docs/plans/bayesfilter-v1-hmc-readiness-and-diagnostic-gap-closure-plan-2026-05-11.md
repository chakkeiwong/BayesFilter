# Plan: BayesFilter v1 HMC Readiness And Diagnostic Gap Closure

## Date

2026-05-11

## Lane

This is a BayesFilter v1 lane plan.  It starts after:

```text
ec4f498 Close v1 post-completion diagnostics
```

Do not use this phase to switch MacroFinance or DSGE over to BayesFilter.
MacroFinance and DSGE remain read-only external compatibility targets until a
separate integration lane defines rollback criteria and ownership boundaries.

Do not edit or stage:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
docs/plans/bayesfilter-structural-svd-12-phase-execution-reset-memo-2026-05-06.md
docs/plans/bayesfilter-structural-sgu-goals-gaps-next-plan-2026-05-08.md
docs/chapters/ch18b_structural_deterministic_dynamics.tex
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

Production implementation remains TensorFlow/TensorFlow Probability only.
NumPy, solve-form, and covariance-form references may be used only in
`bayesfilter/testing` or tests as debugging or parity fixtures.

## Current Status

The previous v1 pass established:

- v1 public API and local QR/SVD/CUT tests pass;
- QR score/Hessian tests pass and compiled parity tests pass;
- optional live MacroFinance compatibility passes read-only on the observed
  external checkout;
- QR score/Hessian first-call cost is dominated by graph warmup, tracing, and
  Hessian/materialization effects;
- parameter dimension and time dimension are confirmed QR derivative cost
  drivers;
- escalated TensorFlow sees GPU, and small GPU-visible/XLA-visible artifacts
  complete, but no broad GPU speedup or derivative-XLA claim is justified;
- Rotemberg and EZ are design-only optional DSGE fixtures;
- SGU remains blocked as a predictive filtering target;
- the first HMC readiness target was selected as:

```text
linear_qr_score_hessian_static_lgssm
```

## Goals For This V1 Phase

1. Turn the selected QR derivative fixture into the first narrow HMC-readiness
   target.
2. Reduce or at least isolate the QR derivative graph-build and memory cost.
3. Complete the missing state/observation scaling diagnostic only after the
   QR derivative memory guard is in place.
4. Add SVD-CUT branch-frequency diagnostics over a small parameter box without
   promoting SVD-CUT to HMC readiness.
5. Preserve external-project independence: MacroFinance and DSGE stay
   read-only/optional.
6. Encode runtime boundaries so fast CI remains small and expensive HMC,
   branch, GPU, and external checks stay opt-in.
7. Produce auditable artifacts, update the v1 reset memo and source map, and
   commit only v1-lane files when the phase is complete.

## Gaps Remaining

### Gap 1: No HMC Smoke For The Strongest QR Target

The selected target has value, score, Hessian, and compiled-parity evidence,
but no sampler-level evidence.

Closure target:
- build a fixed-seed TFP HMC or NUTS smoke for
  `linear_qr_score_hessian_static_lgssm`;
- record finite objective/gradient events, acceptance, step size, and basic
  posterior recovery diagnostics.

### Gap 2: QR Derivative Memory Cost Is Diagnosed But Not Controlled

The parameter ladder reached multi-GB high-water RSS.  This is acceptable as a
diagnostic result but not as a default development path.

Closure target:
- separate value, score, and Hessian costs;
- test whether Hessian materialization, parameter dimension, or repeated graph
  tracing is the dominant controllable cost;
- define a safe shape envelope for the first HMC target.

### Gap 3: State/Observation Scaling Is Still Missing

The state/observation ladder was deferred because the parameter ladder already
pushed memory pressure high.

Closure target:
- run a smaller state/observation ladder only after the safe shape envelope is
  defined;
- stop if memory thresholds are exceeded.

### Gap 4: SVD-CUT Smooth-branch Prevalence Is Unknown

SVD-CUT derivative tests pass on a smooth separated-spectrum branch, but HMC
needs evidence over a region, not one point.

Closure target:
- aggregate branch labels, active-floor counts, spectral gaps, support
  residuals, deterministic residuals, and parity status over a small parameter
  box;
- keep HMC blocked unless the smooth inactive-floor branch dominates.

### Gap 5: GPU/XLA Evidence Is Too Narrow

GPU-visible and XLA-visible value rows completed, but derivative XLA and
device-placement claims are not established.

Closure target:
- keep GPU/XLA optional and escalated;
- test only matching shapes after CPU evidence is stable;
- distinguish "GPU-visible process" from "confirmed GPU placement".

### Gap 6: External Compatibility Is Not A Switch-over Contract

MacroFinance read-only compatibility passed on an observed checkout, but the
checkout was not a clean release certificate.  DSGE fixtures are still
design-only.

Closure target:
- continue optional read-only compatibility checks;
- add only BayesFilter-local fixture descriptors or tests;
- do not import external project internals into production code.

### Gap 7: CI Tiering Is Documented But Not Enforced

The runtime policy exists, but tests and benchmark scripts still need markers
or command recipes that match the policy.

Closure target:
- tag fast, focused, extended, external, GPU, and HMC checks;
- make default local CI avoid GPU, external checkouts, and heavy HMC.

## Hypotheses To Test

| ID | Hypothesis | Gap | Test | Close If |
| --- | --- | --- | --- | --- |
| H1 | The QR static LGSSM target has stable finite value, score, and curvature on a small parameter box. | G1 | Fixed parameter-box objective/gradient/Hessian scan. | Nonfinite count is zero or explained by intended parameter-domain guards. |
| H2 | A short fixed-seed TFP HMC/NUTS smoke can run on the QR target with useful acceptance and no nonfinite transition failures. | G1 | CPU-only HMC/NUTS smoke with logged acceptance, step size, and posterior moments. | Chains complete with finite diagnostics and posterior recovery within loose smoke tolerances. |
| H3 | QR derivative memory is driven primarily by Hessian materialization and parameter dimension, not the filtering recurrence itself. | G2 | Value-only, score-only, Hessian-only, and score/Hessian benchmark rows at fixed shapes. | Hessian/parameter rows dominate warmup/RSS while value/score rows remain materially smaller. |
| H4 | A safe first-HMC shape envelope can be defined without losing the ability to detect derivative failures. | G2 | Shape envelope benchmark with explicit RSS/time vetoes. | A small target shape stays below the agreed RSS/time thresholds and still exercises value, score, Hessian, and sampler code. |
| H5 | State/observation scaling is secondary for the first HMC target once parameter dimension is fixed low. | G3 | Small state/observation ladder after H4. | Rows finish under thresholds or the hypothesis is rejected with evidence. |
| H6 | SVD-CUT derivative readiness is blocked by branch prevalence, not by missing value-filter behavior. | G4 | Parameter-box branch-frequency diagnostic. | Smooth inactive-floor branch dominates, or blocked labels quantify why HMC stays blocked. |
| H7 | GPU/XLA should remain optional until CPU HMC and derivative diagnostics are stable. | G5 | Compare CPU evidence against escalated GPU-visible matching smoke only after H1-H4. | GPU is either useful for the same target shape or explicitly deferred without affecting CPU closure. |
| H8 | External compatibility can remain read-only while v1 HMC readiness progresses locally. | G6 | Optional MacroFinance/DSGE checks use only local adapters and test fixtures. | No production dependency or external edit is introduced. |
| H9 | CI tiers can protect normal development from expensive diagnostics. | G7 | Marker/command audit and focused test run. | Fast local CI is small; extended/GPU/HMC/external checks are opt-in and documented. |

## Execution Plan

Run each phase as:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Continue automatically only when the primary criterion passes and no veto
diagnostic fires.

## Tightening From Pre-execution Review

The plan is approved with these clarifications.

- A private dense QR score-only diagnostic path is in scope because this phase
  must isolate first-order derivative cost from Hessian materialization.  Public
  score-only API exposure is out of scope until a separate API-freeze review
  covers dense and masked semantics together.
- The full QR score/Hessian path remains the curvature and parity diagnostic.
  HMC smoke diagnostics may use TensorFlow autodiff on the QR value path, with
  the analytic score/Hessian path used to check parity and curvature at the
  fixed target point.
- Cost artifacts must materialize the tensors they claim to measure.  A
  score-only row materializes log likelihood and score; a score/Hessian row
  materializes log likelihood, score, and Hessian.
- The state/observation ladder is run only after the score-only cost row stays
  inside the small CPU envelope.  The first envelope is diagnostic, not a
  release performance target.
- SVD-CUT branch-frequency evidence remains diagnostic-only.  A smooth tiny box
  can motivate a later target-specific SVD-CUT HMC plan, but it does not promote
  SVD-CUT HMC in this phase.
- GPU/XLA derivative work remains deferred unless CPU evidence identifies a
  matching shape worth testing with escalated device visibility.

### Phase 0: Lane And Worktree Audit

Actions:
- run `git status --short --branch`;
- confirm the branch starts from or includes `ec4f498`;
- classify dirty files as v1-lane or out-of-lane;
- do not stage shared monograph, structural, Chapter 18/18b, MacroFinance,
  DSGE, sidecar, or local image files.

Primary criterion:
- all required work can stay under `bayesfilter`, `tests`,
  `docs/benchmarks`, `docs/plans/bayesfilter-v1-*`, and
  `docs/source_map.yml`.

Veto diagnostics:
- a necessary change requires editing MacroFinance, DSGE, monograph, or
  structural chapter files.

### Phase 1: QR HMC Target Specification

Actions:
- define the exact static LGSSM fixture, parameter vector, transforms, priors,
  and target log probability;
- use QR value plus TensorFlow autodiff for the sampler gradient, with the full
  score/Hessian path reserved for parity and curvature diagnostics;
- place reusable test-only helpers under `bayesfilter/testing` or `tests`;
- keep production filtering code unchanged unless a missing TF/TFP API boundary
  is discovered.

Tests:
- value parity against the existing QR likelihood;
- score parity against autodiff or finite differences;
- Hessian symmetry and curvature sanity checks.

Primary criterion:
- the target is deterministic, fixed-seed, CPU-runnable, and fully described.

Veto diagnostics:
- target construction relies on NumPy in production code;
- target depends on external MacroFinance/DSGE imports.
- private score-only diagnostics leak into the public API by accident.

### Phase 2: QR Derivative Cost Decomposition

Actions:
- extend or add benchmark rows for value-only, score-only, Hessian-only, and
  score/Hessian combined calls;
- record graph warmup, first measured call, steady call, RSS, max RSS, shape,
  and parameter dimension;
- add hard stop thresholds for extended ladders.

Tests:
- CPU-only benchmark smoke for the selected HMC target shape;
- diagnostic ladder for parameter dimension at the safe target shape.

Primary criterion:
- the phase identifies which derivative component is the immediate bottleneck.
- the score-only row is materially cheaper than the full score/Hessian row at
  the first HMC target shape, or the result artifact explains why it is not.
  This criterion concerns a diagnostic helper, not a public API promise.

Veto diagnostics:
- benchmark rows exceed the memory threshold before producing interpretable
  metadata;
- results are described as optimization success before parity is preserved.

### Phase 3: Safe Shape Envelope And State/Observation Ladder

Actions:
- choose a conservative target shape for first HMC;
- run the state/observation ladder only inside the safe envelope;
- record a stop reason if memory or runtime guard trips.

Tests:
- `v1_state_observation_ladder` or a smaller replacement ladder;
- JSON/Markdown benchmark artifacts.

Primary criterion:
- there is an explicit CPU shape envelope for first HMC readiness.

Veto diagnostics:
- runtime or RSS makes the next HMC phase disproportionate.

### Phase 4: First QR HMC/NUTS Smoke

Actions:
- implement a CPU-only fixed-seed TFP HMC or NUTS smoke;
- start with very short chains and conservative target shape;
- record acceptance, step size, nonfinite count, posterior moments, and runtime;
- keep this out of fast CI until runtime is known.

Tests:
- target log probability finite at initial point;
- value/score finite over proposed parameter box;
- HMC/NUTS smoke completes.

Primary criterion:
- a narrow QR HMC-readiness artifact exists for
  `linear_qr_score_hessian_static_lgssm`.

Veto diagnostics:
- recurrent nonfinite gradients;
- acceptance collapses for reasons not explained by parameter transforms;
- runtime is incompatible with extended local diagnostics.

### Phase 5: SVD-CUT Branch-frequency Diagnostic

Actions:
- add a small parameter-box diagnostic for SVD-CUT derivative branches;
- record active floor counts, weak spectral gaps, integration rank, point
  count, support residual, deterministic residual, and parity status;
- do not run SVD-CUT HMC in this phase unless the branch gate unexpectedly
  passes and a separate plan is written.

Tests:
- existing SVD-CUT derivative tests still pass;
- branch-frequency artifact is produced.

Primary criterion:
- SVD-CUT HMC remains either explicitly blocked with quantified branch evidence
  or promoted only to a new, separate readiness plan.

Veto diagnostics:
- branch labels are missing from artifacts;
- active-floor or weak-gap cases are hidden by regularization.

### Phase 6: Optional External Fixtures Without Coupling

Actions:
- rerun optional MacroFinance compatibility only if local preconditions are
  available;
- add BayesFilter-local Rotemberg/EZ fixture descriptors if needed;
- keep SGU blocked as a predictive filtering target.

Tests:
- optional MacroFinance read-only compatibility;
- adapter-gate tests for DSGE fixture metadata.

Primary criterion:
- external evidence is recorded without changing external repositories or
  BayesFilter production dependencies.

Veto diagnostics:
- a fixture requires importing private external internals in production code;
- the phase starts becoming a client switch-over.

### Phase 7: CI Tier Encoding

Actions:
- add or audit pytest markers/commands for:
  - fast local CI;
  - focused local regression;
  - extended CPU diagnostics;
  - optional external;
  - escalated GPU/XLA;
  - HMC readiness;
- document exact commands in the result artifact and reset memo.

Tests:
- fast local CI command;
- focused QR/SVD/CUT regression command;
- extended commands as opt-in.

Primary criterion:
- normal development has a small default gate and heavier checks are
  reproducible by command.

Veto diagnostics:
- GPU/CUDA tests run without escalation;
- optional external tests become default CI.

### Phase 8: Final Audit, Reset Memo, Source Map, Commit

Actions:
- update the v1 reset memo with every phase result and next-phase decision;
- update `docs/source_map.yml`;
- run `git diff --check`;
- stage only v1-lane files;
- commit after the phase is complete.

Primary criterion:
- all committed files are v1-lane owned and every claim has an artifact or
  explicit blocked status.

Veto diagnostics:
- out-of-lane files are staged;
- HMC or SVD-CUT readiness is claimed beyond the collected evidence.

## Expected Outputs

Plan/result artifacts:

```text
docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-plan-2026-05-11.md
docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-result-2026-05-11.md
```

Likely code/test artifacts:

```text
bayesfilter/testing/*hmc*_tf.py
tests/test_hmc_linear_qr_readiness_tf.py
tests/test_svd_cut_branch_diagnostics_tf.py
docs/benchmarks/bayesfilter-v1-qr-derivative-cost-decomposition-*.json
docs/benchmarks/bayesfilter-v1-qr-hmc-smoke-*.json
docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-*.json
```

Do not require all likely artifacts if an earlier phase veto blocks them.  A
blocked result with exact diagnostics is a valid phase closure.

## Decision Rule For Completing This V1 Phase

This phase is complete when:

- the QR HMC target has either a passing narrow CPU HMC smoke or a precise
  blocker with finite-value/gradient evidence;
- QR derivative memory has a component-level decomposition and safe shape
  envelope;
- SVD-CUT branch readiness is quantified and still not overclaimed;
- external projects remain read-only;
- CI tiers are command-level reproducible;
- the reset memo and source map are updated;
- a scoped v1-lane commit is made.
