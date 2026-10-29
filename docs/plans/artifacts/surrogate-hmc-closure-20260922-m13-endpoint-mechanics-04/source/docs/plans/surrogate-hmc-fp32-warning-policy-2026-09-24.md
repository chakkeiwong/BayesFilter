# FP32 default and saved-run warning policy — 2026-09-24

Owner directive: the observed saved-run discrepancy is not material enough to
stop the surrogate-HMC campaign. Retain it as a warning. FP32 is the default
for computational speed and memory economy.

The scientific goal remains testing canonical analytical DCTR proposal forces
in surrogate HMC, using the declared endpoint posterior for acceptance. This
change neither certifies the finite DCTR likelihood as the model likelihood nor
qualifies M13 posterior sampling.

## Implementation contract and pre-run audit

Keep the recorded 3e-5 saved-value/score comparison unchanged. Separate its
warning from mandatory finite-program validity, finite FP32 values and scores,
FP64 analytical derivative, trace agreement, and recorded repeatability checks.
Missing required evidence still fails. The fixed-design FP32 repeatability
comparison is optional localization, no longer the next required campaign step.

The M13 target already defaults to FP32, and the existing surrogate force
scopes use FP32 with TF32 and XLA. Verify the actual default callable and its
TensorFlow input/output dtypes. Retain explicitly declared FP64 reference,
derivative-check, and small HMC endpoint/integrator roles; this directive does
not require rerunning closed LG1 or converting independent accuracy authorities
to FP32. No new speedup or memory benchmark is claimed.

Skeptical audit: simply removing the old combined no-harm check would also
remove FP32 validity from the decision. Extract validity and finiteness as
explicit required checks before demoting the numerical mismatch. Do not let
all(empty_checks) pass. Preserve every old JSON result; issue a new decision
over the saved replay with hashes and its original status. Warning acceptance
does not establish posterior agreement or remove scope tuning/reference needs.

The original threshold is retained as an inherited engineering diagnostic,
not a calibrated utility threshold. Owner materiality judgment supplies its
new role. Future numerical failures and downstream posterior effects keep their
own declared criteria. No target, data, numerical controls, or force equations
change.

Validation: focused CPU tests exercise the actual consumer and worker call
chain, warning acceptance, each retained veto, missing/nonfinite evidence, and
the FP32 default. Reassess the existing nine-point repaired replay without new
filtering. Budget: at most 180 CPU-worker seconds and zero GPU, charged within
the current campaign. GPUs are hidden before framework import. Output root:
`artifacts/surrogate-hmc-closure-20260924-fp32-policy-01`.

Primary pass criterion: warnings do not block the engineering decision while
invalid/missing evidence does. Numerical differences and MCSE remain explanatory.
There is no new stochastic comparison, posterior verdict, or method ranking.

Next within the existing master: M13 scope-specific control preparation and
posterior endpoint/reference preparation. The earlier replay's one-attempt cap
does not make optional repeatability a prerequisite under this owner directive.

## Execution result

Implemented in the M13 full-consumer assessment and master readiness reporting.
The worker reaches the assessment through the real full-parity stage. Focused
tests exercise that call chain, the actual factory's FP32 default and tensor
dtypes, warning acceptance, and retained vetoes. The initial suite passed 124 tests; the final suite passed 125 tests. Numerical kernels and controls were not changed.

A fresh assessment of the saved repaired replay returns
M13_ENGINEERING_CHECKS_PASSED_WITH_WARNINGS_TUNING_PENDING. Every required
engineering check passes; all numerical dependency hashes match the saved run.
The original JSON, status and comparison values remain unchanged. No new
filtering, random draws, GPU execution, or M13 posterior sampling occurred.
FP32 repeatability across processes remains unmeasured; the existing deterministic
check covers the same FP64 compiled function only.

Evidence: [assessment](artifacts/surrogate-hmc-closure-20260924-fp32-policy-01/assessment/report.json),
[final test manifest](artifacts/surrogate-hmc-closure-20260924-fp32-policy-01/review-completion/source-manifest.json),
[accounting](artifacts/surrogate-hmc-closure-20260924-fp32-policy-01/accounting.json).
The manifests preserve commands, environment, source hashes, CPU-only choice and
wall time. Policy work charged 70.064439 CPU seconds and zero GPU;
the separate prior materiality postprocessing charge is 0.018487292 CPU.
[Campaign balance](artifacts/surrogate-hmc-closure-20260924-fp32-policy-01/campaign-balance.json)
records remaining 11559.064705 CPU and 193195.452245850 GPU seconds.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept engineering checks with warning | Warning does not block; invalid or missing required evidence blocks | All saved required checks pass; injected-defect tests reject | Downstream M13 posterior behavior and FP32 process repeatability | Existing-master scoped controls and endpoint/reference preparation | M13 posterior correctness or all-model completion |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Required engineering checks pass; validity, finiteness and derivative vetoes retained |
| Statistically supported ranking | None; no new stochastic comparison |
| Descriptive-only differences | Saved numerical discrepancies and previously reported materiality/MCSE |
| Default-readiness | FP32 is the owner-directed precision default; this does not promote untuned M13 controls |
| Next evidence needed | Scoped tuning and downstream posterior tests against a declared endpoint/reference |

Post-run red-team: passing engineering checks could be misread as likelihood or
posterior accuracy. The assessment explicitly limits qualification to engineering
checks and keeps posterior/scoped-control flags false. A downstream materiality
failure would trigger its own declared repair; the saved-number warning cannot
override such evidence. No precision speedup has been measured here.

## Completion audit and existing-master preparation

The final call-chain audit found one more reporting endpoint: the original
all-model coverage register still described the consumer as pending. Update
that register and test it, within this policy's unchanged 180-CPU/zero-GPU
allocation. This changes reporting only and must keep posterior qualification
false. Repeat the focused suite because source has changed.

The next-stage scope audit located the already declared prior in
[surrogate-hmc-all-models-master-program-2026-09-18.md](surrogate-hmc-all-models-master-program-2026-09-18.md),
lines 495–505, applied explicitly to M13 in its scope table. Independent
Normal(0, log(2)^2) priors on lambda_Q and lambda_R define the noise-scale
inference exercise. Thus prior selection is resolved by the existing master;
implementation and full-support qualification remain outstanding. Keep the
four-state/two-observation T20 wrapped model, fixed remaining coefficients and
initialization law. This is not a prior fitted to the nine-point results.

The nine-point likelihood reference cannot certify integration over those
unbounded priors. The next reference must account for posterior tails and the
wrapped-density alias validity range; it must not truncate the prior at the
old grid or treat a numerical alias limit as a support boundary. Seed701 has
already served as qualification data. Scoped tuning needs separate calibration,
validation and untouched claim partitions. These are substantive existing-master
tasks; replay-number agreement is not their prerequisite.

Completion review: the coverage register now agrees with the consumer and
closure-readiness report. Final test result: 125 passed, 21 warnings in 33.28s. Both review runs and
the reassessment are charged; total policy work is 70.064439 CPU seconds.
The final source hashes match the tested files. The master discovers the new
incremental receipt, and its computed balance matches the saved campaign balance.

The [M13 scope preparation record](artifacts/surrogate-hmc-closure-20260924-fp32-policy-01/m13-scope-preparation.json)
records the existing prior and exact pending tasks for continuation. This is a
preparation record, not a tuning artifact or authorization for a posterior claim.
