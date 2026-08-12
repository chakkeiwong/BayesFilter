# Zhao-Cui Moment-Teacher Score Error Versus MCSE Transfer Diagnostic

Date: 2026-07-30
Status: approved by the user's request to run the proposed score/MCSE test
Classification: transfer diagnostic only; not moment-teacher score evidence

## Research Intent Ledger

- Main question: on a complete BayesFilter stochastic score program, is the
  paired TF32 numerical drift small relative to target-specific Monte Carlo
  standard error (MCSE)?
- Candidate: FP32 with TF32 enabled.
- Comparator: the identical FP32/XLA finite program with TF32 disabled.
- Expected failure mode: reduced-precision matrix products perturb small score
  signals, possibly with accumulation through the filtering horizon.
- Promotion criterion: for every final cumulative physical-score coordinate,
  `abs(mean(TF32 - reference)) / MCSE(reference mean) <= 0.1`.
- Promotion veto: either arm is non-finite or fails its existing chart, reset,
  marginal, replay, work-count, or graph validity checks; prepared inputs,
  seeds, program source, or controls differ; fewer than eight independent
  estimator seeds; a reference MCSE is zero while numerical drift is nonzero;
  or any coordinate exceeds the ratio threshold.
- Continuation veto: no interpretable paired score result is produced within
  the two-node attempt budget.
- Repair trigger: a validity failure is repaired only within the same target,
  controls, seeds, hardware class, and attempt budget. A ratio failure requires
  precision localization or a reviewed criterion change.
- Explanatory diagnostics: paired-difference MCSE, per-seed drift, value drift,
  runtime, allocator peak, and absolute/relative score drift.
- Nonclaim: this does not test the Zhao-Cui moment teacher's final score. That
  finite program is not implemented. It tests whether the proposed MCSE-scaled
  criterion is satisfied on the nearest complete canonical Contract-E LGSSM
  score route and may only inform the later moment-teacher gate design.

## Evidence Contract

- Engineering/scientific question: is TF32 score drift negligible compared
  with Monte Carlo uncertainty on a complete score-bearing GPU/XLA route?
- Exact route: canonical Contract E-Chol LGSSM at `T=2`, `N=1024`, FP32, XLA,
  exact transport chunk `K=1024`, 20 Sinkhorn steps, 2 terminal-balance steps,
  active reset, and estimator seeds 81700--81715.
- Comparator: the same route, data, particles, prepared noise, residual design,
  ridge, seeds, and controls with TF32 disabled.
- Primary criterion: maximum coordinate ratio described above is at most 0.1.
- Hard vetoes: all route validity checks and paired-identity checks listed in
  the research ledger.
- Explanatory only: timing, memory, paired-difference confidence, value drift,
  and comparisons with unrelated historical MCSEs.
- Not concluded on pass: no moment-teacher score validity, no TF32 default
  readiness, no HMC readiness, no nonlinear-model result, and no claim that
  intermediate relative errors can always be ignored.
- Artifact root:
  `docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/attempt01/`.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Canonical LGSSM route | repository-owned complete value/score graph | nearest complete stochastic score program using Contract E | transfer may not represent moment-teacher conditioning | explicit transfer-only classification | comparator baseline |
| `T=2` | smallest allowed canonical horizon | exposes score propagation at bounded cost | understates long-horizon accumulation | no long-horizon nonclaim | convenience scope |
| `N=1024`, `K=1024` | prior canonical diagnostic and active exact-divisor policy | supplies measurable seed variation and exact one-block transport | target MCSE differs at other particle counts | exact scope recorded | baseline |
| 16 seeds | prior canonical score campaign size | enough for a descriptive sample MCSE at bounded cost | noisy tail estimate and weak generalization | report sample size and paired-difference MCSE | reviewed baseline |
| ratio threshold 0.1 | criterion proposed in the preceding discussion and accepted by user | numerical drift one order below MCSE is practically negligible for this screen | threshold is a judgment, not a theorem | report raw ratios | hypothesis |
| FP32-no-TF32 reference | isolates TF32 from storage precision | same dtype and program minimizes confounding | not an FP64 mathematical oracle | current source/identity equality | comparator |
| balance 2, Sinkhorn 20 | prior validated canonical control selection | avoids tuning on this comparison | may fail for new seeds | existing validity gates fail closed | reviewed inherited control |

## Skeptical Execution Audit

- Wrong baseline: avoided by comparing the same FP32 program and prepared
  inputs; FP64 would conflate TF32 with ordinary FP32 storage/arithmetic.
- Proxy promotion: the ratio is a transfer diagnostic, not moment-teacher
  admission or default readiness.
- Missing stop condition: stop after one TF32 node, one reference node, and one
  aggregate, or on an unrepaired validity failure.
- Unfair comparison: paired seeds and exact prepared-input identities are
  mandatory.
- Hidden assumption: score MCSE from seed variation is used only for this exact
  scope; it is not transferred from other models or horizons.
- Environment mismatch: both arms use the same TensorFlow environment, GPU,
  XLA setting, source hashes, and memory-growth policy.
- Artifact adequacy: per-seed final scores permit both reference MCSE and paired
  drift calculation, directly answering the stated transfer question.

Audit verdict: pass for the transfer question. It cannot answer the exact
moment-teacher question because that final score program does not yet exist.

## Pre-Mortem And Budget

- Misleading pass: short-horizon cancellation could hide larger long-horizon
  drift. Keep the conclusion at `T=2` and require the later integrated route to
  repeat the test at its claim horizons.
- Non-scientific failure: invalid reset marginals or GPU resource failure.
  Preserve the attempt and allow one localized retry without changing controls.
- Statistical weakness: 16 seeds make MCSE itself uncertain. Report raw
  per-seed scores and paired-difference MCSE; do not claim a universal ratio.
- Compute budget: two GPU nodes plus one CPU aggregation, one localized retry
  per node only if infrastructure or serialization fails, expected under five
  minutes total and hard stop at ten minutes.

