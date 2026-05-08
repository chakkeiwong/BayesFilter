# Audit: differentiable particle filter governing note and Phase 1 plan

## Date

2026-05-08

## Scope

This note audits the following two documents:

- `docs/differentiable-particle-filter-program.md`
- `docs/plans/bayesfilter-differentiable-particle-filter-phase1-audit-plan-2026-05-08.md`

The goal is to determine whether they are aligned with the BayesFilter research-engineering discipline, the agreed differentiable particle-filter direction, and the current codebase contracts.

## Executive assessment

Overall assessment: **proceed with revisions noted below**.

The two documents are directionally strong.  They correctly preserve the main program decisions from the discussion:

- BayesFilter owns the production-quality generic SSM API;
- the implementation ladder is EDH -> EDH/PF with importance weights -> soft/differentiable resampling -> OT -> LEDH/PF with neural OT;
- pure filtering evidence must precede HMC evidence;
- student repositories are exploratory engineering input rather than correctness references;
- TensorFlow/TFP is the intended production stack;
- DSGE and MacroFinance should remain client projects behind adapters.

The documents also correctly separate several claim classes that are easy to blur in this topic:

- exact special-case behavior versus approximate nonlinear filtering;
- importance-corrected particle filtering versus differentiable resampling relaxations;
- filtering quality versus HMC validity;
- exploratory code reuse versus audited production contracts.

That said, both documents still need tightening before they should be treated as the controlling artifacts for implementation.  The main issues are not conceptual disagreement; they are mostly about evidence precision, missing operational thresholds, and one important ambiguity about the first implementation rung.

## What the governing note gets right

### 1. Correct program target

The governing note correctly frames the work as a **production-quality reusable component** rather than a research script.  This matches the agreed intent and fits the existing BayesFilter architecture.

### 2. Correct API ownership

The note is consistent with the current BayesFilter interface boundary in:

- `bayesfilter/structural.py`
- `bayesfilter/filters/kalman.py`
- `bayesfilter/filters/sigma_points.py`
- `bayesfilter/filters/particles.py`

This is the right direction.  The DPF should conform to the current structural-model contract rather than inventing a parallel model interface.

### 3. Correct source hierarchy

The source hierarchy is well judged:

- BayesFilter defines production API and release gates;
- the CIP monograph supplies theory direction;
- dsge_hmc and MacroFinance define client needs;
- student code is exploratory input only.

This is particularly important because the current project is trying to convert frontier academic ideas into industrial-quality infrastructure.  The note captures that distinction well.

### 4. Correct laddering of algorithm phases

The five-rung ladder is a sensible development order.  It preserves the crucial difference between:

- flow transport mechanics,
- importance correction,
- differentiable relaxations,
- OT resampling,
- and the final LEDH + neural OT target.

That ordering should reduce the risk of implementing a mathematically tangled end-state before the simpler value-side path is checked.

### 5. Correct separation of filtering and HMC evidence

The governing note correctly states that pure filtering consistency is not enough to justify HMC claims.  This matches the repo-wide evidence discipline and is especially important here, where differentiability can be mistaken for correctness.

## What the governing note should revise

### A. The HMC criterion is still too qualitative

The note says synthetic-data HMC should recover parameters in a "reasonable predeclared range" and should report usable ESS and R-hat.  That is directionally right, but still too loose for a governing artifact.

It should be revised to require that Phase 1 or Phase 2 define:

- the exact posterior recovery criterion;
- whether coverage of the truth by posterior intervals is required;
- which summary (mean/median/MAP) is judged against truth;
- minimum bulk ESS and tail ESS thresholds;
- maximum acceptable R-hat;
- divergence tolerance;
- whether zero divergences are required for strict convergence or only for promotion beyond smoke use.

Without those thresholds, later phases can quietly weaken the evidence bar.

### B. The first implementation rung is slightly ambiguous

The governing note says the next phase is a Phase 1 audit and implies EDH is the first step, but it does not fully pin down what scalar the first TensorFlow/TFP implementation is supposed to return.

This matters because there are at least three distinct first-rung possibilities:

1. EDH as a deterministic flow approximation with a value-side likelihood surrogate;
2. EDH/PF with importance weights targeting a corrected particle estimate;
3. a linear-Gaussian EDH recovery harness used mainly as a value-side contract test.

The plan mentions these pieces later, but the governing note should say more explicitly that the first implementation rung is expected to be:

> EDH linear-Gaussian recovery plus an EDH/PF importance-weight skeleton, with no differentiable-resampling claim yet.

That removes ambiguity about whether soft resampling is already in scope for the first code pass.

### C. Evidence labels are good but incomplete for this topic

The current labels are useful, but the DPF program would benefit from at least two more labels:

- `filter_consistency_evidence` — empirical filtering agreement with reference filters, without HMC claims;
- `sampler_not_justified` — value or gradient path exists, but HMC promotion remains blocked.

The existing labels are close, but these two would make audit notes clearer and reduce the temptation to upgrade filtering evidence into sampler evidence.

### D. The TensorFlow/TFP section should name graph-break risk more concretely

The note says there should be no silent Python-side graph breaks.  Good.  But for this program, the main practical hazards should be named explicitly:

- Python loops over particles that block scalable compilation;
- host-side random-number generation outside TensorFlow;
- shape-dependent branching that breaks `tf.function` assumptions;
- non-differentiable gather/ancestor logic that is accidentally masked by eager execution.

This is not a major flaw, but adding these concrete hazards would make the governing note stronger as an implementation control artifact.

## What the Phase 1 audit plan gets right

### 1. Correctly stops before implementation

This is the right call.  Given the number of open mathematical and interface questions, starting with an audit phase is consistent with the repo’s research-engineering policy.

### 2. Good decomposition of the audit

The subdivision into:

- theory source audit,
- external implementation audit,
- BayesFilter API/code audit,
- client workflow audit,
- benchmark/tolerance design,
- TensorFlow/TFP readiness,
- Phase 2 handoff,

is strong and appropriate for the work.

### 3. Proper treatment of student repos

The plan correctly treats student repos as exploratory implementation input, not correctness references.  This matches the project requirement and should remain unchanged.

### 4. Correct API emphasis

The plan appropriately centers the audit on the existing BayesFilter structural interface and asks for a first DPF public contract.  That is exactly the right boundary to define before implementation.

### 5. Good benchmark ordering

The benchmark ladder is sensible:

- linear-Gaussian first,
- nonlinear filtering second,
- stress cases third,
- synthetic HMC recovery only after value-gradient gates.

This is well aligned with the repo’s diagnostic-first philosophy.

## What the Phase 1 audit plan should revise

### A. It asks for some outputs that are too large for one phase unless priority is stated

Phase 1 currently asks for all of the following:

- theory claim table,
- external implementation inventory,
- API/code audit table,
- client workflow table,
- benchmark table,
- TensorFlow/TFP readiness table,
- Phase 2 handoff,
- final audit/result note.

That is reasonable in principle, but the plan should state priority order explicitly in case time or context runs short.  Otherwise the phase can degrade into shallow coverage of too many sections.

Recommended priority order:

1. theory claim table;
2. BayesFilter API/public-contract table;
3. benchmark/tolerance table;
4. TensorFlow/TFP readiness table;
5. client workflow audit;
6. external implementation audit;
7. Phase 2 handoff.

This ordering protects the production contract first.

### B. Benchmark tolerances are requested but not yet operationalized

The plan asks for a benchmark table with metrics and tolerances, but it does not yet state how strict those tolerances should be at the first rung.

For example, the linear-Gaussian EDH recovery benchmark should eventually specify whether the comparison is against:

- mean/covariance trajectories,
- per-time-step incremental log likelihood,
- total log likelihood,
- and parameter-gradient agreement.

Similarly, for the nonlinear benchmark, the plan should clarify whether UKF/EKF agreement is judged by:

- predictive log likelihood,
- state RMSE,
- filtered mean trajectories,
- or some weighted combination.

The plan does not need the final thresholds now, but it should explicitly require that the audit produce them rather than only naming the metrics.

### C. The external implementation audit should explicitly forbid direct porting as a default

The current language says "do not port student code" in non-goals, which is good.  Still, Phase 1B should go further and say that the default classification for external code is **not reusable until proven otherwise**, especially because:

- framework mismatch is likely,
- engineering assumptions may be hidden,
- and industrial-quality diagnostics are probably incomplete.

This matters because otherwise the external audit section may accidentally consume too much effort relative to the API and benchmark work.

### D. The plan should require explicit claim mapping for differentiable resampling

Phase 1A covers soft-resampling bias and OT/neural OT, but the plan should ask for one extra explicit deliverable:

- for each resampling method, state whether its gradient is intended for training only, value-side benchmarking, or eventual HMC target construction.

This is the central correctness issue for the later ladder, and it should be named more directly.

### E. Phase 1F should state that compiled readiness is a gate, not a nice-to-have

The TensorFlow/TFP readiness section is good, but because the repo is aiming at HMC use, the phrase "compiled differentiable likelihood path" needs to be treated as a required future gate rather than a performance enhancement.

The plan hints at this, but the review recommends stating it more sharply:

- eager-only success can justify debugging or early value checks;
- it does not justify production HMC promotion.

## Consistency with the current codebase

These two documents are broadly consistent with the current code structure.

### Supporting consistency

Current BayesFilter already has:

- a generic structural model protocol in `bayesfilter/structural.py`;
- Kalman, sigma-point, and particle backends that return result objects plus metadata;
- adapter boundaries for DSGE and MacroFinance;
- tests that already treat metadata and structural contracts as first-class artifacts.

That makes the documents’ API direction credible.

### Current mismatch to watch

The current particle and sigma-point implementations are NumPy-first and value-side oriented, whereas the program note targets TensorFlow/TFP differentiable HMC usage.  That is not a contradiction, but it is a real transition risk.

The audit documents should therefore avoid implying that the current NumPy backends are direct prototypes for the TensorFlow path.  They are better treated as:

- reference behavior,
- fixture sources,
- API shape exemplars,
- and regression baselines.

## Recommended revisions before treating these as controlling documents

1. In the governing note, replace the qualitative HMC recovery wording with an explicit requirement that thresholds be declared in the audit before implementation promotion.
2. In the governing note, state the first implementation rung more explicitly: EDH linear-Gaussian recovery plus EDH/PF importance-weight skeleton, no differentiable-resampling claim yet.
3. In the governing note, add one or two evidence labels for filtering-only evidence versus sampler-blocked status.
4. In the Phase 1 plan, add a priority order across audit outputs.
5. In the Phase 1 plan, require the audit to produce concrete benchmark tolerances, not just benchmark categories.
6. In the Phase 1 plan, make explicit that external repos are non-portable by default until proven reusable.
7. In the Phase 1 plan, add a per-resampling-method claim table stating whether the method is intended for training-only, value benchmarking, or HMC-target use.
8. In the Phase 1 plan, sharpen compiled TensorFlow readiness as a promotion gate rather than a later optimization.

## Audit disposition

Disposition: **approve with revisions**.

The documents are strong enough to guide the next pass, but they should not yet be treated as final controlling artifacts for implementation without the revisions above.  The main reason is not that the direction is wrong; it is that this project is vulnerable to claim drift between filtering quality, differentiable surrogates, and HMC correctness, so the governing documents need slightly tighter operational language.

## Suggested next artifact

The next safest artifact is a short revision pass on the two documents that:

- tightens the HMC and benchmark thresholds language;
- pins down the first implementation rung;
- and adds the missing priority/gate clarifications.

After that, a Phase 1 audit result note can be written against the revised plan with lower risk of ambiguity.