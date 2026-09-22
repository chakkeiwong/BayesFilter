# HMC continuation after M26

This replaces the post-M25 execution agenda. The
[M26 result](bayesfilter-hmc-m26-lifetime-and-policy-result-2026-09-22.md)
closes tested CPU process containment and paired numerical identity. It leaves
posterior sufficiency, broader backend evidence and the earlier scientific gaps
open. No M27 numerical experiment has run. The next executable tranche is
bounded member-assessment repair and GPU compatibility/pricing, before a large
confirmation inventory is frozen.

## Evidence that changes the order

The selected rotated-Gaussian L=25 member fails the short readiness window.
Its independent 60000-draw fixed arm nevertheless has R-hat 1.00114 and median
MCSE as large as .289 against .05. A plug-in precision allocation is about
2.01 million draws per chain. This is descriptive planning evidence, not a
guarantee or a sensible immediate allocation. Longer windows alone cannot
close this failure. The verified siblings are unassessed; do not infer that
they fail and do not let this posterior result delete tuning members.

`first_verified` removes the old missing-L=3 selection problem but is an
arbitrary deterministic representative, not an efficiency policy. A bounded
predeclared sibling inventory is now the smallest useful next experiment.
Neither lugsail nor autocorrelation has been shown superior: the M26 arms used
different members/seeds. The median estimator is shared by the two mean-MCSE
choices, which also prevents attributing this median precision failure to the
choice between them.

Exact operating characteristics of the inherited pointwise two-sided 95%
binomial lower-bound screen at .90 are recorded in
`m26-r1/followup-diagnosis-r1/result.json`. For N fits, invert the lower
Clopper–Pearson endpoint to obtain the smallest passing success count k, then
sum `Pr[Binomial(N,p) >= k]`. The hypothetical p values are sensitivity cases,
not pilot estimates. Missing outputs count as failures in the N denominator.

| Fits per model/estimator | Required successes | Probability of passing one screen if p=.95 | CPU hours for three models and one estimator, at the larger measured pilot cost |
| --- | ---: | ---: | ---: |
| 64 | 63 | .164 | 18.58 |
| 128 | 122 | .541 | 37.16 |
| 256 | 240 | .855 | 74.33 |
| 384 | 358 | .951 | 111.49 |
| 512 | 475 | .989 | 148.65 |

For four quantities each with marginal p at least .95, the union bound gives
at least .804 probability of all four passing with N=384, and .956 with N=512.
This addresses the four coverage screens, not the additional delivery screen
or simultaneous 95% confidence. At N=384, even Gaussian alone costs about
26.98 CPU hours at its larger observed pilot cost, above the remaining 22.12
CPU hours. Two pilot costs do not establish a time ceiling. Do not launch 64 or
128 fits merely because they fit part of the budget and then describe a
nonrejection or occasional pass as closure of the original program.

## M27: bounded sibling assessment and backend pricing

Reserve at most **3600 CPU worker-seconds and 4800 GPU worker-seconds** from
M26's terminal remainder, with at most two numerical workers concurrently.
These are convenience resource ceilings for a development/pricing tranche,
not scientific thresholds or an increase in campaign authorization. Stop or
reduce unlaunched development cells if measured cost consumes the ceiling;
retain every planned/attempted/missing cell separately. Do not reduce the
denominator of an already frozen experiment. Use a new `m27-r1/` output root,
immutable source snapshot and exact command manifests.

1. Add an optional diagnostic-harness member rule that predeclares a finite
   list: sort verified L values increasingly, take the first two available L
   values, then take the smallest candidate ID within each. This is a bounded
   cost-oriented hypothesis, not a mixing ranking or public tuning default.
   Retain every verified member and record the requested list before any
   posterior stream. If fewer than two L values exist, record the shortage;
   never silently choose another rule or use posterior diagnostics to fill it.
   The existing single-member and all-member defaults remain unchanged.
   Extend assessment/reporting so both requested members have separate results
   and missing outputs, while the independent experiment unit stays the
   complete fit. Never pool siblings as independent confirmation replications
   or pick the better posterior from the same assessment draws. Tests must
   cover deterministic ordering, shortage, restart, a failed first sibling
   followed by an assessed second sibling, unchanged tuning membership and
   full-denominator reporting on at least Gaussian and rotated Gaussian.

2. Freeze two new ordinary rotated-Gaussian development fits under that rule.
   Use native broad tuning and the unchanged .05 model-coordinate mean/median
   requirement. Use autocorrelation as the declared diagnostic mean estimator
   because the failure being repaired used it, not because it won a comparison.
   Test a 10000-transition readiness window/minimum, 5000-transition chunks
   and 30000 warmup maximum. The saved full warmup motivates the window; the
   three-window maximum is an explicitly unproven budget margin. Keep the M26
   30000 retained minimum, 60000 cap and 5000 chunks. These counts ask whether
   predeclared siblings are usable within an affordable allocation; they do
   not assert adequacy for the failed L=25 member. Use the explicit count
   budget and preserve independent 2000-discard/60000-retained fixed arms.
   Allow at most 1200 seconds per complete fit; unused M27 CPU capacity covers
   implementation tests and diagnosis. Convenience root seeds 2026092281 and
   2026092282, new design IDs and frozen scopes prevent overlap with M26.
   Record every candidate and requested sibling, including caps and health
   vetoes. Do not run a larger confirmation if the mechanism remains unexplained.

3. Price the current complete-fit process boundary on trusted GPU/XLA after
   readiness/memory-growth checks. Use one fresh Gaussian and one fresh
   beta-binomial fit identity, each executed persistently and in an isolated
   child, four executions total, at most 1200 seconds each including teardown.
   Use the M26 model-specific counts, lugsail and `first_verified` as frozen
   baseline settings, with new GPU scopes. These pairs test compatibility and
   cost, not statistical superiority. Match the numerical design, seeds,
   target, starts and source within each pair; require complete candidate and
   receipt agreement, exact saved tensor parity, correct archive/restart and
   normal exits. A mismatch triggers diagnosis before any tolerance is changed.
   Never compare CPU and GPU streams as though they were the same experiment.
   A GPU resource failure does not invalidate completed CPU engineering evidence.
   Preserve GPU hardware identity, placement, XLA, TF32 and verified memory
   growth in every child manifest. Do not reuse M26's CPU-only harness without
   adding an explicit tested GPU route or use its recorded CPU settings for
   a GPU claim.

4. Reconcile all attempts, run a terminal inventory/source/test audit, and
   refresh the next phase from member-specific outcomes and actual GPU costs.
   Only then freeze a larger coverage inventory with its exact operating
   characteristics and conservative total cost. If adequate evidence still
   exceeds the remaining ceiling, classify that confirmation as under-budgeted;
   continue affordable engineering work without weakening its criterion.

Use the public validation CLI for plan validation and execution after generating
the tested suites. The command form is
`python -m bayesfilter.testing.inference_validation plan <suite> --output <plan.json>`
followed by `run <suite> --output <fresh-root> --max-workers <bounded-count>`.
Record the resolved suite files, source root and exact commands before launch.
CPU commands use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, GPU hidden before
import and one numerical thread per worker. GPU commands require trusted access,
`TF_FORCE_GPU_ALLOW_GROWTH=true` before import, verified growth and XLA enabled.
No long run begins from this prose without its concrete validated suite.

## Review, decisions and stop rules

The primary engineering criteria are correct call-chain wiring, complete
retention, archive/restart integrity and bounded normal process exit. The
posterior development criterion is delivery of the unchanged requested
precision/health checks for each declared sibling, reported separately.
Promotion to a general policy still requires fresh coverage evidence with an
adequate denominator. Runtime, pilot coverage, ESS, R-hat and curvature help
explain allocation; none can affect tuning membership or tuning repair.

Skeptical review passed for bounded development, with large confirmation
explicitly deferred until pricing. The wrong baselines would be counting
siblings as replications, comparing unpaired estimators as a causal test,
relabeling the failed M26 stop using a larger retrospective window, or
transferring CPU evidence to GPU. The new sibling rule has a cost rationale
but no established mixing benefit; its earliest check is the two complete
rotated fits. GPU pricing might fail for compilation or resource reasons;
zero normal exits or incomplete native searches cannot price successful fits.
Changing defaults remains outside these development conclusions.

Source/checkpoint corruption, invalid target/Jacobian/reference, absent required
inputs for the affected cell, unavailable trusted GPU for GPU cells, or budget
exhaustion are continuation vetoes. A slow or failed sibling, warmup/precision
cap, normal statistical rejection or finite pilot failure is a repair trigger
and promotion veto, not rejection of the target or the tuning program.
Fix localized infrastructure failures within the unchanged total budget, using
fresh attempt directories and focused regressions.

## Remaining work after this tranche

Supplied-map tails still require an analytically checked bounded-residual map
under a supported frozen codec, with exact whitening as the positive control,
or predeclared sibling assessments of the existing maps. Target/Jacobian/score
and inverse-start checks must precede its downstream posterior tests. Model
precision costs do not transfer from latent Gaussian coordinates. Learned-map
quality remains a separate upstream training program.

Global exploration still needs declared mode occupancy/crossings and independent
same-target reference evidence. Local readiness does not prove that unknown
modes were found. Full-fit subtle-defect power remains unaffordable under its
existing 384-fit-by-at-least-17-experiments-per-arm design at measured rates;
any cheaper redesigned statistic needs inspected justification and fresh null
calibration. M22's frozen Gaussian null screen remains its own closed cell.

Exact MacroFinance integration still requires the newly qualified matching
target/source, data, prior, coordinates and independent uncertainty-bearing
reference. Do not rerun the consumed bootstrap run or manufacture that evidence
from synthetic tests. Current-source multi-model/GPU coverage and measured
maintenance remain incremental work; no blanket refactor or new transition
family is implied.

After each tranche: reconcile costs and workers, classify the outcome, repair
confirmed implementation errors, record the result, refresh the next concrete
design, review its assumptions, then execute within the same authorization.
