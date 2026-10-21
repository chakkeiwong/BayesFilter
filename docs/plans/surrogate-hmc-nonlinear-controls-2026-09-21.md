# M10 scoped DCTR controls and inference feasibility — 2026-09-21

The question remains whether an imperfect full canonical DCTR position-only
field supports endpoint-corrected surrogate HMC. This stage selects numerical
protections for the actual M10 T2 noise-scale scope and measures the cost of
the subsequent B/R and C/D tests. It does not test posterior equivalence.

## Evidence contract and intent

Use the same nonlinear law, independent Normal(0,log(2)^2) log-noise prior,
N1024 exact-divisor transport, Contract E reset, analytical canonical score,
FP32/TF32 force and FP64 integration. Preserve the shared canonical call chain.
The independent R from posterior attempt05 remains frozen. The old local513
reference endpoint is mechanics-only; it can supply an optimistic cost screen
but cannot authorize B/R inference. C uses the selected frozen DCTR scalar as
its endpoint and needs an independent D reference before a C/D claim.

On the frozen calibration and validation data, separately, evaluate protection
multipliers1,.1,.01,.001 through the shared repository selector. Compare each
adjacent pair's log target, analytical score, filtering moments and reset
moments. Select the first pair passing both partitions: healthy nominal rows,
maximum relative change |stronger-weaker|/(1+|weaker|)<=1e-3, flagged stress
failures, and no new failure under stronger protection. The stronger setting
is selected. This is a non-harm criterion; reference error, acceptance and
runtime cannot select controls. No stable pair triggers a fresh scoped repair,
not rejection of the surrogate-HMC research direction.

Nominal coordinates are(0,0),(.15,-.1),(-.15,.1),(-.3,-.3), on nominal
observations. Stress coordinates are the four corners of +/-2log(2), on the
separately frozen stress observations. Calibration and validation each retain
both original dataset hashes in a wrapper carrying their actual partition and
attempt1. Claim observations never enter selection or cost measurements.

Purely explanatory diagnostics include reference-force disagreement, stability
outside these probes, and the measured timings. Consumer mechanics, finite
states, explicit status codes, unchanged frozen inputs, bounded resources and
source/data integrity can veto execution. Failed D/R cannot veto C/D. Missing
full-prior B endpoint or independent D blocks only its dependent comparison.

The heuristic adversaries for eventual posterior tests are direct independent
reference quadrature, prior independence MH, fitted-Gaussian independence MH
and zero-force HMC. Their rationale is respectively an available small-model
oracle, a cheap prior proposal, covariance adaptation without DCTR and the
position-field limit. Evaluate centrally and in both marginal tails on each
declared data regime. They remain NOT_RUN in this preparation stage and cannot
be claimed passed by calibration.

## Defaults, audit and pre-mortem

The multiplier ladder and1e-3 tolerance come from the shared protection selector,
not M10 evidence; they are hypotheses until this target's non-harm curves pass.
The fixed bank comes from M10 mechanics; replay, row independence and direct
canonical parity must still pass after extending accepted tuning partitions.
The nominal coordinates are checked mechanics points; stress corners test
small/large process and observation noise. A small grid may miss pathologies,
so its success grants only scoped experimental controls. N1024 and T2 are this
explicit limited scope, with no transfer to larger horizons or other models.

Adding filter/reset moment diagnostics is observability only. Extend the
fixture to accept calibration/validation nominal/stress data with original
roles recorded; retain the local mechanics consumer's nominal-calibration
restriction. Requalify the changed consumer source. Old source snapshots remain
historical; a new passing report must supersede them before current-source reuse.

Cost checks use complete four-chain, one-leapfrog transitions, one compilation
and four steady evaluations for B and C. The optimistic lower screen is
1.5*(compilation+3250*minimum steady cost) per arm, covering only250 warmup,
1000 verification and2000 retained transitions. It omits nomination, references,
heuristics, analysis and extra leapfrogs, so fitting the screen cannot authorize
inference; exceeding the remaining family allocation defers that arm. A full
forecast is required before any tuning/retained launch. Invalid transitions
invalidate timings rather than supply a cheap forecast.

Skeptical audit: this stage tests protection non-harm and feasibility, not score
accuracy or posterior validity. R preparation must not displace the actual HMC
question; an unaffordable B still leaves C/D independently meaningful. The
largest misleading-pass risk is treating sparse control probes or a local
endpoint as full-prior evidence; explicit statuses forbid both conclusions.

## Execution and budget

Fresh roots use surrogate-hmc-all-models-20260921-m10-controls-* and a second
consumer qualification root. Total new allocation:30 CPU seconds for focused
tests,90 CPU/90 GPU seconds for consumer requalification,350 CPU/450 GPU seconds
for one control selection and cost screen. All count within the unchanged
M10 family1800 CPU/7200 GPU ceiling; before this stage1061.166598019 CPU and
6793.015465505 GPU seconds remain. No additional inference attempt is consumed.
Record failed work and use fresh output roots for any localized retry under
these same ceilings. Stop on a scientific validity failure or budget exhaustion.

Use /home/chakwong/anaconda3/envs/tftwogpu/bin/python. CPU tests deliberately
hide GPU. GPU jobs use escalated access to GPU1 UUID
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, memory growth verified before device
initialization, TF32 enabled and XLA. Cap peak allocator memory at4GiB.
Save exact commands, seeds, model/data/source hashes, manifests, logs, measured
wall and CPU charges, moment/status traces, selection and forecasts. Result:
docs/plans/surrogate-hmc-nonlinear-controls-result-2026-09-21.md.

September21 closeout allocation: after the control worker completed, move20
unused CPU seconds from its350-second allowance to focused orchestration tests
and checked public-master reuse. Total stage and family ceilings are unchanged.
The completed worker's manifest and source snapshot preserve its original plan.
Audit before execution: reuse must verify numerical sources, original tuning
data roles, measured selected controls and non-harm decisions; mixed-master
control costs must be counted once. No GPU job, tuning attempt or new posterior
look is authorized by this closeout. Expected public-master status remains
INCOMPLETE_IMPLEMENTATION with all four M10 comparisons NOT_RUN.
