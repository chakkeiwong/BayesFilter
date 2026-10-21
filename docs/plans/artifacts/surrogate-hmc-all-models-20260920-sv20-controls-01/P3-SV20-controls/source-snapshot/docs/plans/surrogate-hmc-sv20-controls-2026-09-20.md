# SV20 protection calibration and inference-cost screen — 2026-09-20

The main question remains whether an imperfect full canonical DCTR proposal
field supports endpoint-corrected HMC parameter inference. This preparation
stage evaluates numerical protections on actual SV20 and asks whether even a
minimal B-arm inference run fits its existing allocation. It does not decide
posterior equivalence or rank samplers.

Use the qualified actual-SV consumer, T20, N=K=1024, stationary x0, raw-return
weight density, Uniform(.1,.9) prior and probit chart. GPU1 FP32/TF32 DCTR,
FP64 numerical reference/integrator, XLA and verified memory growth. Read only
the frozen calibration and validation datasets under
`artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01/P3-SV20-reference`.
Claim data must not affect this selection.

## Evidence contract and default audit

Compare full-route protection multipliers 1, .1, .01, .001, via the existing
`protection_controls` factory: ridge 1e-5*m, LM damping 1e-3*m, scale floor
1e-6*m, trust radius .1/m, pairwise cap 2/m and coordinate cap 16/m. This
geometric curve is a calibration hypothesis transferred from the existing
campaign; its transfer grants no SV validity. The principled criterion is
non-harm to healthy value/force/filter/reset moments together with flagged,
bounded behavior on stressed calculations. It is not primary-metric tuning.
The largest protection on the first stable adjacent pair is selected only if
both disjoint partitions pass. Failure triggers scope-specific repair, not
rejection of surrogate HMC.

Nominal physical positions (.3,.3), (.5,.5), (.7,.7), (.8,.2) span persistence
and observation scale. Stress positions are the four combinations of .1008 and
.8992, one thousandth of prior width inside its boundary. These convenience
positions expose sensitivity at low/high persistence and return scale; they do
not establish continuous-prior coverage. All nominal rows must have healthy
original value/force status and finite bounded filtering/reset moments. For
each adjacent pair, maximum component change divided by 1+abs(weaker) must be
<=.001. This inherited tolerance is a non-harm hypothesis in fixed model units,
not a posterior approximation bound. Under stronger protection, every stress
row healthy under the weaker setting must stay healthy; other failed rows must
carry original failure codes. A gross absolute 1e6 rejection screen is a
debugging bound only; no output is clipped. Full tensors/statuses are retained.

All other controls (ten flow substeps, finite transport iterations, strengths)
remain frozen warm-start hypotheses. The curve cannot establish their adequacy.
Reference values/gradients explain sensitivity but never select controls.
Acceptance and force error are explanatory only. Selected controls are eligible
only for the imperfect-proposal experiment, never exact-score admission or a
default. This is the prompt Class C evaluation owed by qualification's warm
start; it changes neither canonical algorithm nor endpoint law.

If a pair is selected, measure a complete four-chain, one-leap endpoint-corrected
transition with step .02 and identity mass: one compile plus four steady calls.
Use the minimum observed wall and CPU costs times3250 transitions times1.5.
The 3250 count covers only250 fresh warmup+1000 verification+2000 retention for
one B arm, excluding nomination, repairs, other arms and preparation. This is
an optimistic feasibility screen, never a complete forecast or tuning artifact.
Failing even the full original7200GPU/1800CPU allocation defers inference;
passing requires a complete forecast using the actual remaining family budget.
Invalid transition, fallback or divergence prevents a cost conclusion.

Source/data mismatch, wrong law, invalid artifact and resource exhaustion are
continuation vetoes. A failed protection pair is a repair trigger. Failed cost
screen means UNDER_BUDGETED for this allocation, not a scientific rejection.
No force/reference discrepancy or D/R failure alone may veto continuation.
Heuristic comparisons and MCSE calibration remain outstanding, so no promotion
is possible here. Later inference must construct prior-only, bootstrap-PF and
moment-filter adversaries with conditional low/high-scale/persistence results.

## Budget, audit and artifacts

Entry use: SV family243.377750GPU /322.792706CPU seconds, including tests;
whole campaign8520.168984GPU /9010.816930CPU seconds. Bound this stage across
one execution and one infrastructure retry to600GPU/600CPU seconds, plus
60CPU seconds of focused tests. These fit the original family7200GPU/1800CPU
and campaign226800GPU/54000CPU caps. CPU cost includes the GPU worker. No
budget reallocation or extra retained looks is authorized here. Max GPU peak
allocated memory4GiB; stop on exceeded measured resources.

Prelaunch review found that separate stage ceilings alone did not enforce the
combined SV20 ceiling. The master now checks cumulative family spending,
including dedicated test roots, prior mixed-run SV jobs, parent CPU overhead
and current-run jobs before each SV launch. Regression checks cover cross-stage
exhaustion and mixed-run accounting. The two focused batches pass65 and53 tests;
their combined41.850499CPU seconds fit the60-second test allowance. Actual
prelaunch family remaining:6956.622250GPU/1435.356795CPU seconds. Campaign
charged:8520.168984GPU/9052.667429CPU seconds. The600/600 reservation fits.

Pre-mortem and skeptical audit: the actual consumer passed14GPUchecks and56
tests. No KSC substitution, omitted stationary derivative, logistic chart or
claim-data tuning is present. Reference error and control non-harm are separate
from downstream inference validity. A cost screen can understate expensive
future trajectories, so passing cannot launch inference. Weak or failed controls
need fresh scoped repair; their failure does not refute the HMC mechanism.
The plan's baseline, diagnostic roles, unchanged budget and stop conditions
answer this preparation question; audit passes for bounded execution.

Run through master stage `P3-SV20-controls --scope SV20 --reference-root <above>
--qualification-root <sv20-consumer-01/P3-SV20-qualify> --output <fresh>` with
the tftwogpu interpreter, GPU1 UUID, growth and thread settings from the consumer
plan. Fresh root: `artifacts/surrogate-hmc-all-models-20260920-sv20-controls-01`.
Save exact commands/environment/seeds, source snapshots, per-partition curves,
original statuses, selection, transition traces, cost screen, accounting and
result. Only the shared public tuner can issue a later HMC tuning artifact.
