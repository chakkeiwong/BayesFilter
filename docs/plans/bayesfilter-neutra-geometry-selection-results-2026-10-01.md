# NeuTra geometry and checkpoint selection: execution findings

Status: audit, repairs, bounded execution and terminal review completed.
Rare-region sampling remains unresolved for both mixture seeds. The v3 screen
passed one case, but its rare-event tolerance was too permissive; the repaired
v4 check rejects those saved draws. No production-readiness claim is supported.
Plan: `bayesfilter-neutra-geometry-selection-plan-2026-10-01.md`.
Cycle: `artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/geometry-selection-20261001-r1/`.

## Checked mathematics and code

For a frozen invertible map x=T(z), the code computes

\[
 \ell_z(z)=\log p(T(z))+\log|\det J_T(z)|,\qquad
 r(z)=J_T(z)^T\nabla_x\log p(T(z))+
       \nabla_z\log|\det J_T(z)|+z.
\]

The target is the exact benchmark density, not a density approximation from
the flow. The manual chain rule in `neutra_transport_core.py` and
`pullback_gaussianization_diagnostic` includes both score terms. Independent
autodiff agrees with it at all 36,000 probed rows from nine saved maps. Maximum
scaled discrepancy is 9.31e-14; the log-Jacobian score discrepancy is 3.01e-15.
At 576 selected ordinary/extreme rows, central differences at the recorded
four step sizes give maximum best-step scaled discrepancy 7.55e-9. The largest
inverse/forward roundtrip error is 9.61e-15. These finite checks do not prove
correctness at every possible input, but the checked large residuals are not
explained by incorrect differentiation.

There are two 1000-row Gaussian banks per map, with common rows within each
training seed, and two disjoint 1000-row subsets of the existing development
posterior reference, inverted through each map. Full rows, physical locations,
residuals, score terms, Jacobians, singular values and finite differences are
saved. The standard 1000-point checks remain separate from posterior-reference
probes. No final-reference values entered this diagnosis.

The code also computes the standard reparameterized reverse-KL gradient
correctly at the symbolic level: `reverse_kl_evaluate` differentiates
`-log p(T(z))-logdet` using the supplied exact physical score as a stopped
gradient carrier, with the explicit logdet derivative. This cycle did not
retrain or revalidate every parameter derivative. The earlier directional
checks and the current input-score checks have different scopes.

## Why good KL does not imply whitening

An explicit counterexample separates objective value from derivative quality.
Let phi be a standard-normal density and

\[
 p_k(z)=Z_k^{-1}\phi(z)\exp\{a_k\sin(kz_1)\},\quad
 a_k=k^{-1/2},\quad
 Z_k=E_\phi[\exp\{a_k\sin(kZ_1)\}].
\]

Since exp(-a_k)<=Z_k<=exp(a_k), the log density ratio has absolute value
at most 2a_k. Both KL directions are therefore at most 2a_k and tend to zero.
However,

\[
 \nabla\log p_k(z)+z=a_k k\cos(kz_1)e_1,\qquad
 E_\phi\|\nabla\log p_k(Z)+Z\|^2
   =\frac{k}{2}(1+e^{-2k^2})\longrightarrow\infty.
\]

The last equality uses cos²(t)=(1+cos(2t))/2 and the Gaussian characteristic
function. Additional regularity would be needed to infer score quality from
KL. This example establishes that logical gap; it is not a model of the actual
learned error. Forward KL can assess density fit, and RKL can train a useful
transport, without either loss alone certifying HMC geometry.

A large Jacobian is also insufficient evidence of a defective transport.
For the one-dimensional mixture marginal
p_1(x)=phi(x+5)/3+2phi(x-5)/3, its exact increasing quantile transport obeys

\[
 F_1(T(z))=\Phi(z),\qquad
 T'(z)=\frac{\phi(z)}{p_1(T(z))}.
\]

At x=0, p_1(0)=phi(5)=1.48672e-6,
F_1(0)=0.3333334289 and z=-0.4307270, giving T'(z)=244565.17.
The exact transformed score still equals -z because the two chain-rule terms
cancel appropriately. This is an exact marginal example, not a lower bound
on every two-dimensional transport's Jacobian.

The inspected learned maps do leave large uncancelled terms. For seed11's
RKL2048 map, the largest residual in Gaussian bank 0 is 1506.46 at
x=(-1.54474,2.81980). The target-score pullback norm is 1596.21, whereas the
logdet-score norm is 90.10; the largest Jacobian singular value is 360.24.
In posterior bank 0, the largest residual is 428.58 at
x=(-1.68919,-0.41292), with term norms 625.50 and 197.32. Thus the issue also
appears at these posterior reference points. Conditional tanh scale caps do
not bound conditioner derivatives or off-diagonal shear. Their saturation
summaries cannot establish small r(z).

## Saved-map geometry

Each entry contains the two recorded banks. These are descriptive observations,
not a statistically supported checkpoint ranking. The posterior banks for a
target use common physical points; the Gaussian banks use common latent points
within a training seed. Extreme quantiles vary substantially between banks.

| Map | Gaussian residual q99 | Gaussian maximum | Posterior-reference maximum |
|---|---|---|---|
| Mixture11 warm |20.52 /137.09|817.59 /1661.16|536.02 /61.19|
| Mixture11 RKL256 |15.82 /155.68|708.29 /1461.19|359.82 /51.31|
| Mixture11 RKL1024 |54.08 /22.05|1447.83 /1304.76|410.41 /43.73|
| Mixture11 RKL2048 |35.31 /20.82|1506.46 /1392.03|428.58 /61.46|
| Mixture37 warm |99.09 /327.47|783.50 /754.03|498.42 /143.16|
| Mixture37 RKL256 |129.65 /141.99|759.94 /807.67|480.68 /152.40|
| Mixture37 RKL1024 |93.41 /192.03|821.68 /831.51|481.05 /133.60|
| Mixture37 RKL2048 |227.12 /38.58|947.28 /781.31|479.70 /160.88|
| Wiggle23 selected |35.98 /36.22|75.17 /60.45|47.05 /34.82|

## Implemented repair

Refinement now emits `checkpoint-candidates.json`, preserving every
shape-passing, numerically valid RKL checkpoint and the warm fallback, each
with a map-bound 1000-point report. The consumer tests them in declared order
RKL2048, RKL1024, RKL256, warm, with fresh tuning per map and up to three
verified members per map. It selects on actual sequential HMC diagnostics,
then opens the final reference once. Forward KL is retained as a compatibility
nominee and reported loss; it no longer determines the sole downstream map.

The explicit profile `moments_regions_shape_v3` extends the mixture screen
from five to 18 quantities. It adds nine fixed x0 CDF events, the valley event
|x0|<2 and three second-coordinate CDF events, with the inverse warp where
applicable. They use the existing typed binary-probability R-hat/ESS treatment.
Absent events are not given invented convergence or precision values.
The old profile remains available for historical readers and existing scopes.

The regression counterexample uses symmetric laws at ±sqrt(26) and at
{-sqrt(51),-1,1,sqrt(51)}, with equal weights and matched second-coordinate
patterns. Both have identical first and second moments and right-half-space
probability. Their valley masses are 0 and .5. The legacy feature means are
identical; the new valley feature separates them. This is a precise limitation
of the old observables, not a claim that its earlier reported values were wrong.

## Executed downstream results and the validation defect

| Case | Maps / tested members | v3 result | Retained result |
|---|---|---|---|
| Mixture11 |4 /12|All failed; final reference unopened|Only one member reached retained sampling;0 valley visits in40000 draws|
| Mixture37 |2 /5|RKL1024, epsilon.0625, L18 passed the declared v3 rule|7000 warmup and6000 retained/chain;2 valley visits in24000 draws, both in chain4|

Eleven of seed11's twelve members had zero valley visits in their final warm-up
window; the other entered retained sampling but accumulated none there. Its
right-half-space frequencies in the retained run were .6706,.6678,.6633,.6580,
showing how plausible mode weights can coexist with missing valley evidence.
Seed37's v3 pass had maximum R-hat1.003143, minimum bulk/information-tail
ESS1310.53 and maximum MCSE/SD.027695. These numbers describe the declared
observables under that rule; they do not rescue its rare-event coverage claim.

The old forward-KL selector would have kept seed37's warm map (development
FKL.016886) and discarded RKL1024 (FKL.027753). The repaired shortlist actually
tested that RKL map downstream. This verifies the selection repair, but does
not establish that the RKL map is superior or sufficiently accurate.

The terminal review inspected the actual draws and found why the v3 pass was
too weak. The valley estimate was .0000833333, versus .0012207031 in its fresh
reference and the exact mass .0013498980. The error against that reference was
.0011373698, while the allowed tolerance was .0018880080:

\[
 4\sqrt{\widehat{\operatorname{MCSE}}^2+
          \widehat{\operatorname{Var}}_{\mathrm{ref}}/32768}
 + .03\sqrt{\widehat{\operatorname{Var}}_{\mathrm{ref}}}.
\]

For a rare Bernoulli probability p, the additive term is about .03 sqrt(p),
which can exceed p. With zero chain MCSE, even a zero estimate passes whenever
approximately p<=sqrt(p(1-p))(.03+4/sqrt(32768)). Also, MCSE/SD<=.03 allows
relative MCSE .03 sqrt((1-p)/p), about .816 at the true valley mass. Thus the
rule cannot be interpreted as a reliable relative-probability requirement.
The code computed the v3 rule as specified; its ability to verify rare-region
coverage was inadequate. The initial plan review missed this inferential gap.

The prospective profile `moments_regions_shape_v4` repairs two points. It
requires both outcomes in every retained chain and compares binary event
probabilities using the four combined standard errors without the additive
reference-SD allowance. It leaves the continuous-moment allowances intact and
does not require per-chain rare events in the short warm-up window. On the
saved seed37 draws, the event-observation check fails and the revised reference
tolerance is .0008404911, below the .0011373698 error. This is a retrospective
diagnostic rejection, not a fresh v4 confirmation. The historical v3 artifact
and all failed members remain unchanged.

The stronger rule is still an operational finite-data screen. Requiring an
event in each chain does not prove mixing or precision. Four estimated standard
errors do not supply exact small-count, simultaneous or sequential coverage.
No current result establishes accurate relative valley probabilities.

The failed rare-event check is a required validation failure, not proof of a
particular long-run sampling bias. Dependence may make a rare-region estimate
unreliable even when common observables appear stable. The two visits provide
insufficient information to estimate that dependence convincingly.

## Exact-sampling controls and engineering review

The actual v3 assessment passed63/64 independent exact-mixture warm-up windows
and64/64 retained assessments. The sole failure was a zero-event warm-up
window. After the v4 repair, the actual assessment, per-chain callback and
independent-reference comparison passed53/64 replications at2000 retained
draws/chain (Wilson95% interval .7179–.9012) and63/64 at4000/chain
(.9167–.9972). At4000, all64 passed posterior diagnostics and one failed the
independent-reference comparison. These are finite iid controls; they do not
calibrate the diagnostics under the correlated HMC runs.

48 distinct focused regressions passed across the recorded suites. They include
manual-score corruption detection, finite differences, equal-moment/wrong-valley
counterexamples, inverse-warp events, preserving RKL maps despite worse FKL,
map/probe identity, actual consumer holdout isolation and v4 callback wiring,
the observed rare-probability false pass, and preserving numerical vetoes on
resume. The result is an engineering repair, not evidence of learned-map quality.

The terminal review checked631 frozen source files,109 input hashes,18 complete
standard1000-point probes and17 archived member outcomes with no integrity
errors. Numerical GPU workers used GPU1, FP64 analytic targets, batch-native
TF/XLA, fixed identity mass, and verified memory growth; no new training ran.
The actual fixed-transport HMC adapter's value/score composition was inspected:
it adds the logdet to the physical target value and both chain-rule terms to
the score. The large residual findings do not establish a score-implementation
bug. GPU0/2 belonged to other work. No packages, commits or pushes were made.

`source/` preserves the numerical v3 execution. `source-v4/` preserves the
localized prospective validation repair and its CPU reference controls; it
records exactly which source files changed. The new v4 checks were exercised
on exact iid controls and saved draws, not on a fresh HMC campaign.

The numerical campaign used11 GPU workers and2 CPU reference workers; the
two additional exact-iid controls and terminal tensor inspection ran CPU-only.
No infrastructure retries were needed. Charged work totals1209.986 GPU
process-seconds (.336107 hours) and1778.424 CPU core-seconds (.494007 hours).
Remaining shared conservative allocation is81480.811 GPU process-seconds
(22.633559 hours) and170544.791 CPU core-seconds (47.373553 hours).
This respects the4/8-hour cycle sub-cap. Routine regression suites consumed
98.874 summed wall-seconds including repeated affected checks; their individual
CPU usage was not metered and is recorded separately from research workers.
Exact commands, seeds, environments, source/input hashes, device evidence,
allocator readings and times are preserved in attempt manifests, control result
files, the shared ledger and `checks/`.

## Decisions and remaining work

| Decision | Primary criterion | Veto/required diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept engineering repair |48 focused tests pass|Source/input and complete-probe checks pass|Finite tests do not exhaust all inputs|Use shortlist and v4 in the next reviewed run|Production readiness|
| Accept checked score identities |36,000 rows and576 FD locations pass|No detected derivative or roundtrip failure|Unexamined inputs|Investigate residual geometry and actual kernel dynamics|A global derivative theorem|
| Reject seed11's tested candidates for wider posterior use |All12 members fail v3|Valley observations/precision missing|Other epsilon/L pairs and maps remain untested|Local valley transition/energy-resolution diagnostic|NeuTra impossibility|
| Withhold seed37 rare-region qualification |v3 pass fails retrospective v4|Three empty chains; stricter probability agreement fails|Sparse-event MCSE and mixing|Same local diagnostic using the saved RKL1024 map|Accurate valley mass or general mode coverage|

| Inference status | Finding |
|---|---|
| Hard veto screen |No detected nonfinite geometry or derivative defect; posterior-required event/precision checks remain unmet|
| Statistically supported ranking |None; adaptive checkpoint/member selection is not a method comparison|
| Descriptive-only differences |All KLs, residual tails, HMC timing and observed map/member differences|
| Default readiness |Not established; v4 is an explicit benchmark profile|
| Next evidence needed |Discriminating local transition/energy checks in the valley, followed by a fresh frozen-candidate posterior run if a repair is justified|

The failures invalidate candidate qualification, not the target, exact score
identity, or NeuTra as a research direction. A plausible explanation is that
globally acceptance-tuned trajectories do not resolve the narrow latent regions
corresponding to the physical valley. The checked large derivatives and near-zero
visits support investigating that explanation, but do not prove causality.
The next smallest discriminating experiment should hold the map fixed, include
valley and ordinary posterior starts, and measure local HMC energy/transition
behavior as step size changes. Step sizes should be justified by measured local
curvature and freshly verified through the public tuner. A training-objective
change is not yet justified by this audit alone.

Post-run red team: v4 could still pass inadequate rare-probability estimates
because sparse-event autocorrelation/MCSE estimates can be misleading. A fresh
well-mixed run with stable per-chain rare-event counts and independent-reference
agreement would weaken the current sampling diagnosis. The weakest evidence is
the finite selected set of checkpoints/kernels and the absence of a direct
local-dynamics experiment. No untested initializer, architecture, objective,
or epsilon value is promoted as the remedy.
