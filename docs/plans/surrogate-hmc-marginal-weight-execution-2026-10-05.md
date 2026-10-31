# Marginal importance weights: implementation and staged accuracy experiment

Date: 2026-10-05. Owner authorization: implement, review and execute the proposed
change on LGSSM, KSC SV, then range–bearing. This is a component investigation
for **testing surrogate HMC**: improve its deterministic likelihood/analytical
score computation before spending more time sampling. It is not a new HMC
convergence claim or a change of the observation model.

## Question and candidate

Does replacing ancestor-specific importance ratios by exact marginal mixture
ratios repair the observed range–bearing integration failure without materially
degrading LGSSM or the repository's seven-component KSC SV likelihood and score?
The derivation and limitations are in
`surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex` and its saved audit.

For incoming particles and weights `(a_j,w_j)`, Gaussian transitions `f_j`, and
the actual affine auxiliary-path LEDH maps `F_j(u)=B_j u+c_j`, define
`p(x)=sum_j w_j f_j(x)` and `q(x)=sum_j w_j f_j(F_j^{-1}x)/|det B_j|`.
The candidate uses `V_i=w_i*g(x_i)*p(x_i)/q(x_i)` for the existing one-child-per-
ancestor design. The baseline uses `w_i*g(x_i)*f_i(x_i)/q_i(x_i)`.
The outside `w_i` is essential for this sampling scheme. Both use the same full
UKF, flow, Contract E, OT, GenUT, pairwise correction and caps.

Implement an explicit optional `marginal_mixture` importance-weight policy;
`ancestor` remains the existing default. The candidate is a method extension,
not a claim that Li's original formula was algebraically wrong. Accumulate
`B,dB` in the existing flow loop; its auxiliary endpoint already gives `F_j(m_j)`
and its total tangent. Evaluate Gaussian mixtures and their responsibility-
weighted total derivatives with exact tiling. Do not differentiate through
autodiff in the runtime. Reject unsupported structural, custom-transition,
annealed or resampled-ancestry configurations rather than guessing their law.

## Evidence contract and research intent

| Item | Predeclared meaning |
|---|---|
| Primary question | Accuracy and non-deterioration of this change in the actual full filter |
| Comparator | Existing ancestor weights, identical data, designs, N, precision and controls |
| Expected failure | Marginal weights may still miss regions or amplify finite-cloud reset error; no variance dominance is proved for stratified/fixed designs |
| Primary metrics | Absolute total log-likelihood error; RMS parameter-scaled score error against an independent reference |
| Promotion criterion | All LG/KSC conditional cells pass paired non-inferiority for both metrics; M13 needs separately supported reference improvement |
| Promotion veto | Nonfinite/invalid output, derivative mismatch, failed oracle refinement, or material deterioration; heuristic dominance failure also vetoes general accuracy promotion |
| Continuation veto | Invalid reference/harness, broken mathematical assumptions, corrupted evidence, or exhausted budget; repair localized defects within budget |
| Repair trigger | A candidate accuracy failure triggers localization and the next planned model test; it does not stop the entire research direction |
| Explanatory only | ESS, pre-observation normalization, runtime, peak memory, pointwise errors and small development samples |
| Not concluded | Posterior correctness, HMC convergence, universal non-inferiority, unbiased deterministic log likelihood, or default readiness |

The simple adversaries are (1) the exact Kalman filter on LGSSM, (2) converged
one-dimensional forward integration on KSC, (3) no-flow bootstrap importance/
particle filtering, and (4) independent first-step quadrature on M13. These
are constructed/evaluated where their model permits them. Oracle distance is
the accuracy criterion; outperforming another complicated filter is insufficient.
Report each parameter situation separately rather than pooling it away.

## Requested order and scope

1. Mathematical/engineering checks: Gaussian mixture value and total tangent,
   composition of the actual affine map, identity-map limit, nonuniform weights,
   FP64 finite differences of the full recursive consumer, and old-default parity.
2. LGSSM `(d,m,T)=(1,1,2)`, then `(3,3,2)` and `(3,3,10)`. Scalar LG1 retains
   `F=phi,Q=1,R=.5,P0=1/(1-phi^2)`; evaluate phi `.4,.8,.9`. The 3D fixture uses
   coupled symmetric `A` (diagonal .6, adjacent .1), `F=a*A`, `Q=exp(2b)I`,
   `H=I,R=.5I`, stationary P0; evaluate `(a,b)=(1,0),(.8,-.5),(1.2,.3)`.
3. KSC SV `T=10,20`: `x_t=gamma*x_(t-1)+eta_t`, `eta~N(0,1)`, stationary P0,
   and the **same** seven-component mixture for `log(y^2)-x-2log(beta)`.
   Physical `(gamma,beta)=(.5,.5),(.8,.3),(.3,.8)`. Preserve the corrected beta
   derivative. Compare grid refinements and boundary extension before trusting it.
4. M13 range–bearing: the existing 4D state/2D wrapped observation fixture,
   first `T=1` (independent quadrature), then `T=20` (independent particle
   reference with replicated likelihood and Fisher-identity score uncertainty).
   Include nominal and the previously localized low-Q point. A noisy reference
   can leave the full-horizon score result inconclusive; never call it exact.

Use separate data seeds for calibration, validation and untouched claim;
record exact observations. Same data across methods within a paired comparison.
Random-design replication estimates integration uncertainty conditional on each
saved dataset, not generalization over every possible dataset. Start N=128 for
engineering/development and N=256 for the scope comparison; M13 includes an
N=4096 reproduction if budget permits. N changes are distinct scopes. Existing
controls only seed calibration. Freeze the selected common controls on disjoint
calibration/validation, write a scope artifact, then evaluate untouched claim
designs. This estimates the effect of changing weights at fixed controls, not
each method's independently optimized achievable performance. No default or
HMC admission artifact will be issued by this diagnostic campaign.

## Statistical judgement

Use 32 paired random designs per conditional claim cell. Analyze differences
`D=error_marginal-error_ancestor`, so positive means deterioration. A nonsignificant
test is **not** evidence of non-inferiority. The log-likelihood error margin is
0.05 nat in total (a 5.1% multiplicative likelihood change); the scaled-score
error margin is 0.05. Score scales are a local physical displacement of .1 for
each tested parameter. Thus the score margin measures first-order log-likelihood
error across a small parameter displacement; it is not an HMC convergence test.
These are explicit materiality hypotheses, not previously owner-approved universal
tolerances. Report raw errors so they can be reassessed without rerunning.

Use paired t intervals with their normal-mean approximation disclosed, plus a
paired bootstrap sensitivity check. Correct the one-sided family error .05 over
the 15 LG/KSC conditional cells, two metrics, and up to three looks (32,64,96
replications). Continue to 64/96 only when the first interval is inconclusive
and budget permits. Non-inferiority requires the simultaneous upper bound below
the margin. Lower bound above zero reports statistically supported deterioration;
lower bound above the margin reports material deterioration. Otherwise say
inconclusive. Disagreement with bootstrap or severe outlier dependence makes
the statistical verdict inconclusive. Never remove a failed replicate.

Reference numerical error must be well below the margins (.001 nat and .001 in
scaled score for LG/KSC). Independent Kalman and KSC grid scores may use AD as
reference checks only. Full-engine analytical score FD checks use FP64 and
multiple step sizes, including the parameter-dependent initial distribution.
FP32/TF32 is the candidate default; compare selected FP64 reference evaluations
to localize precision failures. No assertion of equal runtime: marginal weights
add quadratic work. Report compile cost, steady time and memory explicitly.

## Defaults, assumptions and early checks

| Choice/provenance | Justification / risk | Early check / status |
|---|---|---|
| Gaussian full-rank transition | Exact mixture law; singular/custom law invalidates it | Static rejection plus checked Cholesky; assumption |
| Auxiliary flow frozen conditional on ancestors | Makes each proposal affine; dependence on its own innovation would change the Jacobian | Reconstruct actual flowed children; checked prerequisite |
| Existing complete correction controls | Controlled comparison, may be poorly tuned at new horizons | Separate scope calibration/validation, never label transferred settings tuned |
| N=128/256; optional M13=4096 | Bounded development and replication; low N can leave approximation error | N refinement and bootstrap adversary; hypothesis |
| FP32/TF32/XLA GPU | Owner default; can obscure cancellation and ill-conditioning | FP64/FD and validity flags; required default with explicit reference exceptions |
| Fixed data, independent design seeds | Paired uncertainty for integration error; not dataset-population inference | Preserve seed/data partitions; explicit scope |
| .05 margins / .1 score scales | Local likelihood materiality; not posterior-effect proof | Raw errors and sensitivity retained; hypothesis |
| Finite KSC grid, M13 PF | Independent tractable reference; discretization/truncation/MC error | Grid/boundary refinement and replicated PF ladder; must pass reference checks |

## Execution, budget and phase refresh

New authorization is this request; prior HMC deadlines and the old 1,575.9-second
diagnostic balance are historical. Bound this campaign to six elapsed GPU hours
including compilation/retries and 30 CPU-reference minutes, with at most two
localized repair attempts per failed implementation check. No paid compute,
package/environment changes, external review or HMC sampling. Stop launching
work when it cannot fit the remaining budget. Unique root:
`docs/plans/artifacts/surrogate-hmc-marginal-weight-20261005-01/`.

Executable driver: `scripts/run_surrogate_hmc_marginal_weight_campaign.py`.
Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; escalated GPU
access, verified memory growth, FP32/TF32 and XLA. Save logs, JSON observations,
seeds, controls, source hashes, git/dirty provenance, environment, timing,
diagnostics and phase decisions. Subprocess workers isolate compiled graphs;
resume skips only completed matching phases. At each phase boundary update the
checkpoint and `next-phase.json` with evidence, budget and next command. A failed
candidate blocks promotion but continues the planned discriminating phases.

Exact launch after focused checks:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_marginal_weight_campaign.py --output docs/plans/artifacts/surrogate-hmc-marginal-weight-20261005-01 --budget-seconds 21600`.

## Skeptical audit before implementation/execution

**Pass with explicit restrictions.** The comparison changes the real full-engine
weights; it does not test an unused helper. The baseline is current dirty-tree
code, preserved before editing, not invalid pre-August results. Keeping the old
default unchanged is a regression check, not evidence that the candidate is
non-inferior. Original ancestor weights are valid, but inaccurate integration
is possible. Marginal weights have no proved fixed-design variance advantage.
No inference is drawn from a nonsignificant p-value alone. KSC at T=10/20 needs
a new converged reference; the old T=2 enumeration cannot answer it. M13 needs
both first-step and recursive tests. Statistical uncertainty and numerical
validity remain separate. Runtime cost cannot be hidden by an accuracy result.

Pre-mortem: a pass can mislead if both filters share biased resets, the reference
is too coarse, selected seeds look favorable, or the actual score omits map/
initial-law dependence. Independent references, untouched seeds, scope records,
and full-consumer FD are the early discriminators. A failure can be low particle
count, mistuned scope controls, FP32 instability or the proposed estimator itself;
preserve all outcomes and distinguish these explanations before interpretation.

## Executable preflight audit (before first launch)

Reviewed the full call chain and driver. Focused five-test CPU diagnostic suite
passes (GPU intentionally hidden). Full-consumer FP64 finite differences and
saved-source old-default comparisons precede accuracy runs. Tiny N=16 is used
only for these derivative/regression checks, not statistical accuracy claims.
Scope comparisons use N=256. Calibration selects between the two declared
complete control hypotheses using calibration data only; validation is a finite
output veto. This narrow two-hypothesis calibration supports a controlled
comparison, not optimal tuning/default readiness. Student intervals account for
both tails as well as metrics, cells and planned looks. Bootstrap sensitivity
uses 50,000 resamples and is approximate, especially at corrected extreme tails.

The GPU UUID selects the idle RTX 4080 SUPER explicitly: CUDA ordinal ordering
differs from nvidia-smi ordering on this host. All workers verify memory growth.
The old-code worker imports the saved pre-change package, while using the exact
same external diagnostic driver and input generation as the new-code worker.
No read-only review worker or external service is used.

## Localized reporting repair and continuation

The first root preserves all 15 LG/KSC non-inferiority passes and both
engineering passes. M13 exposed a serialization defect on an invalid trial
setting. Preserve invalid values as explicit diagnostic strings, reject them
before selection, and record candidate-wide rejection without abandoning later
models. The pure strict-JSON regression passes. Runtime mathematics and
statistical criteria are unchanged. Continue M13 in fresh root ending `-02`
with 20,775.45 of the original 21,600 seconds remaining. Completed earlier
results are referenced from root `-01`, never overwritten.


## Final master-program audit — 2026-10-05

The execution is complete. A post-run audit found and repaired a provenance
edge case in the driver: a fresh run without a saved baseline could otherwise
allow the current checkout to serve as both comparator sides. The driver now
fails before TensorFlow import when the baseline is missing or is the current
checkout; it requires the key package files, hashes the supplied Python/shared
library snapshot, verifies the old-code import origin, binds particles/GPU/phases
and baseline identity on resume, requires `regression-before` before a direct
`regression-after`, and records `pid=null` after a worker exits. Six focused
CPU tests cover these guards. This repair changes no numerical result or
scientific criterion.

The skeptical audit found no remaining issue that invalidates the completed
experiment. The remaining M13 error is a scientific follow-up question, not a
failed execution step.
