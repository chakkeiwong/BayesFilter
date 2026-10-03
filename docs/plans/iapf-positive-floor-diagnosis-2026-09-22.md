# Positive-floor strength in the fitted iAPF consumer

Status: COMPLETE. Formula checks and all 114 frozen-guide replays pass; positive
floor suppression is present, but lowering the floor does not repair the
conditional heuristic losses. No default changes. See the result and manifest
under `artifacts/iapf-positive-floor-20260922-01/`.
Parent: `younis-kdm-score-master-program-2026-09-14.md`.
Prior result: `artifacts/iapf-initialization-isolation-20260922-01/result.md`.

## Question, source and mathematical target

Does the local floor suppress the Gaussian proposal, even with a valid fitted
guide? Separate this from QR initialization, optimizer stopping and adaptive
particle-count failures. GJL (2017), Section 5.1, equations 15–16, adds a positive
constant c(N,m,V) to the Gaussian to protect importance-weight tails. It does
not specify the numerical rule in the inspected text. Local paper copy:
`.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt`,
lines 907–949. The frozen independent R reconstruction uses a chi-square tail
rule (`reference_iapf_paper.R`, lines 308–310). Neither local rule is established
as an original-author choice.

The actual TF fitter sets epsilon=rho*N(c;c,V), rho=.01. The shared consumer
`fitted_twist_tf.py::normalizer` integrates the guide against N(x;m,Q), giving
p_G=N(c;m,Q+V)/(N(c;m,Q+V)+epsilon). At m=c and Q=qV this is
1/[1+rho*(1+q)^(d/2)]. Away from the center, add exp(r_S^2/2) to that denominator
factor, where r_S^2=(c-m)'(Q+V)^(-1)(c-m). This is a convolution probability,
distinct from the pointwise fraction of Gaussian mass above epsilon. Reducing
the floor may restore twisting while worsening tail protection; it is not
automatically a repair for likelihood accuracy.

## Evidence contract and research-intent ledger

- Primary question: establish whether and where this suppression mechanism is
  present. Baseline: exact saved guides/draws and their actual rho=.01 floors.
- Candidates: rho=1e-6 (a sensitivity hypothesis, not a selected value), and
  rho=exp[-chi-square-upper-tail-quantile(N^-2,d)/2], matching the frozen R
  reconstruction. Keep all floors strictly positive. No default change.
- Stage A: compare the actual TF/XLA normalizer and analytical tangent with
  independent closed-form formulas at d=2,5,10,20,40,80; q=.1,1,10;
  r_S^2=0,d,4d; N=1024; all three floors. Repeat CPU/GPU on identical inputs.
  Require log probability and log normalizer errors <=1e-9 and absolute tangent
  error <=1e-9; one trace per dimension. A known negligible-floor pair
  rho=1e-24/1e-30, d2/q1, must give normalizer/transition outputs within 1e-12.
- Stage B: replay all 19 completed previous consumers, both fitted and heldout
  observations, on GPU FP64/XLA. Keep guide centers/covariances, count, theta and
  actual saved random arrays fixed. Evaluate the three floors through the
  shared actual kernel with its existing numerical trace. Baseline value/score
  must reproduce saved results within 1e-9. This changes the final positive
  guide only; it is not refitting, retuning or a new adaptive algorithm.
- Record conditional p_G distributions, actual Gaussian mixture fractions,
  CDF-derived ESS, label changes, exact-Kalman log-value errors and same-count
  heuristic errors. Also reconstruct each baseline particle's predictive mean
  from saved noises/clouds/ancestor indices and evaluate all floors at those
  same means, separating direct floor effects from changed later trajectories.
  Check reconstructed baseline probabilities against the actual trace <=1e-9.
- Promotion criterion for the mechanism: formula/consumer replay checks pass
  and measured probabilities demonstrate or falsify the hypothesized effect
  in each declared situation. There is no candidate/default promotion in this
  diagnostic. Define near-disabled twisting as p_G<.01 for reporting only;
  report continuous probabilities too, so this threshold cannot select a floor.
- Promotion vetoes: nonfinite quantities, wrong normalization/tangent, baseline
  replay failure or heuristic losses. A formula/replay/source-identity failure
  is also a continuation veto until repaired. Explanatory-only diagnostics:
  ESS, mixture shares, conditional errors and label changes; they do not rank
  stochastic methods. Failed prior adaptive candidates remain rejected and
  absent from this accepted-guide replay; do not count them as successes.
- Constructed heuristic adversaries: exact Kalman oracle (certifying value),
  BPF (no lookahead), constant guide in the same consumer (route sanity), and
  one-step optimal Gaussian proposal (uses current data exactly). Reuse their
  matched-count, same-input saved outputs. Conditional situations: dimension,
  seed, initializer, objective units, fitted/heldout observations and time.
- Repair trigger: measured floor suppression nominates a fresh, explicitly
  configured floor study with disjoint calibration/validation; persistent
  failure despite restored twisting points back to guide-family/fit error.
  Candidate failure is not rejection of iAPF or a continuation veto.
- Nonclaims: paper replication, author implementation failure, unbiased score,
  statistically supported ranking, robust optimal floor, TF32/default readiness,
  canonical LEDH or HMC status. Frozen-guide score differs from model score.

## Assumption audit, non-harm and skeptical review

The baseline .01 is inherited and already unpromoted; its hypothesized failure
is exponentially falling mixture share, tested first by the formula. The 1e-6
choice provides four orders of magnitude sensitivity, not an optimum. The R
tail rule has explicit N,d dependence and provenance but no known author
endorsement. Its weakness is reduced protection in remote tails. Lowering a
floor changes the scientific object and is therefore an explicit Class C
experiment, never a silent guard or a metric-tuned repair. The negligible-floor
case supplies a cheap non-harm check where both choices should agree; unstable
guide cases retain finite positive floors and are flagged, not clipped.

Reusing completed cases is intentional paired mechanism diagnosis, with strong
survivor bias for method-quality inference. New random inputs would confound
the causal comparison. Existing model and particle-count choices are diagnosis
scopes only; they do not transfer paper-model validity. CPU/GPU FP64, stable
signatures and XLA are the previously verified arithmetic reference; the prior
TF32 veto persists. R quantiles are independent diagnostic inputs, not an
admitted runtime dependency. Base R qchisq is used without new packages.

Skeptical review PASS: the plan compares convolution probabilities, not a
pointwise-density surrogate; frozen-ancestor replay separates floor strength
from feedback; saved default-value replay checks call-chain and data identity.
Heuristic results remain conditional descriptive vetoes. Known-target success
can coexist with general guide failure. Lower likelihood error cannot select
or promote a floor. The original data/model and unpublished author-choice gaps
remain separate. No arbitrary particle/iteration-cap expansion is included.

## Execution, artifacts and stopping

Root: `artifacts/iapf-positive-floor-20260922-01/`. Starting balance approximately
45.855 CPU and 47.845 GPU hours, exact transfer in budget.json. Suballocation:
one CPU hour and fifteen GPU minutes, at most six launches / three localized
repairs. Each launch limited to 300 seconds, requires remaining phase/campaign
budget, and uses a new directory with command/source/input hashes and elapsed
time. Stop on corrupted/missing required evidence or budget exhaustion; repair
local harness failures inside the same contract and budget.

Commands: `tftwogpu/bin/python docs/benchmarks/diagnose_iapf_positive_floor.py
--attempt <unique> --device cpu|gpu --mode oracle|replay`. The launcher records
the full interpreter path and environment. CPU deliberately hides GPU;
GPU UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b requires escalation, verified
memory growth, FP64, TF32 off and XLA. No runtime/module/default edits planned.
Result.md, decision.json, verification.json and manifest.json preserve the
decision/inference tables, source limits, uncertainty and exact next action.
