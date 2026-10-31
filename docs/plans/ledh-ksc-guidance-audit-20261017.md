# KSC future-guidance diagnostic — 2026-10-17

## Question and scope

Can the shared covariance-mixture consumer use a non-Gaussian future message
while preserving the seven-component KSC likelihood and the analytical total
derivative of its finite value program? This is an engineering and mechanism
diagnostic, not fitted KDM evidence or canonical LEDH admission. The checkout is
`ledh-refactor-with-policy-fix` in `ledh-iapf-execution-20260922`.

The Gaussian localization stage is closed. Its lack of guidance improvement
does not establish a numerical blocker or justify a precision-repair campaign.
Here the target is `sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1`, not the
native stochastic-volatility likelihood. Write gamma = Phi(theta[0]) and
theta[1] = log beta. The prior and process variance are one; observations
follow propagation and have location x + 2 log beta plus the seven-component
mixture in `sqmc_ksc_tf.py`.

## Exact two-observation future message

For observations y[0], y[1], the message at the first filtering time is

    h(x) = sum_j w_j Normal(y[1]; gamma*x + 2*log(beta) + m_j, 1 + v_j).

For gamma > 0 this is, up to the common factor 1/gamma, a Gaussian mixture in
x with weights w_j, means (y[1] - 2*log(beta) - m_j)/gamma and variances
(1 + v_j)/gamma^2. Integrating the additive transition noise gives this
identity directly. The terminal message is constant. A second guide replaces
this mixture by its moment-matched Gaussian. The mixture guide's spatial
gradient is exact for this representation, but its responsibility-averaged
positive precision is not the negative Hessian. Neither representation makes
the LEDH proposal exact.

Guide coefficients are frozen at theta=(1.5, 0) for this diagnostic, including
finite-difference perturbations. Analytical derivatives include moving-anchor
dependence; they do not differentiate a refitting operation. The claimed score
is the derivative of that frozen-guide finite estimator. Distance from the
model score is a separate, empirical question.

## Evidence contract and research intent

- Baseline ladder and constructed heuristic adversaries: unguided identity
  (predictive importance sampling), defensive global flow (one shared Gaussian
  geometry), defensive local flow (per-particle geometry), and equal mixture
  (simple diversification). These are also the exact same-consumer comparators
  for Gaussian-future and mixture-future guides at global and equal beta.
- Conditional situations: fixed ordinary (0.2, -0.4), negative-future-tail
  (0.2, -8), and positive-future-tail (0.2, 5) observation pairs. These are
  constructed fixtures, not draws supporting a population claim.
- Primary engineering criteria: actual likelihood/call-chain parity; refined
  grid versus exact 49-branch reference agreement within 1e-6 in value and
  1e-5 per score coordinate; finite valid outputs; analytical finite-program
  score versus centered differences at two step sizes within
  2e-7 + 2e-5*abs(finite difference) per coordinate.
- Continuation vetoes: wrong likelihood or timing, failed reference or
  derivative/wiring checks, invalid factors/moments/correction caps, or
  non-finite outputs. A localized harness failure triggers repair and a fresh
  attempt within budget. No bypass of a scientific validity check is allowed.
- Promotion criterion for this stage: engineering criteria only. Conditional
  guide score-error comparisons are descriptive; eight clouds cannot establish
  a ranking. A guide losing to any heuristic in any conditional score cell is
  an explicit promotion veto, not a continuation veto for later fitting work.
- Explanatory diagnostics: per-cloud and per-coordinate score errors, likelihood
  errors, mixture precision variation, runtime and memory. No beta selection,
  score-based tuning, default change, HMC, long-horizon or TF32 readiness claim.
- Expected failure: exact future information may fail to improve finite-particle
  score error because flow approximation, reset and Monte Carlo noise remain.
  Such a result rejects promotion of this candidate, not the research direction.
- Artifacts: versioned attempts under
  `docs/plans/artifacts/ledh-ksc-guidance-20261017/`, with manifest, checks,
  per-cloud results, conditional heuristic verdict, decision and result note.

## Defaults, assumptions, and skeptical audit

| Choice | Provenance and purpose | Failure mode / early diagnostic | Status |
|---|---|---|---|
| T=2, N=32 | Small exact 49-branch target and bounded shared-consumer test | Horizon/particle transfer is unsupported; keep scope explicit | Engineering baseline |
| Three fixed observation pairs | Constructed ordinary and opposite-tail situations | Unrepresentative fixtures; no population inference | Diagnostic design |
| Eight independent clouds per pair | Seeds 617000+100*condition+replicate, shared across arms | High uncertainty; publish every paired row, no ranking | Bounded diagnostic |
| FP64 GPU/XLA | Reference exception to FP32/TF32 production direction | Does not demonstrate production behavior; record dtype/device | Reference baseline |
| OT epsilon=1, iterations=5, chunk=N | Owner setting and exact-divisor policy | Incomplete balancing can affect error; retain validity diagnostics | Frozen comparison baseline |
| Flow steps=16, correction steps=2, rate=.1, radius=.25, basis=4 | Existing optional consumer settings | KSC adequacy and safety calibration unresolved; retain cap/moment checks | Uncalibrated baseline, not target default |
| Identity, defensive global/local and equal beta | Same admissible heuristics as the Gaussian stage; global/local retain 10% predictive density | The 10% fraction is a comparison hypothesis, not KSC tuning; retain density-ratio diagnostics | Baselines |
| Frozen exact/moment guide at theta | Isolates representation without training | Full refitted-guide derivative is a different target; freeze for FD | Mechanism hypotheses |
| CPU grid/enumeration, no XLA | Independent small reference exception | Reference disagreement; check before GPU comparison | Reference authority only |

Skeptical audit: the seven-component density, predict-first timing, prior and
transition variance match the KSC spec and canonical model callback. Proposal
moment matching is not used as the reference likelihood. One shared numerical
consumer must serve every arm; executable wiring and density checks are
required. The oracle guide is an information diagnostic, not a trained KDM
component. No result from before the LEDH invalidation is a comparator. The
budget, explicit derivative target, conditional heuristic set and validity
stops make this diagnostic informative even if all guided candidates fail.

## Execution and budget

New allowance: 900 total GPU wall seconds, at most two launches, each capped at
450 seconds; 180 CPU wall seconds for focused tests, references and analysis.
Measure all attempts; use a fresh directory on retry. Check GPU occupancy,
configure and verify memory growth before initialization, disable TF32 for
this reference exception, and record source SHA-256, commit, environment,
command, seeds, device, elapsed time and plan/result paths. No environment
mutation or external reviewer is required.

1. Generalize the diagnostic wrapper's spec/dimensions and guide representation,
   preserving the shared numerical consumer and existing Gaussian API.
2. Check the actual KSC likelihood and tangent, exact/reference agreement,
   wrapper wiring and frozen-guide derivatives with focused CPU diagnostics.
3. Run all eight arms on the three conditions and eight paired clouds using
   GPU/XLA. Run two-step finite differences on representative guided outputs.
4. Save conditional MSE/likelihood tables and an explicit heuristic verdict;
   audit artifacts and update master/checkpoint with budget and next action.

Commands (from the LEDH worktree):

```sh
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_guided_mixture_diagnostic_tf.py tests/highdim/test_ksc_guidance_diagnostic_tf.py
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_ksc_guidance_diagnostic.py --output docs/plans/artifacts/ledh-ksc-guidance-20261017/attempt01 --budget-seconds 450
```

## Next scientific stage

After validity closure, specify a target-specific fitting protocol before
training: independent pilot and disjoint calibration/validation/final clouds,
positive-quadratic controlled-SMC versus mixture KDM representation, explicit
loss/scaling/capacity/bandwidth/optimizer search and total budget. Fit future
likelihood without counting the current observation twice. Freeze controls
before downstream score comparisons, retain all four conditional heuristics,
and distinguish representation checks from downstream score evidence. No
training settings transfer automatically from Gaussian work.

## Execution record

The first focused CPU check took 29.73 seconds: 14 passed and two wrapper
checks failed before filtering because the new fixture used beta[0]=0. The
shared consumer correctly requires positive defensive mass. This is a harness
error, not evidence against guidance or a reason to weaken the guard. Repair:
global beta=(.1,.9,0), local beta=(.1,0,.9), matching the existing Gaussian
comparison convention. Thus q_beta >= .1*p_predictive in these arms. The
fraction is a frozen baseline hypothesis; all paired guided/unguided arms use
the same beta, and it is not selected using KSC score error. The skeptical
audit passes after this correction. Logs: `artifacts/ledh-ksc-guidance-20261017/cpu-tests01.log`.
The corrected rerun passed all 16 checks in 39.50 seconds. Total focused-test
CPU usage is 69.23/180 seconds; 110.77 seconds remain. GPU allowance remains
900 seconds across at most two launches. Log:
`artifacts/ledh-ksc-guidance-20261017/cpu-tests02.log`.

Attempt01 stopped after 4.2101 seconds, before any filter evaluation. The
runner passed lists to the existing `lru_cache`-decorated reference API, which
requires hashable arguments. Both T=1 and T=2 call sites now use tuples. This
is an infrastructure failure, not a mathematical or numerical result. No
target, guide, control, tolerance, data, or promotion criterion changes. The
original failure is preserved in `artifacts/ledh-ksc-guidance-20261017/attempt01/`.
Attempt02 is the final permitted launch, capped at 450 seconds; 895.7899 GPU
wall seconds and 110.76 CPU reference seconds remain conservatively. The
repair survives the skeptical audit: it restores the documented reference
API without weakening any scientific check.

## Completed execution and terminal audit — 17 October 2026

Attempt02 completed 192/192 valid GPU/XLA rows in 40.9831 seconds. All 24
analytical finite-difference checks passed at the two declared step sizes;
maximum absolute error was 6.403e-9. Independent grid/exact reference
differences were at most 1.777e-15 in value and 1.166e-15 in score. The
standard-library terminal auditor passed coverage, pairing, all 120 heuristic
comparisons and 72 source-file checks. The 16 focused tests passed.

Each guided arm has at least one descriptive heuristic loss (34/120 in total).
No statistically supported score ranking or improvement is established.
There is no numerical continuation veto. In particular, the exact future
mixture also fails promotion; increasing representation capacity is not an
established repair. The research program continues with consumer-mechanism
and target-specific fitting checks, not a repeated precision campaign.

The stage is closed at 45.1932/900 GPU wall seconds and 2/2 launches, with
approximately 70.12/180 CPU seconds used. The remaining wall allowance does
not create a third launch. No process is running.

The MathDevMCP structural audit was inconclusive because prose identifiers
were treated as required code symbols. It is not a semantic verification.
See the [complete result and decision/inference tables](artifacts/ledh-ksc-guidance-20261017/result.md),
[terminal audit](artifacts/ledh-ksc-guidance-20261017/terminal-audit.json), and
[next protocol](ledh-ksc-fitting-consumer-protocol-20261017.md).
