# Skeptical Review: C2 Transformed-Guide TT-DMIS Plan

Date: 2026-08-29

Reviewed plan:
`docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-implementation-test-plan-2026-08-29.md`

Original reviewed plan SHA-256:
`0b7e609cdfaf67ceb595f5839afe3984f6941ff92978efbd870fc580ddad9495`

Mathematical specification:
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/ukf_guided_defensive_tt_dmis_analytical_gradient.tex`

Original reviewed manuscript SHA-256:
`3d6f8e5312f02f72150aa556f8e8cd91aef5d7bd9ed0a6ca3a1e219921fe12a7`

## Verdict

`PASS_FOR_BOUNDED_IMPLEMENTATION_AND_DIAGNOSTIC_EXECUTION`

The revised plan is internally consistent with the proposition-proof
manuscript and is sufficiently specific to begin the shared-core
implementation and bounded C2 diagnostic. This verdict does not promote the
method, authorize HMC or posterior claims, or predict that the method will
improve ESS. At the time of this review no implementation or GPU campaign had
been executed; the later execution record is an explicit follow-up and is not
retroactively part of this verdict.

## Decisive Mathematical Finding

The proposed raw-observation UKF cannot guide the C2 model. For

```text
Y = beta * exp(X / 2) * E,   E ~ N(0, I),   E independent of X,
```

the conditional expectation of `E` is zero and therefore `Cov(X, Y) = 0`.
Any Gaussian conditioning rule matching that population covariance has zero
gain. The plan correctly treats this as a rejected negative control and uses
instead

```text
log(Y^2) - 2 log(beta_star) - E[log(E^2)] = X + centered log-chi-square noise.
```

Matching the exact first two noise moments produces an affine
transformed-observation Kalman guide. The raw observation likelihood remains
in every importance numerator. This is an `extension_or_invention`, not a
claim about Zhao and Cui's algorithm.

## Material Findings and Dispositions

| Severity | Finding | Disposition |
| --- | --- | --- |
| Critical | A raw C2 UKF has exactly zero population gain and would merely reproduce the transition prior while appearing data-guided. | Rejected. The manuscript proves the zero covariance; Phase 2 implements it as a negative control and uses the log-square guide. |
| High | The pure squared-polynomial density `q_H` was conflated with the current callable proposal, which actually samples the normalized floor mixture `q_floor`. | Repaired. The first implementation defines `q_TT = q_floor`; extracting `q_H` is out of scope. |
| High | The current shared APF evaluator hard-codes `-log(N)` and cannot represent fixed banks with masses `omega_s/N_s`. | Exposed as the first implementation gate. Phase 1 generalizes the shared endpoint and protects the uniform route with parity and wiring tests. A C2-local fork is forbidden. |
| High | An ancestor-specific alpha cannot be represented by one component base mass per sampled row without changing the joint stratified design. | Repaired. The implementation uses one alpha per time across all ancestors. Ancestor-specific allocation is explicitly out of scope. |
| High | `J(alpha) = integral pi^2/q_alpha` was initially at risk of being treated as the exact variance of equal fixed banks. | Repaired. `J` is a convex nomination diagnostic; selection uses the exact independent-bank DMIS variance on separate validation draws. |
| High | A conditional alpha pilot could silently omit the auxiliary-particle factor. | Repaired. The joint target is on `(ancestor, state)`, and the executable `J_t` summand contains `(W_j/a_j)^2`. The ordinary importance weight contains one `W_j/a_j`; the second moment contains its square. |
| High | Using the UKF covariance directly as a Student scale changes the proposal covariance when `nu` changes. | Repaired. Finite-`nu` arms use `S_D = ((nu-2)/nu) P_D`, so all tail arms share covariance `P_D`. |
| High | A fixed branch could be mislabeled as an exact pseudo-marginal likelihood. | Blocked by the target statement and result vocabulary. The finite score is exact only for the declared frozen scalar; randomized-likelihood and posterior evidence require a later plan. |
| Medium | The old categorical retained-TT route and the generalized-core retained-TT route were duplicated as scientific baselines. | Repaired. The old route is now only the engineering regression authority; the generalized route is the final scientific TT baseline. |
| Medium | The draft treated common random numbers as a prerequisite for paired inference. | Repaired. Replicate grouping defines the paired design; common random numbers are used only as tested marginal-preserving variance reduction. Method-specific streams remain explicit. |
| Medium | A Student-tail or alpha choice could be selected from the final banks or the short GPU smoke. | Blocked. The smoke fixes `(alpha, nu)=(1/2, 8)`; nomination and validation use independent banks; the final banks are untouched. |
| Medium | The pilot-selection rule was underdetermined. | Repaired. The plan freezes the alpha grid, tail set, pilot sizes, independent validation, bootstrap count, stability conditions, and fallback. It also requires a strictly positive baseline variance. |
| Medium | A stale attempt05 retained snapshot could be presented as the current proposal object. | Blocked. The campaign requires one fresh fingerprinted current-route snapshot and explicitly disallows a historical-repair claim. |
| Medium | The fixture identity `zc24_sv_vector_extension_v1` could be confused with the executable model ID `c2_sv_gamma_log_beta_stationary_v1`. | Repaired. The scope binds both identities, `sigma=1`, and the realized coupling-matrix digest. |
| Medium | `T=20` was ambiguous between a terminal index and a row count. | Repaired. The scope is 20 observation rows, which is terminal index `T=19` in the manuscript. |
| Medium | Silent covariance ridge, inflation, clipping, or log offset could alter the proposal while retaining its name. | Blocked. The first route uses zero alteration and fails closed. Any alteration is a separately calibrated Class-C policy with a distinct identity. |
| Medium | One fresh TT fit cannot establish fit-to-fit variability. | Repaired as a nonclaim. The campaign evaluates proposal randomness conditional on one fingerprinted fit. |
| Medium | Descriptive ESS minima and data-selected worst times could be promoted into inferential conclusions. | Blocked. The predeclared paired criterion uses full replicate contrasts; selected worst times are explanatory only. |
| Medium | GPU/XLA failures could be misclassified because of sandboxed device access or memory preallocation. | Repaired. The plan requires trusted device probes, memory-growth verification before initialization, explicit XLA provenance, and fresh output roots. |

## Required Skeptical Checks

| Check | Result | Reason |
| --- | --- | --- |
| Wrong baseline | Pass after revision | The PF reference is a compatibility screen, retained TT is the direct mechanism baseline, and bootstrap, Gaussian-hint, stationary Gaussian, and transformed-guide-only arms form the practical adversary set. |
| Proxy promoted as criterion | Pass after revision | `J(alpha)`, smoke ESS, validation loss, innovations, and selected worst times are nomination or explanation only. The exact bank variance validates selection; final paired ESS is the mechanism criterion. |
| Missing stop conditions | Pass | Target mismatch, base-mass error, incomplete denominator, nonfinite scale/scalar/score, uniform regression, provenance failure, and budget exhaustion are continuation vetoes. Low candidate ESS alone is not. |
| Unfair comparison | Pass | Scientific arms share data, retained snapshot, total particle count, runtime parameter, auxiliary construction, branch replicates, and target densities. Compile and repeated-evaluation costs are reported separately. |
| Hidden assumptions | Pass with recorded residuals | Alpha, tail set, ridge policy, pilot sizes, branch count, dtype, horizon, snapshot policy, and auxiliary-law freeze are all classified in the default audit. |
| Stale context | Pass | The plan excludes the unreconstructable attempt05 TT snapshot and treats pre-2026-08-21 LEDH evidence as historical only. |
| Environment mismatch | Pass | CPU diagnostics intentionally hide CUDA; GPU work requires trusted access, TensorFlow memory growth, float64, and XLA. Production float32/TF32 is explicitly outside this campaign. |
| Artifact insufficiency | Pass | The driver must emit resolved configuration, manifests, JSON, Markdown, logs, fingerprints, seeds, device policy, timing, and a machine-readable heuristic verdict. Unrecorded overrides fail closed. |

## Manuscript-to-Plan Consistency

| Mathematical object | Plan realization | Status |
| --- | --- | --- |
| Exact model remains the target | Raw C2 transition and observation densities stay in each numerator | Match |
| Current normalized TT component | Complete retained proposal including its internal floor | Match |
| Full-support defense | Covariance-matched Student around the frozen transformed-observation guide | Match |
| Smooth outer mixture | One frozen timewise alpha in `[0.10, 0.90]`; fixed half mixture is the theory-led baseline | Match |
| No categorical outer switch | Fixed equal TT and transformed-guide banks | Match |
| Complete balance denominator | Both component densities evaluated at every draw | Match |
| Nonuniform base measure | `log(omega_s)-log(N_s)` owned by the prepared shared branch | Match |
| Joint APF allocation target | Pilot samples ancestor and state jointly and retains the squared `W/a` factor | Match |
| Same-scalar analytical score | Existing centered manual recursion differentiates the generalized finite scalar with all proposal objects frozen | Match; executable finite-difference and eager/XLA checks pass |
| Initial-time special case | Existing one-component uniform initial proposal is permitted; two banks begin at transitions | Match |
| Exactness boundary | Frozen branch is a deterministic approximate likelihood; pseudo-marginal testing is deferred | Match |

## Remaining Risks

These are empirical risks, not defects in the revised plan:

- The log-chi-square Gaussian moment closure may place the guide correctly but
  still miss skewness and tails badly enough to lose to the Gaussian-Hermite
  hint or bootstrap proposal.
- The retained TT may add no useful geometry. The transformed-guide-only arm
  is required to reveal that result.
- Twelve branch replicates may yield an interval too wide for the predeclared
  mechanism criterion. The correct outcome is inconclusive, not a silent
  budget extension or a descriptive ranking.
- Float64 C2 evidence does not establish the separate float32/TF32 production
  lane.
- A single retained fit does not establish robustness to TT fitting
  randomness.

## Review Boundary

The plan may now proceed through implementation, focused CPU checks, and the
bounded trusted GPU/XLA diagnostic under its stated budget. Any change to the
target, raw versus transformed guide, alpha policy, tail family, particle or
replicate budget, precision lane, pseudo-marginal objective, or posterior
claim requires an explicit plan revision before the affected run.

## Execution Follow-Up (2026-08-30)

The plan was executed under the unchanged fixed-half scope. Focused tests passed
25/25, the corrected GPU/XLA smoke passed its engineering screens, and the
serious N=8192 run completed with 72 finite branch records and verified GPU
memory growth. Two implementation defects found during execution were repaired:
the transformed Student chi-square draw now uses TensorFlow's rate
`beta=0.5`, and the TTSIRT block combiner preserves resolved base masses.

The serious driver's original heuristic table was discovered to compare every
arm with retained TT rather than compare the DMIS candidate with the declared
heuristic adversaries. The raw run is preserved unchanged; a diagnostic-only
post-run audit recomputes the correct comparison from `branch_results.json`.
It finds a mean log minimum-ESS ratio of `5.3839341272`, a 95% bootstrap
interval `[5.1832197155, 5.5999535634]`, and 12/12 positive contrasts, while
bootstrap, transformed Student, and Gaussian-hint arms beat DMIS in per-step
reference error at `t=3`, `t=4`, and `t=11`. The paired mechanism criterion is
therefore met, but the heuristic-dominance promotion veto fires.

The follow-up audit initially recorded that the driver did not implement the
Phase 5 alpha/nu pilot. A separate bounded pilot driver has since completed
that phase under the unchanged contract. Its point minimum was `(nu=16,
alpha=0.6)`, but bootstrap minimizer stability was `0.2865`, below the
predeclared `0.80` gate, so the result correctly falls back to fixed half and
does not issue a selected allocation. The preserved attempt01 also predates
the later per-time maximum-normalized-weight field; the current driver now
emits it. These records do not reinterpret the fixed-half result as a default.

The pilot artifact is
`docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/pilot-attempt02/`.
