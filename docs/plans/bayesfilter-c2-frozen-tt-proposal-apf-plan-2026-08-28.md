# BayesFilter C2 Frozen-TT Proposal APF Plan

Date: 2026-08-28

Status: executed; terminal audit complete; candidate rejected for
claim-bearing use (2026-08-29)

Mathematical specification:
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`,
Section "Frozen-TT Proposal Correction with an Analytical Score"

Terminal serious output root:
`docs/benchmarks/artifacts/c2_frozen_tt_proposal_apf_20260828/attempt02/`

Retained-output smoke root:
`docs/benchmarks/artifacts/c2_frozen_tt_proposal_apf_20260828/smoke-attempt03-final/`

## 1. Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Does demoting the failed C2 retained squared TT from an evidence approximation to a normalized full-support proposal remove the 103.64-nat n=4 failure while retaining an analytical gradient of the exact finite fixed-branch scalar? |
| Mechanism under test | Sample each post-initial state from the normalized retained Hermite/Student-t mixture, then restore the exact SV transition and observation densities through APF importance ratios. |
| Expected failure mode | The fitted TT may remain a poor proposal, causing low ESS or large branch variance even though proposal correction removes fitted-mass bias in expectation. |
| Primary engineering criterion | The implemented value is exactly the LaTeX fixed-branch scalar and the manual recursive score matches centered finite differences of that same scalar. |
| Primary diagnostic criterion | Across independent frozen branches, finite corrected totals are compatible with the screened PF reference under the predeclared uncertainty rule, with per-step ESS and errors reported at t=3, t=4, and the full T=20 path. |
| Promotion criterion | None. This campaign can establish an implemented, viable diagnostic candidate only. |
| Promotion veto | Any failure of support, sampler/density parity, same-scalar score, XLA parity, or the heuristic-dominance table. |
| Continuation veto | Broken source snapshot, call-chain mismatch, non-positive proposal normalizer, invalid conditional CDF, non-finite proposal density or APF scalar, invalid PF comparator, GPU/XLA provenance failure, or exhausted budget. |
| Repair trigger | A finite but low-ESS TT proposal triggers proposal-quality work; a correct sampler with wrong same-scalar score triggers model-score repair; a sampler/density mismatch triggers Hermite-KR repair. |
| Explanatory diagnostics | Component choice, inverse-CDF residual, conditional denominator, CDF endpoint margin, normalized ESS, weight spread, per-step error, direct-TT increment, and branch-to-branch dispersion. |
| Must not be concluded | Exact pseudo-marginal inference, exact posterior targeting with one permanently frozen branch, HMC readiness, default readiness, source-faithful reproduction of the full Zhao-Cui solver, superiority, or a universal n=4 TT result. |

The failed direct TT candidate is evidence motivating the repair. It is not a
continuation veto because this plan tests the correction designed for that
failure.

## 2. Source And Claim Boundary

The following source operations were inspected before execution:

| Source | Technical anchor | What it supports |
| --- | --- | --- |
| Zhao and Cui (2024), local text under `.localresources/papers/` | Eq. (13), lines 539-573 | Squared TT plus positive defensive reference density. |
| Same paper | Proposition 2 and Section 3.1, lines 592-670 | Marginals and conditional KR construction by paired core contractions. |
| Same paper | Eqs. (20)-(23) and Algorithm 3, lines 807-923 | Proposal sampling followed by exact target/proposal correction. |
| Author code `deep-tensor.dev/src/@TTSIRT/marginalise.m` | lines 25-85 | Coefficient-Gram marginalization with no branch or sample-count division. |
| Author code `deep-tensor.dev/src/@TTSIRT/eval_irt_reference.m` | complete routine | Inverse triangular proposal sampling. |
| Author code `models/full_sol.m` | lines 33-42 | Inverse-map sample and exact target divided by `eval_pdf`. |

The squared density, marginalization, conditional proposal, and importance
correction are source-faithful operations. The state-independent retained
marginal proposal, frozen genealogy, deterministic fixed-branch scalar,
two-parameter C2 model, and recursive analytical score are BayesFilter
`extension_or_invention` work. The plan will not label the complete route
source-faithful.

The official JMLR landing page was checked on 2026-08-28 and exposed the paper
and code links without a visible correction or retraction notice. Live forward
citation search failed twice with an upstream HTTP 502, so forward-snowball
coverage is unavailable and is not claimed. The local paper and pinned author
code are sufficient to anchor the operations implemented here.

## 3. Evidence Contract

### 3.1 Exact target and comparators

The claim-bearing diagnostic target is the deterministic scalar
`sum_t(logsumexp(log_weight_t) - log(N))` for a repository-issued fixed branch.
Its score is the analytical derivative of that same scalar with states,
proposal values, auxiliary laws, ancestors, and random numbers fixed.

The frozen C2 scope is:

- model `zc24_sv_vector_extension_v1`;
- state dimension 4 and observation dimension 4;
- model seed 52 and observation seed 42;
- horizon 20;
- reference parameter `(gamma, log(beta)) = (0.6, log(0.4))`;
- direct TT degree 6, rank 6, 8192 fit rows, 32 ALS sweeps, ridge `1e-10`,
  fitter seed `98000 + 100*4 + 10*6 + 6 = 98466`, tau `1e-6`, and the
  existing C2 Student-t rule `student_t_nu_criterion(0.8, 12.0)`;
- TensorFlow float64, GPU, memory growth, and XLA JIT.

The T=20 reference comparator is the valid 800,000-particle, ten-replicate arm
in
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/reference_n4_s42.json`:
total `-66.6979549734` with recorded total SE `0.0041582` and per-step means.
That original file does not contain per-step covariance. For the salient
t=0..4 window, use the independently preserved covariance reference at
`docs/benchmarks/artifacts/c2_n4_root_cause_20260828/attempt03/pf_per_step_reference.json`.
It is an 800,000-particle, ten-replicate CPU-only diagnostic generated from
the same model/observation identities and exactly reproduces the first five
means in the T=20 reference. Do not infer T=20 per-step covariance from this
T=5 artifact.

The candidate ladder is:

1. failed direct TT mass recursion, retained only as the motivating negative
   control;
2. frozen TT retained-marginal proposal APF;
3. same-N frozen bootstrap proposal APF;
4. same-N frozen Gaussian-hint marginal proposal APF;
5. same-N frozen stationary-Gaussian independence proposal APF;
6. screened high-N bootstrap PF reference.

The high-N PF is the reference, not a tuning arm. The three same-N simple
proposals are the constructed heuristic adversary set and are not training
targets.

### 3.2 Pass, veto, and uncertainty rules

Engineering correctness passes only if all of the following hold:

- retained proposal `Z_H`, suffix Gram, tau, map, and defensive law are
  reconstructed from the seven production transition outputs, with no extra
  XLA output or summary in the fitted graph;
- incomplete Hermite Grams agree with independent quadrature on degree 0-6
  fixtures and approach identity at the right endpoint;
- conditional CDFs are finite, monotone on an audit grid, bracket every fixed
  uniform, have positive denominators, and pass inverse/forward residual
  `<= 2e-10` in float64;
- samples and logged densities use the same complete mixture, including the
  unselected component;
- proposal normalization agrees with an independent low-dimensional
  integration fixture within `2e-8`;
- the generic APF call chain includes selected previous weight, selected
  auxiliary probability, exact transition, observation, and proposal terms;
- the C2 local manual scores match centered finite differences on independent
  points within `2e-6` relative/absolute tolerance;
- the complete recursive score matches centered finite differences of the
  same frozen-branch scalar within `2e-5` relative/absolute tolerance;
- eager, TensorFlow graph, and XLA results agree within `2e-10` in float64;
- repeated calls do not retrace beyond the setup-static signatures; and
- all fail-closed negative fixtures fire.

The T=20 candidate is diagnostically viable if four independent N=8192
branches are finite, the mean corrected total differs from the screened PF
mean by at most `max(1.0 nat, 3 * combined_SE)`, and no time step has mean
normalized ESS below `0.0025`. These thresholds nominate a repair that removed
the gross pathology; they do not establish posterior or default readiness.

For stochastic comparisons, totals, ESS, weight spreads, and tails are
descriptive unless a paired interval is reported. The result must state
whether any ranking is statistically supported. Four branches are not enough
to rank methods by extreme quantiles.

Hard vetoes are non-finite values, invalid reference identity, support failure,
sampler/density mismatch, same-scalar gradient failure, branch mutation, or
missing required diagnostics. Low ESS is a candidate rejection and repair
trigger, not an implementation-invalidity finding and not rejection of
importance correction as mathematics.

### 3.3 Heuristic-dominance situations

The salient situations are t=3, where frozen diagnostics identified excess
TT Gram energy; t=4, where the direct discrepancy jumps by about 7.15 nats;
the lowest-ESS time; and the full T=20 total. For each situation report all
three simple proposal arms beside the TT proposal. Losing to any heuristic in
any salient situation is a promotion veto and the headline of the result.
Passing this weak screen proves little.

### 3.4 Nonclaims

Even a pass does not establish exact randomized-likelihood unbiasedness for
the finite-precision inverse sampler, exact posterior targeting, HMC
convergence, source-faithful reproduction of Zhao-Cui Algorithm 3, a new
default, or statistical superiority. A permanently frozen branch defines the
declared deterministic approximate likelihood only.

## 4. Mathematical-To-Code Crosswalk

| LaTeX object | Required endpoint | Executable evidence |
| --- | --- | --- |
| Eqs. `proposal-map` through `q-physical` | `GaussianHermiteRetainedProposal` construction and complete-mixture `log_density` | exact snapshot wiring, Gram/tau/map parity, density normalization |
| Eqs. `incomplete-hermite-gram` through `incomplete-hermite-closed` | normalized incomplete Hermite Gram kernel | quadrature and endpoint tests for every pair through degree 6 |
| Eq. `hermite-kr-cdf` | paired-left/right conditional CDF and inverse | monotonicity, endpoint, inverse/forward, and sample-law tests |
| Eqs. `apf-log-weight` and `frozen-apf-scalar` | existing `_evaluate_core` in `zhao_cui_frozen_proposal_apf_tf.py` | direct recomposition and nonuniform-auxiliary wiring tests |
| Eqs. `score-recursion` and `apf-score` | existing centered-mark recursion | same-scalar finite difference and XLA parity |
| Eqs. `lyapunov` and `lyapunov-dot` | C2 stationary covariance and derivative solves | residual, scalar oracle, and centered finite difference tests |
| Eqs. `initial-gamma-score` through `observation-xi-score` | C2 manual local score methods | pointwise finite difference and complete-program finite difference |
| Conditional mean identity | branch compiler with fixed positive auxiliary law and proposal samples | Monte Carlo one-step identity fixture; diagnostic only |

No reduced evaluator may reimplement the APF scalar. The C2 compiler must
produce `PreparedFrozenProposalBranch`, and the claim-bearing value/score call
must resolve to the existing generic `_evaluate_core` through
`FrozenProposalAPFProgram.compiled(jit_compile=True)`.

## 5. Default And Assumption Audit

| Choice | Provenance and status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| Degree 6, rank 6, 8192 rows, 32 sweeps | Failed attempt05 configuration; negative-control baseline, not a default | Poor proposal or excess off-design energy | CDF validity and N=2048 ESS smoke |
| Student-t `nu` | Existing C2 tail criterion with alpha cap 0.8 and margin cap 12; inherited hypothesis | Bulk dilution or inverse-sampling mismatch | component mass, ESS, sample-law checks |
| Tau | Existing attempt05 retained floor; frozen baseline | Too little useful defensive mass despite support | component count and log-weight spread |
| 64 bisection iterations | numerical hypothesis derived from a 24-wide float64 bracket | root error or CDF stagnation | inverse/forward residual and bracket margin |
| Initial bracket `[-12,12]` with fail-closed expansion to `[-24,24]` | convenience hypothesis | misses an extreme uniform or suffers cancellation | endpoint coverage before sampling |
| N=2048 then N=8192 | bounded diagnostic ladder; N=8192 matches fit-row scale but is not tuned | too few particles for stable total | cross-branch SE and min ESS |
| Four serious branches | minimum uncertainty screen, not ranking evidence | wide branch uncertainty | paired intervals and explicit descriptive-only status |
| Auxiliary law `a=W(theta_star)` | standard fully resampled APF baseline | reference-specific genealogy degrades away from theta_star | finite-difference neighborhood and weight spread |
| Runtime parameter neighborhood | stability-domain constrained around theta star | Lyapunov solve invalid outside stability | spectral-radius and Cholesky guard |

No Class-C clipping, damping, Gram ridge, conditional-mass floor, or CDF
monotonicity repair may be introduced silently. Invalid values fail closed.

## 6. Skeptical Pre-Execution Audit

The first draft was revised before execution for the following material risks:

- A bounded Legendre transport would test the wrong reference measure; the
  plan now implements the Gaussian-Hermite incomplete Gram.
- Reconstructing the retained proposal from an independent fitted object would
  violate the call-chain rule; the plan now consumes production-captured full
  cores and checks the resulting `Z_H` against the snapshot.
- A t=0 TT capture would add an unnecessary lane; the plan now freezes the
  exact stationary prior at the reference parameter, as allowed by the APF
  derivation.
- Proposal ESS could be mistaken for scalar correctness; score/density tests
  are hard engineering gates and ESS remains proposal-quality evidence.
- Agreement from one branch could be mistaken for correction; the serious arm
  requires four independent branches and uncertainty reporting.
- A weak direct-TT-only comparison would hide external inadequacy; the plan
  now constructs three same-N heuristic proposals and evaluates salient steps
  conditionally.
- Fixed branch likelihood could be mislabeled pseudo-marginal exact; the
  target and nonclaims now explicitly forbid that conclusion.
- A transferred particle count or bisection budget could be treated as a
  default; both are hypotheses with early diagnostics and bounded escalation.
- The commands could succeed without checking GPU placement or memory growth;
  both are hard manifest fields and serious-run vetoes.
- The first scope table conflated observation seed 42 with the fitter seed.
  Attempt05 constructs the fitter seed as
  `98000 + 100*n + 10*degree + rank`; the executable scope and this plan now
  bind `98466`, while retaining observation seed 42 separately.

Wrong baselines, proxy promotion, missing stop conditions, unfair comparisons,
hidden defaults, stale pre-2026-08-21 LEDH evidence, environment mismatch, and
artifact relevance were explicitly checked. LEDH evidence is not used. The PF
reference is valid for this exact C2 fixture, and every planned artifact
answers either implementation correctness, proposal-law correctness, or the
declared diagnostic question.

SKEPTICAL_AUDIT: PASS after the revisions above. This pass authorizes the
bounded diagnostic campaign; it is not a scientific promotion.

## 7. Implementation Stages

### Stage 0: document and math closure

1. Compile the revised LaTeX twice with no errors or undefined references.
2. Run MathDevMCP on importance cancellation, Hermite product and
   antiderivative identities, scalar Lyapunov differentiation, transition and
   observation scores, and centered marks.
3. Preserve a concise math-audit ledger in the result note, distinguishing
   backend-proved scalar identities from hand-derived matrix/measure results.

### Stage 1: Gaussian-Hermite proposal primitive

Implement a new TensorFlow module under `bayesfilter/highdim/` containing:

- normalized incomplete Hermite Gram evaluation;
- paired right environments terminating in the stored suffix Gram;
- batched paired-left conditional CDF evaluation;
- setup-static XLA bisection inverse with fail-closed diagnostics;
- complete Hermite/Student-t mixture density in reference and physical
  coordinates; and
- a constructor from the retained production snapshot that consumes the
  production suffix Gram directly and checks snapshot `Z_H`; the legacy full
  snapshot constructor remains a read-only diagnostic route.

Do not add NumPy, pfor, `tf.vectorized_map`, autodiff, a ridge, or a clipped
accepted result.

### Stage 2: C2 model and branch compilers

Implement:

- the C2 `(gamma, log(beta))` TensorFlow model with stationary Lyapunov solve,
  differentiated Lyapunov solve, and manual local scores;
- the TT retained-proposal branch compiler;
- bootstrap, Gaussian-hint, and stationary-independence branch compilers using
  the same frozen APF branch schema; and
- repository-computed compilation manifests binding snapshots, proposals,
  random seeds, states, q values, auxiliary laws, and ancestors.

All compiler randomness is generated once from TensorFlow stateless random
operations. At each compilation step, obtain the reference normalized weights
by evaluating the already-built branch prefix through the same generic
`FrozenProposalAPFProgram`; do not maintain a second weight recursion in the
compiler. Use those weights as the next positive auxiliary law. Runtime
evaluation receives only the completed fixed branch and theta.

### Stage 3: focused CPU-only checks

Run with `CUDA_VISIBLE_DEVICES=-1` and record that choice:

```text
pytest -q tests/highdim/test_c2_gaussian_hermite_proposal_tf.py
pytest -q tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py
pytest -q tests/highdim/test_c2_sv_frozen_fixture_diagnostic.py
pytest -q tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py
pytest -q tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py
```

The new tests include independent quadrature/reference calculations only in
test code, consistent with the NumPy diagnostic-only policy.

### Stage 4: trusted GPU/XLA smoke

Run the exact n=4 T=5 retained-output route and one N=2048 branch of each
proposal. The direct control must pass the production seven-output
call-chain/parity check. It is not a comparison to a historical T=5 scalar:
the only preserved T=5 scalar came from a full-core observability route whose
extra XLA outputs perturb the ill-conditioned ALS trajectory, so that route is
not an acceptable smoke baseline. Historical T=20 compatibility is checked
separately in Stage 5.
Require TensorFlow memory growth before logical-device initialization, actual
GPU placement, XLA compilation, finite outputs, and same-scalar score finite
difference. Stop if any engineering veto fires.

### Stage 5: trusted GPU/XLA T=20 diagnostic

Capture all t=1..19 fitted transitions from the unchanged attempt05 route,
compile four independent N=8192 frozen branches per proposal family, and
evaluate each at the reference parameter. Record:

- total and per-step finite scalar;
- analytical score and centered finite-difference residual;
- ESS and log-weight spread by step;
- proposal component counts and CDF diagnostics;
- absolute error against PF by step and total;
- paired descriptive differences against every heuristic;
- uncertainty intervals and inference status; and
- compile, first-call, repeated-call, fit, and branch-generation wall time.

### Stage 6: terminal call-chain and code audit

Trace the serious harness from captured production cores to proposal
construction, branch issuance, generic APF program, C2 local score, and output
artifact. Add an executable wiring test for every boundary. Compare every
LaTeX equation in Section `frozen-proposal` with the endpoint and test listed
in Section 4. Record `correct`, `wrong relative to the target`, `unsupported`,
or `not checked`; do not use a prose-only implementation verdict.

### Terminal execution and disposition (2026-08-29)

Stage 0 closed the mathematical specification, compiled the manuscript, and
ran the MathDevMCP scalar checks.  Stage 3 closed with 34 focused CPU-only
tests passing.  The first GPU smoke attempt was an infrastructure failure;
the repaired mask-0 run used the retained-output path and completed its
finite, XLA, placement, memory-growth, proposal-law, and same-scalar checks.

The terminal serious run is `attempt02`.  All 16 proposal branches (four
families and four frozen seeds) passed their proposal/CDF, finite-value,
eager/XLA, repeat, and analytical-score checks.  The retained-TT mean total
was `-66.9766151961`, compared with the screened PF value
`-66.6979549734`; its minimum mean normalized ESS was
`0.0006033922`, below the predeclared `0.0025` viability threshold.  The
retained proposal is therefore rejected as a viable candidate under this
scope.  The simple proposal arms are descriptive comparators, not a ranking
claim.

The repaired smoke artifact's direct field is an internal seven-output
call-chain/parity result; it has no historical scalar comparator. The serious
artifact owns the historical T=20 compatibility check.

The direct negative control also failed its historical compatibility check:
the current retained seven-output route gives `31.1071889584` while the
preserved attempt05 scalar is `36.9423456524`.  A second current attempt gave
`27.1328858463`; the two current increment sequences agree to about
`1.2e-12` through the first 12 steps and diverge from step 12 onward.  This
is a hard reproducibility/compatibility veto for the historical direct claim,
not a refutation of the proposal equations or APF call chain.  The terminal
code audit marks the proposal implementation and score route `correct`, and
marks historical trajectory compatibility `wrong relative to the preserved
comparator / not established`.

Overall disposition: no promotion, no default change, and no HMC or exact
pseudo-marginal claim.  The tested candidate failed; the proposal-correction
direction remains viable as a research direction.  The next discriminating
experiment is a fixed-runtime replay with per-step intermediate tensor
snapshots, comparing the first divergent ALS core/Gram/reduction.  A new
T=20 proposal run is not justified until that compatibility boundary is
resolved or explicitly redefined under a fresh evidence contract.

## 8. Compute Budget And Stop Conditions

- CPU-only implementation tests: at most 2 wall-clock hours across repairs.
- GPU/XLA smoke: at most 3 attempts and 45 total GPU minutes.
- T=20 serious diagnostic: at most 2 fit attempts, 6 branch-generation
  attempts, and 4 total GPU hours.
- Output attempts are fresh versioned directories; no prior artifact is
  overwritten.
- Localized harness, serialization, or XLA-compatibility failures may be
  repaired and retried within this unchanged budget and evidence contract.
- Stop for owner direction if the model, data, proposal mathematics,
  comparator, promotion rule, privacy boundary, hardware class, or total
  budget must change.

## 9. Required Artifacts

The terminal `attempt02/` must contain (the earlier `attempt01/` remains
historical evidence):

- `run_manifest.json` with git commit/status, command, conda environment,
  TensorFlow/TFP versions, CPU/GPU status, TF32, XLA, memory policy, seeds,
  wall times, source and snapshot fingerprints, plan path, and result path;
- `proposal_snapshot_manifest.json`;
- `engineering_tests.json`;
- `branch_results.json` with per-branch and per-step records;
- `result.json` with decision and inference-status tables;
- `result.md` with direct scientific interpretation, strongest alternative
  explanation, overturning evidence, weakest evidence, and next action;
- logs for the fit, smoke, serious run, and focused tests;
- `trajectory_compatibility_audit.md`, `code_math_audit.md`, and
  `mathdevmcp_cli_verification.md`; and
- no claim-bearing HMC or default-readiness field.

The result decision table must state the engineering criterion, diagnostic
criterion, hard veto status, main uncertainty, next justified action, and
nonconclusions. The inference-status table must state hard vetoes, whether a
ranking is statistically supported, descriptive-only differences,
default-readiness, and next evidence needed.
