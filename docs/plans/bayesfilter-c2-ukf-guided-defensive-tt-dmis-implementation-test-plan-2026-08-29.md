# C2 UKF-Guided Defensive TT-DMIS Implementation and Test Plan

Date: 2026-08-29

Status: skeptically reviewed on 2026-08-29. Verdict:
`PASS_FOR_BOUNDED_IMPLEMENTATION_AND_DIAGNOSTIC_EXECUTION`. The review permits
the stated implementation and evidence-gated campaign; it is not a scientific,
default-readiness, HMC, or posterior promotion. The implementation, fixed-half
smoke, serious run, and independent 33-point alpha/nu calibration pilot have
now executed. The pilot estimates were finite, but bootstrap minimizer
stability was `0.2865` (required `>=0.80`), so the predeclared rule fell back
to the fixed `(alpha, nu)=(1/2,8)` candidate. No selected-allocation or
promotion verdict was issued.

Mathematical specification:
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/ukf_guided_defensive_tt_dmis_analytical_gradient.tex`

Plan review:
`docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-plan-review-2026-08-29.md`

Planned output root:
`docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/`

## 1. Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can a frozen data-guided mixture of the retained squared-TT proposal and a full-support transformed-observation UKF/Kalman Student proposal materially reduce the C2 `n=4` importance-weight degeneracy while preserving the analytical score of the same finite likelihood scalar? |
| Candidate mechanism | At every transition, draw equal-size deterministic banks from the already-normalized retained TT proposal (including its internal source floor) and a per-ancestor Student proposal whose moments come from the C2 log-square transformed-observation UKF/Kalman update; use one timewise alpha common to all ancestors, and weight all draws by the complete outer mixture density and exact raw C2 transition/observation densities. |
| Expected failure mode | The retained TT may be so poor that even a balanced mixture wastes particles; the Gaussian log-chi-square moment closure may miss nonlinear or heavy-tailed mass; or the generalized base masses may be wired incorrectly. A raw-observation UKF is already ruled out mathematically because its population gain is zero. |
| Primary engineering criterion | The shared APF evaluator implements the manuscript's nonuniform-base scalar, the uniform route is unchanged, and the manual recursive score matches centered finite differences of that same scalar. |
| Primary mechanism criterion | Across independent paired branch replicates, the selected DMIS arm has a positive uncertainty-supported change in log minimum normalized ESS relative to retained TT alone, while remaining finite and reference-compatible. This nominates a candidate; it is not a default or superiority claim. |
| Promotion criterion | None in this campaign. A pass permits an optional diagnostic candidate and a separate randomized-likelihood/posterior plan. |
| Promotion veto | Any loss to a constructed simple proposal adversary in a salient situation, any same-scalar mismatch, or any proposal-law failure. A promotion veto does not automatically stop the repair ladder. |
| Continuation veto | Invalid target/reference identity, broken source snapshot, nonnormalized base masses, incomplete mixture denominator, non-positive proposal scale, non-finite scalar/score, failed uniform-route regression, GPU/XLA provenance failure, or exhausted budget. |
| Repair trigger | Correct scalar plus low ESS triggers proposal repair; failed transformed-guide covariance checks trigger guide-policy review; failed finite differences trigger score/call-chain repair; a candidate loss triggers the next predeclared arm rather than a scientific-direction rejection. |
| Explanatory diagnostics | Per-time component contributions, ESS, log-weight spread, maximum normalized weight, transformed-observation innovation and covariance margins, TT/defensive density ratio, pilot second moment, score residual, and branch dispersion. |
| Must not be concluded | End-to-end differentiation through UKF/TT fitting, exact inference from one permanently frozen branch, exact pseudo-marginal HMC, posterior correctness, source-faithful reproduction of the complete Zhao--Cui solver, universal performance for arbitrary data, default readiness, or statistical superiority. |

The retained-TT candidate's prior failure is a repair trigger, not a
continuation veto. The proposed method is allowed to fail scientifically after
its implementation is shown correct.

## 2. Claimed Target and Actual Computation

The claimed target is the finite frozen-proposal log likelihood

```text
sum_t logsumexp_i(
    log_base_mass[t, i]
  + selected_previous_log_weight[t, i]
  + exact_transition_log_density[t, i]
  + exact_observation_log_density[t, i]
  - selected_auxiliary_log_probability[t, i]
  - complete_mixture_log_density[t, i]
)
```

with the initial-state analogue at `t=0`. For a TT sample,
`log_base_mass = log(1-alpha) - log(N_TT)`; for a defensive sample it is
`log(alpha) - log(N_D)`. Each time row must satisfy
`reduce_logsumexp(log_base_mass) == 0` within dtype tolerance. The old uniform
route is the special case `log_base_mass = -log(N)`.

Here `q_TT` is the current complete retained proposal, including its small
internal Hermite/Student floor. The new outer alpha does not reuse that floor's
categorical probability. Extracting a pure squared-polynomial component would
be a separate untested candidate and is outside the first implementation.

The score is the analytical derivative of this exact finite scalar with
respect to runtime `(gamma, log(beta))`. The observed data, reference
parameter, UKF moments, TT fit, alpha, Student degrees of freedom, proposal
densities, states, component labels, auxiliary laws, and ancestors are frozen.
Autodiff may be used only as an independent diagnostic. It is not the
claim-bearing score path.

The target is not the direct TT normalizer and not an end-to-end derivative of
the proposal compiler.

## 3. Source and Classification Boundary

| Operation | Source anchor | Classification |
| --- | --- | --- |
| Squared TT plus a positive defensive density | Zhao--Cui (2024), Eq. (13), local paper lines 548-573; author `TTSIRT/marginalise.m:25-85` | `source_faithful` operation |
| Conditional TT transport and exact target/proposal correction | Zhao--Cui Proposition 2, Eqs. (20)-(23), Algorithm 3, local lines 592-670 and 825-924; author `full_sol.m:33-42` | `source_faithful` operation |
| Frozen randomness/settings | Zhao--Cui sampling route adapted for a deterministic parameter evaluator | `fixed_hmc_adaptation` only for freezing |
| Retained current-state marginal independent of ancestor | Current C2 route, not Zhao--Cui's full conditional map | `extension_or_invention` |
| Log-square transformed-observation UKF/Kalman Student defense, separate alpha, equal deterministic banks, generalized APF base mass, and analytical score | Manuscript derivation | `extension_or_invention` |

The complete route must never be called source-faithful. The local paper and
pinned author code remain the source authorities; no pre-2026-08-21 LEDH result
is used as evidence. LEDH is an architectural analogy only.

## 4. Evidence Contract

### 4.1 Exact scope

The first C2 scope is:

- data-generating fixture identity `zc24_sv_vector_extension_v1` and
  value/score model identity `c2_sv_gamma_log_beta_stationary_v1`;
- state and observation dimension 4;
- observation seed 42 and model seed 52;
- process-noise standard deviation `sigma=1`, with the realized coupling
  matrix and its digest recorded in every proposal and run manifest;
- 20 observation rows (`time_steps=20` in code), corresponding to terminal
  index `T=19` in the manuscript's inclusive `t=0,...,T` notation;
- runtime/reference parameter `(gamma, log(beta)) = (0.6, log(0.4))`;
- TensorFlow/TFP implementation, float64 for correctness and diagnostic runs;
- XLA JIT enabled on the GPU route;
- TensorFlow GPU memory growth configured and verified before device
  initialization; and
- proposal compilation at `theta_star`, followed by a runtime API that accepts
  only theta.

The existing attempt05 T=20 PF reference
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/reference_n4_s42.json`
is the total-likelihood comparator. Its reported total is `-66.6979549734`
with total SE `0.0041582`. It is not a tuning target and does not provide the
per-step covariance required for a full paired time-series inference claim.

The retained TT source for this campaign must be a newly fingerprinted output
of the current retained-output route. The preserved attempt05 scalar
`36.9423456524` is historically incompatible with the current route and cannot
be treated as the same snapshot. Therefore this campaign may answer whether
DMIS repairs the current retained proposal, but it must not claim to repair the
unrecoverable historical TT realization unless its exact snapshot is first
reconstructed and verified.

### 4.2 Baseline ladder and heuristic adversaries

Use the same observations, total particle count, runtime theta, auxiliary-law
construction, branch seeds, and target densities for all eligible arms:

1. retained TT categorical proposal from the current implementation, used
   only as the protected engineering-regression authority for arm 2 rather
   than as a second scientific arm;
2. retained TT only through the generalized base-mass evaluator, the final
   scientific baseline;
3. transformed-observation UKF/Kalman Student only with the predeclared
   `nu=8` defensive baseline;
4. fixed `alpha=1/2`, `nu=8` TT/guide DMIS, justified by the manuscript's
   factor-two population second-moment bound; `nu=8` has finite fourth moments
   while retaining heavier-than-Gaussian tails and is a baseline, not a tuned
   default;
5. pilot-selected interior-alpha TT/transformed-guide DMIS;
6. bootstrap conditional proposal;
7. current Gaussian-hint marginal proposal; and
8. stationary Gaussian independence proposal.

Arms 3, 6, 7, and 8 form the cheap heuristic adversary set. Their rationale is
respectively data-guided classical filtering, exact-transition propagation,
the strongest simple prior C2 proposal, and a proposal requiring no current
data. A complex mixture losing to any of them in a salient situation is a
promotion veto and must be the headline of the result.

The salient situations are the historical discrepancy times `t=3` and `t=4`,
the campaign's lowest-ESS time, the largest transformed-observation innovation
time, and the full 20-row total. The first two times are predeclared; the
campaign minimum and largest-innovation times are data-selected explanatory
diagnostics and cannot carry a separate inferential claim. Comparisons must be
reported conditionally for each situation, not only as an average.

### 4.3 Engineering gates

All of the following must pass before interpreting ESS or likelihood error:

- every resolved base-mass row is finite, has shape `[particle]`, is immutable
  after branch issuance, and log-sums to zero;
- setting every base mass to `-log(N)` reproduces the old scalar, score,
  normalized weights, ESS, and XLA result within `2e-12` in float64;
- unequal component counts and masses reproduce direct high-precision
  recomposition on analytic one-dimensional fixtures;
- permuting component labels and their samples leaves the scalar and score
  unchanged;
- every TT-bank and transformed-guide-bank sample is evaluated under the same complete
  `logaddexp(log(1-alpha)+log_q_tt, log(alpha)+log_q_defensive)` denominator;
- the raw-observation zero-cross-covariance negative control passes;
- the log-chi-square mean/variance, transformed observations, Kalman solve
  residuals, symmetry, and positive-definite scale checks pass;
- Student sampling and density agree on closed-form moments where they exist,
  normalization fixtures, and whitened-radius diagnostics;
- the branch compiler obtains each next auxiliary law through the shared
  generalized APF prefix evaluator, not a compiler-local weight recursion;
- C2 local analytical scores pass independent centered finite differences;
- the complete recursive score passes centered finite differences of the same
  frozen branch at the reference point and two independent directions with
  absolute/relative tolerance `2e-5`;
- eager diagnostic, graph, and XLA values agree within `2e-10` in float64;
- the traced hot path has a stable explicit signature, no pfor,
  `tf.vectorized_map`, sample-wise Python loop, NumPy runtime path, or silent
  eager fallback; and
- malformed scale matrices, nonnormalized base masses, stale proposal IDs,
  incomplete mixture densities, or non-finite values fail closed.

Tolerances above are inherited from the already passing frozen-proposal route
and are regression hypotheses, not universal defaults. If healthy old-route
fixtures fail them, investigate numerical scaling before relaxing them.

### 4.4 Mechanism and uncertainty criteria

The fixed half-mixture is the first discriminating arm because its role follows
from the manuscript's component-robust second-moment proposition and
half-mixture corollary, not from tuning. The pilot-selected arm is tested only
after the half-mixture is correct.

For the bounded 20-row diagnostic, use twelve paired branch replicates after
the smoke passes. Every replicate evaluates every scientific arm on the same
target data and retained snapshot, with an explicit seed map and disjoint named
substreams for proposal-specific randomness. This replicate grouping is the
pairing unit even when the within-pair Monte Carlo streams are independent.
Use common random numbers only where a tested, marginal-preserving map exists
for both proposal laws; otherwise keep the method-specific streams independent
without discarding the paired experimental design. Report paired intervals
for:

- `log(min_normalized_ess_candidate / min_normalized_ess_retained_tt)`;
- total log-likelihood difference from the PF reference;
- maximum normalized weight; and
- wall time, separately for compilation and repeated evaluation.

The primary mechanism criterion is that a predeclared 95% paired bootstrap
interval for the mean log minimum-ESS ratio of the selected DMIS arm against
retained TT excludes zero on the positive side and at least 10 of 12 paired
ratios are positive. Report the corresponding exact paired sign-test result.
This supports only the claim that the candidate improves this degeneracy
diagnostic in the frozen C2 scope. Extreme per-time minima remain noisy; all
raw branch values must be shown.

Reference compatibility requires finite totals and an absolute mean total
difference no larger than `max(1.0 nat, 3 * combined_SE)`. This is a continuation screen
against gross target mismatch, not proof of likelihood correctness or
superiority.

No viable candidate is ranked as best unless the paired uncertainty analysis
supports that ranking. Passing a hard screen means only that the arm remains
viable.

### 4.5 Exactness nonclaims

The artifact must state that one fixed branch defines a deterministic
approximate likelihood. An exact pseudo-marginal claim would require fresh
randomized likelihood evaluations or an invariant extended-state update, plus
separate unbiasedness and posterior tests. Neither is in this campaign.

## 5. Default and Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| `alpha=1/2` | Theory-derived diagnostic baseline from the symmetric factor-two second-moment bound; not a default | Wastes half the budget if one component is uniformly poor | one-step pilot moment and N=1024 smoke |
| Interior alpha range | Fixed grid `[0.10, 0.90]` in increments of `0.025`; the 0.10 floor caps each component-relative second-moment factor at 10. This is a bounded calibration hypothesis, not a default | Too narrow prevents useful allocation; too wide weakens defense | held-out pilot objective, boundary frequency, and component contribution table |
| Alpha selection | Use convex `J(alpha)` only to nominate; select or reject with the manuscript's exact equal-bank DMIS variance estimate on an independent validation pilot, then freeze before final banks | Treating a random-mixture proxy as exact DMIS variance, pilot overfit, or noisy boundary choice | both objective curves, component variance terms, independent validation, boundary frequency |
| Equal bank counts | Fixed `N_TT=N_D=N/2` to remove data-dependent discrete counts and guarantee both banks are represented | Computational waste when alpha is extreme | contribution ESS by bank and fixed-half comparison |
| Timewise common alpha | Required by the derived row-normalized base-mass schema; ancestor-specific gates are not implemented | Misses useful ancestor-specific adaptation | stratify diagnostics by ancestor innovation, but do not change the gate in this campaign |
| Student degrees of freedom | Fixed defensive baseline `nu=8`; calibration candidates `{4, 8, 16, Gaussian limit}` with scale `((nu-2)/nu) * P_D` for finite nu, so every arm has UKF covariance `P_D`. The baseline has finite fourth moments and non-Gaussian tails; it is not a universal default | Heavy tails dilute bulk or light tails miss support; `nu=4` makes covariance diagnostics noisier | covariance and radial-tail checks plus pilot second moments |
| C2 guide convention | Frozen `transformed_log_square_moment_ukf_v1`: exact `m_chi=-EulerGamma-log(2)`, `v_chi=pi^2/2`, exact conditional transition moments, and the affine transformed-observation Kalman update. No sigma-point tuning is needed because the transformed map is affine | A raw-observation UKF has zero gain; log-chi-square Gaussian closure may be a poor shape approximation; zero observations make the exact log transform undefined | raw-zero-gain proof fixture, log-chi moment checks, nonzero-data check, transformed solve residuals, and comparison with the GH hint |
| Covariance ridge/inflation | Zero is the initial fixture value because the checked transformed-observation conditional covariance is the specified proposal scale and changing it would change that object; factorization fails closed instead of silently altering it. Any nonzero value is a Class-C candidate requiring a bias/robustness calibration and manifest identity | Silent proposal alteration or Cholesky failure | minimum relative eigenvalue and no-fire healthy regression |
| Current retained TT snapshot | One fresh campaign input and negative-control geometry, not the historical attempt05 object | Repeats an ill-conditioned or nonreproducible fit | snapshot fingerprint and direct call-chain diagnostic; fit-to-fit variability is not measured or claimed in this budget |
| N=1024 smoke, N=8192 serious | Smoke is bounded mechanics; serious N matches the prior diagnostic scale but is not tuned | Too few particles for stable tails | paired branch interval and maximum weight |
| Twelve serious branches | Bounded paired uncertainty arm, not enough for universal tail ranking | Interval remains too wide | report interval width; do not extend budget silently |
| Auxiliary law at `theta_star` | Existing frozen-branch convention | Poor genealogy away from reference theta | neighborhood score checks and weight spread |
| Float64 C2 diagnostic | Existing correctness lane; production float32/TF32 is outside this campaign | Hides production precision effects | separate future float32/TF32 plan |

The selection pilot may use the observed data because it changes only the
fully accounted proposal. It must use random banks independent of the final
evaluation. Reusing final banks for alpha or nu selection is forbidden.

## 6. Mathematical-to-Code Crosswalk

| Manuscript object | Required code endpoint | Executable evidence |
| --- | --- | --- |
| Eqs. `smooth-mixture`, `defensive-student` | New normalized Student component and complete log-mixture density | normalization, support, logaddexp, and sampling-law tests |
| Eqs. `c2-raw-zero-cross-covariance`--`c2-transformed-guide-covariance` and `student-scale` | C2 transformed-observation guide compiler using checked batched TensorFlow solves and the covariance-matched Student convention | zero-gain negative control, exact log-chi moment oracle, covariance residual, SPD and zero-observation negative tests, and empirical/closed-form covariance checks |
| Eqs. `initial-dmis-weight`, `sequential-dmis-weight` | Resolved base-mass tensors in `PreparedFrozenProposalBranch` and shared `_evaluate_core` | direct recomposition, unequal masses/counts, permutation invariance |
| Eq. `uniform-apf-special-case` | Default uniform base mass issued by `prepare_frozen_proposal_branch` | old/new full regression and call-chain wiring test |
| Eqs. `increment-score`--`total-sequential-score` | Existing centered-mark recursion after base-mass generalization | complete same-scalar finite difference and XLA parity |
| Eqs. `joint-pilot-target`--`joint-pilot-estimator` and `dmis-variance` | Offline nomination plus exact equal-bank variance validation on the joint ancestor-state proposal, separate from runtime | convexity grid for joint `J_t`, explicit squared `W/a` term checks, component-stratified variance curve, independent pilot/validation, frozen selection manifest |
| Propositions `dmis-unbiasedness`, `sequential-dmis-unbiased` | C2 deterministic-bank compiler | exact discrete/linear-Gaussian fixtures and Monte Carlo diagnostic |

Required call chain:

```text
benchmark driver
  -> current retained snapshot + C2 transformed-observation proposal compiler
  -> C2 deterministic-bank branch compiler
  -> prepare_frozen_proposal_branch (repository-issued base masses and ID)
  -> prepare_frozen_proposal_apf_program
  -> FrozenProposalAPFProgram.compiled(jit_compile=True)
  -> shared _evaluate_core
  -> C2 manual exact-model score methods
```

No C2-local evaluator may duplicate `_evaluate_core`.

## 7. Implementation Phases

### Phase 0: freeze the specification and regression baseline

1. Compile and render-inspect the LaTeX document.
2. Complete the MathDevMCP audit and record backend limitations in its
   appendix.
3. Preserve current focused-test output for the generic APF and C2 branch
   compiler under a new versioned artifact.
4. Record the current branch schema, scalar, score, normalized weights, and
   branch fingerprint on small deterministic fixtures.

Exit: the document builds, equations have no unresolved references, and the
old uniform evaluator has an executable protected baseline.

### Phase 1: generalize the shared APF measure

Modify `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` so the prepared
branch owns resolved tensors:

- `initial_log_base_mass: [particle]`;
- `transition_log_base_mass: [time-1, particle]`.

The repository factory accepts omitted masses only as a backward-compatible
request for uniform masses, resolves them immediately, validates each row by
`reduce_logsumexp == 0`, and includes them in the branch fingerprint and
manifest. `_evaluate_core` adds each log base mass and removes the hard-coded
`-log(N)` increment. The normalized-weight and centered-score recursion stays
shared.

Update all branch combiners to preserve one shared base measure. A product of
state-coordinate proposal blocks must not add base masses once per block.

Exit: old uniform-route tests and new nonuniform fixtures pass. This is a hard
continuation gate.

### Phase 2: implement the frozen transformed-observation Student component

Add a TensorFlow-only component under `bayesfilter/highdim/` with:

- an executable negative control proving that the raw C2 augmented-observation
  covariance and gain are zero within tolerance;
- exact TensorFlow constants for the mean and variance of `log(chi_square_1)`;
- setup-time `log(y^2)` transformation at `theta_star`, failing closed on an
  exact zero rather than silently adding an offset;
- batch-native per-ancestor prior moments and the affine transformed-observation
  Kalman update from the manuscript, with no sigma-point hyperparameters;
- checked conditioning solves and Cholesky factorization;
- multivariate Student sampling from stateless TensorFlow random inputs;
- normalized log density with the same scale convention as the sampler;
- complete provenance for transformed-guide convention, log-chi-square moments, degrees of freedom,
  covariance policy, transform identity, data/time/ancestor identity, and
  `theta_star`; and
- explicit `extension_or_invention` and `scout_not_truth` status.

Use `tf.function` with an explicit stable signature for repeated numerical
kernels and evaluate XLA after eager diagnostic parity. Do not introduce pfor,
`tf.vectorized_map`, NumPy, or a sample-wise Python loop in the hot path.

Exit: moment, density, sampler, support, finite, and fail-closed tests pass.

### Phase 3: implement the deterministic-bank compiler

Extend the C2 compiler without adding a second APF evaluator:

1. Keep the initial state proposal and initial base mass uniform.
2. At each transition, form equal fixed TT and transformed-guide banks with disjoint
   stateless random streams.
3. Draw ancestors for each bank from the same shared auxiliary law.
4. Evaluate both `q_TT` (the complete current retained density, including its
   internal floor) and `q_D` at every generated state and store the complete
   outer mixture log density, regardless of generating component.
5. Store the exact component base masses `log(1-alpha)-log(N_TT)` and
   `log(alpha)-log(N_D)`.
6. Obtain the next auxiliary law by evaluating the completed prefix through
   the shared APF program at `theta_star`.
7. Bind component labels and every proposal/control fingerprint in the
   compilation manifest, while the claim-bearing branch receives only the
   resolved states, densities, base masses, auxiliary laws, and ancestors.

Exit: direct recomposition, permutation, component-denominator, branch-ID,
same-scalar score, and shared-call-chain tests pass.

### Phase 4: fixed-half mixture diagnostic

Run the smallest exact and stochastic diagnostics in this order:

1. one-dimensional Gaussian-mixture integral with an analytic normalizer;
2. two-step linear-Gaussian state-space model with a Kalman likelihood oracle;
3. C2 `n=1`, `T<=3`, `N<=128` CPU-only fixture;
4. C2 `n=4`, `T=5`, `N=1024` trusted GPU/XLA smoke.

Use only `alpha=1/2` and the fixed `nu=8` Student baseline for the C2 smoke.
The analytic fixtures may exercise all predeclared tail conventions, but no
tail or allocation choice may be tuned from the smoke.

Exit: every engineering gate passes. A finite low-ESS half mixture continues
to Phase 5 unless a continuation veto fires.

### Phase 5: offline alpha and tail calibration

For each time step, use one alpha common to every ancestor and independent
pilot banks from the half mixture to estimate the manuscript's joint
ancestor-state `J_t(alpha)` on the 33-point grid `{0.10, 0.125, ..., 0.90}`.
For every nu candidate, use 4,096 pilot draws total, split equally between
components. The `J_t` pilot summand
must include the square of the exact compile-time
`W_{t-1}^j / a_{t-1}^j` factor, as in the manuscript's
`joint-pilot-estimator`; averaging
per-ancestor conditional objectives is the wrong target. Verify numerical
convexity against the analytical second derivative sign and treat this as
nomination evidence only. On a second independent validation pilot, estimate
every component mean and variance from another 4,096 total draws in the
manuscript's exact equal-bank DMIS variance formula. Use 2,000 stratified
bootstrap resamples to assess the validation curve. Advance one selected pair
only if all estimates are finite, the fixed-half baseline variance estimate is
strictly positive, the bootstrap relative standard error of the candidate's
variance estimate is at most 20%, at least 80% of bootstrap minimizers retain
the same nu and an alpha within one grid step, and the one-sided 95% interval
for its variance ratio against the fixed `(alpha=1/2, nu=8)` arm is below one.
Otherwise omit the selected-alpha arm from the final comparison and retain the
fixed-half baseline. Freeze any advanced time-indexed allocation and tail
policy in a repository-issued artifact before creating final banks.

No runtime or theta-dependent retuning is allowed. A noisy, flat, or
boundary-unstable objective falls back to the fixed half mixture as a
diagnostic candidate, not as a promoted default.

Exit: selection identity, pilot/final independence, objective curve, and
validation decision are preserved.

### Phase 6: trusted GPU/XLA 20-row diagnostic

Use one fresh current-route retained snapshot and twelve independent paired
branch replicates for every final scientific arm. Apply the seed-map and
common-random-number rule in Section 4.4, record every method-specific stream,
and preserve all twelve replicate-level contrasts. Record all evidence named in
Sections 4.2-4.4. The driver must emit a run manifest, structured result JSON, readable
result Markdown, focused logs, snapshot and proposal manifests, and the
machine-readable heuristic-dominance verdict.

The run manifest records the git commit and dirty-tree state, exact command,
conda environment and package versions, CPU/GPU device identity, memory-growth
verification, dtype/TF32/XLA settings, data and snapshot fingerprints, every
pilot/final seed, wall time, plan and result paths, and all output paths.

Exit: issue either `CANDIDATE_VIABLE_FOR_RANDOMIZED_LIKELIHOOD_TESTING`,
`CANDIDATE_REJECTED_PROPOSAL_VARIANCE`, or an engineering/continuation failure.
Do not issue a default, HMC, posterior, or exactness verdict.

### Phase 7: terminal call-chain and math audit

Trace every claim-bearing consumer endpoint through the shared APF evaluator,
base-mass tensors, complete mixture density, C2 exact model, and analytical
score. Add an executable wiring test. Compare every crosswalk row above to its
code and test. Classify each as `correct`, `wrong relative to the stated
target`, `unsupported`, or `not checked`.

## 8. Planned Files

Expected implementation files, subject to existing module boundaries found at
execution time:

- modify `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py`;
- modify `bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py`;
- add `bayesfilter/highdim/c2_transformed_observation_student_proposal_tf.py`;
- add `docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_20260829.py`;
- add `docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_pilot_20260830.py`;
- modify `tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py`;
- add `tests/highdim/test_c2_transformed_observation_student_proposal_tf.py`;
- add `tests/highdim/test_c2_ukf_guided_tt_dmis_tf.py`.

Do not modify unrelated TT fitting code merely to make this candidate pass.
Work with the current dirty tree and preserve all user changes.

## 9. Exact Planned Commands

The implementation turn must first inspect the active conda environment and
record its package/device versions. The intended environment is `tftwogpu`.

Focused CPU-only tests deliberately hide CUDA before TensorFlow import:

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -n tftwogpu \
  pytest -q tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py

CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -n tftwogpu \
  pytest -q tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py

CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -n tftwogpu \
  pytest -q tests/highdim/test_c2_transformed_observation_student_proposal_tf.py

CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -n tftwogpu \
  pytest -q tests/highdim/test_c2_ukf_guided_tt_dmis_tf.py
```

Before any GPU debugging, run trusted/escalated `nvidia-smi` and an escalated
TensorFlow device/memory-growth probe. The planned smoke and serious commands
are:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/chakwong/anaconda3/bin/conda run -n tftwogpu \
python docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_20260829.py \
  --mode smoke \
  --output-root docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/smoke-attempt01

TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/chakwong/anaconda3/bin/conda run -n tftwogpu \
python docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_20260829.py \
  --mode serious \
  --output-root docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/attempt01
```

Both GPU commands require escalated/trusted device access under `AGENTS.md`.
Every retry uses a fresh versioned output directory. The driver must resolve
each mode to the exact dimensions, horizon, particle counts, seeds, calibration
grid, branch count, dtype, and XLA policy declared in this plan; it must emit
that resolved configuration before computation and fail closed on an
unrecorded CLI override.

## 10. Compute and Attempt Budget

- CPU-only implementation and focused tests: at most 2 wall-clock hours across
  repairs.
- GPU/XLA smoke: at most 3 attempts and 45 total GPU minutes.
- Pilot calibration: at most 2 independent pilot/validation attempts and 60
  total GPU minutes; no grid expansion beyond the declared `C` and nu sets.
- 20-row diagnostic: at most 1 fresh TT fit and 12 paired branch sets, with at
  most 2 infrastructure retries and 3 total GPU hours.
- No HMC, posterior sampling, package installation, environment mutation, or
  expanded particle/seed campaign is authorized by this plan.

A localized harness, serialization, or resource failure may be repaired and
retried within this budget if the target, data, method, criteria, hardware
class, and privacy boundary remain unchanged. Record the failure and repair.

## 11. Stop Conditions and Interpretation

Stop implementation interpretation immediately for a failed uniform-route
regression, incomplete proposal density, nonnormalized base mass, invalid guide
scale, same-scalar score failure, or corrupted artifact. Repair within budget
before continuing.

Do not stop the research ladder merely because retained TT or half-DMIS has low
ESS. That is an expected candidate result and motivates the next predeclared
repair. Stop the campaign when a true continuation veto fires, the budget is
exhausted, or the remaining action would change the scientific target or
method.

The final result note must answer separately:

- Did the shared implementation and call chain pass?
- Did the current candidate pass the proposal-quality criterion?
- Did any heuristic-dominance veto fire?
- Is any ranking statistically supported?
- What differences remain descriptive only?
- What exact claim is not being made?
- What next evidence is justified?

## 12. Pre-Mortem

The run could pass while misleading us if the complete mixture density is
correct only on generated points, if alpha is selected on final random banks,
if the current snapshot is mislabeled as the historical attempt05 object, if a
high average ESS hides one catastrophic time, or if proposal compilation time
is excluded from the cost comparison. The earliest checks are independent
density integration, pilot/final seed separation, snapshot fingerprints,
per-time conditional tables, and separate compile/repeated-evaluation timing.

The run could fail for engineering rather than scientific reasons if the new
base mass is counted once per state block, the Student scale convention differs
between sampling and density, the transformed-guide covariance loses positive definiteness,
or the XLA graph retraces. The direct fixtures and stable-signature tests occur
before the T=20 candidate is interpreted.

## 13. Post-Run Red-Team Requirement

The result must state the strongest alternative explanation, the observation
that would overturn the conclusion, and the weakest evidence. In particular,
an ESS improvement may come entirely from the transformed-guide component and show that the
TT is unnecessary; the defensive-only arm is required to expose that outcome.

## 14. Execution Addendum (2026-08-30)

The Stage 1 Claude review in `/tmp/c2_ukf_dmis_stage1_review.md` was checked
against the current source. Its manuscript verdict is accepted: the three
clarity findings F1--F3 were repaired in the LaTeX source, and the document
rebuild and MathDevMCP rerun are recorded separately. The review did not audit
the later executable implementation.

The implementation audit found and repaired two concrete defects before the
serious run:

1. TensorFlow's `stateless_gamma` uses a rate parameter. The transformed
   Student sampler therefore uses `beta=0.5` for a chi-square draw, not
   `beta=2.0`; an empirical covariance regression now protects this identity.
2. The fixed-TTSIRT block combiner previously discarded resolved nonuniform
   base masses. It now verifies equality across blocks, carries the first
   resolved masses into the combined branch, and has a regression test.

These repairs do not change the declared C2 target or the fixed-half route.
The focused suite currently passes 25 tests. The corrected `smoke-attempt03`
passed the engineering, finite-density, eager/XLA, repeat, trace, GPU, and
centered-score screens, but its heuristic-dominance veto fired and it remains
descriptive one-seed evidence only.

### Skeptical pre-run audit

The serious command is allowed because the target, fixture, current-route
snapshot policy, proposal families, `(alpha, nu)=(0.5,8)` arm, N=8192,
twelve seeds, float64/XLA/GPU lane, and three-hour campaign budget are
unchanged from the reviewed contract. The following risks are recorded rather
than hidden:

- At the time of this pre-run audit the driver did not yet implement the Phase
  5 33-point alpha/nu pilot, independent validation variance, or bootstrap
  selection. The serious result was therefore intentionally fixed-half and
  not a selected-allocation comparison; the follow-up pilot is recorded in
  Section 16 below.
- The mechanism screen is a nomination diagnostic, not a superiority test; the
  serious result must preserve all replicate values and report no statistical
  ranking unless the predeclared paired uncertainty calculation is present.
- The current code's result summary is a bounded diagnostic and does not by
  itself establish the full Phase 6 paired-bootstrap criterion. A heuristic
  arm beating the DMIS arm in a salient situation is a promotion veto, while a
  low ESS or a failed candidate is only a repair trigger unless a continuation
  veto fires.
- The preserved serious run predates the later additive observability and
  reporting fixes. Its manifest binds the exact source snapshot that generated
  the branch values; a stale or incomplete manifest would be an artifact
  failure, not evidence. The current driver now emits per-time maximum
  normalized weights and uses the corrected heuristic comparison for future
  runs.

The earliest checks are the focused tests, the corrected smoke artifact, the
trusted GPU/memory-growth probe, and the serious run's per-branch engineering
fields. No posterior, pseudo-marginal, HMC, source-faithfulness, or default
claim is authorized by this execution.

### Executed commands and artifacts

The corrected smoke was executed at:

`docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/smoke-attempt03/`

The bounded serious command is:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_20260829.py \
  --mode serious \
  --output-root docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/attempt01
```

The serious result, if the command completes, is preserved under the fresh
`attempt01` directory with its own run manifest, proposal manifest, branch
records, logs, and result note. Any failed or superseded attempt remains
readable and is not overwritten.

The terminal result note is
`docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-execution-result-2026-08-30.md`.

## 15. Terminal Execution Record (2026-08-30)

The serious command completed in the trusted `tftwogpu` environment. It used
one fresh retained fit, N=8192, 20 rows, six families, and the twelve seeds
`9201`--`9212`. Wall time was `2236.63` seconds, including `1081.60` seconds
for the retained fit. Both visible GPUs were initialized with verified memory
growth; the placement probe and all branch evaluations ran on GPU with XLA
enabled and float64 tensors.

The raw run artifact is:

`docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/attempt01/`

The original driver result reports `engineering_pass=True` and a finite fixed-
half mechanism screen. Its heuristic table was found to compare arms against
retained TT rather than compare the DMIS candidate against the declared simple
adversaries. That table is not used for the terminal decision. The diagnostic-
only post-run auditor corrects the comparison and preserves its output at:

`docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/attempt01/paired_audit.md`

The authoritative paired result is:

- mean log minimum-ESS ratio (DMIS/retained TT): `5.3839341272`;
- 95% deterministic percentile-bootstrap interval: `[5.1832197155,
  5.5999535634]` from 20,000 resamples (seed `20260830`);
- positive contrasts: `12/12`; exact two-sided paired sign-test p-value
  `0.00048828125`; and
- mean absolute DMIS/PF total gap: `0.0157258384`, below the `1.0`-nat
  compatibility tolerance.

Thus the fixed-half candidate passes the predeclared paired ESS nomination
criterion and is classified as
`CANDIDATE_VIABLE_FOR_RANDOMIZED_LIKELIHOOD_TESTING`. This is not a promotion:
the corrected heuristic table finds that `bootstrap_conditional`,
`defensive_student`, and `gaussian_hint_marginal` have lower mean absolute
per-step error than DMIS at each salient time `t=3`, `t=4`, and the retained-
TT lowest-ESS time `t=11`. The heuristic-dominance veto therefore fires.

The preserved attempt01 did not include the Phase 5 alpha/nu pilot or the
later per-time maximum-normalized-weight field. The current driver now emits
the latter. The alpha/nu pilot was then executed as a follow-up in
`pilot-attempt02`; its selection gate failed on bootstrap minimizer stability,
so no selected alpha or tail value is supported and `(0.5, 8)` remains the
theory-led diagnostic baseline. The run does not establish pseudo-marginal
exactness, posterior correctness, HMC readiness, source-faithful Zhao--Cui
reproduction, default readiness, or statistical superiority.

## 16. Alpha/Nu Pilot Follow-Up (2026-08-30)

The previously deferred Phase 5 pilot was implemented in
`docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_pilot_20260830.py` and
executed under the unchanged pilot budget. The first attempt stopped before
numerical work because imports initialized TensorFlow logical devices before
the memory-growth helper; it is preserved as
`pilot-attempt01/SUPERSEDED.md`. The driver was repaired to configure memory
growth immediately after the TensorFlow import, and the fresh retry completed
at:

`docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/pilot-attempt02/`

The retry used one fresh retained fit, 19 transition times, each declared
`nu` candidate (`4`, `8`, `16`, Gaussian limit), the 33-point alpha grid
`0.10,...,0.90`, 4,096 objective draws split into two 2,048-member banks,
an independent validation bank of the same size, and 2,000 deterministic
bootstrap resamples. The joint pilot summand included the exact squared
`(W_j/a_j)^2` factor; every objective and validation value was finite, and the
analytical second-derivative estimates were nonnegative. The aggregate point
minimum was `nu=16, alpha=0.6`, with validation variance
`1.4544334272e-05` versus the fixed-half value `1.9400602566e-05` and a 95th
percentile variance ratio of `0.913913`. The selected-pair bootstrap relative
standard error was `0.075244`, but minimizer stability was only `0.2865`, so
the required `0.80` gate failed. The pilot status is therefore
`fallback_fixed_half`; no tuned pair was replayed into final banks and no
allocation was promoted.

The exact command was:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_ukf_guided_defensive_tt_dmis_pilot_20260830.py \
  --output-root docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/pilot-attempt02
```

The result, proposal manifest, snapshots, and run manifest are preserved under
that output root. The pilot is calibration evidence only; the fixed-half
serious result and its heuristic veto remain the terminal scientific decision.
