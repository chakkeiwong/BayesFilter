# C2 Mixture-UKF/APF Phase 5: Recursive Lagged Moment Map

Date: 2026-09-04  
Governing plan: `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Phase 4 close: `docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-backward-error-repair-close-20260904.md`  
Status: `EXECUTED_PASS_MECHANICS_C2_WIRING_PHASE5A_READY`

## Question and boundary

The Phase 4 defensive proposal is a valid finite program but fails the
predeclared ESS/promotion screen. The next question is narrower: can a
model-independent, lagged transition-moment map provide a numerically sound
coordinate system for recursive fitting, and can its behavior be separated
from proposal variance and C2 observation effects?

This phase does not retune the Phase 4 proposal, change the exact C2 target,
or claim that a better coordinate map improves filtering. The linear fixture
is a mechanics oracle. The C2 run is a small integration diagnostic whose
outputs are descriptive only.

## Mathematical object

Let (x_j) be a fixed carried cloud with normalized weights (w_j), and let
(m_j^-) and (Q_j) be the conditional transition mean and covariance for
ancestor (j). The predicted moments are

\[
 m^- = \sum_j w_j m_j^-, \qquad
 P^- = \sum_j w_j\left[Q_j+(m_j^- - m^-)(m_j^- - m^-)^{\mathsf T}\right].
\]

The implementation symmetrizes (P^-), checks finite values and strict
positive definiteness, and forms the lower Cholesky factor (L) with
(LL^{\mathsf T}=P^-). The physical/reference maps are named explicitly:

\[
 R_t(u)=m^-+Lu, \qquad T_t(x)=L^{-1}(x-m^-).
\]

The map used at time (t) is computed from the carried state at (t-1) and
is frozen before fitting or score evaluation at (t). It is therefore lagged;
there is no same-step fixed-point iteration in this phase.

For a linear transition (m_j^-=Ax_j+a), (Q_j=Q), the law of total
covariance gives (m^-=A m+a) and (P^-=APA^{\mathsf T}+Q) whenever the
carried cloud moments are (m,P). This is the primary executable parity
criterion. A map made from these moments has zero predicted-coordinate mean
and identity predicted-coordinate covariance in exact arithmetic.

## Research intent and evidence contract

| Field | Declaration |
| --- | --- |
| Main question | Does the generic lagged moment/Cholesky map reproduce known linear moments and remain finite through a short recursive C2 probe? |
| Candidate | `recursive_lagged_moment_map_v1`, with a static initial-map comparator |
| Mechanics baseline | Independent linear Kalman moment recursion and a direct dense covariance calculation |
| Primary pass criterion | Linear mean/covariance and map inverse/forward parity within `2e-12`; all recursive rows finite, SPD, and shape-valid |
| Promotion criterion | None in this phase; no proposal or filtering promotion is sought |
| Hard vetoes | target/data identity change, nonfinite or non-SPD map, failed linear parity, failed forward/inverse identity, missing records, or corrupted artifact |
| Explanatory diagnostics | map shift, Cholesky diagonal/eigenvalue margins, condition number, standardized-coordinate residual, posterior-cloud ESS, and observation response |
| Nonclaims | no posterior correctness, likelihood unbiasedness, ESS superiority, basis improvement, HMC readiness, or production/default readiness |
| Artifact | fresh `phase5-recursive-map-attempt02/` with manifest, raw JSON, result JSON/Markdown, command, formal sidecars, and source hashes; attempt 01 is preserved as a superseded harness run |

The C2 probe uses the exact transition and observation log densities from the
existing C2 model adapter. It does not replace the exact numerator with a UKF
closure and does not feed a fitted TT normalizer into the cloud update.

## Default and assumption audit

| Choice | Provenance and role | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| Float64 TensorFlow | repository correctness lane | GPU/CPU or XLA rounding mismatch | eager/XLA parity and manifest | reviewed diagnostic default |
| Fixed cloud size `N=128` | bounded Phase 5 budget | empirical moment noise hides map behavior | linear oracle plus a second `N=256` smoke if needed | convenience pilot |
| Equal symmetric linear sigma cloud | exact moment mechanics fixture | nonnegative weights do not represent every density | direct dense moment comparison | mechanics-only baseline |
| Stateless normal C2 bank | deterministic frozen diagnostic cloud | finite-cloud tails can distort map | ESS, covariance margins, and static-map residual | candidate diagnostic |
| Constant process covariance in C2 | C2 model contract (`sigma^2 I`) | wrong adapter for a state-dependent model | source call-chain inspection | source fact for C2 only |
| Lagged map, one update per step | recursive-map proposal in the governing plan | feedback can amplify approximation error | per-time map shift/residual | candidate |
| Static initial map | cheap comparator | can be unfair if its scale is not recorded | same cloud, same observations, map condition | comparator only |
| XLA compiled moment kernel | repository hot-kernel policy | unsupported operation or retracing | compile/eager parity and trace count | required for GPU probe |
| Frozen map/cloud for diagnostics | analytical-gradient boundary | differs from a fully adaptive algorithm | explicit manifest and finite-difference scope | required; score not claimed |

## Implementation contract

1. Add `bayesfilter/highdim/recursive_moment_map_tf.py` with a fixed-shape
   TensorFlow moment kernel, strict weight/shape checks, safe SPD reporting,
   an affine-map constructor, and an optional analytic moment/Cholesky tangent
   helper. The module must be model independent and contain no NumPy,
   pfor, `vectorized_map`, or sample-wise Python numerical loop.
2. Add focused tests for the linear law-of-total-covariance identity, map
   forward/inverse identities, invalid-weight and invalid-covariance
   fail-closed behavior, tangent finite differences, and eager/XLA parity.
3. Add `docs/benchmarks/run_c2_mixture_ukf_apf_phase5_recursive_map_20260904.py`.
   It runs (a) a recursive linear oracle using the existing Phase 0 fixture
   and (b) a three-step, `N=128` C2 cloud probe using the exact C2 model
   factors. The driver writes one record per time and never overwrites an
   existing output root.
4. Keep proposal selection, TT fitting, basis choice, and reference density
   unchanged. A future fitter integration must consume this map through the
   generic callable rather than a C2-local copy.

## Linear oracle protocol

Use the Phase 0 matrices and a fixed observation sequence formed from the
fixture observation plus a declared deterministic offset. At every step:

1. represent the current Gaussian moments by a symmetric (2D)-point cloud;
2. evaluate conditional transition means/covariances and call the generic
   moment kernel;
3. compare returned moments with the independent (A m+a) and
   (APA^{\mathsf T}+Q) calculations;
4. construct the lagged map and test (T_t(R_t(u))=u) on fixed points;
5. update the independent Kalman posterior moments and continue.

The static comparator reuses the first predicted map at every time. For both
maps record standardized predicted mean/covariance residuals, map condition,
minimum Cholesky diagonal, and shift from the previous lagged map. The lagged
map should have residuals at roundoff level in the oracle; this is a mechanics
check, not a claim about nonlinear density fitting.

## C2 integration probe

Construct a deterministic stationary Gaussian cloud with a stateless normal
bank and equal initial weights. For three observations, use the exact C2
transition matrix and process covariance to form the conditional arrays, call
the same generic moment kernel, generate a fixed map cloud, and update cloud
weights with the exact C2 observation log density. Carry the weighted cloud to
the next step. Record map margins, map shifts, standardized residuals under
the static initial map, and cloud ESS/observation response.

The probe is deliberately too small to support a proposal-efficiency claim.
It only answers whether the generic map is wired to a real nonlinear adapter,
whether the map remains SPD under recursive empirical moments, and whether the
observation changes the carried cloud. Any failure is classified separately as
adapter, map, or finite-cloud diagnostic failure.

## Skeptical audit and pre-mortem

Audit disposition: `PASS_FOR_BOUNDED_PHASE5_MECHANICS_AND_INTEGRATION`.

The plan survives the required skeptical checks:

- the baseline is an independent linear Kalman recursion, not a weak TT-only
  comparator;
- no proxy (ESS or map residual) is a promotion criterion;
- map SPD, forward/inverse, finite, source identity, and record-completeness
  stop conditions are explicit;
- the static and lagged maps use the same cloud, observations, dtype, and
  bank, so their descriptive comparison is paired;
- `N=128`, three C2 steps, and one GPU-hour are bounded and do not silently
  reopen the closed `N=8192` ladder;
- all C2 target factors remain exact, and the map is only a coordinate change;
- no adaptive-total gradient is claimed; the map is frozen for this diagnostic;
- a model-specific C2 binding is isolated in the driver, while the callable
  under test accepts only generic transition moments.

Pre-mortem:

| Misleading outcome | Distinguishing check | Disposition |
| --- | --- | --- |
| Static map looks worse only because of a scale convention error | direct forward/inverse identity and linear oracle | repair map orientation/Jacobian before interpreting residuals |
| C2 map remains SPD because invalid rows are silently replaced | per-row validity, raw minimum eigenvalue, and fail-closed host check | hard veto; no identity fallback in accepted rows |
| Empirical C2 cloud collapses and is mistaken for map failure | report ESS and observation response separately from map margins | classify finite-cloud/proposal diagnostic; do not reject mechanics |
| XLA result differs from eager result | identical fixed bank and elementwise parity | repair kernel/backend before any C2 interpretation |
| A map residual improves while recursive fitting worsens later | per-time residual and shift history | reject map as a recursive candidate and open Phase 5A unchanged-map arm |

## Budget, commands, and stop rules

Budget: at most 1.0 GPU hour and 1.0 CPU hour, with one fresh attempt. A
localized harness or serialization repair may be retried under the same
contract; a target, measure, map definition, or budget change requires a new
plan. Larger particle counts and basis/reference sweeps remain closed.

Focused CPU checks:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
  python -m pytest -q tests/highdim/test_recursive_moment_map_tf.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase4_repair.py
```

Bounded GPU/XLA probe:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase5-map \
  /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
  python docs/benchmarks/run_c2_mixture_ukf_apf_phase5_recursive_map_20260904.py \
  --output-root docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5-recursive-map-attempt02 \
  --rows 128 --horizon 3
```

The command must record the actual environment, memory-growth policy, device,
XLA setting, seed, fixture/source hashes, elapsed time, and all raw per-time
records. It must fail closed on missing or nonfinite values. A valid run with
large empirical residuals is a descriptive repair trigger, not a continuation
veto; a failed linear identity, map factorization, target adapter, or artifact
contract is a continuation veto until repaired.

## Phase-close and refresh

At close, write a decision table and an inference-status table that classify
engineering correctness, numerical validity, and scientific interpretation
separately. Preserve failed records and hashes. If the mechanics pass but the
C2 cloud is noisy, refresh Phase 5A with the same map and a representation
diagnostic; do not tune the defensive proposal on the C2 probe. If the map
fails, repair only the generic callable and rerun the linear oracle. If all
checks pass, the next permitted action is a one-step fixed-map representation
ladder (Hermite/RBF) on disjoint rows; a Student TT reference remains deferred.

## Execution close (2026-09-04)

The final GPU/XLA attempt was executed in the fresh directory
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5-recursive-map-attempt02/`.
Attempt 01 is preserved as an intermediate harness run; attempt 02 is the
authoritative result after correcting the initial mean-shift report and adding
explicit model/fixture transition and process-covariance checks.  The focused
CPU regression passed `9/9` tests before the final replay.

| Field | Final disposition |
| --- | --- |
| Result | `PASS_PHASE5_MECHANICS_AND_C2_WIRING` |
| Linear oracle | mean `5.551115123125783e-17`, covariance `1.1102230246251565e-16`, round trip `2.220446049250313e-16` maximum absolute errors |
| C2 adapter | transition and process errors `0.0`; stationary covariance error `3.552713678800501e-15`; all rows finite/SPD |
| Continuation | `CONTINUE_NO_REAL_BLOCKER` |
| Formal audit | MathDevMCP structural match on the actual module; SymPy scoped identities proved; Lean exit status 0; boundaries recorded in `phase5-recursive-map-attempt02/formal-audit.md` |
| Budget | 2.944 seconds elapsed for the final GPU run; within the one-hour GPU and one-hour CPU caps |
| Next phase | Phase 5A fixed-map representation diagnostic; no Phase 4 promotion and no Student TT reference |

The result and close note are descriptive for the C2 cloud.  ESS, map shifts,
and finite-bank standardized residuals do not establish proposal efficiency,
posterior correctness, or a better recursive filter.
