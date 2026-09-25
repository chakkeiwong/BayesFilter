# C2 Phase 8 exact-likelihood Laplace-mixture APF plan

Date: 2026-09-04  
Governing program:
`docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Status: `PHASE8C_COMPLETE_PHASE8D_REPLICATION_READY`  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Question and reason for the repair

Can a fixed-topology proposal built from the exact observation likelihood's
state score and curvature avoid the transformation-tail failure of the
log-square UKF while retaining the exact transition/observation target, a
complete conditional proposal denominator, and the existing analytical score
of one frozen finite APF program?

Phase 7 showed that the UKF proposal is observation-responsive and that the
importance denominator is correct. It also localized the common UKF collapse
to time 14, where a nearly zero observation becomes an extreme log-square
innovation. The exact likelihood score remains bounded there. This plan
changes the proposal construction; it does not change the target and it does
not introduce a small-observation exception.

The original C2 path, including time 14, is now diagnostic/calibration data.
It may test whether the mechanism was repaired, but it cannot support an
untouched claim. Any later comparison uses fresh, predeclared observation
seeds.

## Mathematical proposal

For retained ancestor `j`, use the exact transition conditional

\[
 p_j^-(x)=f_t(x\mid x_{t-1,j})=\mathcal N(x;m_j^-,P_j^-)
\]

when that conditional is Gaussian. A later non-Gaussian transition adapter
must state its own local approximation and remains a different route. Let

\[
 \ell_t(x)=\log g_t(y_t\mid x),\qquad
 s_t(x)=\nabla_x\ell_t(x),\qquad
 J_t(x)=-\nabla_x^2\ell_t(x).
\]

The generic model adapter supplies all three quantities with batch-native
TensorFlow operations. The claim-bearing C2 adapter implements the score and
curvature analytically; autodiff is allowed only as an independent test.

For a fixed tempering schedule `0 < lambda_1 <= ... <= lambda_R = 1`, define

\[
 \Phi_{j,r}(x)=
 \frac12(x-m_j^-)^\mathsf T(P_j^-)^{-1}(x-m_j^-)
 -\lambda_r\ell_t(x).
\]

The kernel records the sign-reversed log-posterior objective
\[
 \mathcal L_{j,r}(x)=-\Phi_{j,r}(x)
 =-\frac12(x-m_j^-)^\mathsf T(P_j^-)^{-1}(x-m_j^-)
  +\lambda_r\ell_t(x).
\]
Consequently, its per-step Newton diagnostic is an ascent check
\(\mathcal L_{j,r}(x_{r+1})\geq\mathcal L_{j,r}(x_r)\), up to the declared
floating-point tolerance; it is not a claim that the positive quantity
\(\Phi\) increases.

Starting from a declared fixed point, one fixed Newton update is

\[
 H_{j,r}=(P_j^-)^{-1}+\lambda_rJ_t(x_{j,r}),
\]

\[
 x_{j,r+1}=x_{j,r}-\eta_r H_{j,r}^{-1}
 \left[(P_j^-)^{-1}(x_{j,r}-m_j^-)
       -\lambda_rs_t(x_{j,r})\right].
\]

Every schedule, step fraction, start, and iteration count is fixed before a
claim run. Each `H` must be finite and SPD. An invalid factorization or a final
stationarity residual outside its scale-aware bound fails closed; it does not
silently add a ridge, clip an eigenvalue, or fall back to the transformed UKF.

At `lambda=1`, the local Gaussian component is

\[
 q_{t,j,k}^{\rm L}(x)=\mathcal N(x;\widehat x_{j,k},C_{j,k}),
 \qquad C_{j,k}=H_{j,k}^{-1}.
\]

The first mechanics route has one start at `m_j^-`. The fixed-mixture route
adds a regular-simplex start bank. For dimension `D`, its `K=D+1` standardized
vertices satisfy

\[
 \frac1K\sum_{k=1}^K v_k=0,
 \qquad
 \frac1K\sum_{k=1}^K v_kv_k^\mathsf T=I_D,
\]

and starts from `m_j^- + delta L_j v_k`, where
`L_j L_j^T=P_j^-`. The topology remains fixed even if multiple starts reach
the same mode; no pruning, merging, or argmax is allowed.

For a component mode, the Laplace log-evidence diagnostic is

\[
 e_{j,k}=\log p_j^-(\widehat x_{j,k})+ell_t(\widehat x_{j,k})
 +\frac D2\log(2\pi)+\frac12\log\det C_{j,k}.
\]

The K=1 route uses `e_j` as APF lookahead. The mixture route compares equal
component weights with a smooth `softmax(e_jk)` arm. Any finite-precision
zero weight is invalid; no unrecorded floor is inserted. The ancestor law is

\[
 a_{t,j}\propto \bar w_{t-1,j}\exp(\widehat e_j).
\]

The exact importance weight remains

\[
 \widetilde w_t=
 \frac{\bar w_{t-1,J}f_t(X\mid x_{t-1,J})g_t(y_t\mid X)}
 {a_{t,J}\,q_{t,J}(X)},
\]

where `q_tJ` is the sum of every realized Gaussian component and, when
declared, one full-support Student component. The selected component density
is never substituted for the complete density.

## Why the near-zero C2 limit is a required oracle

For one C2 coordinate,

\[
 \ell(x;y)=-\xi-\frac{x}{2}
 -\frac{y^2}{2}\exp(-x-2\xi)+C,
\]

so

\[
 s(x;y)=-\frac12+\frac{y^2}{2}\exp(-x-2\xi),
 \qquad
 J(x;y)=\frac{y^2}{2}\exp(-x-2\xi).
\]

Thus, as `y -> 0`, the K=1 Laplace mode and covariance approach

\[
 \widehat x_j\longrightarrow m_j^- -\frac12P_j^-\mathbf 1,
 \qquad
 C_j\longrightarrow P_j^-.
\]

This finite limit is the direct regression for the Phase 7 mechanism. It is
not a C2-only branch in the generic kernel.

## Gradient boundary

The first claim-bearing route freezes observations, random rows, ancestors,
component labels, Newton schedule, converged proposal locations and
covariances, lookahead probabilities, and complete proposal densities at the
reference parameter. The existing analytical recursion differentiates the
same exact numerator and carried normalized weights. This is the analytical
score of the frozen finite program, not the total derivative of adaptive
Laplace construction.

A later adaptive-total route must include the implicit mode derivative. If

\[
 F(x,\theta)=(P^-)^{-1}(x-m^-)-\nabla_x\ell(x;\theta)=0,
\]

then, when `H=partial_x F` is nonsingular,

\[
 \frac{d\widehat x}{d\theta}=-H^{-1}\partial_\theta F,
 \qquad
 dC=-C(dH)C.
\]

It must also differentiate component weights, ancestor probabilities, and
the complete proposal log density. That route is outside this plan.

## Research intent ledger

| Field | Declaration |
| --- | --- |
| Main question | whether exact-likelihood local geometry repairs the transformation-tail proposal mismatch |
| Candidate | K=1 fixed-schedule Laplace proposal, followed conditionally by a fixed regular-simplex mixture |
| Expected failure | Newton nonconvergence, indefinite local precision, missed modes, or a valid but inefficient Gaussian approximation |
| Primary Phase 8A criterion | exact linear-Gaussian result, correct C2 state derivatives and near-zero limit, fail-closed invalid-curvature fixture, fixed-signature/XLA mechanics, and complete-density tests all pass |
| Phase 8B nomination criterion | on calibration data only, the exact-likelihood arm removes the time-14 collapse without a new all-time heuristic loss and without a validity failure |
| Promotion veto | any complex arm loses to a cheap heuristic in any predeclared salient regime or time on untouched data |
| Continuation veto | wrong exact target or denominator, derivative formula mismatch, hidden fallback, nonfinite accepted output, broken call chain, corrupt artifact, or exhausted new budget |
| Repair trigger | valid but low ESS, duplicated modes, poor Laplace residual, or retracing/performance warnings |
| Nonclaims | no default, general-model success, posterior correctness, adaptive-total gradient, or statistically supported superiority |

## Baselines and salient situations

The heuristic adversary set is constructed before the run:

| Baseline | Rationale |
| --- | --- |
| bootstrap conditional | exact transition is the minimal filtering proposal |
| transformed-Student `nu=8` | current robust data-guided heuristic |
| Gaussian-hint marginal | cheap global moment proposal |
| transformed UKF K=1 | direct predecessor whose failure is being repaired |
| stationary independence | weak broad-support sanity check |

Report all horizon indices, not only four checkpoints. Conditional summaries
must separately show ordinary observations, the smallest absolute-observation
event, largest transformed innovation, lowest candidate ESS, early time, and
terminal time. A one-path calibration contrast cannot establish superiority.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| exact transition conditional `N(m-,P-)` | locally optimal APF factorization for the C2 Gaussian transition | unavailable for a non-Gaussian model | adapter capability check | target-specific exact input, generic interface |
| K=1 start at `m-` | smallest discriminating repair | misses multiple modes | symmetric bimodal fixture | baseline hypothesis |
| regular-simplex K=D+1 | dimension-symmetric fixed topology with first two moments | starts merge or miss distant modes | simplex identities and bimodal fixture | conditional hypothesis |
| tempering/iteration schedule | no justified inherited default | too few steps or excessive compile/runtime | residual versus fixed schedule ladder | calibration hypothesis |
| Newton step fraction | Class-C numerical choice, including one | overshoot or needless damping bias | objective/residual trust curve on quadratic and nonlinear fixtures | must be selected by non-harm calibration |
| no ridge/eigenvalue clipping | accepted result must represent the stated Hessian | rejects a potentially usable case | indefinite-Hessian fixture | fail-closed baseline, not a claim of universal usability |
| equal versus evidence weights | no established choice for duplicated modes | poor mixture allocation or underflow | normalization, duplication, and held-out ESS | comparator arms |
| Student defense | Phase 4 warm start only | dilutes well-localized bulk | local-only versus defense | optional hypothesis, not a transferred default |
| frozen proposal score | existing exact finite-program contract | omits adaptive construction derivative | central difference of the frozen program | claim boundary |

## Phases

### Phase 8A: routine mechanics

Implement one model-independent TensorFlow module with fixed-signature XLA-on
factories for the Laplace kernel and proposal sampler. Add a C2 derivative
adapter, but do not yet run a C2 particle campaign. Required tests are:

1. one Newton update recovers the exact Gaussian conditional mean, covariance,
   and evidence in a linear-Gaussian fixture;
2. analytical C2 state score and curvature match independent finite
   differences, including nonzero `xi`;
3. the `y -> 0` mode/covariance limit above holds;
4. invalid or indefinite accepted precision fails closed without ridge,
   clipping, or transformed-UKF fallback;
5. regular-simplex moment identities hold for dimensions 1, 2, and 4;
6. K=1 and K=D+1 complete densities normalize and are invariant to component
   permutation;
7. fixed random rows reproduce samples and proposal densities;
8. the public C2 compilation endpoint resolves to the generic kernel; and
9. graph tracing is bounded to one concrete function per shape/configuration.

Phase 8A is routine implementation/test work and consumes no serious campaign
budget. It may use a tiny escalated GPU/XLA smoke after the CPU suite passes.
The admissible run is recorded at
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8a-mechanics-attempt03/`;
attempts 01 and 02 are retained as superseded implementation snapshots.

### Phase 8B: calibration and mechanism regression

Use the already-seen C2 path and separate synthetic near-zero/bimodal fixtures
only as calibration. Select the fixed schedule and step fraction from a small
one-factor ladder by validity, stationarity residual, nondecreasing log-posterior
objective \(\mathcal L\), and runtime before considering ESS. The selection rule
is: retain configurations valid on the linear-Gaussian and C2 near-zero
fixtures, reject any with a stationarity or ascent violation, then choose the
lowest worst residual, breaking ties by runtime and finally by iteration count.
The bimodal fixture is explanatory and must not be used to tune K=1 into a
multimodal claim. Then run `N=1024` over all 20 times for
K=1, transformed UKF K=1, bootstrap, transformed Student, and Gaussian hint.
Only if K=1 remains valid may the simplex and Student-defense arms open.

The time-14 result is a mechanism regression, not untouched promotion
evidence. Preserve full per-time weights, proposal moments, exact score/parity,
Newton residuals, factorization margins, traces, and runtime.

The Phase 8B budget is one GPU-hour, one `N=1024` branch per listed family,
and at most two localized harness repairs. A repair may not change the target,
data, family list, selection rule, or denominator. If the budget is exhausted,
close the phase with the partial evidence and do not start Phase 8C.

### Phase 8C: fresh paired diagnostic

This phase requires a cost refresh after Phase 8B. Freeze fresh observation
seeds and all selected controls before launch. Run at most three paired
`N=8192`, horizon-20 branches under a new cap of one GPU-hour and two localized
harness repairs. Report uncertainty as descriptive unless the replication is
expanded under a separate plan. No default or broad scientific promotion is
available from this phase.

## Evidence contract

| Field | Declaration |
| --- | --- |
| Exact comparator | the repository frozen APF evaluator with exact C2 `f*g`, realized ancestor law, generalized row mass, and complete conditional `q` |
| Validity authority | analytical identities, independent finite differences, exact density recomposition, non-XLA/XLA parity, and endpoint wiring tests |
| Explanatory diagnostics | ESS, max weight, Newton residual, Hessian margin, component separation, runtime, and graph trace count |
| What will not be concluded | a passing mechanism test does not prove posterior accuracy, unbiased likelihood, generality, total adaptive gradients, or superiority |
| Phase 8A artifact | focused test capture and tiny GPU smoke under a fresh `phase8a-*` directory |
| Phase 8B artifact | fresh `phase8b-calibration-attemptNN/` with plan snapshot, manifest, raw all-time records, result, and close note |
| Phase 8C artifact | fresh `phase8c-paired-attemptNN/` with frozen observation identities and complete run manifest |

## Skeptical audit before execution

Disposition: `PASS_FOR_PHASE8A_MECHANICS_ONLY`.

- The plan does not treat Phase 7 ESS as evidence that the denominator is
  wrong; that call chain already passed.
- It does not tune a threshold around the near-zero observation. The proposed
  kernel consumes exact likelihood derivatives for every datum.
- The original C2 path is explicitly calibration data after inspection; any
  later claim uses fresh observation seeds.
- The bootstrap, transformed Student, Gaussian hint, and predecessor UKF are
  retained, and every time index enters the heuristic table.
- ESS is a proposal-efficiency diagnostic and promotion veto, not a
  correctness criterion or optimizer for the safety calibration.
- Fixed Newton damping is not silently inherited. A non-harm trust curve is
  required before selection; invalid curvature fails closed.
- A K=1 Gaussian can miss multimodality, so the symmetric bimodal fixture and
  conditional simplex phase are explicit.
- The exact likelihood and complete proposal denominator remain unchanged.
- The first analytical gradient is plainly the frozen finite-program score;
  the implicit adaptive derivative is derived but not claimed.
- The new method family and serious GPU budget are separated from the closed
  Phase 7 campaign. Phase 8A cannot silently launch Phase 8C.

## Planned commands

After implementation, Phase 8A uses:

```bash
CUDA_VISIBLE_DEVICES=-1 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python -m pytest -q \
  tests/highdim/test_exact_likelihood_laplace_apf_tf.py \
  tests/highdim/test_c2_exact_likelihood_laplace_adapter.py \
  tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py
```

The Phase 8A GPU/XLA smoke command (using a fresh attempt directory for each
source snapshot) is:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true \
MPLCONFIGDIR=/tmp/mpl-c2-phase8a-mechanics-attemptNN \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_exact_likelihood_laplace_phase8a_20260904.py \
  --output-root \
  docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8a-mechanics-attemptNN
```

The driver uses `N=32`, all 20 frozen observations, one fixed K=1 schedule,
and identical frozen random inputs for XLA/non-XLA parity. It checks the exact
frozen analytical score against central differences, verifies GPU output and
memory growth, and requires one trace each for the Laplace kernel and sampler.
This is a mechanics smoke; it cannot provide ESS, proposal-efficiency, or
promotion evidence. The existing observation path is already-seen data.

Prelaunch audit disposition:
`PASS_FOR_TINY_PHASE8A_GPU_XLA_MECHANICS_SMOKE`. The command cannot overwrite
an existing directory, does not run Phase 8B/8C, uses the same exact evaluator,
does not select a proposal setting, and writes source hashes, dirty-state
identity, device/memory/XLA metadata, diagnostics, and nonclaims. All Phase
8B/8C commands remain to be frozen before those runs start.

## Phase 8A close

The fresh attempt-03 artifact reports `PASS_PHASE8A_GPU_XLA_MECHANICS`, with
15 focused CPU tests passed and the Phase 8 Lean file compiling with exit 0.
All declared checks passed, including one trace each for the Laplace kernel and
sampler, exact frozen-score finite differences, XLA/non-XLA parity, and the
per-step nondecreasing log-posterior objective. MathDevMCP sidecars and the
formal boundary are recorded in the attempt-03 `formal-audit.md` and
`phase8a-close-20260904.md`. Phase 8B is ready; no proposal-efficiency or
promotion claim is made.

## Phase 8B prelaunch audit and commands

Skeptical disposition: `PASS_FOR_PHASE8B_CALIBRATION_AND_N1024_MECHANISM_RUN`.
The runner uses the same exact frozen evaluator and complete conditional
denominator for every family, freezes the schedule before reading candidate
ESS, includes the constructed heuristic set, writes a fresh output directory,
and caps the campaign at one GPU-hour with two localized repairs. The
already-seen C2 observations are explicitly calibration/mechanism data, not
untouched promotion data. The Gaussian-hint comparator is loaded from a
versioned Phase 7 snapshot with metadata hashes; a missing or mismatched
snapshot fails closed. The bimodal fixture is explanatory and cannot promote a
K=1 claim.

The calibration-only smoke command is:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase8b-calibration-attempt01 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py \
  --calibration-only --output-root \
  docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8b-calibration-attempt01
```

After that smoke passes, the bounded comparison command is:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase8b-comparison-attempt01 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py \
  --particle-count 1024 --branch-count 1 --output-root \
  docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8b-comparison-attempt01
```

Both commands require XLA (the default), verify memory growth and GPU
placement, and refuse to overwrite an existing artifact directory.

## Phase 8B close

The admissible `phase8b-comparison-attempt01` run completed in 72.947 seconds
on the RTX 4080 SUPER with XLA and verified memory growth. The calibration
selected `quarter_long` before ESS was inspected. All six N=1024 branches
passed the exact-target, complete-denominator, APF identity, finite-
difference, and parity checks. The exact-likelihood Laplace K=1 arm had
minimum ESS `208.607746` and time-14 ESS `980.594322`; the transformed UKF
arm had minimum/time-14 ESS `55.285247`. These are one-realization
calibration/mechanism diagnostics. Proposal-specific random-key offsets mean
the paths are not common-random-number paired, and the Gaussian-hint arm uses
an already-seen Phase 7 snapshot. The result therefore remains
candidate-only and cannot support a ranking or default change.

Attempts 01 and 02 were localized harness failures (curvature shape and
non-finite placeholder serialization); the repairs left the target, data,
selection rule, and denominator unchanged. Their directories remain preserved
and are excluded from interpretation. The formal MathDevMCP sidecars,
certificate, and detailed close record are in the comparison directory.

## Phase 8C refresh and skeptical audit

The Phase 8B cost refresh is `72.947 s` for one N=1024 branch per six
families, including calibration. A conservative eightfold particle scaling
would be about `9.7 min` for one N=8192 branch before compilation variance;
three branches remain below the one-GPU-hour cap, with a hard wall-clock stop
at 55 minutes. This estimate is a budget check, not a runtime guarantee.

The fresh phase uses a new C2 observation seed and a versioned TensorFlow-
generated fixture. It must not load the Phase 7 Gaussian-hint snapshots. The
fresh comparator set is therefore the exact Laplace K=1 candidate, transformed
UKF K=1, bootstrap conditional, transformed Student `nu=8`, and stationary
independence. A newly generated Gaussian hint may be added only after its
fresh-data construction and metadata are independently recorded; it is not
silently substituted by the old snapshot.

Before launch, the following skeptical checks passed:

1. The scientific question and target remain unchanged; only the observation
   path and proposal random seeds are refreshed.
2. Schedule selection remains calibration-first and cannot inspect fresh ESS.
3. Every branch uses the exact C2 numerator and complete conditional
   denominator; omitting the stale Gaussian hint cannot invalidate the target.
4. The candidate remains diagnostic-only. A heuristic loss blocks promotion,
   not continuation; a nonfinite program, target mismatch, missing fresh
   fixture, or budget exhaustion is a continuation veto.
5. Three branches at most are allowed, with independent proposal seeds and a
   shared fresh observation path. Because the compiler APIs do not yet accept
   caller-supplied random tensors, common-random-number pairing is not claimed;
   the result must report this uncertainty and remain descriptive.
6. No numerical protection, ridge, clipping, adaptive derivative, or model-
   specific small-observation exception is introduced.

The refreshed Phase 8C plan and fixture generator must be reviewed before the
N=8192 launch. The exact command and fresh fixture hash belong in the new
`phase8c-paired-attemptNN` manifest.

## Phase 8C close and Phase 8D entry

The repaired `phase8c-paired-attempt02` run passed all `15/15` finite-program
records on a fresh TensorFlow-generated C2 path in `145.694 s`. The candidate
was nonworse than every listed heuristic at all `60` branch/time comparisons;
its minimum ESS values were `3593.162`, `3633.416`, and `3548.778`, while the
UKF minima were `156.148`, `293.105`, and `176.737`. The time-14 candidate ESS
values were `7753.404`, `7712.425`, and `7725.611`. These contrasts remain
descriptive because there are only three proposal randomizations on one data
path and no uncertainty interval over observation paths.

The first fresh attempt is superseded for its heuristic verdict by a repaired
branch aggregation and K=1 random-stream alignment; the second attempt is the
admissible result. MathDevMCP and Lean checks, focused tests, source hashes,
and the repair history are in its artifact directory. No target, denominator,
schedule-selection rule, or gradient boundary changed.

Phase 8D is the next frozen-control replication. It uses at least three new
observation seeds, the already selected `quarter_long` schedule without fresh
ESS-based retuning, the same five-family comparator set, and new output roots.
The purpose is robustness and uncertainty description, not default promotion.
