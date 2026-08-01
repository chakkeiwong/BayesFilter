# LGSSM Value And q_scale Bias Hypothesis Audit

Date: 2026-07-20

## Research Intent Ledger

| Item | Audit statement |
| --- | --- |
| Main question | Which mathematical or implementation mechanisms can produce the persistent value and `q_scale` disagreement with the exact Kalman program? |
| Candidate under test | Canonical Contract E--Chol value and total-score finite particle program at `T=50`, with `N=5000` and `N=10000` claim scopes. |
| Expected failure mode | Common finite-particle proposal/importance-normalizer error, repeated transport/reset amplification, cancellation-sensitive long-horizon score error, or a missed scope-specific numerical control. |
| Promotion criterion | None from this audit. It is an explanatory diagnostic, not a correctness or default-readiness decision. |
| Promotion veto | A target mismatch, omitted total derivative, invalid artifact, or failed same-scalar derivative identity would veto interpretation of the campaign as evidence about the intended finite program. |
| Continuation veto | None fired by the trace. The finite program is finite, replayable, and has passed the existing local derivative checks; the scientific mismatch remains unresolved. |
| Next discriminating artifact | Paired time-local value/score decomposition for active Contract E, no-reset weighted recursion, and exact Kalman, using identical observations and prepared random streams. |
| Nonclaims | This note does not establish posterior correctness, equivalence, HMC readiness, a convergence rate, or a cause with certainty. |

## Executed Mathematics

The canonical route initializes particles as

\[
x_i^{(0)} = z_i\,q/\sqrt{1-\phi^2},
\]

then, at every time index, propagates with the diagonal transition matrix and
process noise, applies the exact linear-Gaussian flow, forms the importance
logits, accumulates a `logsumexp` likelihood increment, transports the cloud,
and optionally applies Contract E before replacing the log weights by
`-log(N)`. The code is at
`bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:697-856`.

The importance correction is the finite-program quantity

\[
\ell_t = \log\sum_i \exp\{\log w_{t-1,i} +
 \log p(x_{t,i}\mid x_{t-1,i}) + \log p(y_t\mid x_{t,i}) -
 \log q(x_{t,i}\mid x_{t-1,i}) + \log|J_t|\}.
\]

The score recursion differentiates this same program, including particle
tangents, normalized-weight tangents, transport tangents, and Contract E reset
tangents (`..._canonical_lgssm_tf.py:1080-1160` and the continuation of that
routine).

Contract E does not preserve the full weighted cloud. It computes weighted
source moments, uniform transported-cloud moments, a ridged covariance gap,
residual injection, and a final affine covariance map
(`bayesfilter/highdim/ledh_contract_e_reset_tf.py:44-102`). The exact identity is

\[
A(\widetilde\Sigma+\lambda I)A^\top=\Sigma_w+\lambda I,
\]

not exact raw covariance equality. The raw residual is

\[
\Sigma_{out}-\Sigma_w=\lambda(I-AA^\top),
\]

so moment matching does not make the finite particle filtering recursion equal
to the Kalman recursion.

The DGP and Kalman oracle both use the stationary covariance
`diag(q^2/(1-phi^2))` (`scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py:137-183`
and `docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py:101-127`).
The particle loop is transition-first (`..._canonical_lgssm_tf.py:778-790`),
whereas the DGP records an observation from the initial state. For this
stationary, time-homogeneous model, shifting the stationary Markov sequence by
one transition leaves its joint observation law unchanged. This is therefore a
conditional target-ordering risk for nonstationary or time-varying models, but
not a confirmed mismatch for this LGSSM benchmark.

## Hypothesis Set

| Rank | Hypothesis | Why it can produce the observed result | Current status |
| ---: | --- | --- | --- |
| 1 | Common finite-particle proposal/importance-normalizer error | The linear flow is the conditional Gaussian proposal, but finite particles still estimate the predictive normalizers through random ancestor states and carried weights. At `T=2`, paired Contract E and no-reset errors were nearly identical, which points upstream of reset. Increasing `N` can reduce dispersion without monotonically removing a long-horizon mean error. | Leading hypothesis class; exact common component not isolated. |
| 2 | Log-normalizer Monte Carlo bias | The reported value is a sum of logs of finite random normalizer estimates. Even if an unlogged normalizing-constant estimator were unbiased, Jensen's inequality does not make its log unbiased. Deterministic transport/reset changes the particle law, so even the usual sign intuition need not apply. The score bias is the parameter derivative of this finite-`N` log-bias function and can be large when the exact terminal score is cancellation-small. | Intrinsic finite-`N` mechanism; magnitude not isolated. |
| 3 | Repeated Contract E reset as a long-horizon amplifier | Equal-weight moment-restored clouds discard higher moments and weighted dependence after every period. The discarded shape error feeds into later increments. However, paired `T=2` results did not show reset as the dominant source; no corresponding `T=50` no-reset comparison exists. | Serious but unisolated; not justified as the sole leading cause. |
| 4 | Incomplete tuning objective | The tuner selects the first pair passing `TV_col` and row-error gates only (`docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py:589-718`). Those gates test coupling marginals, not value or total-score agreement. The observed pairs were `(20,5)` at `N=5000` and `(20,8)` at `N=10000`; both still have detected value and `q_scale` mean bias. | Confirmed tuning gap. |
| 5 | Untuned terminal entropic geometry | `epsilon=0.5`, annealing `scaling=0.9`, and the geometry-derived `epsilon0` schedule are fixed in the harness (`docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py:31-37`), while only Sinkhorn and balance counts are searched. `epsilon0` depends on sample extrema (`..._canonical_lgssm_tf.py:564-585`), so changing `N` also changes the annealing path. Marginals may be excellent while barycentric locations and their derivatives remain entropy-biased. | Plausible; untested in these scopes. |
| 6 | Reset covariance/ridge and residual-design effects | Contract E restores a ridged identity, uses a fixed centered residual design, and depends on the realized transported cloud. The ridge is small (`7.301568984985351e-09`), so its direct magnitude alone is unlikely to explain a 10--16% `q_scale` relative bias, but ill-conditioning or residual injection can amplify downstream effects. | Plausible; telemetry is not preserved in claim aggregates. |
| 7 | Long-horizon recursive amplification and score cancellation | A `T=50` run repeatedly feeds approximation error forward. The `q_scale` score is especially sensitive because `q` enters both the stationary initial scale and transition covariance. The exact Kalman score increments have substantial positive/negative cancellation, so local errors can be small in absolute terms but large relative to the terminal sum. | Strong explanatory mechanism; source of local error still unknown. |
| 8 | FP32/TF32 numerical drift | The claim route is float32 with TF32 enabled. Prior paired precision diagnostics show TF32 can materially change score components in some shapes. Deterministic replay proves repeatability, not closeness to FP64. | Plausible secondary mechanism; not isolated at these exact scopes. |
| 9 | Full-recursion total-derivative omission | A local derivative bug could bias only the score while leaving value affected by a separate mechanism. Primitive and same-scalar finite-difference checks, including the prior `T=2,N=32` check, strongly support that the score differentiates the executed finite scalar. A time-local decomposition is still needed to close this hypothesis at `T=50`. | Lower probability; not fully ruled out. |
| 10 | Off-by-one observation semantics | The implementation propagates before assimilating the first observation. Stationarity makes the generated observation law invariant to this shift for this benchmark. | Not a confirmed cause here; becomes real under nonstationary initialization/dynamics. |
| 11 | RNG, batching, or prepared-input drift | Prepared streams are stateless, hashed, and replayed; claim seeds use singleton microbatches. Batch merge checks preserve per-seed outputs. | Largely ruled out for the current artifacts. |
| 12 | `q` versus `q*`, `q^2`, or `log q` parameterization | The exact chain rule gives `q*(dL/dq*) = q(dL/dq)` for the proposed `q*` at the DGP, so the physical value and `q` score are invariant up to a positive coordinate rescaling. | Ruled out as an independent repair. |

## Tuning Audit

The executed campaign tuned only:

- `sinkhorn_steps`;
- `balance_steps`.

The following controls were fixed or inherited rather than selected with a
scope-specific value/score criterion:

- terminal `epsilon`;
- annealing `scaling` and the geometry-derived `epsilon0` schedule;
- Contract E reset cadence (active at every period);
- fixed residual-design construction and centering;
- prepared ridge magnitude and ridge policy;
- float32/TF32 backend choice;
- XLA and chunk policy (these are correctly treated as execution-policy
  identities, not free scientific knobs);
- the selection objective itself, which used marginal feasibility and did not
  use a Kalman-blind value/score stability or no-reset comparison diagnostic.

The repository tuning registry makes this explicit: the LGSSM route declares
`("sinkhorn_steps", "balance_steps")` tunable and `("epsilon", "scaling",
"prepared_ridge", "particle_count")` fixed (`bayesfilter/highdim/ledh_tuning_registry.py:25-38`).
That is sufficient for an engineering marginal screen, but insufficient to
support a claim that all material numerical controls were tuned for the value
and total-score target. `particle_count`, dtype/backend, and chunk policy must
remain scope identity fields; they should not be silently optimized away.

Even the two searched counts were tuned only for feasibility. At `N=5000`, the
first tested pair `(20,5)` passed, so no higher Sinkhorn count was compared. At
`N=10000`, `(20,5)` failed a marginal gate and `(20,8)` passed, after which the
search stopped. Thus there is evidence that the selected counts are adequate
for the declared marginals, but no evidence that they minimize value or score
bias. The fixed ridge, `epsilon`, and `scaling` were inherited from the repaired
`T=2,N=32` lower rung
(`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-lower-rung-continuation-result-2026-07-14.md:23-30`),
which is warm-start provenance rather than target-specific tuning evidence.
Moreover, because `epsilon0` is computed from the finite cloud's maximum and
minimum scaled coordinate, an `N` change changes the realized annealing
schedule. Consequently the observed nonmonotone `N=5000`/`N=10000` bias is not
a clean particle-count convergence comparison with all other numerical
geometry held fixed.

## Evidence Classification

| Evidence class | Finding |
| --- | --- |
| Hard veto evidence | No non-finite result, replay mismatch, invalid chart, or failed required chunk identity in the accepted `N=5000`/`N=10000` claim artifacts. |
| Statistical evidence | The simultaneous 95% intervals reject zero mean value bias and zero mean `q_scale` bias at both scopes. The estimates are evidence of persistent bias, not a proof of its mechanism. |
| Descriptive evidence | Increasing `N` reduced some seed dispersion and changed the bias, but the `q_scale` mean remained negative (`N=5000` about `-9.9%`; `N=10000` about `-15.9%`). This is not a demonstrated monotone `1/N` law. |
| Engineering correctness | Same-scalar checks support differentiation of the executed finite Contract E program. They do not show that the finite program equals Kalman. |
| Scientific interpretation | Candidate failure is currently a failure of the finite algorithm/control configuration against the exact Kalman target. Short-horizon paired evidence points to a common upstream finite-particle error, while long-horizon reset amplification is not yet separated. This is not evidence that the LGSSM target or Contract E research direction is invalid. |

## Smallest Discriminating Experiment

Use the same observations and prepared streams for three arms:

1. active Contract E reset;
2. no-reset weighted recursion;
3. exact Kalman.

Record per time step:

- value increment and cumulative value;
- total `q_scale` score increment and cumulative score;
- initial stationary contribution;
- transition/proposal contribution;
- observation likelihood/normalization contribution;
- carried previous-weight contribution;
- transport contribution; and
- Contract E moment/weight/reset contribution.

For every reported partial score, perform a same-scalar finite-difference check
of that partial finite scalar. This separates a reset/finite-particle error from
an omitted derivative term and shows whether the `q_scale` discrepancy is local
or accumulated by cancellation. Only after this decomposition should a small
paired sensitivity arm vary terminal `epsilon`, TF32, and ridge/transport reset
policy. A future tuning scope should bind those controls and must select on an
untouched value/score diagnostic in addition to marginal feasibility.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Treat current bias as unresolved finite-program error | Value and `q_scale` simultaneous zero-bias CIs reject zero | No engineering veto in accepted scopes | Reset versus transport geometry versus precision | Run the three-arm time-local decomposition | No claim of a single proven root cause |
| Expand tuning scope | Current tuner omits material numerical controls and uses marginal-only selection | Do not promote current settings as universal | Cost and feasibility of value/score-aware calibration | Add small, Kalman-blind sensitivity grid before another large ladder | No retroactive retuning on claim data |
| Preserve current canonical route | Contract E identity and total derivative remain the eligible route | No route-identity veto | Whether Contract E can meet Kalman at finite `N` | Diagnose before changing reset semantics | No HMC/default-readiness claim |

## Post-Run Red Team

The strongest alternative explanation is precision drift or a subtle full-score
composition omission that happens to survive local checks. A paired FP64 or
FP32-no-TF32 run and the time-local same-scalar decomposition would overturn the
reset-first interpretation if they remove the bias without changing the reset
or transport approximation. The weakest part of the present evidence is that
claim artifacts retain final values/scores and marginal gates but not the full
per-time reset covariance/conditioning telemetry needed to correlate local
numerical error with the terminal `q_scale` bias.
