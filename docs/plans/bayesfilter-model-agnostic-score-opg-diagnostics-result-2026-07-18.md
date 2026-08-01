# Model-Agnostic Score OPG Diagnostics Result

Date: 2026-07-18

Status: `ENGINEERING_PASS_SCIENTIFIC_DIAGNOSTIC_ONLY`

## Outcome

The two score-comparison amendments are documented and implemented:

1. global score-vector error relative to the total reference-score norm, with
   absolute error and a cancellation-resistant predictive-increment-energy
   companion; and
2. a regularized average predictive-score OPG metric whose scale-aware ridge is
   `max(epsilon_0/T, epsilon_min) D`.

The implementation is TensorFlow-only, accepts batched candidate scores, has no
scientific hyperparameter defaults, fails closed on invalid shrinkage/ridge
inputs, and emits the unregularized spectrum, realized ridge, floor status, and
regularized condition proxy.  CPU-hidden XLA compilation passed.

The current frozen `T=2,p=5` LGSSM campaign was used only as a diagnostic
witness. Exact differentiated Kalman prefix scores produce two predictive
increments whose sum agrees with the existing total oracle. Their average OPG
has numerical rank two, as required by the rank bound. The regularized metric
is positive definite, but three parameter directions are not independently
identified by this two-step record. No acceptance threshold was selected.

## Claimed And Computed Quantities

| Item | Claimed target | Quantity actually computed | Verdict |
| --- | --- | --- | --- |
| global norm error | Euclidean error of candidate and reference total scores in identical coordinates | total-score norm ratio plus absolute norm | correct under checked direct calculation |
| cancellation-resistant norm | error relative to root summed predictive-score energy | Frobenius norm of the reference increment matrix | correct; coordinate-dependent diagnostic |
| average OPG | `T^-1 sum_t s_t s_t^T` from reference predictive-score increments | exact differentiated Kalman prefix-score differences for the witness | correct for the LGSSM witness |
| regularized metric | diagonal-shrunk average OPG plus `max(epsilon_0/T,epsilon_min)D` | TensorFlow matrix construction and Cholesky solve | correct under positive diagonal `D` and positive `epsilon_0` |
| particle-seed variation | Monte Carlo uncertainty only | per-seed candidate diagnostics; no seed covariance enters OPG | correctly separated |
| scientific equivalence | a future model-class-specific score-accuracy criterion | no threshold or equivalence test selected | not checked |

## Documentation Review

The new chapter section defines the coordinate contract, both norm metrics,
the average OPG, shrinkage, total-score scaling, rank limitation, ridge floor,
and the distinction between predictive increments and particle seeds. It gives
proposition-and-proof statements for finite-horizon positive definiteness and
the asymptotic meaning of the `1/T` ridge.

The skeptical review caught two mathematical risks before execution:

- applying `epsilon_0/T` to the summed OPG would make the ridge relative order
  `1/T^2`, so the amendment applies it to the average OPG; and
- fixed nonzero diagonal shrinkage does not vanish asymptotically, so recovery
  of the unshrunk OPG requires `lambda_T -> 0`.

The full LaTeX book built successfully to
`/tmp/bayesfilter-score-opg-latex/main.pdf`. The build retains unrelated
repository-wide undefined-citation and duplicate-label warnings. None names a
new OPG label, and every new section/equation/proposition label appears once in
the auxiliary file.

## LGSSM Witness

The exact HMC-coordinate predictive-score increments are:

```text
t=1 [ 1.4886979449, -0.2691321398,  0.1872759983,  2.1133791521,  6.1432716353]
t=2 [ 0.3548734530,  0.0011658539, -0.2601850366, -0.5673551071, -1.1311113341]
sum [ 1.8435713979, -0.2679662859, -0.0729090383,  1.5460240451,  5.0121603011]
```

For the explicitly diagnostic convenience settings `lambda=0`,
`epsilon_0=1`, `epsilon_min=0`, and `D=I`:

| Quantity | Contract E mean | No-reset mean |
| --- | ---: | ---: |
| absolute score error norm | `0.0321088` | `0.0313107` |
| relative total-score norm error | `0.576805%` | `0.562467%` |
| relative increment-energy error | `0.471756%` | `0.460029%` |
| RMS regularized-OPG error | `0.0126886` | `0.0127075` |
| maximum diagonal standardized error | `0.0270412` | `0.0269302` |

The unregularized average-OPG eigenvalues are, up to float64 roundoff,

```text
[-2.52e-15, -3.94e-18, 7.20e-16, 0.230271, 22.932139]
```

so its numerical rank is two. The realized average-metric ridge is `0.5`, the
floor is inactive, and the corresponding total metric is positive definite.
These settings were selected to demonstrate the formula and rank-deficient
case, not to make either candidate look favorable. The historical simultaneous
screen remains `inconclusive` for both arms and its global reset direction
remains `mixed_or_inconclusive`.

## Checks

| Check | Result |
| --- | --- |
| reusable formula, shape, invalid-input, rank, zero-denominator, and CPU-XLA tests | `10 passed` |
| witness plus historical-screen regression tests | combined focused suite `17 passed`, two TensorFlow Probability deprecation warnings |
| Python compilation | pass |
| diff whitespace check | pass |
| LaTeX full-book build | pass; PDF emitted under `/tmp` |
| witness SHA-256 | `9210d3f5f441775d8db8e8c6d4e40485b9d6b4c90bf22cd6501c82ab18501d4e` |

## Inference Status

| Question | Status |
| --- | --- |
| hard veto screen | engineering checks pass; historical LGSSM scientific screen remains inconclusive |
| statistically supported ranking | none |
| descriptive-only differences | every Contract E versus no-reset metric in the witness |
| default readiness | no `lambda`, ridge, ridge scale, or acceptance threshold is a scientific default |
| next evidence needed | predeclare model-class settings and thresholds; expose reference predictive-score increments for nonlinear comparator lanes; use longer or pooled records when full-rank geometry is required |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| admit the generic API as a diagnostic utility | direct formulas, input guards, batching, XLA, and serialization pass | none for diagnostic use | behavior under large nonlinear-model score scales | wire into one reviewed nonlinear comparator with frozen coordinates/settings | no scientific default or correctness claim |
| retain both norm and OPG views | they expose different failure modes | global norm can hide a coordinate; regularized OPG can mask missing rank | parameterization and reference quality | always emit absolute error, rank/spectrum, ridge/floor, and maximum diagonal statistic together | no coordinate invariance |
| preserve the old LGSSM result | new metrics were not predeclared for that campaign | historical screen unchanged | low power and short horizon remain | treat witness only as formula validation | no retrospective pass or ranking |

## Run Manifest

| Field | Value |
| --- | --- |
| git commit | `15170e1573d19b235d96f3ed3525fa2071f58320` with scoped uncommitted changes in a shared dirty worktree |
| environment | TensorFlow `2.19.1`; project environment |
| CPU/GPU | CPU only with `CUDA_VISIBLE_DEVICES=-1`; no GPU evidence claimed |
| data | frozen LGSSM dataset seed and observations inherited from the source aggregate |
| estimator seeds | frozen source arms `81500..81515`; used for Monte Carlo diagnostics only |
| command | `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_score_diagnostics_tf.py tests/highdim/test_model_agnostic_score_opg_lgssm_witness.py tests/highdim/test_canonical_lgssm_kalman_certification.py` |
| witness command | `CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/emit_model_agnostic_score_opg_lgssm_witness.py --aggregate docs/benchmarks/artifacts/canonical_lgssm_balancing_kalman_repair_20260717/phase3/t2/aggregate_attempt01.json --output docs/benchmarks/artifacts/model_agnostic_score_opg_diagnostics_20260718/attempt01/lgssm_t2.json` |
| plan | `docs/plans/bayesfilter-model-agnostic-score-opg-diagnostics-plan-2026-07-18.md` |
| result | this file |
| artifact | `docs/benchmarks/artifacts/model_agnostic_score_opg_diagnostics_20260718/attempt01/lgssm_t2.json` |

## Post-Run Red Team

The strongest alternative explanation for the small global LGSSM numbers is
that the large `r_scale` score dominates the Euclidean norm; this is why the
maximum diagonal OPG statistic remains mandatory. Conversely, the regularized
OPG result depends strongly on a ridge that supplies three unobserved
directions at `T=2`; it must not be read as a data-identified five-dimensional
distance.

The result would be overturned as an engineering certificate by a direct
formula mismatch, a prefix-increment sum mismatch, a non-positive total metric
under the documented assumptions, or a changed historical screen. None
occurred. The weakest scientific evidence is the single short LGSSM record and
the convenience ridge. It supports implementation validation only.
