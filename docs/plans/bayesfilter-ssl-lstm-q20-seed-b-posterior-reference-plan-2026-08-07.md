# SSL-LSTM q=20 seed-B untouched posterior/reference plan (2026-08-07)

## Decision sought

Determine whether the already-retained seed-B NeuTra/fixed-HMC draws agree with
an independently constructed posterior reference for the exact four-coordinate
q=20 target.  This phase tests posterior agreement; it does not retrain the
transport, retune HMC, or use retained draws to design the reference.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | Do the 4,000 retained seed-B model-coordinate draws agree, within predeclared uncertainty and numerical tolerances, with a target-only deterministic posterior reference? |
| Candidate | Seed-B checkpoint 4000 / optimizer step 6250 with the frozen `L=3`, step-size `0.8115211181271775` NeuTra HMC kernel. |
| Comparator | TensorFlow float64 Gauss-Hermite quadrature over unbounded `R^4`, centered at a target-only multistart MAP and scaled by a target-score finite-difference Hessian. |
| Expected failure mode | The learned transport/HMC can converge to a biased region; alternatively quadrature can miss another mode or be under-resolved. |
| Promotion criterion | The reference passes every reference-validity veto, and the 99% chain-aware moving-block-bootstrap upper confidence bounds pass all four posterior-equivalence margins. |
| Promotion veto | Any failed reference-validity check, any posterior-equivalence upper bound outside its margin, any retained archive/hash mismatch, or any nonfinite comparison result. |
| Continuation veto | Broken target/archive identity, corrupted inputs, non-XLA target evaluation, visible GPU, invalid reference target rows, inability to obtain an SPD stable local Hessian, unresolved quadrature across the full order ladder, wall-time cap, or implementation/test failure. |
| Repair trigger | A valid quadrature reference that disagrees rejects this seed-B posterior candidate. Quadrature invalidity or nonconvergence triggers a separately planned freshly tuned plain fixed-HMC reference; it does not reject seed B or the research direction. |
| Explanatory diagnostics | MAP start outcomes, Hessian step stability, quadrature ESS/max weight, order/scale deltas, point counts, posterior point estimates, and bootstrap distributions. |
| Must not be concluded | Model adequacy, broad posterior correctness, sampler superiority, cross-seed robustness, native zero divergences, or default readiness. |

## Evidence contract

- Scientific question: posterior agreement for target signature
  `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
- Exact candidate: the two retained seed-B chunks in the r2 sequential archive;
  all 2,000 warm-up transitions per chain remain excluded.
- Exact comparator: target-only deterministic quadrature.  Reference creation
  must finish and be hash-bound before comparison code may read retained tensor
  values.
- Primary criterion: simultaneous 99% moving-block-bootstrap upper bounds no
  larger than `0.10` reference SD for maximum standardized mean error, `0.10`
  for maximum relative SD error, `0.15` for covariance Frobenius error
  normalized by reference covariance Frobenius norm, and `0.15` reference SD
  for maximum standardized error over marginal q05/q50/q95.
- Reference vetoes: target/adapter mismatch; any non-XLA worker; any visible
  GPU; fewer than five successful MAP starts or missing axial coverage; a
  competing mode within 20 log-density units; MAP score infinity norm above
  `1e-5`; non-SPD or step-unstable negative Hessian; any nonfinite or
  target-invalid quadrature row; normalized weight ESS below 50; maximum weight
  above 0.05; or lack of adjacent-order and cross-scale stability.
- Reference stability thresholds: maximum standardized mean delta `<=0.02`,
  maximum relative SD delta `<=0.02`, normalized covariance Frobenius delta
  `<=0.03`, and maximum standardized q05/q50/q95 delta `<=0.03`.
- Explanatory only: raw point estimates, signs of differences, MAP location,
  quadrature log normalizer, and individual bootstrap quantiles.
- Nonclaims: passing establishes agreement only for this archived candidate and
  synthetic q=20 target under these margins. Failure distinguishes candidate
  disagreement from inconclusive/invalid reference evidence.
- Preserved artifact: versioned JSON progress/reference/comparison/result under
  `docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r2/`.

The equivalence margins are convenience-chosen hypotheses, not inherited facts.
They demand small errors relative to posterior scale and a simultaneous 99%
uncertainty bound. They are suitable for this candidate gate but are not a
universal BayesFilter default.

## Method and frozen defaults

### Reference construction

1. Set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and fail if a GPU is
   visible. Use TensorFlow/TFP float64 and XLA-compiled target calls.
2. Run TFP L-BFGS from the prior center and its eight axial points at plus/minus
   one prior standard deviation. This nine-start set is derived from the target
   prior, not from NeuTra draws.
3. Require the center and at least one start on each coordinate axis to converge
   to the same best basin: maximum endpoint coordinate difference `<=1e-4` and
   maximum log-density difference `<=1e-6`. These are convenience numerical
   tolerances tied to the `1e-5` score gate, not scientific defaults. A distinct
   converged stationary point within 20 log units vetoes the single-proposal
   quadrature claim.
4. Form the local negative Hessian by central differences of the analytic target
   score at steps `1e-3`, `3e-4`, and `1e-4`. Require SPD and normalized
   Frobenius change `<=1e-3` for the last two estimates. These steps/tolerance
   are convenience numerical hypotheses; all three are recorded.
5. Use the inverse terminal negative Hessian as the Gaussian proposal covariance.
   TensorFlow Golub-Welsch eigendecomposition constructs Gauss-Hermite nodes and
   weights, avoiding a NumPy runtime. Convert Gaussian expectations to target
   integrals with the required `exp(log_target - log_proposal)` importance
   correction; omitting that correction computes the wrong integral.
6. Evaluate proposal scales `1.0` and `1.5` at orders `7, 9, 11, 13`. Stop at
   the first order at or above 9 whose adjacent-order summaries and same-order
   cross-scale summaries pass every frozen stability threshold. If order 13
   fails, quadrature is inconclusive.
7. All quadrature target calls use the existing persistent 25-worker x 4-row
   CPU batch-native pool, `tensorflow_eigh`, and XLA. The reference contains no
   sample-wise fallback, `tf.map_fn`, or NumPy numerical route.

### Posterior comparison

1. Verify the sequential summary, archive manifest, every retained tensor
   receipt, target signature, and the reference artifact/hash before parsing
   retained tensors.
2. Preserve the array as 1,000 transitions x 4 chains x 4 coordinates. Compute
   observed mean, SD, covariance/correlation, and q05/q50/q95 differences.
3. Use 2,000 deterministic-seed circular moving-block bootstrap replicates,
   resampling separately within each chain. Block length 32 is derived as the
   nearest integer to `sqrt(1000)`, rather than tuned from observed agreement.
4. The 0.99 quantile of each replicate's maximum discrepancy is its simultaneous
   upper confidence bound. The four frozen margins must all pass.

### Numerical provenance/default audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Four coordinates, identity transform, prior center/SD | Target source; reviewed default | Exact target definition | Wrong target binding | Signature and parameter-name checks |
| Nine prior-derived MAP starts | Derived baseline | Probes center and both directions of every axis without NeuTra data | Remote/off-axis mode missed | Basin ledger; competing-mode veto; scale agreement |
| L-BFGS max 200 iterations, tolerance `1e-10` | Convenience hypothesis | Tight target-only stationarity search in 4D | False convergence or wasted calls | Convergence flag, iterations, score norm |
| Hessian steps and `1e-3` stability | Convenience hypothesis | Three-scale cancellation/roundoff check | Bad local proposal geometry | SPD and step-delta vetoes |
| Orders `7,9,11,13` | Budget ladder | 2,401 to 28,561 points per proposal scale | Shared under-resolution | Adjacent-order plus cross-scale vetoes |
| Proposal scales `1.0,1.5` | Hypothesis | Tests sensitivity to Laplace-tail mismatch | Both miss a remote mode | Multistart screen and scale stability; otherwise plain HMC |
| Weight ESS 50 / max 0.05 | Convenience guard | Detects concentrated unresolved sums before summary comparison | False confidence from a few nodes | Recorded ESS and maximum weight |
| Equivalence/stability margins | Convenience scientific hypotheses | Scale-free, strict candidate gate | Outcome sensitivity to margins | Report raw discrepancies and margin ratios |
| 2,000 bootstrap replicates | Budgeted default | Stable 0.99 empirical upper bounds at low cost | Tail Monte Carlo noise | Record seed and order-statistic resolution |
| Block length 32 | Derived | `round(sqrt(1000))` for 1,000 draws/chain | Residual dependence longer than block | Existing ESS/R-hat plus sensitivity at blocks 16 and 64 as explanatory only |

## Baseline ladder

- Naive baseline: prior-centered Gaussian/Laplace description, explanatory only.
- Tuned classical authority if needed: newly tuned plain fixed-HMC with its own
  sequential screen; not run unless quadrature is invalid/inconclusive.
- Proposed method: trained seed-B NeuTra with fixed-HMC.
- Enhanced proposed method: not introduced in this validation phase.

## Compute, attempts, and commands

- Campaign wall cap: 18,000 seconds.
- CPU allocation: 25 pinned target workers plus the parent/supervisor; no GPU.
- Quadrature evaluation budget: at most 104,328 target rows across both scales
  and all four orders. The observed 25x4 warm throughput of about 6.4 rows/s
  projects about 4.5 hours at the complete ladder and about 2 hours through
  order 11, excluding startup/MAP overhead.
- One preflight attempt and one reference attempt are authorized. A localized
  harness failure may be repaired and retried inside the same cap; scientific
  threshold changes require a new versioned plan/output.
- Planned commands:

```bash
CUDA_VISIBLE_DEVICES=-1 conda run -n tfgpu python docs/benchmarks/run_ssl_lstm_q20_seed_b_posterior_reference_2026_08_07.py --mode preflight --output-root docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r2/preflight --cap-seconds 900
CUDA_VISIBLE_DEVICES=-1 conda run -n tfgpu python docs/benchmarks/run_ssl_lstm_q20_seed_b_posterior_reference_2026_08_07.py --mode reference --output-root docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r2 --cap-seconds 18000
CUDA_VISIBLE_DEVICES=-1 conda run -n tfgpu python docs/benchmarks/run_ssl_lstm_q20_seed_b_posterior_reference_2026_08_07.py --mode compare --output-root docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r2 --cap-seconds 1800
```

## Pre-mortem

- Misleading pass: all Gauss-Hermite ladders resolve one local mode but miss an
  off-axis mode. The prior-derived multistart basin ledger is the cheap guard;
  unresolved evidence causes a plain-HMC reference rather than a pass.
- Implementation failure mistaken for science: target status fails at extreme
  nodes or Hessian differencing is unstable. These invalidate quadrature only.
- Tuning leakage: retained samples influence center, covariance, order, scale,
  or thresholds. Phase separation and reference hash creation before archive
  parsing veto the comparison.
- Apparent disagreement from autocorrelation: raw errors are not the gate; the
  chain-aware block-bootstrap upper bound is.
- Apparent agreement from weak margins: all margins and their convenience
  provenance are frozen here and raw margin ratios remain visible.

## Skeptical audit before execution

Audit status: **passed after revision**.

- Wrong baseline: no existing q=20 plain-HMC posterior exists. The comparator is
  the same signed target density, not training loss or a weak historical run.
- Proxy promotion: MAP, Hessian, ESS, and quadrature stability only admit the
  reference. They cannot establish posterior agreement by themselves.
- Stop conditions: target/archive mismatch, invalid rows, unstable geometry,
  unresolved order/scale, corruption, non-XLA/visible GPU, tests, and wall cap
  all stop claim-bearing comparison.
- Fairness: the reference uses no NeuTra transport, checkpoint, tuning kernel,
  warm-up, or retained values. Only the target source and prior define it.
- Hidden assumptions: remote modes, proposal tails, numerical Hessian, margins,
  and bootstrap dependence are explicit above.
- Environment: CPU is intentional for independent reference evaluation; XLA is
  required, GPU is hidden, TensorFlow/TFP float64 is required, and the target
  signature is fixed.
- Artifact sufficiency: separate reference and comparison artifacts preserve
  the phase boundary, inputs, hashes, diagnostics, uncertainty, and verdict.
- Revision made: finite-box quadrature was rejected because apparent grid
  stability would not bound truncation. Unbounded Gauss-Hermite plus independent
  proposal-scale stability replaces it. A quadrature failure triggers plain HMC
  rather than a candidate verdict.
- Preflight repair note: the first implementation attempt stopped before any
  quadrature row. Review found exact-float basin comparison and a missing
  `target/proposal` importance correction. Both were wrong for the stated
  reference. The r2 harness uses tolerance-based basin identity, the correct
  proposal-density correction, focused Gaussian-moment tests, first-admitted-
  order stopping, and a fresh output root. The scientific target, comparator,
  vetoes, equivalence margins, hardware class, and campaign cap are unchanged.
- r2 locator result: no quadrature row ran because fewer than five starts met
  the combined optimizer-stop/stationarity classification. The r3 diagnostic
  preserves every start as it completes and separates optimizer stop flags,
  score stationarity, and basin agreement. It does not relax the r2 reference
  veto or use retained values. A locator-policy revision is allowed only if the
  preserved target-only diagnostics show stationary same-basin endpoints and
  identify the stop flag, rather than geometry, as the failed condition.
