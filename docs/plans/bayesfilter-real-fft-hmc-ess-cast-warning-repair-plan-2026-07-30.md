# Real-FFT HMC ESS cast-warning repair plan

Date: 2026-07-30

Status: `COMPLETED`

## Research and engineering intent

| Field | Binding decision |
| --- | --- |
| Question | Can BayesFilter compute the existing split-chain cross-chain bulk/tail ESS without TFP's warning-producing `complex128 -> float64` cast while preserving the current numerical gate? |
| Mechanism | Replace only the three `tfp.mcmc.effective_sample_size` calls in `hmc_convergence.py` with a repository-owned real-input FFT implementation using `tf.signal.rfft` and `tf.signal.irfft` |
| Baseline | TFP 0.25.0 `effective_sample_size(..., filter_beyond_positive_pairs=True, cross_chain_dims=1)` |
| Pass criterion | Representative finite real chains match TFP ESS within `rtol=1e-10, atol=1e-10`; PP-UKF-style diagnostics emit no complex-to-real cast warning; existing convergence tests pass |
| Veto | Formula, shape, dtype, finite/nonfinite behavior, gate decision, or candidate diagnostic changes outside declared numerical tolerance |
| Explanatory only | Runtime and small floating-point differences below the parity tolerance |
| Nonclaims | No new convergence estimator, improved sampling, posterior correctness, candidate ranking, or change to the completed attempt-11 result |

## Evidence contract

The reference formula is the installed TFP 0.25.0 implementation. It centers
each chain, estimates lag autocovariance by zero-padded FFT, combines chains
using Vehtari et al.'s cross-chain variance expression, applies the existing
Geyer initial-positive-pair truncation, and returns total cross-chain ESS. The
repair must reproduce those operations. Only the FFT representation changes:
real FFT/inverse-real FFT replaces explicit complex FFT followed by an
unconditional complex-to-real cast.

The result artifact is the focused test output and this plan's completion
record. Existing attempt-11 samples and public result remain immutable.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| TFP cross-chain ESS formula | Current BayesFilter convergence gate and TFP 0.25.0 source | A rewritten formula changes promotion decisions | Direct parity on independent, correlated, tied, and tail-indicator-like chains | Binding baseline |
| Real `float64` inputs | HMC convergence contract | Complex model draws would need different semantics | Reject non-real/non-`float64` coercion through the existing draw contract | Binding scope |
| `rfft`/`irfft` | TensorFlow real-signal API | Padding or normalization differs from TFP | Same next-power-of-two padding and lag-denominator parity tests | Reviewed implementation choice |
| `1e-10` parity tolerance | Float64 FFT roundoff allowance | Tolerance hides a material estimator change | Report maximum absolute/relative differences in tests | Verification tolerance only |
| No attempt-11 rerun | Diagnostic implementation repair after immutable completion | Historical artifact is silently reinterpreted | Preserve attempt-11 artifacts and make no result-file edits | Binding non-action |

## Implementation

1. Add a private, TensorFlow-only real-FFT cross-chain ESS helper in
   `bayesfilter/inference/hmc_convergence.py` reproducing the installed TFP
   cross-chain and positive-pair equations.
2. Route the existing bulk, lower-tail, and upper-tail ESS calculations through
   the helper. Keep R-hat, quantiles, thresholds, schemas, and field meanings
   unchanged.
3. Add focused parity tests against TFP and a regression test proving the
   BayesFilter diagnostic emits no `complex128 -> float64` warning.
4. Run focused convergence and PP-UKF/NeuTra controller tests, compile checks,
   and `git diff --check`.
5. Record the measured parity and completion verdict here. Commit only this
   lane's plan, implementation, and tests.

## Skeptical plan audit

- **Wrong baseline:** the comparator is the exact installed TFP function used
  by the completed campaign, not a different ESS definition.
- **Proxy promotion:** warning removal is an engineering criterion; numerical
  parity and unchanged pass/fail behavior are the scientific criteria.
- **Missing stop conditions:** any parity failure, nonfinite drift, schema
  change, or existing-test regression blocks completion.
- **Unfair comparison:** both implementations receive identical tensors and
  use the same chain axis, lag weighting, and positive-pair rule.
- **Hidden assumptions:** the helper is deliberately scoped to static rank-3
  real `float64` HMC draws, which is the existing caller contract.
- **Environment mismatch:** CPU tests are sufficient because this is a
  deterministic diagnostic formula, not a GPU sampling result; the production
  TensorFlow API remains GPU-compatible.
- **Artifact adequacy:** direct output parity and captured warning absence
  answer the stated question; a new HMC campaign would not add discriminating
  evidence.

Audit verdict: `PASS_FOR_IMPLEMENTATION`. The warning was independently
reproduced from TFP ESS on a real tensor without PP-UKF or HMC, and the observed
attempt-11 count is exactly explained by 24 diagnostics times three ESS calls.

## Completion record

The repository-owned helper now reproduces TFP 0.25's cross-chain
initial-positive-pair ESS formula using `tf.signal.rfft`/`tf.signal.irfft`.
The first parity run correctly caught a missing `N-k` lag denominator; no code
was accepted until that TFP normalization was restored.

Verification:

- independent, correlated, and tied-indicator fixtures matched TFP within
  `rtol=1e-10, atol=1e-10`;
- the final attempt-11 `L=9` model-coordinate tensor had maximum direct ESS
  difference `3.64e-11` from TFP;
- the full real-FFT HMC diagnostic on that tensor reproduced
  `max_rhat=1.0069797083237662`, `min_bulk_ess=18487.046136262437`, and
  `min_tail_ess=2881.5381539385935`, with zero complex-cast warnings;
- focused convergence, shared NeuTra HMC, PP-UKF validation-driver, and
  preflight tests passed: `41 passed` after adding the correlated/tied parity
  cases;
- Python compilation and `git diff --check` passed.

Completion verdict: `PASS`. The estimator, thresholds, schemas, and completed
attempt-11 decisions are unchanged. New diagnostics identify the implementation
as `Geyer initial positive pairs; TFP 0.25 formula parity via TensorFlow real
FFT`. No HMC rerun is required because the preserved real samples reproduce the
terminal diagnostic and the change affects only deterministic post-sampling
ESS computation.
