# SSL-LSTM NeuTra Principled Predictive-Test Repair Result

Date: 2026-07-17

Decision: `ENGINEERING_AND_MATHEMATICAL_REPAIR_PASSED_SCIENTIFIC_CALIBRATION_PENDING`

## Outcome

The state-space LSTM chapter is now self-contained for the locked scalar model,
the NeuTra experiment history, the predictive estimand, the fixed-batch defect,
the growing-bandwidth repair, and the proper-score confidence-region decision.
The code now implements the prospective method in TensorFlow/TFP `float64` with
XLA enabled by default.

The repair closes the known mathematical inconsistency in the historical
fixed-16 uncertainty route. It does not select a scientifically acceptable
forecast-loss budget, validate finite-sample coverage/power on the complete
20-feature decision, calibrate MMD, acquire HMC draws, or open the blinded G/H
confirmation suffix.

## Main Mathematical Result

For a unit-variance AR(1) sequence with correlation `0.6`, the true long-run
variance is `4.0`, whereas the fixed-16 Bartlett/batch limit is
`3.531382239526912`, an `11.7154%` underestimate. This bias does not vanish as
the draw count grows.

The prospective estimator uses per-chain centering and bandwidth
`floor(kappa_HAC * N^(1/3))`. Under the chapter's explicit stationarity,
ergodicity, moment, strong-mixing-summability, and CLT assumptions, a growing
bandwidth with `ell_N -> infinity`, `ell_N/N -> 0`, and `ell_N^2/N -> 0` is
consistent for the raw long-run covariance. A positive fixed ridge is not covered by that
claim: the implementation records it as numerically admissible but inference
inadmissible. Confirmation requires zero ridge unless a separately proved
vanishing loading is added.

The proper-score rule uses

```text
L(delta) = sum_h lambda_h * (0.5 * delta_mean,h^2
                              + 0.25 * delta_log_variance,h^2).
```

It computes the exact minimum and maximum of this quadratic loss over the joint
Wald ellipsoid. Equivalence requires the maximum to lie below the frozen loss
budget; material difference requires the minimum to lie above it; overlap is
inconclusive. The false-equivalence and false-material-difference arguments
reduce to noncoverage of the joint confidence region and are therefore
asymptotically controlled at its declared level under the stated assumptions.

## Controlled Evidence

The deterministic fixed-seed AR(1) diagnostic used 32 independent chains,
1,024 burn-in steps, 32,768 retained draws, and stateless seed
`[20260717,1901]`.

| Estimator | Draws/chain | Bandwidth | Estimate | Absolute error from 4.0 | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Growing Bartlett HAC | 4,096 | 16 | `3.394244148698` | `0.605755851302` | Development diagnostic |
| Growing Bartlett HAC | 32,768 | 32 | `3.799503134873` | `0.200496865127` | Primary controlled mechanics check |
| Historical fixed batch | 32,768 | 16 | `3.537976593971` | `0.462023406029` | Failed historical baseline |

The realized continuous estimates are descriptive for one fixed synthetic
stream. The theorem and analytic AR(1) limit, not a stochastic ranking, establish
the defect and the appropriateness of a growing-bandwidth repair.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit the additive HAC and proper-score APIs for controlled calibration | Passed 76 focused tests, analytic fixtures, XLA/eager parity, KKT checks, tamper checks, and AR(1) limit check | No implementation, formula, finite-value, XLA, or confirmation-leakage veto | Finite-sample bandwidth performance beyond the controlled fixture | Design a fresh direct coverage/power calibration | G/H equivalence, posterior correctness, or target sample sufficiency |
| Reject fixed-16 intervals as a general confirmatory basis | Analytic counterexample proves nonvanishing bias | No ambiguity in the AR(1) calculation | Historical receipts used the estimator only for feasibility/design work | Preserve receipts as historical diagnostics; do not promote them | Historical runs are useless or their point estimates are invalid |
| Keep positive-ridge HAC outside inference | Fail-closed status implemented and tested | Positive ridge is an inference veto, though it may be numerically diagnostic | A future vanishing-loading theorem could change this boundary | Require zero ridge in the next direct validation | Every singular finite-sample covariance makes the method scientifically false |
| Keep MMD explanatory | No scientific tolerance or repaired MMD uncertainty was calibrated | Promotion boundary remains closed | Whether MMD adds useful shape power after primary loss calibration | Calibrate separately after the primary decision is viable | Joint predictive-law equivalence |
| Keep Phase 9 and HMC acquisition closed | No loss budget `K` or direct finite-sample design exists | Authority/evidence boundary intact | Application meaning of acceptable forecast regret | Obtain a domain choice of `K`, then validate coverage/power prospectively | G/H predictive outcome or need for 4,096/8,192 target draws |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Passed for the bounded implementation and controlled mechanics checks |
| Viable method | Growing Bartlett HAC plus exact proper-score ellipsoid bounds is a viable calibration candidate |
| Statistically supported ranking | None; no stochastic method superiority ranking was attempted or supported |
| Descriptive-only differences | Realized AR(1) estimates, runtimes, and finite-sample errors |
| Default-readiness | Not established; no public/default change was made |
| Next evidence needed | Prospective choice of `K`, horizon weights, HAC multiplier, sample ladder, fresh seeds, simultaneous coverage target, false-decision bound, and direct controlled validation of the complete decision |

## Equation-To-Code Audit

| Document object | Implementation anchor | Audit result |
| --- | --- | --- |
| Augmented state, gate order, sigmoid/tanh activations | `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:791` and `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:830` | Matches Chapter 28a gate and state ordering |
| Cell, hidden, and latent transition mean | `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:505` | Matches `c_t`, `a_t`, and `z_t` mean equations |
| Observation mean | `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:522` | Matches `C z_t + e` |
| Stochastic/deterministic partition | `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:357` | Only latent coordinates receive innovation; hidden/cell coordinates are deterministic completion |
| Parameter order and softplus scales | `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:123` and `bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:171` | Matches the 24-entry scalar chart and positive-scale transform |
| Four free parameters and prior center | `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py:79`, `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py:105`, and `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py:138` | Matches `(A,d,C,e)` and the locked center |
| SVD-UKF likelihood plus `N(psi_0,4^2 I)` prior | `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py:765` | Matches the posterior target equation |
| Transition-then-observation forecast timing | `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py:1070`, `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py:1109`, and `bayesfilter/nonlinear/ssl_lstm_predictive_tf.py:1121` | Matches the documented one-to-ten-step forecast law |
| Growing bandwidth rule | `bayesfilter/inference/predictive_equivalence.py:1310` | Matches `max(1,floor(kappa_HAC*N^(1/3)))`; `math.cbrt` avoids exact-cube rounding errors |
| Bartlett HAC | `bayesfilter/inference/predictive_equivalence.py:1323` and `bayesfilter/inference/predictive_equivalence.py:1418` | Matches the chapter's lag weights, per-chain centering, symmetry, and pooled-mean scaling |
| Scientific loss matrix | `bayesfilter/inference/predictive_equivalence.py:2196` | Matches mean coefficient `1/2` and log-variance coefficient `1/4` in mean-then-log-variance order |
| Exact confidence-region extrema | `bayesfilter/inference/predictive_equivalence.py:2232` and `bayesfilter/inference/predictive_equivalence.py:2404` | Cholesky transform, eigenbasis secular solve, hard-case handling, and KKT residual checks match the trust-region derivation |
| Three-way decision | `bayesfilter/inference/predictive_equivalence.py:2540` | Matches upper-below-budget, lower-above-budget, otherwise inconclusive logic |

## Verification

| Check | Result |
| --- | --- |
| Focused predictive statistics | `76 passed` in `36.69 s`; two dependency deprecation warnings only |
| Lightweight model/document algebra | `7 passed` in `3.03 s`; two dependency deprecation warnings only |
| Python compilation | Passed for the repaired module and new tests |
| Diff whitespace check | Passed for scoped files and repository-wide `git diff --check` |
| LaTeX build | `docs/main.pdf`, 404 pages, 1,591,016 bytes |
| New chapter citations/references | Resolved, including `neweywest1987` and `flegal2010batchmeans` |
| Broader compiled model suites | Resource-stopped at the plan's 10-minute CPU cap after 39% with no observed failure; a narrower compiled retry was also stopped; neither is counted as a pass |

The full book still reports 11 undefined citations and four multiply defined
labels in unrelated chapters. They predate this lane and were not modified.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `b1606a3ec19643356705cf9d08ccf7c6495b6186` with unrelated dirty worktree preserved |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TFP `0.25.0` |
| Device | CPU-only reference; `CUDA_VISIBLE_DEVICES=-1`; GPU intentionally hidden |
| Numerical policy | TensorFlow/TFP `float64`; XLA default APIs checked against non-XLA reference |
| GPU status | No GPU run; no GPU performance or production evidence claimed |
| Data version | Locked 30-observation scalar SSL-LSTM fixture for source audit only; no confirmation forecasts opened |
| Random seeds | AR(1) stateless seed `[20260717,1901]`; no HMC/NeuTra seeds used |
| Primary command | `CUDA_VISIBLE_DEVICES=-1 ... python -m pytest -q tests/test_predictive_equivalence.py tests/test_predictive_equivalence_principled_repair.py` |
| Document command | `cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` |
| Output artifacts | This result, matching reset memo, `docs/main.pdf` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-principled-predictive-test-repair-plan-2026-07-17.md` |
| Result | This file |

## Negative-Result Separation

- Implementation failure: not supported.
- Mathematical inconsistency in the new raw HAC estimator: not supported under
  the stated assumptions and prospective bandwidth conditions.
- Historical inferential-design failure: supported for fixed-16 general
  consistency and for midpoint-only scientific margins.
- Tuning/calibration failure of the repaired method: not tested.
- Evidence against NeuTra, HMC, the SSL-LSTM, or predictive-functional
  validation: not supported.

## Post-Run Red Team

The strongest alternative explanation is that the repaired asymptotics are
correct but finite-sample performance remains inadequate for the target's
dependence and 20-dimensional loss region. The AR(1) diagnostic cannot establish
coverage or power for forecast influence sequences, and a reverse-KL transport
pair can still share a missing posterior mode.

This closeout would be overturned by a formula mismatch, a trust-region
counterexample, a reproducible XLA/eager discrepancy, or direct validation
showing unacceptable false-decision rates under the frozen design. The weakest
remaining element is scientific, not engineering: no domain-approved loss
budget `K` exists. Without it, the method can quantify uncertainty and loss but
cannot declare practical equivalence.

## Handoff

The next plan should be a direct finite-sample calibration plan, not HMC
acquisition. It must freeze:

- the acceptable symmetric log-score regret budget `K`;
- horizon weights;
- HAC multiplier and zero-ridge rule;
- sample-size ladder and fresh seeds;
- simultaneous coverage and false-decision targets;
- complete decision families, including whether MMD remains explanatory or is
  separately calibrated as co-primary.

Only a passing direct validation can justify a separately authorized target
draw acquisition or G/H confirmation forecast.
