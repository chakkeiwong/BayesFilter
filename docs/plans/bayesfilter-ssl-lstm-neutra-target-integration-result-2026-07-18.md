# SSL-LSTM NeuTra Target Integration Result

Date: 2026-07-18

Decision: `TARGET_ADAPTER_PREFLIGHT_PASSED_GH_CONFIRMATION_CLOSED`

## Scope

Phases 1--5 of the target-integration plan were executed.  The work derived
the conditional observation variance from the frozen SSL-LSTM parameter chart,
implemented the target adapter, calibrated independent horizon scales,
compared path and conditional-moment features on fresh target-shaped banks,
and performed a numerical split-region preflight.  No HMC or NeuTra training,
retained posterior artifact, G/H comparison, or confirmation result was read.

## Review And Repairs

The pre-execution audit found and repaired three plan ambiguities:

- `sigma_y` is now sourced from `unpack_ssl_lstm_parameters` and not inferred
  from realized observation innovations.
- Moment standardization uses per-horizon calibration-path mean and unbiased
  standard deviation, not the MMD pairwise-distance bandwidth helper.  Four
  three disjoint seed domains are explicit: calibration `4101--4104`, paired
  evaluation `4201--4204`, and robustness `4301--4304`.
- HAC on the fixed ten-row A2 fixture is classified as engineering
  shape/conditioning evidence only because those rows are not stationary
  posterior chains.

The first implementation run exposed a rank-two variance broadcast error and a
test fixture that zeroed only one calibration chain.  Both were repaired and
the focused adapter suite passed.  A later audit caught a tautological paired
check based on averaging centered influences; it was replaced by the actual
feature-estimate difference against the standard error of per-cluster influence
differences.  The initial and intermediate receipts were preserved; repair-04
is authoritative.

## Evidence Contract Result

| Evidence role | Result |
| --- | --- |
| Primary integration evidence | Repair receipt `target-integration-preflight-repair-04.json` with source hashes, calibration provenance, target adapter outputs, covariance checks, and `GH_CONFIRMATION_CLOSED` status |
| Hard vetoes | None in the declared adapter/preflight scope |
| Target confirmation | Still closed; Phase 6 was not executed |
| Explanatory only | Path/Rao feature differences, independent-bank differences, scale values, condition number, and runtime |
| Statistical ranking | None; the fixture is target-shaped engineering evidence, not a posterior-chain comparison |
| Nonclaims | No posterior correctness, HMC validity, NeuTra quality, G/H equivalence/material difference, model adequacy, sampler ranking, or default readiness |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit target adapter for further planning | All adapter/preflight hard checks passed in repair-04 | No adapter, provenance, covariance, or artifact veto | Fixture is not a stationary posterior chain | Draft a separate Phase 6 authorization plan if desired | G/H or posterior correctness |
| Keep calibration scales frozen | Four disjoint calibration chains and positive scales passed | No scale provenance veto | Calibration fixture is engineering-only | Reuse only the hashed receipt | Universal scale optimality |
| Keep G/H confirmation closed | Boundary explicitly preserved | Confirmation authority gate not executed | Actual retained-chain covariance remains untested | Require a new reviewed confirmation plan | Equivalence/material difference |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Passed for target adapter and engineering preflight; no invalid values, provenance mismatch, covariance failure, or artifact overwrite |
| Statistically supported ranking | None; no sampler or estimator ranking was attempted |
| Descriptive-only differences | Path/Rao feature differences, independent-bank differences, scales, condition number, and runtime |
| Default readiness | Not established |
| Next evidence needed | Stationary retained-chain covariance and a separately authorized G/H confirmation |

## Preflight Checks

| Check | Result |
| --- | --- |
| Conditional observation variance | Passed; `s2[d,r,h] = sigma_y[d]^2`, finite and positive |
| Feature order | Passed; mean horizons followed by log-variance horizons |
| Calibration scale provenance | Passed; four calibration chains, ten rows, two replications, positive per-horizon scales |
| Seed separation | Passed; calibration/evaluation/robustness domains disjoint |
| Paired six-MCSE diagnostic | Passed descriptively on 40 evaluation draw clusters |
| Independent robustness bank | Generated and recorded; differences are descriptive only |
| HAC covariance | Finite, SPD, zero-ridge admissible; engineering-only interpretation |
| Covariance condition number | `272.6164`, below `1e8` limit |
| Minimum covariance eigenvalue | `6.0338e-05`, positive |
| Split alpha allocation | `0.025 + 10*0.0025 = 0.05` |
| Split bounds/KKT | Authenticated and admissible; exact residuals are in the repair receipt |
| CUDA/JIT provenance | CPU-hidden XLA reference forecasts and eager reference statistical adapter; no GPU or production claim |

## Run Manifest

| Field | Value |
| --- | --- |
| Command | `CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_target_integration_2026_07_18.py --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-integration/target-integration-preflight-repair-04.json` |
| Environment | conda `tfgpu`, TensorFlow `2.20.0`, TensorFlow Probability `0.25.0` |
| Execution role | CPU-hidden XLA reference forecasts; eager reference statistical adapter |
| Trust basis | `cpu_hidden_reference_exception_not_gpu_evidence` |
| Wall time | `177.43` seconds |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-target-integration-plan-2026-07-18.md` |
| Plan SHA-256 in receipt | `6900692a99a02b8057b0cd26ea902e6d8430396d59d9b4211c669202077fe251` |
| Repair receipt SHA-256 | `a2b269ab9595adfa5a2ba9db4f033301b1faa947a21e7bdf2211dc6d5874f59d` |

## Artifact Hashes

The initial receipt is preserved at:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-integration/target-integration-preflight.json`.

The authoritative repaired receipt is:
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-integration/target-integration-preflight-repair-04.json`.

The receipt records exact hashes for the runner, adapter,
predictive-equivalence, forecast, and parameter-adapter source files.  It also records calibration
center/scale hashes, innovation-bank signatures, split-region KKT residuals,
and all nonclaims.

## Phase Handoff

Phases 1--5 hand off successfully to a separate authorization decision.  The
next plan, if authorized, must use the repaired receipt and source hashes,
retain the independent scale and covariance policies, and predeclare the
actual stationary retained-chain covariance gate.  It must not treat this
target-shaped engineering receipt as G/H or posterior evidence.

Phase 6 is explicitly closed pending that separate plan.

## Post-Run Red Team

The strongest alternative explanation is that the fixed A2 ten-row fixture is
too small and nonstationary to represent retained posterior chains.  The
preflight can therefore pass while the actual G/H covariance or forecast
comparison later fails.  This is why the HAC result is labeled engineering
only and why no confirmation was opened.  The conclusion would be overturned
by a source-hash mismatch, overlapping seed domains, a feature-order error, or
an observation/process variance mix-up; repair-04 checks each of those
conditions.  The weakest remaining evidence is external validity to actual
retained chains.
