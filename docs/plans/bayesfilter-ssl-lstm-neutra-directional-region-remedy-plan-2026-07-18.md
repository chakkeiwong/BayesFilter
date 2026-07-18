# SSL-LSTM NeuTra Directional-Region Remedy Plan

Date: 2026-07-18

Status: `CONTROLLED_AUDIT_PASSED_TARGET_CONFIRMATION_CLOSED`

## Research Intent Ledger

| Field | Prospective contract |
| --- | --- |
| Main question | Can dimension-matched average and horizon confidence regions, prospectively calibrated HAC bandwidth, and observation-noise Rao-Blackwellization recover controlled decision power while retaining familywise false-decision control? |
| Exact baseline | The closed July 17 procedure: one 20-dimensional 95% Wald region for average and all horizon losses, `kappa_HAC=1`, zero ridge, and two simulated forecast replications |
| Candidate mechanism | One 20D average region at `alpha_avg=0.025`, ten 2D marginal horizon regions at `alpha_h=0.0025`, development-only HAC nomination, and conditional-moment influence estimation |
| Expected failure mode | The split geometry may repair local power while persistent equivalence remains limited by HAC bias; Rao-Blackwellization may help only when observation noise is a material variance component |
| Promotion criterion | In a locked fresh-seed controlled audit, every required family certifies coverage at least 90%, required-decision probability at least 80%, false-decision probability at most 5%, and invalid probability at most 5% with prospectively allocated exact-binomial bounds |
| Promotion veto | Any required family misses any primary operating target; exact-boundary decisive leakage exceeds its declared ceiling; scale provenance is not independent |
| Continuation veto | Invalid controlled-law algebra, non-finite or non-positive-definite covariance, failed KKT checks, corrupted/overwritten artifact, GPU/XLA provenance failure, or resource-cap exhaustion |
| Repair trigger | A valid development miss nominates the next declared mechanism; a valid locked-audit miss rejects that candidate but not the validation direction |
| Explanatory diagnostics | Point losses, continuous bounds, condition numbers, observed coverage and decisions without simultaneous certification, runtime, and path-versus-Rao variance ratios |
| Forbidden conclusion | No HMC validity, G/H predictive equivalence, posterior correctness, NeuTra quality, sampler superiority, model adequacy, or default readiness follows from this program |

## Mathematical Contract

Preserve ten horizons, equal horizon weights, TensorFlow/TFP `float64`, zero
ridge, and

`K_avg = K_max = 0.0068491` (computed exactly in code from the frozen anchors).

The regions allocate `0.025 + 10 * 0.0025 = 0.05`. If the average region and
all horizon marginal regions cover their respective truths, neither a false
equivalence nor a false material-difference decision is possible. The union
bound therefore controls either false-decision type by at most 0.05
asymptotically without an independence assumption.

For conditional means `m` and variances `s2`, use

`mu = mean(m)`, `v = mean(s2 + m**2) - mu**2`,

with cluster influences `mean_r(m)-mu` and
`mean_r(s2 + (m-mu)**2)/v - 1`. Target use requires a scale frozen from an
independent calibration-only bank; confirmation values may not set the scale.

## Evidence Contract

| Evidence role | Contract |
| --- | --- |
| Primary promotion evidence | One fresh-seed locked controlled audit of one frozen candidate, with exact simultaneous binomial bounds and all required families |
| Promotion veto | Any miss of coverage, required decision, false decision, invalid procedure, or boundary leakage target |
| Continuation veto | Only the validity, artifact, environment, authority, or resource failures listed in the intent ledger |
| Repair trigger | Development evidence may nominate geometry, HAC, or Rao-Blackwell repairs but may not promote them |
| Explanatory only | July 17 outcomes, development-arm rates, continuous loss summaries, tail metrics, and runtime |
| Result artifact | Fresh JSON receipts under `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/directional-region-remedy/` and one result/reset note beside this plan |
| Nonclaims | Passing certifies only finite-sample behavior on the declared controlled laws; target HMC and G/H confirmation stay closed |

## Phases

### 1. Document and prove the remedy

Revise Chapter 28a to define conditional-moment influences, independent scale
provenance, finite-sample HAC bias, split-region geometry, the union-bound
proof, and boundary behavior. Audit all numerical constants and cross
references, then build the book.

### 2. Implement focused statistical APIs

Add authenticated scalar and batched split-region bounds and classification to
`predictive_equivalence.py`. Add conditional-moment mean/log-variance
influences. Do not edit the closed July 17 runner or the unrelated
`dense_directional_score_geometry.py` lane.

### 3. Add controlled tests and runner

Test region dimensions/radii, covariance projections, union allocation,
tamper rejection, scalar/batched parity, path/conditional-moment algebra,
boundary truth construction, scale boundaries, and zero-ridge/HAC contracts.
Add a fresh runner with `smoke`, `development`, and `audit` modes. The runner
must refuse overwrite and any HMC, retained-sample, confirmation, network, or
model-file input.

### 4. Development nomination

On development-only seed domains compare, without ranking claims:

1. full-20D July 17 geometry, path estimator, `kappa=1`;
2. split geometry, path estimator, `kappa=1`;
3. split geometry, path estimator, one multiplier nominated from
   `(1.0, 1.5, 2.0, 3.0)` using coverage and validity before power;
4. the nominated split/HAC design with conditional-moment influences.

Use the smallest rung that discriminates mechanisms. Development evidence may
select one locked candidate but is not promotion evidence and supports no
stochastic ranking.

### 5. Audit-capacity preflight and locked audit

Freeze the exact number of families, looks, and operating claims before the
audit. The 8,192-draw development nomination must not proceed directly if its
weakest observed primary decision rate is below the observed rate required to
certify the audit target. In that case, run the selected mechanism only at a
fresh-seed capacity ladder `(12,288, 16,384)`, stopping at the first rung with
minimum primary required-decision rate at least 85%, maximum
false/wrong-direction/boundary rate at most 2%, zero invalid procedures, and
pooled descriptive coverage across the gated families in `[0.93,0.97]`.
These are nomination thresholds, not promotion evidence. Family-specific
coverage remains an audit criterion; it is not screened by the minimum of 96
development replications. Requiring every one of 22 observed family rates to
exceed 93% would falsely stop about 99.35% of the time under independent exact
95% coverage and is therefore forbidden as a capacity gate. If neither rung
qualifies, keep audit closed and write a blocker.

Use at least 1,536 replications per family when the simultaneous claim
count is at most 120: at tail probability `0.05/120`, this gives about 96.1%
probability of certifying a 5% upper target when the true event rate is 2.5%.
For the current 77-claim ledger, the 80% lower-bound target requires at least
1,279/1,536 decisions (83.27% observed). Use one capacity-nominated draw count,
one frozen HAC multiplier and estimator, and fresh audit seeds. No post-audit
candidate selection or rerun as prospective evidence is allowed.

### 6. Result and consistency review

Write the run manifest, decision table, inference-status table, negative-result
separation, and post-run red team. Re-run focused tests, compile checks, LaTeX,
and a code-to-equation audit. Record any residual issue explicitly.

## Commands And Environment

Focused CPU-hidden checks:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_predictive_equivalence_principled_repair.py \
  tests/test_ssl_lstm_neutra_directional_region_remedy.py
```

Document build:

```bash
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

GPU commands will be frozen after runner review. They must use the repository
TensorFlow/TFP GPU/XLA path, record device/TF32/JIT/trust provenance, and use
fresh output paths. Combined development plus audit resource cap is 2 trusted
GPU-hours; smoke cap is 10 minutes. Sequential stopping applies after a valid
locked candidate passes all gates. No HMC or NeuTra training is authorized.

## Pre-Mortem

- A split-region pass could be misleading if marginal covariances are sliced
  in the wrong feature order. Scalar analytic and batched parity tests must
  verify `(mean_h, log_variance_h)` selection.
- HAC tuning could overfit the audit. Only development seeds may nominate a
  multiplier; audit seeds and results remain inaccessible until freeze.
- Rao-Blackwellization could remove process uncertainty by mistake. Its input
  variance is conditional observation variance only; terminal/process
  simulations remain in the conditional means.
- Exact-boundary families could be mislabeled as equivalence or material.
  Their required outcome is inconclusive; either decisive outcome is leakage.
- A high observed pass rate could be promoted without certification. Only the
  simultaneous exact bound is primary.
- A controlled pass could be presented as target evidence. Artifacts must
  state that HMC and G/H confirmation were neither read nor executed.

## Skeptical Pre-Execution Audit

| Audit question | Finding |
| --- | --- |
| Wrong baseline? | No. The exact closed July 17 full-region/path/`kappa=1` procedure is the baseline; projection-only historical estimates are not used as promotion evidence. |
| Proxy promoted? | No. Development and Rao variance reductions only nominate; one fresh end-to-end audit is primary. |
| Missing stop condition? | No. Invalid math/artifacts/device, resource exhaustion, or a locked-audit primary miss stop promotion. Candidate failure alone does not reject the direction. |
| Unfair comparison? | All development arms use the same laws, draws, seed blocks, loss thresholds, zero ridge, and operating summaries. No superiority ranking is claimed. |
| Hidden assumptions? | Gaussian controlled-law conditional moments, stationary AR dependence, independent chains/arms, Wald asymptotics, fixed scales, and local proper-score interpretation are explicit. |
| Stale context? | The plan incorporates the July 17 immutable result and corrects its conservative 20D horizon geometry and underpowered 256-replication certification. |
| Environment mismatch? | CPU checks hide CUDA; serious execution is trusted TensorFlow/TFP GPU/XLA `float64`, consistent with repository policy. |
| Do artifacts answer the question? | Yes for finite-sample controlled behavior. They cannot answer posterior correctness, target equivalence, or model adequacy, which remain explicit nonclaims. |

Audit decision: `PASS_FOR_IMPLEMENTATION_AND_BOUNDED_CONTROLLED_EXECUTION`.
The principal repaired baseline and evidence roles are explicit; no proxy is a
promotion criterion; all material continuation vetoes and nonclaims are
prospective.

## Execution Close

Implementation, controlled development, capacity repair, and the locked audit
are complete. The result and post-run review are recorded in
`docs/plans/bayesfilter-ssl-lstm-neutra-directional-region-remedy-result-2026-07-18.md`.

The locked audit passed the controlled-law evidence contract. HMC, NeuTra
training, retained samples, G/H forecasts, and target confirmation were not
read or executed. The next authorized work is target-side scale provenance and
conditional-variance integration only; confirmation requires a separate
reviewed evidence contract.
