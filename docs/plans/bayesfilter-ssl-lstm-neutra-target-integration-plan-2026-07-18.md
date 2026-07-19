# SSL-LSTM NeuTra Target Integration Plan

Date: 2026-07-18

Status: `PHASES_1_TO_5_COMPLETE_PHASE_6_GH_CONFIRMATION_CLOSED`

## Scope and Boundary

This plan is the handoff from the locked controlled directional-region audit.
It integrates the repaired predictive statistic with the frozen scalar
SSL-LSTM forecast path.  It does not authorize HMC acquisition, NeuTra
training, retained-sample selection, or a G/H confirmation.  A confirmation
may be proposed only after the final target-side gates in this plan pass and a
separate execution decision is recorded.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can the locked split-region/Rao-Blackwell/HAC procedure be evaluated on the actual frozen SSL-LSTM forecast representation without changing its estimand or using confirmation data to tune it? |
| Candidate mechanism | Target adapter built from the existing ten-horizon forecast outputs, with observation-noise conditional moments and independently frozen horizon scales. |
| Exact baseline | The closed July 17 full-region/path procedure for historical context, and the locked controlled candidate: one 20D average region plus ten 2D horizon regions, zero ridge, Bartlett `kappa_HAC=3.0`. |
| Expected failure mode | Observation variance is extracted from the wrong parameter chart, process uncertainty is accidentally removed, feature order is permuted, scales depend on confirmation data, or target covariance is too ill-conditioned for the fixed region calculation. |
| Promotion criterion | All target adapter, scale provenance, covariance, finite-value, and path/Rao consistency gates pass on fresh target-shaped fixtures and an independent calibration bank; only then may a separate G/H confirmation plan be considered. |
| Promotion veto | Any feature-order mismatch, non-positive/non-finite conditional variance, scale provenance violation, inadmissible covariance, failed path/Rao algebra check, or use of confirmation data for tuning. |
| Continuation veto | Broken frozen forecast contract, missing required fixture inputs, corrupted artifact, unavailable declared execution environment, or resource exhaustion before the preflight artifact is complete. |
| Repair trigger | A valid target-shaped failure nominates only the declared repair (variance extraction, feature mapping, scale handling, or covariance conditioning); it does not justify opening G/H confirmation. |
| Explanatory diagnostics | Path/Rao point differences, conditional-to-total variance ratios, HAC condition numbers, scale magnitudes, runtime, and memory use. |
| Forbidden conclusion | No posterior correctness, HMC validity, NeuTra quality, model adequacy, sampler ranking, or G/H equivalence/material-difference claim follows from this adapter work. |

## Evidence Contract

| Evidence role | Required evidence |
| --- | --- |
| Primary integration evidence | A fresh target-shaped adapter test artifact showing exact feature ordering, finite conditional variances, independent scale provenance, and path/Rao agreement within predeclared Monte Carlo tolerances. |
| Promotion veto | Any hard numerical, provenance, algebra, or artifact failure listed above. |
| Continuation veto | Only the validity, authority, environment, and resource failures listed above. A candidate adapter miss is a repair signal, not rejection of the predictive-validation direction. |
| Explanatory only | Continuous loss differences, variance-reduction ratios, condition numbers, and runtime without a predeclared uncertainty analysis. |
| Preserved artifact | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-integration/` containing calibration-bank metadata, adapter/preflight JSON, hashes, and a result note beside this plan. |
| Nonclaims | Passing establishes only that the declared statistic is wired to the frozen forecast representation under the tested target-shaped fixtures. It does not open G/H confirmation. |

## Frozen Mathematical Contract

For draw `d`, forecast replication `r`, and horizon `h`, the existing forecast
API returns an observation mean `m[d,r,h]` and an observation innovation
`e[d,r,h] * sigma_y[d]`, where `e` is standard normal and `sigma_y` is the
positive observation standard deviation from the embedded parameter draw.
The adapter obtains `sigma_y` from the already authenticated
`paths.terminal.full_parameters[d]` by applying
`unpack_ssl_lstm_parameters(..., std_floor=1e-4)`; it must not infer a scale
from the realized innovation. This is anchored in
`bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py:171-208`, where the
parameter chart locates `observation_std_start`, applies the softplus map and
floor, and constructs `observation_covariance` at lines 274-284. The forecast
recursion that multiplies the standard-normal observation bank by this
`observation_std` is at
`bayesfilter/nonlinear/ssl_lstm_predictive_tf.py:1105-1126`.
Therefore the conditional observation variance for the Rao-Blackwell adapter
is

`s2[d,r,h] = sigma_y[d]**2`.

This is conditional on the parameter draw, terminal-state draw, and realized
process path used to produce `m`.  It includes observation noise only.  The
terminal and process uncertainty remains represented by the draw and
replication clusters; it must not be replaced by `s2`.  For vector
observations, the adapter must use the declared observation covariance and
the exact feature projection, rather than silently taking a scalar norm.

The feature order is frozen as
`(mean_1,...,mean_H, log_variance_1,...,log_variance_H)` and must be passed to
`conditional_mean_log_variance_influence` without reordering.  Horizon
standardization is defined prospectively, not by the MMD pairwise-distance
helper.  From a calibration-only bank of complete observation paths, for each
horizon `h` compute the pooled mean `a^0_h` and unbiased standard deviation
`a_h`; require every `a_h` to be finite and strictly positive, then freeze and
hash the vectors before evaluating the independent evaluation bank.  The same
`a^0_h` and `a_h` transform path observations, conditional means, and
conditional variances (`s2/a_h**2`).  The calibration bank uses four chain
roots `(20260718, 4101)` through `(20260718, 4104)`; the independent evaluation
bank uses `(20260718, 4201)` through `(20260718, 4204)`.  No seed or materialized
innovation tensor may appear in both domains.  A robustness-only independent
bank uses `(20260718, 4301)` through `(20260718, 4304)` and is also disjoint
from both prior domains.  Each bank has four chains, ten
fixed A2 parameter draws, and two forecast replications per draw.  The fixed
predictive boundary remains `K_avg=K_max=0.0068491`; the split alpha allocation
remains `0.025 + 10*0.0025 = 0.05`; the HAC multiplier remains `3.0`; ridge
remains zero.

## Phases

### Phase 1: Freeze equations and inspect the adapter boundary

Read the frozen forecast dataclasses and compiled/eager paths.  Derive the
conditional variance from the actual parameter embedding and observation
equation, with source line anchors in the result note.  Confirm shapes,
dtype, horizon convention, cluster unit, and that terminal/process uncertainty
is not discarded.

Required checks: source inspection; scalar shape assertions; eager/XLA parity
on a tiny fixture; finite and strictly positive `sigma_y**2` checks.

Handoff: equations, feature order, and cluster semantics are written in the
adapter test/plan and no unresolved source ambiguity remains.

### Phase 2: Build the target conditional-moment adapter

Add the smallest TensorFlow/TFP adapter needed to transform forecast outputs
into the locked mean/log-variance influence representation.  Prefer existing
typed APIs.  Reject wrong dtype, rank, horizon, observation dimension,
non-finite values, non-positive variances, and provenance mismatches.  Keep
the path estimator available as a comparator; do not change the closed
controlled runner.

Required artifacts: source implementation, focused unit tests, and an
adapter-contract JSON receipt.  No HMC or retained posterior input is allowed.

Handoff: focused tests pass and the adapter returns authenticated features with
the declared ordering and status.

### Phase 3: Independent scale calibration

Generate a calibration-only bank from fresh seeds and the fixed ten-row A2
parameter fixture.  The bank must be independent of any future evaluation or
confirmation bank and must record seed, draw/replication counts, forecast
configuration signature, innovation hashes, worker count, and output hash.
Estimate the per-horizon center and unbiased standard deviation directly from
the complete observation paths, then freeze them before evaluating the
independent evaluation bank.  This is a scale contract, not a claim that the
calibration bank estimates the posterior predictive law.

Required gates: no overlap between calibration and evaluation seeds; finite,
positive scales; reproducible replay; no scale derived from a G/H decision or
from the confirmation bank.

Handoff: immutable calibration metadata and frozen scale receipt exist.

### Phase 4: Target-shaped path versus Rao validation

Use independent target-shaped fixtures with the same ten-horizon dimensions,
parameter chart, terminal covariance conventions, and innovation roles as the
frozen forecast API.  Compare path and conditional-moment estimates with
shared innovations for a paired precision diagnostic and independent
innovations for a robustness diagnostic.  For the shared-bank paired
diagnostic, compare the actual 20-feature path and conditional-moment estimates.
Estimate their paired Monte Carlo standard errors from the per-(chain, draw)
difference of their influence sequences over the 40 draw clusters, divided by
`sqrt(40)`, and require every finite feature-estimate difference to lie within
six such standard errors, with an absolute floor of `1e-12`.  Centered
influences must not themselves be averaged as the feature difference because
that would make the check tautological.  This is an
explanatory integration gate for the declared fixture, not a statistical
ranking or target equivalence claim.  Independent-bank estimates are reported
descriptively only.  No estimator is ranked from this run.

Required gates: exact conditional-variance algebra on a hand-computable
fixture; paired and independent estimates finite; the six-MCSE paired bound;
no process uncertainty loss; feature projection and scale invariance tests
pass; and calibration/evaluation seed domains are disjoint.

Handoff: target adapter result note states whether the candidate is viable,
what failed if not, and keeps G/H confirmation closed in either case.

### Phase 5: Covariance and resource preflight

Construct the locked split-region covariance from the evaluation-bank
target-shaped influence clusters using Bartlett HAC multiplier `3.0` and zero
ridge.  Check symmetry,
positive definiteness, eigenvalue floor, condition number, KKT residuals,
alpha allocation, and exact bound authentication.  Measure the declared
resource envelope without starting confirmation; stop before the cap.

The fixed ten-row A2 fixture is not a stationary posterior chain.  Its HAC row
order is therefore an engineering shape/conditioning check only.  Even if it
is numerically admissible, it cannot establish target long-run covariance or
authorize a G/H confirmation.  Actual retained-chain covariance remains a
separate gate in the later confirmation plan.

Handoff: a preflight receipt has all hard checks, source/config hashes, device
and JIT provenance, resource estimate, and an explicit `GH_CONFIRMATION_CLOSED`
status.

### Phase 6: Separate confirmation authorization gate

Only if Phases 1--5 pass may a new plan propose one locked G/H confirmation.
That plan must freeze the target scale receipt, adapter version, covariance
policy, seeds, family counts, stopping rule, and resource cap before reading
confirmation results.  Any failed gate writes a blocker/repair result and does
not open confirmation.

## Resource And Execution Policy

Phases 1--5 use focused CPU-hidden tests and small target-shaped fixtures.  Any
GPU/XLA run must use the trusted repository TensorFlow/TFP path and record
device, TF32, JIT, trust basis, wall time, and artifact paths.  External sample
generation is multicore CPU work by default.  No HMC runtime, posterior
acquisition, network access, model-file edit, or confirmation evaluation is
authorized by this plan.

## Skeptical Pre-Execution Audit

| Audit question | Finding |
| --- | --- |
| Wrong baseline? | No. The closed directional-region candidate and the existing frozen forecast API are named explicitly. |
| Proxy promoted? | No. Path/Rao agreement and variance ratios are integration gates/diagnostics; they cannot establish posterior or G/H correctness. |
| Missing stop condition? | No. Invalid algebra, provenance, covariance, artifact, environment, and resource vetoes are explicit; confirmation remains closed by default. |
| Unfair comparison? | No. Shared innovations are paired only for precision; independent banks are required for robustness, with the same chart and horizon convention. |
| Hidden assumptions? | Observation-only conditional variance, retained process/terminal clusters, independent scale provenance, and fixed HAC/alpha policy are stated. |
| Stale context? | No. This plan incorporates the 2026-07-18 controlled audit and its explicit target-side handoff. |
| Environment mismatch? | CPU checks hide CUDA; GPU/XLA provenance is mandatory for any serious run. |
| Do artifacts answer the question? | Yes for adapter wiring and target-shaped numerical validity; they explicitly cannot answer G/H or posterior claims. |

Audit decision: `PASS_FOR_TARGET_ADAPTER_PREFLIGHT_ONLY`.

## Amendment Record

The initial review found three material ambiguities and repaired them before
execution: (1) the conditional variance now cites the exact parameter-unpack
and forecast-recursion sources and forbids estimating it from realized noise;
(2) horizon scaling now uses per-horizon calibration-path mean and unbiased
standard deviation, with fixed seed domains and a six-MCSE paired check; and
(3) HAC on the fixed A2 row fixture is classified as engineering-only because
those rows are not a stationary posterior chain. The existing pooled
pairwise-distance routine is explicitly not used for moment standardization.
These amendments preserve the evidence contract and do not open G/H
confirmation.

## Close Record Requirements

At the end of each executed phase, record the exact command, environment,
seeds, wall time, artifact paths/hashes, hard-veto status, descriptive
diagnostics, decision, next handoff, and nonclaims in a result note.  Do not
rewrite immutable calibration or confirmation receipts after evaluation.

## Execution Close

Phases 1--5 completed with authoritative receipt
`target-integration-preflight-repair-04.json` and result note
`bayesfilter-ssl-lstm-neutra-target-integration-result-2026-07-18.md`.
All declared adapter/preflight hard checks passed.  Phase 6 was not executed;
G/H confirmation, HMC/NeuTra execution, retained-sample access, posterior
claims, and sampler ranking remain closed.

The authoritative receipt binds SHA-256
`6900692a99a02b8057b0cd26ea902e6d8430396d59d9b4211c669202077fe251`
of the pre-close plan.  This status/close section was added afterward and did
not change the executed algorithm, seeds, thresholds, source files, or receipt.
