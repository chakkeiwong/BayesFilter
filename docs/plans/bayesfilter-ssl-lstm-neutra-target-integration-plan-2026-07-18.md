# SSL-LSTM NeuTra Target Integration Plan

Date: 2026-07-18

Status: `PROPOSED_TARGET_ADAPTER_WORK_GH_CONFIRMATION_CLOSED`

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
`conditional_mean_log_variance_influence` without reordering.  Horizon scales
are estimated only from an independent calibration-only innovation/draw bank,
then frozen and hashed before any confirmation-shaped evaluation.  The fixed
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

Generate a calibration-only bank from fresh seeds and a declared target-shaped
fixture.  The bank must be independent of any future confirmation bank and
must record seed, draw/replication counts, forecast configuration signature,
innovation hashes, worker count, and output hash.  Estimate one scale per
horizon/feature using the existing scale API or a documented equivalent, then
freeze it before evaluating confirmation-shaped data.

Required gates: no overlap between calibration and evaluation seeds; finite,
positive scales; reproducible replay; no scale derived from a G/H decision or
from the confirmation bank.

Handoff: immutable calibration metadata and frozen scale receipt exist.

### Phase 4: Target-shaped path versus Rao validation

Use independent target-shaped fixtures with the same ten-horizon dimensions,
parameter chart, terminal covariance conventions, and innovation roles as the
frozen forecast API.  Compare path and conditional-moment estimates with
shared innovations for a paired precision diagnostic and independent
innovations for a robustness diagnostic.  Predeclare Monte Carlo tolerances
from the calibration-bank effective sample size; do not rank estimators from a
single descriptive run.

Required gates: exact conditional-variance algebra on a hand-computable
fixture; paired and independent estimates finite; expected path/Rao difference
shrinks with bank size; no process uncertainty loss; feature projection and
scale invariance tests pass.

Handoff: target adapter result note states whether the candidate is viable,
what failed if not, and keeps G/H confirmation closed in either case.

### Phase 5: Covariance and resource preflight

Construct the locked split-region covariance from the target-shaped influence
clusters using Bartlett HAC multiplier `3.0` and zero ridge.  Check symmetry,
positive definiteness, eigenvalue floor, condition number, KKT residuals,
alpha allocation, and exact bound authentication.  Measure the declared
resource envelope without starting confirmation; stop before the cap.

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

## Close Record Requirements

At the end of each executed phase, record the exact command, environment,
seeds, wall time, artifact paths/hashes, hard-veto status, descriptive
diagnostics, decision, next handoff, and nonclaims in a result note.  Do not
rewrite immutable calibration or confirmation receipts after evaluation.
