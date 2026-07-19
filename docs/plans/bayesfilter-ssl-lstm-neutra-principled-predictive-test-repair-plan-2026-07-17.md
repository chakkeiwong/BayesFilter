# SSL-LSTM NeuTra Principled Predictive-Test Repair Plan

Date: 2026-07-17

Status: `EXECUTED_ENGINEERING_PASSED_DIRECT_CALIBRATION_PENDING`

## Research Intent Ledger

| Field | Prospective contract |
| --- | --- |
| Main question | Can the SSL-LSTM predictive comparison replace its inconsistent fixed-length uncertainty estimator and arbitrary coordinate-margin aggregation with an asymptotically valid, scientifically interpretable decision surface? |
| Exact closed baseline | Historical `chain_batch_long_run_covariance(..., block_length=16)` and componentwise Phase 8 decision machinery, as recorded in `bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-result-2026-07-17.md` and `bayesfilter-ssl-lstm-neutra-phase-8-sample-size-margin-preflight-result-2026-07-17.md`. These receipts are not mutated. |
| Candidate mechanism | Per-chain-centered growing-bandwidth Bartlett/Newey-West HAC plus exact quadratic proper-score loss extrema over a joint Wald ellipsoid. |
| Expected failure mode | Short dependent sequences can retain appreciable HAC bias; covariance can be singular; trust-region root solving can fail at zero gradients or repeated eigenvalues; a scientifically acceptable loss budget remains unavailable. |
| Promotion criterion | Engineering API passes analytic/reference/XLA tests; growing-bandwidth conditions are enforced; controlled AR(1) evidence shows repaired HAC error decreases with sample size while fixed-16 retains its theoretical bias; exact loss extrema match independent low-dimensional references; decision branches satisfy constructed false-equivalence/material-difference fixtures. |
| Promotion veto | Formula mismatch with the LaTeX chapter; nonfinite output; non-positive-definite decision covariance after the declared ridge ladder; trust-region KKT residual failure; wrong chain pooling; XLA/eager mismatch; controlled evidence contradicts the AR(1) limit; confirmation/HMC data are opened. |
| Continuation veto | Broken target/test harness, corrupted artifacts, an implementation result that cannot be independently checked, resource cap, or discovery that the proposed theorem conditions do not cover the implemented estimator. Candidate underperformance alone triggers focused repair, not abandonment. |
| Repair trigger | Any focused unit/reference failure is repaired locally and rerun. Controlled coverage/power failure leaves confirmation closed and triggers bandwidth/sample-size calibration in a new plan. |
| Explanatory diagnostics | Finite-sample HAC bias, eigenvalues, ridge choice, condition number, trust-region KKT residuals, interval widths, MMD point discrepancies, runtime. |
| Must not be concluded | G/H predictive equivalence or difference; posterior correctness; mode coverage; NeuTra or HMC superiority; model adequacy; default readiness; adequacy of any loss budget; sufficiency of 4096 or 8192 target draws. |

## Evidence Contract

- Scientific/engineering question: whether the repaired estimator and decision implement the mathematics in Chapter 28a and remove the known fixed-bandwidth inconsistency.
- Comparator: the exact historical fixed-16 estimator for the AR(1) counterexample and independent analytic/brute-force fixtures for new functions.
- Primary pass criteria: all focused tests pass; for unit-variance AR(1) with `phi=0.6`, the analytic fixed-16 limit is `3.5313822395` versus truth `4.0`, and the growing-HAC absolute error decreases across a deterministic increasing sample ladder; trust-region bounds agree with reference solutions within the declared numerical tolerance.
- Hard vetoes: malformed/static-shape/dtype errors, nonfinite tensors, inadmissible covariance, failed KKT checks, source/document equation mismatch, confirmation leakage, GPU/XLA provenance failure if a GPU result is produced.
- Explanatory only: observed finite-sample error magnitudes, speed, ridge frequency, and MMD diagnostics.
- Nonclaims: the list in the research ledger applies even if every engineering check passes.
- Result artifact: `docs/plans/bayesfilter-ssl-lstm-neutra-principled-predictive-test-repair-result-2026-07-17.md` and a reset memo at the matching path.

## Mathematical And API Scope

1. Preserve the historical fixed-batch API for receipt replay, but document it as historical and not generally consistent when block length is fixed.
2. Add `growing_hac_bandwidth(draw_count, multiplier=1.0)` implementing
   `max(1, floor(multiplier * draw_count**(1/3)))`, with static prospective inputs and the constraints `bandwidth < draw_count`, growth, and vanishing bandwidth fraction.
3. Add a new result type and `chain_bartlett_long_run_covariance`:
   per-chain centering; lag-`k` covariance denominator `N`; average across chains; Bartlett weights; symmetric result; pooled-mean covariance divided by `C*N`; fail-closed ridge ladder and precision metadata; TensorFlow `float64`; XLA on by default.
   The raw spectral and pooled-mean estimates carry the consistency claim.
   Confirmation requires zero ridge unless a separately justified vanishing
   loading is implemented; a fixed positive ridge is numerical diagnosis only.
4. Add a scientific-loss result and builder using
   `W_L=diag(lambda/2, lambda/4)` in mean-then-log-variance order.
5. Add exact ellipsoidal quadratic-loss bounds. Factor covariance as `A A^T`; transform to a Euclidean ball; solve lower and upper trust-region extrema through symmetric eigendecomposition and deterministic bisection of the secular equation; check feasibility and KKT residuals; fail closed when covariance is not positive definite.
6. Add a three-way decision: `PASS` only when the exact upper loss bound is below the frozen budget; `MATERIAL_DIFFERENCE` only when the exact lower bound is above it; otherwise `INCONCLUSIVE_UNDERPOWERED`; malformed or unauthenticated inputs return a hard veto.
7. Keep MMD as an omnibus shape diagnostic. This plan does not calibrate its scientific tolerance and does not use the historical fixed-block MMD interval for promotion.

## Checks And Controlled Validation

### Focused CPU reference and unit checks

Run with GPU deliberately hidden:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_predictive_equivalence.py \
  tests/test_predictive_equivalence_principled_repair.py
```

Required cases:

- IID and deterministic hand-calculated HAC values;
- independent-chain pooled scaling;
- AR(1) analytic long-run variance and fixed-16 limiting bias;
- bandwidth rule over an increasing draw grid;
- ridge exhaustion and malformed-policy failure;
- eager/XLA equality on the default compiled surface;
- scientific-loss constants and horizon order;
- exact loss extrema for centered, one-dimensional, diagonal, repeated-eigenvalue, interior-minimum, and boundary-minimum cases;
- decision branches and authentication/tamper failure;
- equation constants tied directly to Chapter 28a text.

### Controlled finite-sample diagnostic

Use a deterministic TensorFlow stateless AR(1) fixture with four independent
chains and fresh, non-HMC seeds. Compare fixed-16 and growing-HAC point error at
increasing powers of two. The result is an estimator diagnostic, not a sampler
or predictive-equivalence experiment. If stochastic nonmonotonicity prevents
the predeclared decreasing-error check on one realized path, replace it with a
deterministic innovation sequence or replicated RMSE under a separately
recorded repair; do not select a favorable seed.

No G/H retained archive, confirmation forecast suffix, HMC transition, NeuTra
training, candidate search, or model-file change is authorized by this plan.

### Documentation checks

```bash
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Check undefined citations/references, duplicate labels, equation dimensions,
and the equation-to-code map in the result note.

## Resource Stop

- Focused CPU/reference execution: 10 minutes total.
- Optional trusted GPU/XLA smoke: 5 minutes total and only if CPU XLA cannot
  establish the compiled surface. It is not scientific evidence.
- No HMC, forecast acquisition, training, external disclosure, package install,
  network call, or long sweep.

## Pre-Mortem

- Misleading pass: synthetic AR(1) checks one dependence law only. Mitigation:
  claim estimator mechanics and known-limit repair, not target adequacy.
- Implementation failure mistaken for scientific failure: trust-region hard
  cases can defeat naive root solving. Mitigation: independent low-dimensional
  references and explicit KKT checks.
- Tuning failure mistaken for theorem failure: one bandwidth constant can be
  poor at finite `N`. Mitigation: the theorem claim concerns the growing rule;
  finite-sample multiplier calibration remains a separate controlled task.
- Hidden proxy promotion: decreasing HAC error does not validate the complete
  predictive decision. Mitigation: confirmation remains closed and a later
  direct coverage/power study is required.

## Skeptical Pre-Execution Audit

| Audit question | Finding |
| --- | --- |
| Wrong baseline? | No. The closed fixed-16 implementation is the defect being tested; no weak ordinary-HMC comparator is introduced. |
| Proxy promoted? | No. AR(1), unit tests, and KKT checks establish mechanics only. They cannot promote G/H or a scientific loss budget. |
| Missing stop condition? | No. Time, scope, continuation vetoes, and confirmation/HMC prohibitions are explicit. |
| Unfair comparison? | No sampler comparison occurs. The estimator comparison uses the same controlled sequence and an analytic truth. |
| Hidden assumption? | Stationarity, independent chains, CLT/mixing, finite moments, summable autocovariances, static shapes, and prospective bandwidth are explicit in the chapter and plan. |
| Stale context? | The plan binds the July 17 closed Phase 8 receipts and supersedes only their inferential method prospectively. |
| Environment mismatch? | TensorFlow/TFP `float64`, CPU-hidden reference, XLA-default code, and optional trusted GPU boundary match repository policy. |
| Do artifacts answer the question? | Yes for implementation and asymptotic-method consistency; no target or scientific conclusion is attempted. |

Audit decision: `PASS_FOR_BOUNDED_EXECUTION`. The plan addresses the actual
fixed-bandwidth defect, preserves the blinded confirmation boundary, and does
not confuse controlled diagnostics with a scientific promotion criterion.

## Execution Order And Handoff

1. Implement the additive API and tests.
2. Run focused CPU/reference checks; repair local failures and rerun.
3. Run the bounded controlled AR(1) diagnostic inside the focused tests.
4. Build the LaTeX document and inspect warnings.
5. Write the result and reset memo with decision/inference-status tables.
6. Audit chapter equations against exact code symbols and final line anchors.

The next phase may design a direct finite-sample coverage/power experiment only
if this plan passes. It must choose the scientific loss budget `K`, horizon
weights, HAC multiplier, sample ladder, simultaneous coverage target, false
decision bound, and fresh seeds prospectively. HMC acquisition and G/H
confirmation remain separately closed until that design passes.
