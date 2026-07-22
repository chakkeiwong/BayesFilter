# Zhao-Cui Blockwise TTSIRT-APF Rung-1 Plan

Date: 2026-07-22

Status: `READY_FOR_BOUNDED_DIAGNOSTIC`

## Research Intent

| Field | Contract |
| --- | --- |
| Main question | Can an actually fitted squared-TT and its conditional KR map supply a valid, non-collapsed proposal to the certified fixed-branch APF when replicated across 24 independent state blocks? |
| Mechanism | Fit scalar initial and bivariate adjacent Gaussian targets in algebraic reference coordinates; compile `(x_previous,x_current)` prefix-conditioned TTSIRT blocks; combine 24 blocks under one ancestor genealogy. |
| Expected failure mode | Poor density fit or numerical grid inverse compounds across 24 blocks, causing weight collapse despite correct APF plumbing. |
| Promotion criterion | Finite fitted transports and branch; paired-core conditional `log q`; same-scalar analytical score/FD max error <= 0.03; minimum ESS fraction >= 0.5 at `d=24,T=3,N=256`; no support or measure mismatch. |
| Promotion veto | Bounded physical support, zero defensive mass, non-finite fit/map/value/score, wrong ancestor law, tensor-product suffix-grid conditional density, score/FD failure, or minimum ESS fraction below 0.5. |
| Continuation veto | The fitted TTSIRT cannot define a finite full-support conditional proposal even on the scalar Gaussian block. A failed 24D candidate alone is a repair trigger, not a direction veto. |
| Repair trigger | Candidate failure triggers a fresh degree/rank/scale/defensive-mass tuning scope or a larger structural block, within the attempt budget. |
| Explanatory diagnostics | Calibration and holdout sqrt-density residuals, ranks, conditional log-density error, ESS, log-weight spread, value/score difference from Kalman, compile and warmed time. |
| Must not be concluded | No source-faithful variable ordering or block factorization, Austria SIR, NAWM, nonlinear validity, HMC convergence, default readiness, or superiority. |

## Evidence Contract

The exact baseline is the fully adapted diagonal-Gaussian oracle from
`docs/benchmarks/artifacts/zhao_cui_frozen_proposal_apf_rung0_20260722/gpu_attempt01/result.json`.
The primary criterion is downstream APF ESS plus same-scalar analytical score
identity. Fit residuals and Kalman differences are explanatory; they cannot
replace the downstream criterion. The artifact root is
`docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/`.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Independent scalar blocks | synthetic diagonal LGSSM oracle | Isolates fitted TT/KR mechanics at `d=24` | hides cross-coordinate rank needs | explicit nonclaim and later coupled fixture | diagnostic hypothesis |
| `(previous,current)` order | local prefix-conditioned TTSIRT API | Makes conditional suffix generation exact via Proposition-2 prefix marginal | differs from Zhao-Cui paper order | axis-order manifest and conditional-density tieout | `fixed_hmc_adaptation` |
| Algebraic coordinate map | author `AlgebraicMapping` formula and local P85/P86 checks | Full physical support with finite reference domain | tail distortion or boundary instability | roundtrip/Jacobian tests and finite inverse | reviewed formula, scale to tune |
| Legendre degree/rank | local clean-room fitter | Small bounded candidate grid | underfit or ill-conditioning | disjoint holdout residual and condition veto | hypotheses, not defaults |
| Positive defensive mass | Zhao-Cui Eq. (13) | Enforces support | excessive defensive mass degrades ESS | candidate grid and downstream ESS | hypothesis |
| Grid CDF/bisection | local P83 diagnostic transport | Only implemented fixed TTSIRT inverse locally | numerical density/inverse mismatch | conditional log-density and roundtrip checks | diagnostic-only, nonproduction |
| `d=24,T=3,N=256` | NAWM observable/shock count plus bounded mechanics budget | High-dimensional composition test without NAWM claim | too easy/short for general filtering | explicit scope and later ladder | convenience diagnostic |

## Skeptical Audit

The plan does not use a weak bootstrap comparator: it compares to the exact
fully adapted Gaussian proposal. Fit loss is not promoted to the downstream
criterion. The algebraic map prevents the bounded-support error. The compiler
uses paired-core prefix marginals, not the historical suffix tensor grid. The
24D product target is intentionally easy and cannot establish nonlinear or
cross-coordinate scalability. The command and artifact answer only whether
fitted scalar TTSIRT mechanics survive 24-fold composition.

The audit passes bounded diagnostic execution.

## Tuning And Holdout

Calibration candidates are the Cartesian product:

- degree in `{6, 10}`;
- adjacent rank in `{2, 4}`;
- algebraic scale in `{1.5, 2.5}`;
- defensive mass in `{1e-6, 1e-4}`.

Use deterministic Gauss-Legendre calibration points and a disjoint shifted
midpoint holdout. Select by finite status then minimum heldout relative
sqrt-density RMS; ties within `1e-6` prefer lower degree, rank, and defensive
mass. Freeze the selected controls before a fresh stateless proposal branch
with seed `220723`.

These settings are scoped only to the synthetic diagonal model, `T=3`, scalar
blocks replicated to `d=24`, float64 offline fit, float32 GPU/XLA online
evaluation, and `N=256`. They are not transferable defaults.

## Budget And Stop Conditions

- At most 16 calibration candidates and 2 execution attempts.
- CPU fitting budget: 5 minutes total.
- Trusted GPU/XLA claim-branch budget: 2 minutes.
- Fresh versioned directory per attempt; preserve failures.
- Stop on non-finite target/fit/map, condition-number veto, missing defensive
  mass, same-scalar score failure, or exhausted budget.
- If ESS fails but the scalar block is valid, record candidate rejection and
  design the next rank/scale/block repair; do not reject the research direction.

## Planned Command

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1 \
python docs/benchmarks/run_zhao_cui_blockwise_ttsirt_apf_rung1.py \
  --output-root docs/benchmarks/artifacts/zhao_cui_blockwise_ttsirt_apf_rung1_20260722/gpu_attempt01 \
  --dimension 24 --time-steps 3 --particle-count 256 --seed 220723
```

