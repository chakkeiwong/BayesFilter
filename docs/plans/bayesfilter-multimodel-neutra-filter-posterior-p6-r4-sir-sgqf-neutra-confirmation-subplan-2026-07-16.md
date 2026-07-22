# P6 R4 Subplan: SIR-SGQF NeuTra Confirmation

Date: 2026-07-16

Status: `REVIEWED_READY_FOR_EXECUTION`

## Objective And Entry

Confirm or reject the fresh frozen `dim3_lr1e3` dense-IAF transport by tuning
and running HMC in its latent coordinates, then comparing the resulting
physical posterior means with the admitted same-target plain-HMC comparator.

Entry binds:

- typed target signature
  `0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`;
- comparator result SHA-256
  `621c3d6e748eed38433efaa02ff097a971132de89f323f12702533723e3ce9b2`;
- final training result SHA-256
  `c69b4e4e02b68d13be74f7a87ffc0ec9b1d6a47bc8438d56c048577a78531854`;
- frozen payload SHA-256
  `2ddff1ed2521ec674e64665bb8882a84ebc767e0850d677db72fa05a7e5ccdf4`;
- transport hash
  `dbd29efe786ec23c7b1098ba95ec6cad3a439b4889e04c67eeb2127965949c89`.

## Research Intent And Evidence Contract

| Field | Frozen contract |
| --- | --- |
| Question | Does the frozen SIR-SGQF residual dense IAF support valid HMC samples whose three physical posterior means are practically equivalent to the same-target plain-HMC comparator? |
| Baseline | admitted affine-preconditioned plain-HMC retained archive under the identical typed target |
| Candidate | HMC on the exact target pulled back through the fresh frozen dense IAF plus fixed affine map |
| Physical estimands | `kappa=0.1*exp(theta0)`, `nu=18*exp(theta1)`, observation-noise SD `10*exp(theta2)` |
| Tuning admission | disjoint 1,000 burn-in plus 1,000 retained draws; modern R-hat `<=1.01`; finite health/status; zero declared energy divergences |
| Sampler pass | recent-window warm-up modern R-hat `<=1.05`; cumulative retained max modern R-hat `<=1.01`; min bulk ESS `>=1000`; min tail ESS `>=400`; all health/status gates clear |
| Agreement pass | for all three means simultaneously, Bonferroni 95% upper bound `abs(mean_N-mean_H)+z*sqrt(MCSE_N^2+MCSE_H^2)` no greater than `0.10` comparator posterior SD |
| Promotion | sampler plus agreement pass yields `NEUTRA_CONFIRMED` for this cell at physical-mean scope only |
| Vetoes | identity/transport/comparator hash drift; invalid target status; nonfinite state/value/score/log acceptance/diagnostic; energy divergence; unmoved chain; no tuning admission; warm-up/retained cap; convergence/ESS failure; supported mean disagreement |
| Explanatory only | acceptance, short probes, runtime, source summaries, quantiles, SD/correlation differences, training and heldout loss |
| Not concluded | full-distribution, tail, covariance, or mode equivalence; SGQF exactness; epidemiological calibration; forecasting; robustness; superiority; default/production readiness |

## Fresh Tuning And Sampling

- four latent chains at zero plus fixed offsets `(0.10,-0.10,0.08)`,
  `(-0.10,0.10,-0.08)`, and `(0.16,0.08,-0.12)`;
- step grid `(0.05,0.10,0.20,0.30,0.40,0.50)`, eight leapfrog steps;
- short probes: 64 burn-in plus 128 draws, seeds `(20260716,32000+i)`;
- short probes only order candidates by lowest modern R-hat, then maximum
  minimum bulk ESS and grid order;
- tuning verifiers: 1,000 burn-in plus 1,000 draws in that frozen order, seeds
  `(20260716,32100+i)`; first modern-R-hat `<=1.01` health-valid row admits;
- warm-up root `(20260716,32201)`, chunks 1,000, minimum 2,000, recent window
  1,000, modern R-hat `<=1.05`, cap 10,000;
- retained root `(20260716,32301)`, chunks 2,000, minimum 4,000, cap 10,000;
  each checkpoint jointly tests convergence/ESS and agreement.

All seeds are disjoint from target design, comparator, screens, heldout, and
final training. Tuning, warm-up, and retained samples are archived separately;
only retained samples contribute to posterior summaries.

## Statistical Design

Mean MCSE is posterior SD divided by the square root of split-chain
cross-chain ESS. Family-wise alpha is `0.05` over three two-sided normal
intervals, with critical value `Phi^-1(1-alpha/(2*3))`. The margin is `0.10`
times the comparator SD for the same physical estimand.

At a retained checkpoint:

- all upper bounds within margins: agreement passes;
- any simultaneous lower bound beyond its margin: supported mean
  disagreement;
- otherwise: unresolved precision and sampling continues to the cap.

The interval is used only after convergence/ESS validity. Means do not
determine distributions, and both routes share the same SGQF-approximate
target, so no exact-posterior or full-distribution claim is permitted.

## Required Artifacts And Checks

1. Verify all source recursive hashes and reconstruct the exact typed identity.
2. Verify recipe, final seed, step count, training state hash, payload hash,
   artifact signature, transport hash, parity, heldout status, and no screen
   reuse before loading the transport.
3. Run a compiled GPU/XLA transported-target canary before the tuning grid.
4. Preserve short probes, each disjoint tuning verifier, tuning selection,
   warm-up chunks, retained chunks, cumulative archives, and progress ledgers.
5. Compute modern rank/folded R-hat and bulk/tail ESS in source log-coordinate
   samples; compute agreement in declared physical coordinates using TF/TFP.
6. Write result, manifest, ledger, recursive hashes, decision/inference tables,
   and post-run red-team note; run focused tests and scoped diff checks.

## Repair, Handoff, And Stops

Localized serialization, reporting, path, import-order, or XLA resource defects
may be repaired in a fresh root with unchanged target, transport, grid, seeds,
thresholds, hardware class, and arm budget. No tuning admission, warm-up cap,
retained convergence cap, or supported mean disagreement is a cell-local
sampler blocker, not general evidence against NeuTra. A cap-hit interval that
crosses a margin is `EVIDENCE_BLOCKED_AGREEMENT_PRECISION`.

On `NEUTRA_CONFIRMED`, write the P6 cell result and continue P7 synthesis. On a
terminal block, record the precise layer and continue P7. Reopen target or
training only for identity, score, transport, or artifact invalidity.

## Skeptical Audit

Decision: `PASS` after repair.

The first draft inherited P4's short-probe ESS selector, which could admit a
kernel without modern-R-hat tuning verification. This was the same defect that
invalidated P6 R2 attempt 1. The repaired plan makes short probes ordering-only
and requires a disjoint modern-R-hat `<=1.01` verifier. It binds the exact
same-target comparator, freezes physical estimands and margins before R4 data,
keeps convergence ahead of agreement, distinguishes disagreement from
insufficient precision, uses separate seeds/archives, and preserves all
scientific nonclaims.

Claude remains unavailable at the private-workspace disclosure boundary
already recorded for P5/P6. No external verdict is claimed; local skeptical
review and focused checks govern this bounded trusted-local run.
