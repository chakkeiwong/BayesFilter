# Paper d100 math and code audit plan (2026-08-14)

Status: `COMPLETE`

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Does the target-to-training-to-transformed-HMC code compute the stated mathematical objects, and which checked or untested mechanisms can explain the Gaussian/funnel failures? |
| Claimed target | Funnel: `log p(y,x)=-y^2/2-exp(-2y)||x||^2/2-99y+C`; Gaussian: `log p(theta)=-(theta-mu)' Precision (theta-mu)/2+C`; transformed HMC: `log p_z(z)=log p(T(z))+log|det J_T(z)|`. |
| Exact comparator | TensorFlow autodiff of the same finite target/transport program, exact sampler identities, inverse-roundtrip and Jacobian identities, and preserved HMC archives. |
| Primary audit criterion | For each frozen d100 transport, direct autodiff gradient of `log p(T(z))+logdet(T,z)` agrees with the explicit score provided to HMC within float64 tolerance. |
| Audit veto | Target score mismatch to autodiff, transport roundtrip/logdet mismatch, transformed-score mismatch, archive/hash mismatch, or diagnostic formula mismatch. |
| Explanatory diagnostics | Chainwise Gaussian projection means, batch-means MCSE sensitivity, inverse-flow scale saturation, funnel tail discrepancy, objective gradient clipping, and runtime complexity. |
| Must not be concluded | Passing parity does not establish MCMC convergence, objective superiority, tail correctness beyond diagnostics, or default readiness. |

## Mathematical audit map

1. Verify the funnel normalizing-Jacobian term `-99y`, analytic score, and exact
   sampler `x=exp(y)r` with `r~N(0,I)`.
2. Verify Gaussian row-vector precision score and Cholesky whitening against the
   frozen source constants.
3. Verify each IAF stage: `T_i(z)=z_i exp(s_i(z_<i))+t_i(z_<i>)`,
   `log|det J_T|=sum_i s_i`, sequential inverse, and reversal composition.
4. Verify the objectives: reverse KL minimizes `E_q[log q-log p]`; uniform exact
   replay forward loss minimizes `E_p[-log q]`, equal to forward KL plus the
   target entropy constant.
5. Verify transformed HMC force: `J_T' score_p(T(z)) + grad_z logdet` against
   autodiff of the finite value program for all four frozen d100 states.
6. Inspect preserved failures after parity: Gaussian projection 2, reverse
   funnel tail compression, and forward inverse-flow runtime.

## Hypotheses and discriminators

| Hypothesis | Classification before audit | Discriminator |
|---|---|---|
| Funnel density/score omits a Jacobian | Plausible but low probability | Target score/autodiff and sampler-residual checks |
| Gaussian score/whitening orientation error | Plausible | Source constants, score/autodiff, and exact-draw whitening |
| IAF inverse/logdet sign or composition error | Plausible and serious | Frozen-state inverse/logdet/score-autodiff parity |
| HMC misses `grad logdet` | Plausible and serious | Direct transformed-target force parity |
| Gaussian forward discrepancy is a 99% multiplicity artifact only | Partially falsified | 99.9% archive adjudication and iid calibration; forward remains rejected |
| Gaussian forward discrepancy is an MCSE/chain-bias issue | Plausible | Chainwise means, batch-size sensitivity, fresh starts in a future plan |
| Reverse KL tail compression is objective geometry | Plausible | Exact funnel tail statistics with sampler gates passing; contrast exact-replay forward result |
| Forward runtime is a code defect | Partially supported | Complexity inspection and profile; eliminate duplicate inverse validation before profiler |

## Skeptical audit

| Question | Finding |
|---|---|
| Are we testing the claimed HMC force, not a proxy? | Yes. The audit differentiates the exact transformed scalar value and compares it with the manual HMC score. |
| Could a passing autodiff test hide a shared wrong target? | Yes. Exact source formulas and samplers are separately checked. |
| Does a parity pass explain posterior failure? | No. It rules out an implementation mismatch but leaves finite-chain, tuning, and objective-geometry explanations. |
| Is the forward runtime change semantics-preserving? | Yes. It replaces a second inverse solve with the algebraically identical base-normal plus returned logdet formula, covered by a density equality test. |

Audit verdict: `PASS_FOR_EXECUTION`. This is a CPU diagnostic-only investigation
over frozen artifacts. It does not launch training or HMC.

## Execution record

Command executed on 2026-08-14:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/audit_neutra_paper_d100_math_code_2026_08_14.py
```

Result artifact:
`docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/math-code-audit-r1/`.

All four frozen transports passed direct TensorFlow-autodiff parity of the
transformed value and force, explicit pullback, `grad log|det J|`, and
forward/inverse log-Jacobian identities.  Maximum complete-force residuals
were `3.55e-14` to `7.11e-14`; roundtrip-coordinate residuals were
`2.66e-15` to `1.09e-14`.

The audit therefore rules out a target-score, Jacobian-sign, transport
composition, or manual-HMC-force discrepancy for the frozen runs.  It does
not prove that either chain has reached stationarity.

The reverse-funnel proposal is directly underdispersed before HMC:
`E_q[y^2]=0.68394`, `P_q(y<-2)=0.000824`, and
`P_q(y>2)=0.002319`, compared with exact values `1` and `0.022750`.
Its retained HMC output remains underdispersed, while the forward-KL proposal
and retained output both cover the funnel tails. This is consistent with the
known directionality of reverse KL, but is not proof that the objective alone
caused the finite-chain failure.

The Gaussian proposals are close to the normalized Gaussian by proposal
diagnostics, yet both HMC archives have a negative projection-2 mean in every
chain. The forward case changes from a near-zero warmup-average projection to
retained chunk means `-0.0621` and `-0.0381`. This supports finite-chain,
shared-start-bank, or transformed-geometry hypotheses rather than a known
math/code mismatch. Its bounded-scale map is materially saturated on retained
states (forward stages 1/2: `0.222/0.068` at `|s|>=0.999`; reverse stage 2:
`0.297`). These are explanatory diagnostics, not a concluded cause.

All four frozen training states selected their final checkpoint at update
`5000`. Their heldout losses still decreased over updates `4250` through
`5000`; the changes were small, but no MCSE-aware late-loss stopping analysis
was performed. Therefore the historical 5000-update cap is a baseline budget,
not evidence that the transport optimizer had converged. Undertraining remains
a hypothesis, not an established explanation of the HMC results.
