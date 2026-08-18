# Paper d100 NeuTra math/code audit result (2026-08-14)

Plan: `docs/plans/bayesfilter-weighted-forward-kl-paper-d100-math-code-audit-plan-2026-08-14.md`

This is a CPU-only diagnostic. It does not rerun training or HMC.
The transformed force is checked against autodiff of the exact finite value program.

## Parity results

| Case | value | transformed force | roundtrip | decision |
|---|---:|---:|---:|---|
| gaussian-reverse-r1 | 0.000e+00 | 6.969e-14 | 3.220e-15 | PASS |
| gaussian-forward-r1 | 0.000e+00 | 4.444e-14 | 2.665e-15 | PASS |
| funnel-reverse-r1 | 0.000e+00 | 3.553e-14 | 6.051e-15 | PASS |
| funnel-forward-r1 | 0.000e+00 | 7.105e-14 | 1.088e-14 | PASS |

## Interpretation

All parity checks passing rules out an omitted target-score term, a missing `grad log|J|` term, a pullback orientation error, and an inverse/logdet sign error for these frozen states. It does not establish HMC convergence or transport quality.

The remaining failure hypotheses are therefore empirical rather than an established code mismatch:

1. Reverse KL is mode-seeking in the relevant geometry. Its objective `E_q[log q-log p]` penalizes placing q mass in low-density tails, while it does not directly penalize missing p-tail mass; this predicts the observed funnel tail compression.
2. Forward KL uses exact replay and directly penalizes low q density at replayed target-tail rows. Its funnel pass is consistent with that mechanism, while the Gaussian projection-2 drift remains a direction-specific finite-training or finite-chain issue, not evidence of a wrong force.
3. The d100 config bounds every conditional log-scale to `[-1,1]` per stage (`s_max=1`, three stages). This is a capacity/conditioning hypothesis. The scale saturation table is descriptive and is not a promotion criterion.
4. The Gaussian projection failure could still be a common initialization or finite-chain effect. The four chains share the same small initial-state construction in the HMC runner, so chainwise replication with independently dispersed starts is needed before attributing it to the learned transport.

These hypotheses do not rank objectives and do not establish default readiness.

## Archived-chain geometry

| Case | retained scale contact at 99.9% cap by stage | targeted retained diagnostic |
|---|---|---|
| gaussian-reverse-r1 | 0.000, 0.000, 0.297 | projection-2=-0.07216; chain means=[-0.1000711764460011, -0.04195960042620413, -0.10287979725462844, -0.0437439942374124] |
| gaussian-forward-r1 | 0.000, 0.222, 0.068 | projection-2=-0.05009; chain means=[-0.058699929740403146, -0.03427653059491883, -0.04225946910324272, -0.06511613840364931] |
| funnel-reverse-r1 | 0.000, 0.000, 0.000 | E[y^2]=0.83925; P(y<-2)=0.01050; P(y>2)=0.01350 |
| funnel-forward-r1 | 0.000, 0.000, 0.000 | E[y^2]=0.96044; P(y<-2)=0.02275; P(y>2)=0.02325 |

The scale entries are fractions in stages 0, 1, and 2. They show whether the tanh-bounded conditional scales are active on the actual retained HMC states. They are explanatory diagnostics only: a scale cap can make the transport geometry poor, but it cannot bias the invariant transformed target once the exact Metropolis correction and audited force are used.

For the Gaussian, batch-means MCSE is reported for several block counts in `result.json`; sensitivity to block size is evidence about uncertainty estimation, not a proof of stationarity.

## Learned proposal

| Case | proposal diagnostic | log target/proposal SD | importance ESS fraction |
|---|---|---:|---:|
| gaussian-reverse-r1 | whitened means=[0.016039883413867476, -0.0040225523260314765, -0.0062763335946715995, 0.006085337631058691]; seconds=[1.007168915090965, 1.0036179754681265, 0.9843434874330996, 0.9914194005699035] | 0.315 | 8.838e-01 |
| gaussian-forward-r1 | whitened means=[0.009820650690638337, -0.004638835318551065, -0.0036051052067809617, 0.0026165742441890227]; seconds=[1.007542406070068, 1.013167908337642, 1.0041355038801951, 1.0022366076396534] | 0.340 | 8.896e-01 |
| funnel-reverse-r1 | E_q[y^2]=0.68394; tails=0.00082/0.00232 | 0.331 | 4.151e-01 |
| funnel-forward-r1 | E_q[y^2]=1.02613; tails=0.02747/0.02533 | 0.900 | 8.206e-01 |

These are proposal-quality diagnostics from iid `z~N(0,I)` draws, not posterior samples. The target/proposal ratio is unnormalized, but its dispersion and normalized-weight ESS are invariant to the missing target normalizing constant. A poor value explains difficult transformed geometry; it does not change HMC's exact invariant density.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Non-claim |
|---|---|---|---|---|---|
| Frozen transformed-density implementation | direct value/force, Jacobian, and roundtrip parity | see case table | numerical tolerance and finite probe set | retain code; investigate learned geometry and chain-start effects | no posterior correctness claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | No parity veto if all cases pass |
| Statistically supported ranking | None |
| Descriptive-only differences | scale saturation and prior HMC diagnostics |
| Default-readiness | Not assessed |
| Next evidence needed | independent HMC starts and transport-capacity/scale ablation under a new reviewed plan |
