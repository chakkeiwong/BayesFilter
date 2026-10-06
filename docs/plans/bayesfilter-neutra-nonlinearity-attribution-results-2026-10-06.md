# Scalar nonlinearity attribution: completed results and recovery note

The reviewed executable campaign finished on 2026-10-06 at 04:38 Asia/Shanghai.
All 87 workers completed: one pricing worker, two mathematical controls, eight
pilots, 32 confirmation pairs, eight learning-rate pairs, 24 capacity controls,
four longer-training controls and eight transfer pairs. The supervisor exited
successfully; no campaign worker remains active.

**Scalar nonlinearity contributes to better density fitting under the tested
training procedure.** This result survives both tested conditioners, and the
additional controls did not eliminate the observed difference. It does not
establish that missing scalar nonlinearity is the sole cause of the earlier
canonical IAF failures, or that every multivariate affine autoregressive flow
must fail. Two nonlinear confirmation fits still failed the distribution screen.

The experiment uses exact target samples and FP64 TensorFlow GPU/XLA training.
Its primary outcome is density fitting, not posterior estimation or HMC
performance. The common-network affine readout is a diagnostic intervention;
it is distinct from canonical IAF, which has separate capacity controls.

The [plan](bayesfilter-neutra-nonlinearity-attribution-plan-2026-10-05.md) defines
the interventions, derivations, source anchors, budgets and predeclared analysis.
The complete worker tables are in the
[generated report](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/attribution-results.md).
The [terminal audit](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/attribution-terminal-audit-r1.json)
records the final integrity and accounting checks.

## Primary comparison

For each fitted pair, the computed quantity is

\[
\widehat\Delta=\frac1n\sum_{i=1}^n
  [\log q_N(X_i)-\log q_A(X_i)],\qquad X_i\overset{\mathrm{iid}}\sim p.
\]

Its expectation conditional on the two fitted maps is exactly
\(D_{\mathrm{KL}}(p\Vert q_A)-D_{\mathrm{KL}}(p\Vert q_N)\).
The target is the normalized synthetic distribution; the estimate is a Monte
Carlo approximation to this density contrast. Each pair uses 131,072 independent
reference rows. Training seeds, rather than these rows, are the replication
units for the predeclared sign test. All eight paired lower conditional 99%
normal-approximation limits exceed zero in each of the four comparisons.

| Target / conditioner | Positive pairs | Mean KL benefit, nats (descriptive) | Affine / nonlinear screen passes | Bonferroni-adjusted sign p |
|---|---:|---:|---:|---:|
| Two modes / cMADE | 8/8 | 0.0232605 | 6/8 / 8/8 | 0.015625 |
| Two modes / diagnostic Hoffman | 8/8 | 0.0145129 | 8/8 / 8/8 | 0.015625 |
| Three modes / cMADE | 8/8 | 0.3316874 | 1/8 / 7/8 | 0.015625 |
| Three modes / diagnostic Hoffman | 8/8 | 0.0952953 | 0/8 / 7/8 | 0.015625 |

Thus the declared directional density effect is supported in all four settings.
The mean magnitudes and conditioner interactions remain descriptive; this is
not a statistical ranking of overall map quality or sampling performance.
The reference intervals are conditional normal approximations, not rigorous
bounds on population KL. The small number of fitting seeds and two development
geometries limit the claim.

Both nonlinear screen rejections occur on the three-mode target with fitting
seed 449. Their maximum standardized feature discrepancies are 5.38048 for
cMADE and 6.38703 for diagnostic Hoffman, exceeding the fixed threshold of 5.
These are candidate rejections despite positive density contrasts. The maps
remain rejected; no screen was relaxed. The probes and training remained
finite, so these results do not invalidate the common numerical harness.

## Controls and generalization

On the one-dimensional mixture with centers -3 and +3 and unit component
variance, the plan proves that every Gaussian/affine-flow density has forward
KL at least 0.4581454 nats. The fitted nonlinear density has estimated KL
0.0009864 with conditional 99% Monte Carlo interval [0.0006525, 0.0013204].
The affine fit has estimated KL 0.459492. Both readouts pass the Gaussian
positive control with estimated KL approximately 0.00017. The expressivity
proof is restricted to the stated one-dimensional Gaussian-base setting.

The two final learning rates were tested for both matched conditioners and
canonical IAF controls. Validation selected 0.0003 for all matched affine and
nonlinear families. Canonical IAF used widths 64 and 134, plus a width-134
uncapped-scale variant. Of these 24 fits, one uncapped two-mode fit passed the
distribution screen; the others did not. The wider or uncapped configurations
did not consistently repair fitting within this grid. Two pilot seeds and two
rates cannot rule out other successful IAF configurations or optimizers.

Longer affine training reached the nonlinear arm's recorded training time in
all four controls, with 45,056 to 53,248 lifetime updates and time ratios from
1.005 to 1.036. The remaining nonlinear density benefits were 0.00858 and
0.02553 nats for two modes, and 0.17346 and 0.06632 nats for three modes.
Two of the four extended affine fits passed the screen. These are descriptive
controls using pilot seeds and the original paired reference banks. One of
their original nonlinear pilot fits had failed the screen; longer affine
training does not change that rejection.

The initial validation density contrast in the confirmation pairs ranged from
-0.0013164 to -0.0000412 nats: the nonlinear arm did not begin with better
validation cross entropy. Parameters, parameter counts and data were matched
within each pair. Equal initial arrays nevertheless do not define equal initial
functions, and optimization conditioning remains a possible part of the effect.

All eight nonlinear fits on four previously unused target geometries passed
the distribution screen, compared with one of eight matched affine fits. Every
paired density difference was positive. The four geometries contain two or
three random mixture components, each with two fitting seeds. This is a
successful bounded transfer check, not a population-wide success-rate estimate
or a new confirmatory method ranking.

## Terminal audit, decisions and recovery

The terminal review rechecked 1,690 recorded artifact hashes and both frozen
source snapshots, all 146 saved endpoint probes with 1,000 finite valid rows,
checkpoint reload records, GPU memory-growth provenance, batched XLA training,
all four comparison summaries and the full job matrix. No integrity or
execution-provenance error was found. All 14 pending test-charge files were
already settled. The first 11 worker manifests retain the parent campaign plan
pointer; their preserved source snapshots include the attribution plan, and
the later manifests point directly to it. They were not rewritten.

Attribution workers consumed 12.1507 GPU-process hours and 14.1694 CPU-core
hours; these are worker costs, not all earlier campaign work or test charges.
The settled campaign ledger records 10.607 GPU-process and 9.241 CPU-core hours
remaining. Per-worker manifests preserve commands, environment, source commit
and dirty-worktree snapshots, target signatures, seeds, hardware and wall time.
The run environment was the tfgpu interpreter, TensorFlow 2.20.0, GPU index 2,
with memory growth and XLA. TF32 readiness does not follow from these FP64 runs.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept the finite-recipe density effect | Four predeclared sign tests pass | Failed maps remain rejected | Optimization and effective functional capacity | Preserve this attribution result; investigate remaining fit failures before promotion | Scalar curvature is the sole cause of canonical IAF failure |
| Retain viable nonlinear candidates | 30/32 confirmation and 8/8 transfer fits pass the distribution screen | Two nonlinear confirmation maps rejected | Seed and target dependence; score geometry | Diagnose seed-449 discrepancies and validate any intended downstream use separately | Correct posterior sampling or universally reliable training |
| Complete this executable campaign | All declared workers and controls complete | No integrity or numerical invalidity found | Finite optimizer and geometry coverage | Use this note to recover; no repeat attribution launch is needed | Exhaustive architecture or optimizer search |

| Inference status | Result |
|---|---|
| Hard veto screen | Preserve all failed distribution screens, including two nonlinear confirmation fits; numerical/provenance checks pass |
| Statistically supported ranking | Directional density benefit for the nonlinear readout under each declared target/conditioner recipe |
| Descriptive-only differences | Effect magnitudes, conditioner interactions, capacity/rate controls, equal-time controls and transfer counts |
| Default readiness | Not established; canonical IAF remains unchanged |
| Next evidence needed | Target-specific fit reliability, realistic-teacher validation and independent downstream HMC assessment for broader claims |

Post-run skeptical review: the strongest alternative explanation is that the
nonlinear parameterization is easier to optimize in this finite training regime,
in addition to having more scalar shape freedom. A broader calibrated affine
family closing the gap would overturn any necessity claim for the multivariate
targets, while leaving the restricted one-dimensional theorem intact. The
weakest evidence is generalization from four geometries with exact teachers.
The result therefore supports a useful scalar-nonlinearity mechanism under the
tested procedure, not a universal NAF-versus-IAF verdict.
