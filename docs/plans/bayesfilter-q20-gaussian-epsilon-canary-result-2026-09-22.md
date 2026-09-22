# q20 Gaussian-derived step-size canary result

All six Gaussian-derived steps are unsuitable for the current frozen map at
the tested starts. Across both starting banks, none of 192 large-step proposals
was accepted. Every pair failed numerical health checks. The small-step control
accepted 29 of 32 proposals, moved every chain in both banks, and passed the
same numerical checks. This is a rejection of these proposed steps on this map,
not a rejection of NeuTra or evidence that no smaller setting can work.

The [plan](bayesfilter-q20-gaussian-epsilon-canary-plan-2026-09-22.md) was reviewed
before execution. The completed worker ran on September 22, 11:49–11:58 Shanghai
using the preserved r2 snapshot, q20/T30 beta=1 UKF-approximate posterior,
`direct-w16-lr0.0005-r0-beta1-u512`, identity latent mass and float64. Four chains
were batched on host GPU 1 (RTX 4080 SUPER). GPU memory growth was verified
before initialization. The HMC graph used XLA with a stable signature and one
trace across all fourteen calls. No training or numerical setting was changed.

## Results and interpretation

Each table row combines four transitions per chain, four chains, and two banks:
the original prior-drawn physical starts mapped to z, and four fresh N(0,I4)
latent draws from the transport's proposal law. All initial states were valid.
Every pair reset to the same bank. The latter bank is not a posterior sample.

| L | Gaussian-derived epsilon | Accepted / proposed | Invalid proposal targets | Nonfinite log acceptance ratios |
| --- | --- | --- | --- | --- |
| 3 | 1.2595670001 | 0 / 32 | 5 / 32 | 5 / 32 |
| 5 | 1.3053652852 | 0 / 32 | 18 / 32 | 18 / 32 |
| 9 | 1.3474553766 | 0 / 32 | 31 / 32 | 31 / 32 |
| 13 | 1.2069251635 | 0 / 32 | 32 / 32 | 32 / 32 |
| 18 | 1.1997862812 | 0 / 32 | 31 / 32 | 31 / 32 |
| 25 | 1.2113847601 | 0 / 32 | 32 / 32 | 32 / 32 |

The largest individual Metropolis probability across the Gaussian-root calls
was `3.6957818e-14`, at L=3. For L>=5 all reported probabilities underflowed to
zero. Thus zero accepted moves is accompanied by strong numerical evidence of
unsuitable proposals, not merely a run of unlucky uniform acceptance draws.
All retained states and retained target status stayed valid: bad proposals
were rejected. Native divergence flags were not exposed; invalid proposal
status and nonfinite ratios are the observed numerical veto evidence.

For the control `epsilon=.062002709114199195, L=3`, the original-start bank
accepted `[2,4,4,3]` of four proposals per chain; the proposal-start bank accepted
`[4,4,4,4]`. Both had zero invalid proposal targets and zero nonfinite log ratios.
These controls establish that the diagnostic path can produce healthy movement
at the smaller step. Four transitions do not qualify that step, establish .7
acceptance or prove convergence.

The Gaussian formula claims stationary acceptance for an exact N(0,I4) latent
target. The actual calculation used the frozen map's exact transformed q20
value/score from nonstationary starts. Those targets and start laws have not
been shown equal. At the four map-generated starts, the norms of
`grad_z(log pi_T(z))+z` were 5.8564, 6.8575, 6.7515 and 5.6419; an exact
standard-normal target would give zero. At the original starts the norms were
198.7133, 8.4101, 13.2139 and 192.6367. These point checks demonstrate a mismatch
with an exact standard normal at the inspected points. They do not estimate
posterior covariance, quantify global whitening, distinguish every training
failure cause, or certify the implemented target's independent derivative
accuracy. No new independent score-parity test was performed in this canary.

### Why the ideal-Gaussian assumption was unsupported

The frozen export records 512 optimizer updates, `status=hmc_trial_nominee`,
`plateau_observed=false`, `posterior_qualified=false`, and
`map_reliability.posterior_coverage_checked=false`. The inspected
`neutra_training_protocol.assess_training_rung` nominates a trial after numerical
reliability, the minimum update count, and observed improvement from baseline;
it does not require a plateau, Gaussian scores, latent covariance identity or
posterior coverage. `q20_production_training.run_training_cohort` then skips
further updates for that eligible nominee. This establishes the nomination
mechanism, not that 512 updates are inherently insufficient or sufficient.
The map reliability check's `pullback` entry compares trainable and frozen
implementations under an arbitrary cotangent; it is not a posterior-whitening
assessment.

A follow-up using only the saved start-bank values strengthens the pointwise
diagnosis. Write `r(z)=log pi_z(z)+||z||^2/2`, using the unnormalized target.
For an exact standard normal this is constant, and its gradient is
`grad_z log pi_z(z)+z=0`. At the four map-proposal points, the saved r values
span **3.0434910107 log units**. Therefore the density ratio pi_z(z)/phi(z)
varies by **20.9784 times** across those points; unknown normalization cancels
from this comparison. Their latent norms are 1.6203, 2.3904, 2.2910 and 2.0966.
Their score-residual norms, reported above, are 5.64–6.86, compared with ideal
Gaussian score norms equal to those latent norms. These are actual-target
point checks, not a statistical estimate of a global divergence or a test of
Gaussianity on posterior draws. The values and formula are preserved in
[saved-whitening-audit.json](artifacts/q20-gaussian-epsilon-canary-2026-09-22/saved-whitening-audit.json).

At L3/epsilon1.259567, even the thirteen map-proposal trajectories whose final
target status was valid had Hamiltonian errors ranging from 59.0687 to
1.17564e10 (negative saved log acceptance ratios). Invalid status alone does
not explain the rejection. The saved summary does not retain an internal
trajectory's first invalid operation, so the precise origin of every numerical
failure cannot be identified from endpoint telemetry.

The justified verdict is that the standard-normal approximation used to select
these steps is materially wrong at the inspected points and unusable for these
trajectories. The check does not distinguish residual location/scale errors
from nonlinear curvature or tails, and it does not establish which optimizer,
capacity, coverage or score issue caused the mismatch. Covariance whitening
and distributional Gaussianity are separate questions; neither was certified
for this export. No additional target evaluation or GPU work was needed for
this follow-up.

The next justified research action is to examine current-map geometry and
training adequacy, and use measured smaller proposals for any further tuning.
The already rejected .0759375 endpoint remains rejection evidence. The canary
does not supply a .7-acceptance step or authorize importing Gaussian-root
settings into the master program.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject all six Gaussian-root initializations for this map/start scope | Zero of 192 proposals accepted; largest probability 3.70e-14 | Every pair has invalid proposals/nonfinite log ratios; retained states valid | Geometry and training failure mechanism; behavior at independent posterior points | Current-map geometry/training diagnosis and smaller measured proposals | Failure of NeuTra as a research direction or of the ideal Gaussian derivation |
| Small-step control remains a development candidate | All chains moved; 29 of 32 proposals accepted | No recorded numerical veto | Acceptance and mixing beyond four transitions | Public per-L measurement and fresh verification if funded | Tuned kernel, posterior correctness or convergence |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Numerical failure for every Gaussian-root pair in both banks |
| Statistically supported ranking | None; no ranking was attempted |
| Descriptive-only differences | Acceptance counts, per-chain probabilities, residual norms and elapsed times |
| Default-readiness | None of these canary results qualifies a new default |
| Next evidence needed | Target-specific latent geometry and qualifying measured kernels, followed by separate posterior diagnostics |

Post-run review: failure at far prior starts alone would be a weaker reason to
reject a Gaussian-sized proposal near the map's typical output. Failure also
occurred at every map-proposal bank, addressing that alternative for these
points. A qualified map/target/score repair, or healthy independently tested
posterior starts, could change the conclusion in a new scope. The weakest
evidence is coverage: eight starts and four transitions per chain cannot
characterize the posterior or identify a uniquely responsible training defect.

## Execution evidence and cost

The first setup attempt failed before HMC because the diagnostic passed raw
model status to a normalized telemetry checker. It consumed 19.510748 seconds.
The localized harness repair uses the existing public telemetry normalization;
the original script and failure remain archived. The successful retry consumed
558.155205 supervisor seconds, including worker startup and shutdown. Total
charged cost was **577.665953 seconds (9 minutes 38 seconds)**, within the
original cumulative 1,200-second cap. No budget was renewed.

Remaining campaign allowance is **155,934.801794 seconds (43.3152 hours)**;
remaining diagnostics are **1,144.952755 seconds (19.0825 minutes)**. The master
remains at its prior ensemble-affordability pause. The canary's costs are in
the campaign ledger and refreshed terminal balance records.

- [Worker manifest](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00004-gaussian-epsilon-canary/worker/manifest.json): commit, exact source hashes, environment, target/map/data identity, device and memory settings.
- [Full result](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00004-gaussian-epsilon-canary/worker/result.json): seeds, per-call findings, timings and raw tensor trace paths.
- [Starting banks](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00004-gaussian-epsilon-canary/worker/start-banks.json): latent/physical states, scores and status.
- [Budget receipt](artifacts/q20-gaussian-epsilon-canary-2026-09-22/budget-receipt.json) and [failed setup archive](artifacts/q20-gaussian-epsilon-canary-2026-09-22/setup-failure-01/budget-receipt.json).
- [Diagnostic launcher](artifacts/q20-gaussian-epsilon-canary-2026-09-22/run_canary.py); its exact command is preserved in the supervisor receipt.
