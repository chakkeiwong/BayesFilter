# HNN-NeuTra Exact-Gradient Comparison Terminal Result

Superseded for tuning and performance claims, 2026-07-18.  The experiments
below used an ad hoc fixed grid that did not perform native BayesFilter dual
averaging, selected partly by short-chain R-hat, and did not bind the native
mass artifact.  Their tuned runtime, seconds/ESS, speed-ratio, break-even, and
performance-pass statements are `UNSUPPORTED_PENDING_NATIVE_RETUNING`.
Native PP-UKF retuning subsequently reached the owner acceptance band for both
arms but failed modern R-hat, with an additional exact-arm energy-tail veto.
See `bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.

Decision: `PROGRAM_COMPLETE_FOUR_FULL_COMPARISONS_ONE_PARTIAL_STRUCTURAL_CELL`.

## Answer

The corrected experiment now compares exact-gradient NeuTra-HMC with
HNN-NeuTra-HMC on the same frozen chart. LGSSM-KF, PP-UKF, PP-SGQF, and
SIR-SGQF have complete one-seed validity evidence for both arms and descriptive
performance-screen passes. STR-UKF has a valid HNN result and a healthy matched
mechanics speed comparison, but its exact-gradient retained comparator remains
unavailable after two energy-health failures.

| Cell | Same-chart HNN validity | Exact comparator | Direct posterior comparison | Descriptive performance |
| --- | --- | --- | --- | --- |
| LGSSM-KF | pass | pass | complete under historical P3 contract | pass |
| PP-UKF | pass | pass | pass, max `z_MC=1.7864` | pass, `25.435x` matched ratio |
| PP-SGQF | pass | pass | pass, max `z_MC=0.6975` | pass, `25.895x` matched ratio |
| SIR-SGQF | pass | pass | pass, max `z_MC=0.1281` | pass, `25.564x` matched ratio |
| STR-UKF | pass | failed before retention | unresolved | mechanics only, `14.031x` matched ratio |

This is four complete posterior configurations across three model families and
one partial structural configuration. Filters are counted as separate
posterior configurations, not separate model families. No statistically
supported ranking is claimed from one seed.

## Corrected Nonlinear Results

| Cell | Tuned HNN eps/L | Tuned exact eps/L | HNN/exact sampling seconds | HNN/exact seconds per min bulk ESS | Min HNN/exact truth tail |
| --- | --- | --- | ---: | ---: | ---: |
| PP-UKF | `0.4/10` | `0.2/10` | 93.930 / 2362.997 | 0.02051 / 0.33019 | 0.2132 / 0.2017 |
| PP-SGQF | `0.2/10` | `0.2/10` | 46.478 / 1136.869 | 0.006657 / 0.16320 | 0.1927 / 0.1937 |
| SIR-SGQF | `0.8/10` | `0.8/10` | 40.883 / 1034.786 | 0.008033 / 0.20384 | 0.3737 / 0.3742 |

Every complete nonlinear comparison passed exact endpoint parity, finite and
reconstructed full energy, invocation counts, adaptive warm-up, retained
rank-normalized split/folded R-hat at most `1.01`, bulk ESS at least `1000`,
tail ESS at least `400`, generating-truth tails, physical 95% interval overlap,
and pooled-MCSE direct agreement. No nonlinear second seed was required.

STR-UKF HNN used `epsilon=0.2`, `L=12`, 2,000 warm-up and 4,000 retained draws
per chain. It passed with minimum truth tail `0.2885` while preserving the exact
deterministic structural recursion and forbidding artificial state noise. The
exact arm failed warm-up energy health at maximum `|delta H|=4213.19` for
`epsilon=0.2`, `L=8`, and `1307.24` for the localized `epsilon=0.1`, `L=8`
repair. Direct STR HNN/exact agreement is therefore `unsupported`, not failed.

## Cost Interpretation

The matched nonlinear benchmarks fixed chart, initial states, seed, endpoint,
dtype, GPU/XLA route, transition count, step size, and leapfrog count. Three
synchronized warm repeats alternated arm order. The `14.0x` to `25.9x` ratios
are descriptive mechanism-cost observations, not uncertainty-supported
rankings.

HNN preparation means supervision generation plus the target-specific training
grid. Its matched-transition break-even was:

| Cell | HNN preparation | Preparation break-even batches | Full reuse-campaign break-even |
| --- | ---: | ---: | ---: |
| PP-UKF | 152.003 s | 200 | immediate |
| PP-SGQF | 54.901 s | 152 | immediate |
| SIR-SGQF | 110.505 s | 333 | immediate |
| STR-UKF | 251.465 s | 249 | exact posterior comparison unresolved |

“Immediate” applies to the full independently tuned reuse campaign because the
measured exact-gradient tuning grid already cost more than HNN preparation plus
HNN tuning. It does not mean HNN preparation is free. Common NeuTra chart
training was not reconstructed, so from-scratch totals remain unsupported.
The per-result reuse ledgers are scenario comparisons. They do not count both
failed STR exact warm-up attempts as if those failures were part of a normal
reusable workflow; actual campaign spend is recorded separately below.

The historical LGSSM-KF result remains a valid complete same-chart comparator:
learned sampling took 227.164 seconds versus 1792.325 seconds for exact
gradient, and reuse seconds per minimum bulk ESS were 0.2483 versus 0.6193.
Its zero-residual arm was descriptively better than learned residual, so the
combined program still does not prove that residual learning improves an
already strong NeuTra chart.

## Decision Table

| Decision | Status |
| --- | --- |
| Exact comparison repaired | yes for PP-UKF, PP-SGQF, and SIR-SGQF |
| Primary accuracy criterion | passed in four complete configurations |
| STR HNN validity | passed |
| STR direct exact comparison | unresolved after repeated exact energy failure |
| Matched mechanism speed screen | HNN descriptively faster in all measured cells |
| Hard veto status | clear except STR exact arm |
| Statistically supported ranking | none |
| Default readiness | not established |
| Next justified action | stabilize STR exact NeuTra-HMC only if that missing comparison is worth a new campaign |
| Not concluded | universal superiority, calibration, latent-model exactness, from-scratch speed, or production default readiness |

## Post-Run Red Team

The strongest alternative explanation is that the admitted NeuTra charts have
already made these posteriors easy and the learned force mainly removes the
cost of differentiating filters, rather than discovering important residual
geometry. LGSSM's zero-residual result supports that explanation. A different
exact-gradient implementation or more robust structural HMC tuning could also
close STR without changing HNN; the current campaign does not distinguish
those possibilities.

The weakest evidence is STR exact-HMC comparability. A short 500-transition
tuning probe understated rare energy tails at both selected repair candidates.
Future structural tuning should validate shortlisted candidates on a longer
energy-tail probe before promotion. That is a new experiment question and is
not silently added to this completed campaign.

## Artifacts

- Plan: `docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-repair-plan-2026-07-18.md`
- Phase 3 result: `docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-phase3-predator-prey-result-2026-07-18.md`
- Phase 4 result: `docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-phase4-sir-structural-result-2026-07-18.md`
- Artifact root: `docs/plans/artifacts/hnn-neutra-exact-gradient-comparison-repair-20260718/`
- Historical LGSSM result: `docs/plans/bayesfilter-hnn-surrogate-hmc-p3-lgssm-pilot-result-2026-07-17.md`

All nine top-level repair-attempt hash ledgers replay. Serious runs recorded
commit `15170e1573d19b235d96f3ed3525fa2071f58320`, TensorFlow 2.19.1, TFP 0.25.0,
float64, GPU/XLA/TF32, memory growth, commands, seeds, wall time, plan, result,
and managed-session trust basis.

The five serious launches used 32,494.147 seconds, or 9.026 GPU-hours, within
the 24-hour campaign ceiling. STR attempts used 16,538.832 seconds, or 4.594
GPU-hours, within the six-hour STR ceiling.

Terminal engineering verification passed: 39 focused tests, Python syntax,
`git diff --check`, 135 JSON parses, and all nine top-level hash replays. Claude
health probing returned `CLAUDE_PROBE_OK`, but the bounded one-path terminal
review exited successfully with no output. The same reviewer-response
limitation occurred during plan review. It is recorded as unavailable advisory
review, not converted into an agreement verdict or a scientific blocker.
