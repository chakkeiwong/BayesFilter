# SSL-LSTM NeuTra Phase 8 Sample-Size And Margin Preflight Result

Date: 2026-07-17

Decision: `FEASIBILITY_ENVELOPE_FOUND_MARGIN_AND_DIRECT_VALIDATION_REQUIRED`

## Outcome

The earlier concern was correct: neither the 448-draw nor the 1984-draw design
was supported by a prospective sample-size calculation. The new analytical and
synthetic preflight finds no feasible scenario at 1984 draws. It identifies
candidate feasibility envelopes at larger draw counts, but it does not select
a scientific margin, MMD tolerance, or acquisition target.

| Labeled scenario | First simulated feasible rung | First feasible MMD tolerance | Estimated HMC + forecast cost | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Historical original: `0.15/log(1.15)`, symmetric | 8192 confirmation draws/chain | `0.005` | `2.8395` GPU-hours | Feasibility signal only; mean material clearance is small and conservative analytical requirements exceed the grid |
| Historical repair: `0.10/0.5*log(1.25)`, IUT/TOST | 8192 | `0.005` | `2.8395` GPU-hours | Feasibility signal only; persistent true-equivalent mean remains limiting through 6144 |
| Arithmetic anchor midpoint: `0.125/0.5*(log(1.05)+log(1.25))`, IUT/TOST | 4096 | `0.01` | `1.4116` GPU-hours | Computationally attractive sensitivity, but not a scientifically justified margin |

Each cost includes segment-rounded additional HMC for both G/H and forecast
generation for both charts. It is a planning estimate, not authorization.

## What The Preflight Changed

The old “midpoint” label was arithmetically wrong for the declared anchors.
The actual mean midpoint is `0.125`; the actual log-variance midpoint is about
`0.13597`, or variance ratio about `1.1457`. This correction matters because it
substantially changes the precision required to separate the true-equivalent
and material stress cases. It still does not tell us whether those margins are
scientifically meaningful.

The material preflight used four fresh 8192-draw synthetic pilot replications
per family and 20,000 fresh joint-Gaussian Monte Carlo draws per family/rung/
scenario. It simulated the full correlated 20-feature decision and combined
feature/MMD probabilities using dependence-robust Frechet bounds. All 13
families were evaluated; covariance condition numbers were at most `60.31`,
and every pilot covariance used zero ridge.

## Analytical Cross-Check

Draws below are per chain and rounded upward to a multiple of 16.

| Requirement | 80% single limiting coordinate | 90% single limiting coordinate | 80% conservative 20-coordinate bound | 90% conservative 20-coordinate bound |
| --- | ---: | ---: | ---: | ---: |
| Repair mean: true `0.05`, margin `0.10` | 4144 | 5600 | 9904 | 11104 |
| Repair variance: true `1.05`, margin `0.11157` | 2816 | 3792 | 6704 | 7520 |
| Repair material mean: `0.20` beyond `0.10` | 2240 | 2768 | 4208 | 4592 |
| Repair material variance: `1.25` beyond `0.11157` | 1968 | 2416 | 3680 | 4016 |
| Original material mean: `0.20` beyond `0.15` | 8944 | 11024 | 16784 | 18336 |

The analytical and simulated figures are not contradictory estimates of one
exact quantity. The analytical table uses the worst observed 1984-draw
standard error and a deliberately conservative marginal lower bound; the
simulation uses the average fresh pilot joint covariance. Their spread is the
main uncertainty and is why no candidate rung is acquisition-ready.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject 1984 as a planned formal design | No scenario met all operating targets | Feasibility veto fired | None material to that rung | Do not acquire to 1984 for formal equivalence | Descriptive comparison at 1984 is useless |
| Preserve 4096 midpoint as a sensitivity candidate | All simulated screens passed from tolerance `0.01` | No numerical veto | Margin has no loss/utility justification | Derive scientific margin before any direct validation | `0.125/0.13597` is the correct scientific contract |
| Preserve 8192 historical contracts as sensitivity candidates | All simulated screens passed | No numerical veto | Gaussian/`1/N` approximation and pilot-vs-history spread | Direct finite-sample validation after margin choice | 8192 is sufficient on target G/H |
| Keep HMC and Phase 9 closed | No scientifically selected, directly validated design exists | Authority/evidence boundary | Future direct-validation result | No HMC acquisition yet | G/H equivalence or material difference |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Passed: historical bindings, no-confirmation boundary, GPU/XLA, finite outputs, covariance/MMD admissibility, single traces, and receipt integrity |
| Statistically supported ranking | None; scenario differences are sensitivity results under different margins, not fair method candidates |
| Descriptive-only differences | Per-family power curves, analytical-vs-simulation spread, MMD tolerance curves, and resource costs |
| Default-readiness | Not established |
| Next evidence needed | Scientific margin derivation, then fresh direct finite-sample validation of the complete decision at the relevant rung |

## Run Manifest

| Field | Material preflight |
| --- | --- |
| Receipt | `sample-size-margin-preflight-material.json` |
| SHA-256 | `ad13cede2f7ab23f18f956eb7eb39e729f1ed987e4175292cafd7ee59786d89d` |
| Git commit | `b1606a3ec19643356705cf9d08ccf7c6495b6186` dirty |
| Environment | `tfgpu`, Python `3.13.13`, TF `2.20.0`, TFP `0.25.0` |
| Device | physical GPU 1, trusted context, XLA JIT, TF32 enabled, `float64` |
| Seeds | synthetic pilot `(17001,17002)`; parametric Monte Carlo `(18001,18002)` |
| Wall time | `146.4031` seconds under `2400`-second cap |
| Shape | four chains, 8192 synthetic draws, two forecast replications, horizon 10 |
| Monte Carlo | 20,000 per family/rung/scenario |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

The exact command, source hashes, environment, device list, assumptions, and
output path are in the JSON receipt. GPU 1 was concurrently used by a separate
MacroFinance lane, so elapsed time is not performance evidence. The worktree
remained dirty and unrelated changes were preserved.

## Post-Run Red Team

Strongest alternative explanation: the asymptotic simulation looks optimistic
because its average fresh 8192-draw covariance is smaller than the worst
realized standard error in the five historical 1984 runs. Conversely, the
conservative analytical 20-coordinate bound can be pessimistic because it
ignores favorable cross-feature dependence. Only direct finite-sample
simulation at the selected scientific contract can distinguish these.

What would overturn this conclusion: a defect in the `1/N` scaling, joint
feature covariance, MMD degree-of-freedom scaling, or complete decision logic;
or direct finite-sample evidence showing that a smaller/larger rung meets/
misses the prospective targets.

Weakest evidence: the pilot covariance comes from four synthetic replications,
and MMD/feature joint behavior is bounded rather than jointly simulated.
Frechet bounds protect the decision from an independence claim but can make
the screen conservative.

## Handoff

Phase 9 and HMC acquisition remain closed. The next plan must first specify a
forecasting-loss or application-utility rationale for one mean and one
log-variance margin. Then it may run a fresh direct finite-sample validation of
the complete 13-family decision at the smallest relevant candidate rung, with
a higher rung available only as a prospective repair. No G/H confirmation
forecast may be opened during that work.
