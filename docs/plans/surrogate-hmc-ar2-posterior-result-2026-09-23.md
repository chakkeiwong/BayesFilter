# AR2 surrogate-HMC posterior experiment — 2026-09-23

Canonical analytical DCTR successfully drove the AR2 HMC proposals through
8,000 retained draws per chain, with independent Kalman endpoint acceptance.
All 13 estimated posterior differences are inside the predeclared pilot
margins. Ten simultaneous intervals establish equivalence for their functions;
three remain too wide. The overall formal B/R result is INCONCLUSIVE, with
qualified diagnostics and no demonstrated discrepancy. This is an uncertainty
limitation, not a failed sampler or evidence against surrogate HMC.

The tested model has state dimension two, observation dimension one, T=10,
and two inferred noise log scales. Structural deterministic filtering preserves
the noiseless lag. The proposal uses the full canonical N64 FP32/TF32 DCTR
force; HMC coordinates, Kalman endpoint and grid reference use FP64.

| Retained B/R diagnostic | Result |
|---|---|
| Draws | Four chains, 8,000 each; adaptation and verification excluded |
| Acceptance by chain | 0.731875, 0.730375, 0.737875, 0.733500 |
| Modern parameter R-hat | 1.000662, 1.001052 |
| Bulk ESS | 7,296.3; 4,452.8 |
| Tail ESS | 3,921.5; 7,403.2 |
| Retained chunk health / canonical-force fallbacks | 32 of 32 passed / zero |
| Posterior mean, log process-noise scale | 0.311434; reference 0.308717; difference 0.002717 |
| Posterior mean, log observation-noise scale | -0.164843; reference -0.170041; difference 0.005198 |
| Formal function comparisons | 10 equivalent, zero discrepant, three inconclusive |

The unresolved functions are the squared standardized process-noise scale,
the standardized cross moment and the observation-scale median CDF. Their
estimated differences are -0.014259, 0.003948 and 0.000531, respectively;
their margins are 0.1, 0.1 and 0.02. The broad simultaneous intervals, rather
than the point estimates, prevent overall formal equivalence. The first look
at 4,000 draws was also inconclusive. No additional looks are authorized here.

The public tuner selected L=1 and step size 0.7941176042. Fresh verification
passed; four adaptation force fallbacks are archived and excluded from retained
evidence. Fresh verification and retained sampling had no force fallback.
The master review passed 69 tests. The 106-file source snapshot stayed unchanged.
GPU placement and memory growth were verified. The B worker used 2,645.23
GPU-wall seconds, substantially below its conservative cost forecast; this
one-arm observation is not an efficiency ranking.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next existing-master action | Not concluded |
|---|---|---|---|---|---|
| Preserve viable B/R pilot; no formal promotion | All point estimates within margins; 10/13 simultaneous intervals within margins | Actual-chain diagnostic checks and chunk health passed | Remaining second-moment and median-CDF precision; empirical reference allowance | Execute the independently tuned A/R comparator, then the existing finite-target and remaining-model work | Formal whole-family equivalence, owner acceptance of AR2, speed advantage or all-model completion |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Passed; no retained fallback or unhealthy chunk |
| Statistically supported ranking | None |
| Descriptive-only differences | Acceptance, runtime and point-estimate agreement |
| Default readiness | Not established |
| Next evidence needed | Planned exact-gradient comparator; separate finite-target comparisons |

The result is preserved in
artifacts/surrogate-hmc-closure-20260922-ar2-inference-01/ar2-inference-b/report.json,
with tuning/chunk tensors and run manifests in that directory. The exact
command and environment are in the parent run-manifest.json. The pass charged
2,718.357431 CPU-worker and 2,645.228483 GPU-wall seconds. Cumulative recorded
charges are 34,925.457759 CPU and 32,881.453919 GPU against caps of 54,000
and 226,800. Minor recovery inspection/preflight overhead outside the master
is not represented as precisely measured by these receipts.

Pre-execution A/R audit: the comparator is the same model posterior with the
exact Kalman force, independent public tuning and the same held-out reference.
It answers the existing comparison question. A/R is descriptive and does not
expand the primary B/R multiplicity family. Its 3,200 CPU / 3,400 wall worker
allowance plus 200 CPU review fits the remaining inference-pass and campaign
budgets. The dry run passed; launch will use a fresh inference-02 directory.

Post-run red-team assessment: agreement of means and good R-hat could conceal
errors in less precisely sampled second moments. The simultaneous intervals
expose that uncertainty, so no all-functions equivalence is asserted. A future
qualified interval lying outside a margin would overturn the current
no-demonstrated-discrepancy result. The weakest evidence is transfer of
controlled-law MCSE calibration to these actual chains, together with empirical
rather than rigorous grid-integration error bounds.

## Completed exact-gradient comparator

A/R completed in artifacts/surrogate-hmc-closure-20260922-ar2-inference-02,
with four independently tuned chains of 8,000 retained draws. All 32 chunks
passed, with zero force fallbacks. Acceptance was 0.716875–0.729125;
parameter R-hat was 1.000508 and 1.000128; bulk ESS was 5,060.9 and 4,789.5;
tail ESS was 2,301.2 and 5,943.5. All 13 estimated differences were within
pilot margins; eight simultaneous intervals were within margins and none
demonstrated discrepancy. Overall A/R remains descriptively INCONCLUSIVE.
This comparator has no additional formal family-level promotion claim.

A's posterior means were (0.304752,-0.177304), versus the exact reference
(0.308717,-0.170041). Both ordinary and surrogate-force HMC passed the actual
sampling checks with point estimates close to R. The inconclusive whole-family
interval result therefore also occurs with the exact force; it is not evidence
of a surrogate-specific failure. There is no statistically supported ranking.

The A worker used 140.664492 GPU-wall seconds, versus B's 2645.228483.
Kalman evaluation is much cheaper for this small linear model. These are
single-run descriptive costs; the campaign tests correctness and practical
viability, not a speed advantage over an available exact Kalman gradient.
Each master run passed 69 tests and its source-integrity check.

The A pass charged 162.565729 CPU / 140.664492 GPU seconds. Cumulative recorded
charges are now 35088.023488 CPU / 33022.118410 GPU. The next existing-master
action is the finite-target domain/cost preflight under
surrogate-hmc-ar2-finite-target-2026-09-23.md. B and A receive no extra looks.
