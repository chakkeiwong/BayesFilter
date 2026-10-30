# LG2 endpoint qualification and inference cost

LG2 passed all 16 claim-endpoint qualification checks on GPU1. Its full
canonical analytical DCTR proposal field, independent FP64 Kalman endpoint,
claim-data binding and complete transitions are executable through the master.
**LG2 posterior inference remains NOT_RUN.** The measured complete cost of
even the bounded 2,000-draw B/R pilot exceeds the remaining family allocation.

The [plan](surrogate-hmc-lg2-endpoint-2026-09-22.md) preserves the original
two-state, two-observation, T20 model, N=K=1024, attempt1/replicate0 claim,
stationary parameter-dependent initial law, prior, parameter chart and selected
protection multiplier 0.01. The candidate is an imperfect deterministic
position-only DCTR field; acceptance uses the independent model endpoint.
This qualification tests the actual consumer, not force accuracy as a proxy
for HMC inference.

The [report](artifacts/surrogate-hmc-all-models-20260922-lg2-endpoint-01/report.json)
records direct canonical-callable parity, independent typed bindings,
bitwise replay, row independence, permutation, original force and endpoint
statuses, signs, proposal reversal, reference predictive parity and complete
four-chain transitions. Reversal error was zero against the declared
7.6294e-6 tolerance. Five transition probes and five complete retained-diagnostic
probes were healthy. These are mechanics and cost checks, not posterior draws.

| Quantity | CPU-worker seconds | GPU-wall seconds |
| --- | ---: | ---: |
| Qualification charge | 169.579018 | 135.984779 |
| Remaining cumulative LG2 allocation | 1,023.995559 | 6,665.035729 |
| Complete prospective B/R pilot forecast | 15,327.988770 | 16,120.324203 |

The forecast includes 3,586 four-chain transitions for a one-step public-tuned
B arm, eight complete 1,000-position diagnostic evaluations, at least 240
seconds for compilation, 90 seconds for analysis and a 1.5 safety factor.
It includes no C arm. Costs are descriptive measurements of this consumer and
hardware, not an efficiency comparison. The precision audit separately shows
that a 2,000-draw schedule is not an adequately justified joint-equivalence
design. Transferring enough budget for this pilot alone would not resolve that
scientific limitation.

The actual command was the master entry point with `--stage P3-LG2-endpoint
--scope LG2 --output docs/plans/artifacts/surrogate-hmc-all-models-20260922-lg2-endpoint-01`.
The [manifest](artifacts/surrogate-hmc-all-models-20260922-lg2-endpoint-01/run-manifest.json)
preserves the exact command, tftwogpu environment, commit, seeds, twelve frozen
input hashes, 107 archived source dependencies, verified memory growth and
GPU/TF32/XLA settings. No source changed during execution. Peak TensorFlow
allocation was 151,252,224 bytes. The run stayed within its 350 CPU/450 wall
reservation. No public HMC tuning, retained posterior sample or statistical
look was taken.

The master reader verifies all sixteen required checks, archived sources,
frozen inputs, successful accounting and the recomputed complete forecast.
Its reporting-only source may subsequently change; the archived version is
still checked and numerical dependency changes remain disqualifying. A focused
regression now rejects omission of the independent-binding check. The reader
reports DEFERRED_COMPLETE_BUDGET and grants no new inference authority.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve endpoint qualification; defer inference | All 16 preparation checks pass; posterior criteria untested | No observed mechanics or numerical veto; complete inference budget fails | Joint interval precision and actual tuned-chain behavior | Prospectively justify the diagnostic and precision design, then reserve its complete cost and use public tuning | B/R, C/D, D/R and C/R remain NOT_RUN; no rejection of surrogate HMC |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current launch budget is insufficient; qualification itself passes |
| Statistically supported ranking | None |
| Descriptive-only differences | Timings, memory use and numerical probe errors |
| Default readiness | Not established |
| Next evidence needed | A validated prospective uncertainty policy, complete funded schedule, public HMC tuning and actual retained comparisons |

Post-run red team: healthy probe positions and a reversible short proposal
cannot establish healthy posterior trajectories. The forecast may also vary
with tuning and posterior position. The safety factor limits optimism but is
not a bound. A new measured implementation improvement or a funded, adequately
powered design could change the launch decision; relaxing the closed results
or treating preparation as inference would not.
