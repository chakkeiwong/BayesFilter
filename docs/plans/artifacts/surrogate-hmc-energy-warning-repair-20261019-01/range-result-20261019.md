# Range-bearing continuation result

Range bearing completed its declared sampling rule at 2026-10-19 05:08 Hong
Kong. Four chains each have 2000 discarded warmup transitions and 2000 retained
inference draws. The maximum modern R-hat is1.00745857, below1.01. Acceptance
by chain is93.80%,94.45%,94.70%,94.90%; bulk ESS is671.76 and1523.21, and
tail ESS is1310.03 and2504.28 for the two parameters. These are descriptive
diagnostics for the specified fixed endpoint approximation.

The retained maximum absolute energy error1006.58 exceeded the reporting
threshold1000. It was recorded as a warning and did not terminate sampling.
The proposal and MH correction remained unchanged. The result reports zero
arithmetic divergences and one deterministic force fallback. Large finite
energy error is not included in that arithmetic-divergence count.

The restart used9134.72 worker-seconds including its setup. It preserved the
saved1500 warmup transitions per chain and continued under the frozen settings.
The master advanced automatically to SIR, which was alive at the19 October12:00
Hong Kong check, with2750 warmup transitions per chain and no retained draws.
Its latest chunk passed all non-energy health checks; warmup is not yet ready.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Range sampling complete | Warmup ready; cumulative modern R-hat<=1.01 | No terminal non-energy health veto | Exact-posterior agreement not checked | Preserve all retained draws; master continues SIR | Exact-posterior correctness or superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | Passed the existing sampler health checks |
| Statistically supported ranking | No method comparison was run |
| Descriptive-only differences | Acceptance, ESS and runtime describe this run |
| Default-readiness | Not assessed; owner-requested warning policy works in the actual sampler |
| Next evidence needed | Remaining SIR sampling diagnostics; reference agreement remains a separate scientific question |

Red-team note: passing the sampling criterion supports completion under the
declared rule, not exact-posterior agreement. The fixed approximate endpoint
target has not changed. No added validation run is required by this result.

Evidence: ../surrogate-hmc-48h-extension-20261018-01/status-check-20261019-040045.json,
the linked range hmc.json and inference manifest, and the shared budget ledger.
The active SIR run continues under the existing deadline20 October16:00:41
Hong Kong; no additional work direction or budget was introduced.
