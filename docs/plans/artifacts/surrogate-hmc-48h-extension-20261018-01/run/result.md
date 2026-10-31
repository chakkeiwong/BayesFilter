# Surrogate-HMC 48-hour extension result

Status: INCOMPLETE_SEE_MODEL_RESULTS. Updated 2026-10-20T07:37:55.049608+00:00.

| Model | Outcome | Evidence |
|---|---|---|
| LGSSM | Saved sampling rule passed; preserved | Earlier literature-prior run |
| KSC SV | Saved sampling rule passed; preserved | Earlier literature-prior run |
| predator_prey | VERIFIED | /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/run/predator_prey-retry-01/predator_prey/inference/hmc.json |
| range_bearing | VERIFIED | /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/run/range_bearing-energy-resume-01/range_bearing/inference/hmc.json |
| sir_d18 | BUDGET_EXHAUSTED | /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/run/sir_d18-energy-retry-01/sir_d18/inference/hmc.json |

| Decision field | Finding |
|---|---|
| Decision | INCOMPLETE_SEE_MODEL_RESULTS |
| Primary criterion | Recorded adaptive sampling rule; per-model artifacts carry the decision |
| Veto diagnostics | Existing health rules retained; a health-vetoed candidate is rejected |
| Main uncertainty | Mixing within time allocation; posterior/reference agreement not checked |
| Next justified action | Use individual terminal status: complete, time-limited, or candidate repair |
| Not concluded | Exact-posterior accuracy, heuristic dominance or method superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | Read each hmc.json; no health threshold was relaxed |
| Statistically supported ranking | None; this is a continuation, not a comparison |
| Descriptive-only differences | Acceptance, ESS, intermediate R-hat and runtime |
| Default-readiness | Not established by this extension |
| Next evidence needed | Unfinished sampling and independent posterior agreement, as applicable |

No extra validation chain was run. VERIFIED means the recorded sampling rule passed. A rejected candidate does not reject the research direction. All prior attempts are preserved. The strongest alternative explanation for an incomplete result is insufficient time or unfavorable HMC geometry; exact-model posterior agreement remains unevaluated.

## Terminal interpretation, 20 October 2026

All campaign processes have stopped. Four models passed the cumulative
sampling rule; SIR did not finish warm-up. The owner-requested fresh
range-bearing run at target acceptance 70% also passed and supersedes the
95%-target run above for reporting the requested configuration.

| Model | Warm-up per chain | Retained per chain | Maximum modern R-hat | Outcome |
|---|---:|---:|---:|---|
| LGSSM | 2,000 | 1,000 | 1.007082 | Sampling rule passed |
| KSC SV | 2,000 | 1,000 | 1.005998 | Sampling rule passed |
| Predator–prey | 3,000 | 6,000 | 1.007034 | Sampling rule passed |
| Range bearing, target 70% | 2,000 | 1,000 | 1.003620 | Sampling rule passed |
| SIR | 5,500 | 0 | 3.417349 at the last warm-up check (5,000) | BUDGET_EXHAUSTED; warm-up incomplete |

There are four chains per model. SIR's final 5,500-transition chunk passed
numerical health checks but was not a scheduled readiness check. Its retained
posterior diagnostics are unavailable because no posterior draws were retained.
It stopped at 15:37:54 Hong Kong time. The supervisor terminated rather than
leaving an active job or a pending queue. Finite energy errors remained warnings.

The extension ledger charged 161,236.65 worker seconds (44.79 hours) and records
11,563.35 seconds unspent; BUDGET_EXHAUSTED is the worker's terminal decision,
not a claim that the aggregate ledger reached zero. No new run was launched.

Fresh range-bearing evidence:
`../../surrogate-hmc-range-bearing-accept70-20261019-01/run/range_bearing/inference/hmc.json`.
Its retained acceptance rates were 70.2%, 68.0%, 68.4%, and 68.7%.
LGSSM and KSC evidence is in
`../../surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-hmc-01/`,
under each model's `inference/hmc.json`; predator–prey and SIR evidence is
linked above. Passing the sampling rule does not establish agreement with an
independent model posterior. This terminal report adds no validation phase.
