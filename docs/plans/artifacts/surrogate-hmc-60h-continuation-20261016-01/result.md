# Surrogate-HMC 60-hour continuation result

Updated 2026-10-18T00:39:23.973839+00:00. Status: INCOMPLETE_SEE_MODEL_RESULTS.

| Model | Recorded outcome | Evidence |
|---|---|---|
| range_bearing | HEALTH_VETO | range_bearing/master.json |
| predator_prey | BUDGET_EXHAUSTED | predator_prey/master.json |
| sir_d18 | BUDGET_EXHAUSTED | sir_d18/master.json |

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Preserve all attempts | Only VERIFIED records complete the declared sampling rule | Inspect each hmc.json; budget stops are incomplete evidence | Posterior coverage and endpoint approximation | Review failed candidates and unused budget; continue authorized repairs where justified | General posterior correctness or superiority |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Per-model worker results and logs control; no silent reinterpretation |
| Statistically supported ranking | None; no method comparison was run |
| Descriptive-only differences | Acceptance, ESS, R-hat and runtime require their model-specific context |
| Default-readiness | Not established by this continuation |
| Next evidence needed | Review cumulative diagnostics and any incomplete model phases |

Post-run limitation: numerical health and cumulative R-hat do not establish
complete posterior exploration or exactness of an approximate endpoint.
LGSSM/KSC results remain in the preceding campaign. No extra validation chain was added.
