# Force reuse and diagnostic reporting repair, 2026-10-12

The retained-position force cache passes the actual LGSSM N4032/T50 four-chain
consumer comparison. All three paired runs have identical positions, endpoint
potentials, energies, acceptance decisions and status flags. Each two-transition,
one-leapfrog run makes three force calls instead of four. Source snapshots and
serialized outputs are in ../surrogate-hmc-lgssm-force-reuse-20261012-01.

For a frozen position-only field F, retain C=F(q). A proposal uses C for its
initial half-kick and computes C*=F(q*) at its terminal kick. Acceptance selects
(q*,C*) and rejection selects (q,C), row by row. Thus the force at the retained
position remains exactly the force required by the next transition. This
argument requires fixed force identity and coordinates; rebootstrap after either
changes. The Metropolis endpoint and random draws are unchanged. For S
transitions with L leapfrog steps, calls change from S(L+1) to 1+SL. At L=1 the
asymptotic reduction is one half; this alone cannot close a 100-hour HMC forecast.

| Paired run | Uncached seconds | Cached seconds | Interpretation |
|---|---:|---:|---|
| Cold compile/execution | 153.03 | 170.75 | Compilation included |
| Warm, cached first | 176.84 | 96.79 | Descriptive timing |
| Warm, uncached first | 196.71 | 189.14 | Descriptive timing |

These two warm pairs do not support a statistically established speed ratio.
They do establish exact replay and the counted reduction for the exercised case.
Benchmark normal prior and two-transition pilots are mechanics checks, not
posterior convergence or model-level method superiority.

The regression run had 193 passing tests and one stale test that expected plain
tensors where the existing API returns structured force/endpoint values.
Correcting that expectation and rerunning its test passed. The original FAIL
artifact is preserved under ../surrogate-hmc-force-reuse-regression-20261012-01;
this reconciliation does not overwrite it. Additional current checks: 65
launcher and consumer-reporting checks, nine saved-prefix comparison checks,
and four CPU/XLA precision-wrapper checks. These selections overlap; counts
must not be added as unique tests. CPU checks intentionally hid GPU devices.

Reporting now saves numerical outputs, per-lane health and available parity
checks before optional native-B4 execution. A simulated late B4 failure
demonstrates that completed evidence survives. The older timed-out diagnostic
did not archive those comparisons, so its timing evidence is not retroactively
promoted to parity evidence.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain force-cache repair | Exact replay and expected call counts pass | No finite/status/divergence/fallback veto in paired GPU case | Only LG exercised at full scope | Keep regression guard; continue precision/cost repair | Full HMC feasible or converged |
| Keep native B4 diagnostic only | Earlier run timed out | Incomplete comparison | Numerical parity and steady cost not archived | Preserve incremental evidence in future runs | B4 admission |
| Continue SIR precision localization | T10 mixed arm passes 1% | Valid complete rows; source/input checks pass | Accumulation through T50 | Run same arm through T50 | FP32 default repaired |

| Inference status | Finding |
|---|---|
| Hard veto screen | Cache: none observed; SIR pure-FP32 score precision remains a veto |
| Statistically supported ranking | None |
| Descriptive-only differences | Timing pairs above |
| Default readiness | Cache engineering replay supported; no new numerical default promoted |
| Next evidence needed | Full-horizon precision, fresh scope tuning and downstream HMC diagnostics |

Post-run red team: timing changes could reflect GPU clocks or contention; more
repetitions would be needed for a speed ranking. A cache mismatch under any
accepted/rejected/fallback path would overturn the engineering decision.
CPU tests cover these paths; the full-scope GPU run did not exercise every one.
