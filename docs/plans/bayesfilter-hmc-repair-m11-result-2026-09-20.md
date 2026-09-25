# M11 bounded metric-boundary step repair

Status: complete, including all three fresh fits and terminal reconciliation. The active
[master program](bayesfilter-hmc-repair-master-program-2026-09-16.md) records
the pre-run contract, skeptical audit and allocation. Outputs are under
`artifacts/hmc-repair-master-2026-09-16/m11-r1/` (R below).

The metric-boundary repair fixes a reproduced implementation failure. Starting
from the same observed-data position and seed as M10 fresh fit 0, preparation
now completes with three qualified metric updates instead of zero. Each
update still passes the independent new-coordinate step search and all
existing covariance and affine-parity checks.

## Failure and repair

M10's diagnostic replay established that its three metric searches started
from unconstrained dual-averaging step proposals around 1e15--1e30, although
the executed step was capped at `0.0004412119158251195`. Twenty halvings could
not reach even one finite trajectory. The code was carrying an unexecuted
adaptation proposal into a new-coordinate search.

The boundary now nominates the dual-averaging average bounded by the previous
coordinate system's executed ceiling. Comparing in log space before
exponentiation avoids overflow, while unsaturated averages preserve the exact
old calculation. The new-coordinate search can expand above the old ceiling;
it independently qualifies its step using unchanged L, momentum streams,
acceptance bracket, attempt limit and numerical checks. A bad internal average
rejects the proposed metric and retains the qualified incumbent. It cannot
manufacture step qualification.

The matched repaired preparation applies three updates. The probes start at
`0.0004412119158251195`, `0.9036020036098448` and `1.1718268199431512` and pass
after 12, five and five attempts. This repairs the measured failed-search
mechanism. It does not establish that every future preparation is stable.

Failure-only trace diagnostics now identify nonfinite fields, counts and first
indices. A separate diagnostics-only replay of M10 fresh fit 1 reproduces its
completed-window numerical summaries and covariance evidence, then fails on one
nonfinite `log_accept_ratio`, at zero-based index 151 of the next window.
Reported step sizes and retained target values were finite. This distinguishes
the failure from a reported-step overflow but does not identify the integrator
substep or certify the proposed state/score. The hard veto remains unchanged.
Coordinate and metric signatures differ from the original because the corrected
M10 diagnostic report participates in estimate identity; old receipts are not
reused. `R/trace-replay-parity.json` records the comparison and its limitations.

The affected tests also exposed an older reader rejecting the current
reasonable-step payload once its Gaussian fixture began applying a metric.
The reader now accepts both known historical/current shapes, validates probe
counts and unique seeds, checks acceptance ranges, and rejects inconsistent
qualification metadata. It also requires one fixed L/momentum-probe design
throughout a search and rejects mixed old/new attempt formats. None of this
allows an inconclusive or unhealthy probe to qualify a metric.

## Evidence and decisions

The three fresh fits use new seeds, the serious preset, the optional
finite-window covariance policy, no supplied geometry hint, and the same
posteriordb `sblrc-blr` target/data and observed-data initialization as M10.
All candidate measurement/verification, broad L coverage, retained-member
selection and posterior/reference screens are unchanged. Every verified
candidate remains retained; posterior assessment uses the first sorted verified
identity selected before observing its posterior draws. The other members
remain unassessed, not rejected.

Two of the three fresh fits pass the complete declared posterior/reference
screen. The third fails preparation after two qualified metric updates.
All outcomes are preserved in `R/fit-summary.json`.

| Fresh fit | Metric updates | Candidates / verified | Warmup / retained per chain | Complete-screen outcome |
| --- | ---: | --- | --- | --- |
| 0 | 3 | 100 / 26 | 2000 / 2000 | Passed |
| 1 | 3 | 100 / 29 | 2000 / 5000 | Passed |
| 2 | 2 | Not reached | Not reached | Preparation numerical veto |

The two selected retained members have maximum modern R-hat 1.00743 and 1.00673,
meet their lugsail precision requirements, and pass all six reference-mean
comparisons with uncertainty from both chains. Their retained lengths differ
because the same cumulative controller stops only when the declared checks
pass. The 25 and 28 other verified members remain available but unassessed.
The two searches finish the funded 100-candidate search; this is not exhaustive
coverage of the continuous epsilon domain.

Fresh fit 2 reports two nonfinite `log_accept_ratio` values in its next slow
window, first at zero-based index 31. The last qualified epsilon is
1.2237080256119226. Its preceding window had minimum temporal ESS 193.61;
insufficient covariance ESS is therefore not an explanation for that last
update. A locally qualified step still does not guarantee later trajectory
health. The failure remains in the denominator. Three development seeds cannot
establish comparative reliability, statistical superiority or default readiness.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep bounded starting-step repair | Matched counterexample recovered; inactive arithmetic and actual new-coordinate qualification tests pass; two fresh whole-fit screens pass | Third fresh fit has a numerical preparation veto; invalid averages and failed probes still reject | Local qualification does not guarantee later trajectory health | Diagnose the saved failing transition before proposing bounded recovery | General reliability, global stability or convergence |
| Keep failure diagnostics and reader repair | Trace-component, format, corruption and artifact replay checks pass | Unknown/mixed metadata and invalid evidence remain rejected | Historical reader cannot establish missing historical telemetry | Retain explicit source and payload provenance | Retroactive authority for historical artifacts |

| Inference status | Assessment |
| --- | --- |
| Hard veto screen | Original numerical failures remain failed; no veto is relaxed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Metric update counts, candidate counts, ESS, acceptance and runtime. |
| Default-readiness | Not established; finite-window policy remains experimental and nondefault. |
| Next evidence needed | Saved failed-transition diagnosis, broader target/start replication and the master's outstanding calibration work. |

## Tests, sources and documentation

Initial failed tests and their charges remain in R. The first affected batch
found 45 failures from the same historical-reader mismatch; the repaired
reader suite passes. Focused tests cover finite huge log averages, exponential
overflow, exact unsaturated arithmetic, invalid internal states, actual
new-coordinate expansion, rollback, each nonfinite trace field, strict format
validation and artifact replay. The documentation, dispatch and registry
checks pass. One inherited tiny-bootstrap test remains skipped.
The terminal record contains **372 distinct passing test identities**, zero
unresolved failures and one skip across the recorded overlapping batches;
these counts are not additive with M10's 510 tests.

Source r1, used only for failure localization, is
`6dfedf98e9b1470dc98bef00cf1a9ccb70cff0b7c0a984dac4844e5012598c7e`.
Source r2, used for the repaired matched replay and all three fresh fits, is
`14dd4b4abf1b1894200b3e7f9eff71fdcf1dbb2d13a1fd69fae031a390917d5f`.
The final source snapshot is
`49968d882ffe6744a1608be0abc4f617d736a4cbb4ea3222d1c74cc7c47cc406`.
Its two later HMC changes tighten reader consistency and move invalid-average
handling into the existing local rejection boundary. Finite-input arithmetic
and seeds are unchanged. The new invalid-input branch is tested on CPU and
was not exercised by the frozen GPU fits. Concurrent q20 edits are preserved
and distinguished in source manifests.

Numerical runs use GPU 1, TensorFlow/TFP, XLA and verified memory growth before
initialization. CPU reference tests deliberately hide GPUs. Commands, seeds,
upstream data identity, environment and wall times are preserved per attempt.

Both guides and the official 566-page PDF are updated. Physical pages 411--412
were inspected for the new step-boundary explanation. PDF SHA-256 is
`cea65455c7b24e87fb63493e8ba01bd07a19e7afa84a9d1e88eb3579416c1d5b`.
The three inherited unresolved citations remain Afshar2015, Gorinova2020 and
Pakman2014. No configuration default, tuning admission threshold, posterior screen
or ranking rule is changed.

## Terminal accounting and next repair

The terminal audit passes with no invalid artifact or outstanding work. It
checks five GPU attempts, three source snapshots, two complete candidate
inventories, 334 numerical receipts and 229 tensor checksums, along with test,
launch, harness and book provenance. No M10 or M11 campaign worker remains.
All failed test runs and both failed numerical attempts (the diagnostic replay
and fresh fit 2) are charged. The unused sixth GPU slot is released; it was
reserved for a localized harness retry, not a replacement success.

| Worker-wall seconds | M11 charged | Cumulative charged | Remaining campaign allowance |
| --- | ---: | ---: | ---: |
| CPU reference/tests/build/overhead | 1179.8307465079706 | 108729.11150506153 | 150470.88849493847 |
| GPU | 1577.1906843289034 | 114488.41406213288 | 58311.585937867116 |

The CPU charge includes the predeclared 600-second overhead. M11 uses less
than its 4000/7200-second ceilings; approximately 41.80 CPU and 16.20 GPU worker
hours remain in the whole campaign. `R/reconciliation-terminal.json` is the
opening ledger for further work.

The next focused repair must first reproduce the exact failed transition with
accepted/proposed state, potential, score and energy diagnostics. The current
report identifies the failed field, but not the underlying arithmetic.
Distinguish a proposal-only failure from retained/shared target invalidity
before designing any bounded restart. Preserve the failed window, use fresh
streams for a new attempt, retain all candidate and posterior requirements,
and audit the recovery contract before execution. A blind increase in sample
counts or relaxation of the finite-trace requirement is not justified.

Post-run red-team question: the successful matched repair begins from a checked
observed-data start and uses an optional covariance policy. A favorable result
could depend on that initialization, local scale and particular momenta.
Broader preparation robustness, difficult posterior geometry, later stopping
regimes, whole-procedure calibration and exact MacroFinance comparisons remain
separate questions. Another nonfinite proposal would trigger more precise
trajectory diagnosis and a bounded new preparation hypothesis; it would not
justify discarding failed runs or weakening the posterior criteria.
