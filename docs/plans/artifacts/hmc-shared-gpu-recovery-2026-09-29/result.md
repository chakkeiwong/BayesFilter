# Shared-GPU recovery execution

## Terminal pricing status checked September 29

The service exited with status 1 at **05:38:30 Shanghai, September 29**.
Trusted `systemctl --user show` confirms it is failed and inactive; the pricing
receipt confirms child shutdown. There is no active reservation in the settled
GPU ledger. The main campaign has not started (0/32), and none of the eight
pilots completed the declared full fit. There is no complete-fit measurement
from which to give a reliable completion forecast.

All eight mechanics checks passed in 67.078 enclosing seconds. All four public
preflight cells passed in 607.152 enclosing seconds. Pricing used 10,709.555
enclosing seconds (2 h 58 min 30 s). These are recorded elapsed times, including
sharing overhead; no attribution of all elapsed time to contention is justified.

| Case | Pilot outcome | Preserved progress |
| --- | --- | --- |
| K0 | 1,775-second hard fit allowance exhausted | Search complete, five verified members; one of two declared posterior assessments complete with 2,000 warmup and 1,000 retained transitions per chain |
| K1 | Same hard allowance exhausted | Search incomplete, one verified member so far |
| K2 | Same hard allowance exhausted | Search incomplete, four verified members so far |
| K3 | Same hard allowance exhausted | Search incomplete, no verified member yet |
| K4 | Same hard allowance exhausted | Search incomplete, two verified members so far |
| K5 | Bootstrap failed after 24.254 child seconds | First round rejected all proposals with finite log acceptance; next round had nonfinite log acceptance |
| K6 | Bootstrap failed after 20.561 child seconds | First round had nonfinite proposal and log-acceptance diagnostics |
| K7 | Same hard allowance exhausted | Search incomplete, four verified members so far |

Each slow fit received all 600 extension seconds beyond its 1,175 nominal
seconds. All six made durable progress; their last recorded progress ages
ranged from 0.88 to 73.97 seconds. No automatic checkpoint retry occurred because
the original cumulative hard allowance was entirely spent. The sharing and
extension mechanisms therefore operated, but the pilot allowances did not
support a complete fit under the observed load. Persistent sharing cannot be
treated as measured lost compute, and a budget stop cannot establish a sampler
failure. K5/K6 are numerical startup failures and must not enter the contention
retry path.

The grant balance is **190,601.314 GPU seconds (52.945 hours)**, including the
1,200-second NeuTra reserve. Cumulative SSM charges are 18,859.450 seconds,
including previous attempts and waiting. The original new-work, compute and
report deadlines remain September 29 23:03:52, September 30 03:03:52 and
September 30 05:03:52 Shanghai. Available GPU hours do not extend elapsed wall
time. The report deadline is a limit on the investigation, not a promise that
all 32 cases will succeed.

The next repair sequence is to inspect the existing bounded startup initializer
for K5/K6 and to explicitly reallocate pilot/repair time before another launch.
The first runtime measurement must complete the exact declared fit, preserving
the broad search, member counts and posterior requirements. Same-source native
resume should preserve completed evidence where supported; a changed source or
scientific configuration requires a new identity. Previously completed member
assessments must be reused, including any unfavorable result. Affordability must
be recalculated against the original remaining wall time before main admission.

Skeptical audit of this next step: an unchanged relaunch would encounter
exhausted fit budgets; a longer timeout alone would not repair K5/K6 startup;
and the successful short public preflight cannot price the broad search and
serious posterior workload. Do not lower numerical checks, treat incomplete
searches as prices, or move the original clock. Test the initialization and
budget-resume mechanisms before spending on another full batch. This status
check changed records only; it launched no new numerical run.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep main unstarted | Complete same-workload prices unavailable | Pricing is incomplete | Whole-fit cost under shared load | Repair and obtain a complete price within the original limits | Any reliable finish time or 32/32 completion |
| Preserve six resource-limited fits | Durable numerical progress recorded | Fit caps exhausted, not numerical rejection | Remaining search and assessment cost | Review cumulative pilot allocation and native resume | Contention is the only cause of long runtime |
| Diagnose K5/K6 startup | Nonfinite proposals/log acceptance recorded | Current bootstrap handoff invalid | Initial step/geometry versus deeper target failure | Check the existing bounded initialization mechanism | State-space target or HMC method invalid |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | K5/K6 fail current numerical bootstrap; six other pilots are resource-incomplete |
| Statistically supported ranking | None |
| Descriptive-only differences | Per-case runtime, verified-member counts and progress |
| Default readiness | No new default or broad posterior claim |
| Next evidence | Successful bounded startup diagnosis and complete same-workload pricing |

Post-run red-team note: the strongest alternative explanation to GPU contention
alone is the underestimated broad-search and per-member compilation/workload.
Completing an unchanged fit under a bounded continuation would discriminate
resource insufficiency from a numerical stall. K0's single completed member
does not satisfy its two-member pricing contract. The weakest evidence remains
the absence of any uncensored complete price.

## Launch-time record

The continuation launched September 29 at 02:28 Shanghai on GPU UUID
`GPU-4f1220f9-7ba2-21ad-3f9a-b24c2ca4ce91` (GPU 2). Trusted telemetry recorded
foreign PID 6247 and 36% utilization at admission. The campaign began numerical
preflight alongside that work; it did not require an empty GPU. All eight
mechanics checks passed; public-pipeline preflight was then running. Service:
`bayesfilter-hmc-shared-ssm-20260929-r1.service`. Live state is in
`runtime-r1/status.json`; stage evidence is in `prepared-r5/`.

All admission layers now honor the explicit shared-device policy. The
supervisor still extends a progressing fit using observed contention.
An incomplete cooperative resource stop or supervised timeout can additionally
trigger one automatic retry from the same native checkpoint, funded only from
the unspent original base-plus-contention allowance. Source, data, seed and
checkpoint checks remain strict. Attempt caps survive coordinator restarts.
Final assessments and numerical/infrastructure failures are not selected for
retry. All attempts, startup and shutdown time count toward the enclosing caps.

The code previously set zero extension for preflight and pricing. Shared-mode
preflight cells now have 170 nominal plus 170 recovery seconds, with their
existing 10-second outer margin. Pricing has 1,175 nominal plus 600 recovery
seconds and the same margin. These are bounded scheduling hypotheses, financed
by 5,480 seconds of the existing repair allocation. Main allowances retain the
measured 1.5x rule. Pricing now respects the remaining original wall deadline
and cumulative SSM cap. A budget-limited pilot stays unpriced without stopping
independent affordable lanes; unknown failures still stop for diagnosis.

The focused set passed 97 tests in 127.53 seconds, including actual K0 and K7
subprocess recovery after a committed numerical chunk. Existing chunks retained
their checksums, the pipeline finished with verified candidates and correct
warmup exclusion, and subsequent invocation did not rerun completed fits.
These tests used deliberately hidden GPUs and injected contention/interruptions.
They demonstrate recovery mechanics, not GPU throughput or convergence.
The compatibility set passed 110 tests in 149.43 seconds, including actual
Gaussian/beta-binomial isolated pipelines; final scheduling checks passed 50
tests in 1.62 seconds. Sets overlap. JUnit files preserve every result.

The official tuning chapter and reference describe shared admission and recovery.
The chapter PDF builds; its recovery paragraph was visually inspected. One
initial test stopped on a map whose chunk lists were still empty; the corrected
test waits for actual committed evidence. An initial preparation comparison
confused Python tuples with JSON lists; `prepared-r4/` preserves that incomplete
preparation, which launched no numerical process. JSON-normalized comparison
passed before freezing `prepared-r5/`.

The frozen package identity is
`a8c6c6107c2da78d25a144be28a3503431352681465e338e945fca899f0a0e43`.
Only timeout policy, isolated fit recovery and campaign planning/provenance
changed inside the package from the preceding snapshot. The external queue
changed too. Data and independent-reference files retain their checksums;
suite comparison permits only scheduling budgets/policy/metadata differences.
Numerical targets, filters, scores, HMC kernels and scientific criteria did not
change. All ten completed C1 recovery fits are reused.

Before launch, the ledger had 201,986.283 GPU seconds (56.107 hours), including
the existing 1,200-second NeuTra reserve. Prior SSM charges total 7,474.481
seconds, including settled waiting and both earlier preflights. Every previous
runtime/stage root remains in cumulative accounting. The CPU repair charge is
a conservative 1,800 seconds from 41,238.430, leaving 39,438.430 seconds; nested
tests are not charged again. The 36-GPU-hour SSM and 22-hour main ceilings
remain. New work stops September 29 at 23:03:52 Shanghai, computation September
30 at 03:03:52, and reporting is due at 05:03:52. These are original deadlines.

Exact launch command, git commit, environment, source identity, input hashes,
seeds and growth requirement are recorded in `launch-command.json`,
`configuration.json`, `engineering-verification.json` and
`prepared-r5/manifest.json`. The active ledger remains
`hmc-ssm-funded-2026-09-28/grant-ledger.json`. No other user's process was stopped.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Run with shared-device recovery | Scheduling and real checkpoint tests pass; shared admission observed | Hard numerical and resource limits remain | Complete-fit cost under changing load | Fresh GPU preflight, complete prices, affordable main fits | Guaranteed deadline completion |
| Reuse completed C1 recovery | All ten completions checked | Original coverage failures remain | Calibration still unresolved | No C1 rerun | Original confirmation repaired |
| Preserve every case | 32 main slots retained | Incomplete/over-budget lanes remain explicit | Which cases deliver retained output | Per-case terminal assessment | All cases converged |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Numerical preflight and posterior vetoes remain enforced |
| Statistically supported ranking | None |
| Descriptive-only differences | Utilization, runtime and recovery counts |
| Default readiness | No numerical default promotion |
| Next evidence | Complete GPU preflight/prices and per-member posterior diagnostics |

The remaining risk is prolonged contention beyond the funded cap. Another
checkpoint retry cannot create time once that cap is exhausted. Such a result
is recorded as incomplete execution, not evidence that the model or sampler is
invalid. GPU recovery behavior under a naturally occurring timeout remains to
be observed; deterministic and real CPU subprocess tests provide the current
recovery evidence.
