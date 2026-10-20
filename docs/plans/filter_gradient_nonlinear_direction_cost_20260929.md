# Nonlinear direction costs and OS memory accounting

Activate after the original nonlinear cohort passes its healthy/refusal gates
and all three independently calibrated LEDH-family scopes pass untouched
CPU/GPU qualification and readback.
Compare the prior test-only Python calls with the new enclosing analytical
owner for nonlinear UKF and nonlinear LEDH with diagnostics. UKF uses the exact
fixture in `tests/fixtures/filter_repair_nonlinear_directions_20260929.json`.
LEDH uses case0 from the ledh provider in
`tests/fixtures/filter_repair_nonlinear_untouched_20260929.json`, bound to its
independently validated test-scope nomination (04806, eight/eight reset counts).
The failed original LEDH fixture is excluded. Within each case, both arms use
the identical frozen fixture, controls and all nested outputs. One fresh process per arm, with
UKF prior then enclosing and LEDH enclosing then prior, is a descriptive cost
screen only. Require numerical/status parity at1e-9 and exact discrete values
before interpreting timings. Preserve every attempt and do not exclude slow
processes. No statistical ranking or whole-program cost acceptance follows.

Follow the Gaussian cost contract: measure construction, first call and their
sum, three conditioning calls and30 synchronized warm calls; synchronize all
shared outputs; stack only prior values/scores during timing and normalize
auxiliary outputs afterward. Record sources, fixture, environment, affinity,
one-trace reuse and CPU-reference provenance. Primary measurements precede
comparison-owner compilation, HLO export and output serialization.

At each construction/cold/warm/collection boundary record ordinary VmRSS,
VmHWM, RssAnon, RssFile and smaps_rollup Rss/Anonymous alongside getrusage
ru_maxrss. Record read order and acknowledge that non-atomic observations
cannot be exact simultaneous peaks. This addresses the Gaussian screen's
rusage value below sampled VmRSS. Readings may distinguish metric accounting
from observed growth; they cannot identify allocator live tensors or guarantee
true maxima. One default-JIT owner per fresh process, no forced cache clearing,
system changes or cross-process interference. Python collection observations
must not be described as compiler eviction.

Allocation:6 workers/1800 CPU process-seconds, no GPU seconds, inside unchanged
global caps after charging nonlinear qualification. Four measured arms, one
source-bound readback/policy worker and one localized repair allowance;
300-second initial timeouts, one numerical worker at a time, stable runner and
unique numbered artifacts. Any validity, source/input/environment discrepancy
or budget exhaustion stops for localization. GPU costs wait for unshared
non-display hardware. No training/HMC, package/system/cache mutation, subagents,
main merge, live MacroFinance edits or canonical admission.

Skeptical review: the tiny UKF may mostly measure host dispatch and cannot
predict large-filter costs. Collecting memory maps allocates a small amount of
host memory, so it is kept outside timers and cannot explain all compile
residency. The baseline receives no extra timed auxiliary stacking. Same
fixture/source/environment and unchanged numerical gates prevent a reduced
candidate from looking cheaper. Main criterion is trustworthy accounting of
the qualified owners; memory/speed are descriptive repair triggers here.
Self-review found and corrected a stale reference to the failed LEDH fixture.
The new fixture and counts apply equally to both measured arms, and the
source-bound readback must verify the nomination/fixture identity as well as
the complete numerical/status records. Self-review passes for preparation,
conditional on completed qualification. No numerical-source changes during
the cost cohort. The existing test-only prior comparator is the appropriate
baseline for dispatch/compilation costs; no method-quality ranking is sought.

Activated after04821 passed162 combined qualification/readback/policy checks.
The ordinary and diagnostic endpoints now both have healthy CPU/GPU evidence.
Base charges are111078.085746 CPU/97801.619105 GPU seconds; this cohort receives
the declared6 workers/1800 CPU seconds with zero GPU allocation. Exact commands
use the stable runner with group
`nonlinear_direction_cost_{ukf|ledh_diagnostics}_{python_reference|enclosing}_cpu`,
`--device CPU --test-timeout-seconds 300`, then
`nonlinear_direction_cost_readback_cpu`. Results stay in new numbered directories
under the established filter-gradient-repair-20260917 artifact root.
