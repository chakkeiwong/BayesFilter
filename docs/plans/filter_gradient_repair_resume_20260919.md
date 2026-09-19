# Filter execution repair recovery, September 19

Worktree: `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`.
Branch: `repair/filter-gradient-xla-validation-20260918`.
Master: [repair program](filter_gradient_repair_master_20260917.md).
Detailed evidence: [execution record](filter_gradient_repair_execution_20260917.md).
Artifacts remain under the primary checkout's
`docs/plans/artifacts/filter-gradient-repair-20260917/`.

The user authorized execution repairs, preserved numerical algorithms and
tolerances, before/after comparisons, and commit/push. Main must remain gated
until full testing. Canonical LEDH rebuilding is excluded; unsupported claims
remain blocked. Only the two geometry initializers have approval to migrate
their random stream. Other seeded draws remain unchanged.

September 19 owner authorization adds 48 GPU / 24 CPU process-hours to the
original 4 GPU / 8 CPU caps. Active cumulative caps are **52 GPU / 32 CPU
hours**; the earlier 16 GPU / 12 CPU proposal is superseded. Through 01058,
charges are 14,543.687 GPU / 28,013.996 CPU seconds, leaving 172,656.313 GPU /
87,186.004 CPU seconds. No further compute approval is needed within these
caps. Use the driver for authoritative accounting, including interrupted runs
and supplemental charge files.
No campaign worker remains running at this recovery checkpoint.

Use the existing approved command prefix:

```text
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts/run_filter_repair_campaign.py
```

GPU3 is the same-class contention alternative under the existing plan. The
driver checks contention, sets and verifies memory growth, hides GPUs for CPU
diagnostics, enforces timeouts and shares accounting/locking across worktrees.
Do not bypass the driver or launch parallel numerical workers.

Checkpoint `c4351f8c` is committed and pushed. Its shared tensor-program helper
builds full pullback graphs only when requested, including coefficients used
only in custom derivatives. Preparation sampling/push/resampling, coordinate
clipping and target shifting now have stable XLA boundaries. Review restored
route/time/shape/nonfinite vetoes before qualification.

The helper passes 22 CPU and GPU checks; six public pullback checks and all 11
sequential consumer checks pass on GPU. Preparation passes 46 checks on CPU
and GPU. The policy/controller group passes 59 checks. These are focused
results, not all-repository coverage.

The two-/four-date public endpoint measurements 01019/01018 preserve prior
candidate outputs exactly and match the frozen baseline within 8.882e-16.
Compared with the prior candidate, single-process host maxima fall by
245.4/409.8 MiB; GPU allocator peaks are unchanged. The four-date candidate
still exceeds original baseline host memory by 362.6 MiB. Harness review
reconstructed the old hash by reversing only an equivalent dictionary syntax
change. The descriptive analysis is `lazy-pullback-diagnostic-01019.json`;
old artifacts remain stale for terminal comparisons. No three-process result
or isolated causal memory attribution is established.

Full preparation run 01020 was stopped at 854.417 seconds after the first 45
checks, with 27.67 GiB RSS observed at 12:01. The first 36-dimensional P59
assembly test had not finished. Run 01024's 45-second stack localizes fitting
update tracing; the run times out at 120.520 seconds.

The next fitter patch defers existing accepted-update derivative construction,
preserving the fixed-design packed-core/target derivative and rejection rule.
All six focused CPU cases, all 37 existing fitting checks (01031), and all 12
scalar adjacent-TT consumer checks (01032) pass. GPU run 01030 times out after
five progress markers without JUnit; it remains incomplete. Run 01028 also
preserves an external-tape TensorList boundary failure affecting the prior
source. Passing tests enclose value and gradient as the actual scalar consumer
does; they do not establish a new external-tape fitter API.

Run 01033 times out at 301.873 seconds. Its 45-second stack has passed the
initial fit and reached model simulation; RSS snapshots are 6.45 GiB at 01:27
and 10.53 GiB at 04:20. The large consumer still has no passing outcome or
final peak. Capture a later stage/stack before another broad retry.

Run 01034 passes all six final CPU fitter checks after the closure edit;
01035 passes all 59 policy/controller checks. The static guard passes for
171 sources with 1,097 exact schema/reference exceptions; coverage is partial.
Focused Ruff and whitespace checks pass.

The latest committed and pushed checkpoint is `191b9ab6`, including the
budget/watchdog and shared TTSIRT coordinate changes. Run 01038 localizes later graph
growth to retained-sample transport after fitting. The new masked marginal
and coordinate program preserves the grid-CDF extension and total pullbacks;
01041/01043 pass all nine checks on CPU/GPU. Run 01042 passes all six deferred
fitter checks on GPU, closing that focused qualification gap. The current
static guard passes 172 sources / 1,107 exact exceptions, with partial coverage.

Run 01044 completes its assembly assertions in 292.905 seconds (one passing
JUnit case), but the process exceeds the 300-second deadline. The 270-second
RSS high water is 12.50 GiB; final process high-water reporting was added to
the localization worker for the next attempt. Retry this one assembly test
under the existing 900-second ceiling. Runs 01046--01048 pass 6/11/25 public
pullback, sequential and TTSIRT checks on CPU; their GPU qualification is
pending because both GPU2 and GPU3 had unrelated active work.

Uniform log weights and weighted target means now use stable XLA helpers;
all 60 preparation checks pass on CPU (01045). Ten repaired preparation
wrappers were added to the static guard without new exceptions. The policy
group passes all 59 checks (01049). Finish the remaining preparation/callback
audit before freezing source/harness for terminal repeats. Avoid repeating
the broad preparation group until the host-memory regression is localized.

Run 01050 now completes the one-test assembly and exits in 354.963 seconds;
its final process high-water RSS is 16,332,076 KiB (15.58 GiB). The identical
baseline CPU diagnostic 01058 passes in 74.134 seconds at 690,176 KiB
(0.658 GiB). This descriptive single pair triggers graph/cache investigation;
it does not establish terminal timing ratios or isolated XLA overhead.

Uncommitted P72 preparation repairs normalize weights and assemble fit/guard
arrays in stable XLA helpers. Line interpolation, exact first-duplicate
selection and gather use separate compiled stages to preserve comparisons of
realized binary64 columns; the original frozen-design derivative boundary is
retained. Failed attempts 01051--01055 are preserved. All 30 CPU checks pass
in 01057; GPU checks remain pending. Static coverage is 172 sources / 1,108
exact exceptions and remains partial. Review/checkpoint this repair, then
continue the memory and remaining callback audit. Both approved GPUs were
occupied at recovery; do not interfere with unrelated work.

All F01--F20 terminal decisions remain open. Other carried blockers include
the predator-prey residual about 1.578e-9 against the fixed 1e-10 gate,
core-affine/higher-rank slowdown, centered qualification, incomplete callback
coverage, and missing terminal comparisons. The latest earlier comparison
had 18 valid pairs and 840 missing; this is not a current completion result.
No merge, repository-wide policy-compliance or default-readiness claim is justified.
