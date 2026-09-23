# Matched cost measurements for the SVD correctness repair

Question: what compile, warm-call and memory cost does magnitude-normalized,
binary64-convergence SVD add to the complete rectangular SRUKF value route,
and how does XLA compare with graph execution? These measurements concern the
new SVD correctness repair, not a new filtering algorithm or analytical score.

Freeze f4ea46dde as the pre-SVD repair mechanism baseline and the current source
as candidate. Test prior graph, prior XLA, current graph and current XLA in
separate fresh processes on identical linear 3-state fixtures at horizons1/3.
Pin baseline factor and filter modules; the unchanged cubature dependency must
have an exact matching Git hash. Current graph mode is an explicit reference.
Inputs, callbacks, rank cutoffs and support tolerances are identical. Returned
likelihood, mean, covariance, rank and support are checked against independent
closed-form Kalman after costs are recorded. Never rank speed for an inaccurate
arm; preserve both its numerical failure and descriptive resource observations.

Capture construction/trace/compile+first-call time,20 synchronized warm calls,
changed observation operands and return-to-original. Record live/peak allocator
bytes, process RSS/HWM/PSS and graph/HLO size. Snapshots cannot establish precise
native peak or eviction; the existing process-lifetime containment work remains
separate. Observe sharing during GPU execution and reject shared-device cost
comparisons. Use the same physical UUID for all matched GPU arms/repeats, verified
memory growth and recorded thread/TF32/environment settings.

Reserve at most48 numerical workers plus two short analyzer checks /14400
charged seconds under unchanged32CPU/52GPU
campaign caps. Start with CPU pilots, then three fresh-process observations per
arm/horizon/backend where budget/device availability allows. The resulting
continuous differences are descriptive, not statistically proven superiority.
Existing investigation triggers stay: cold above2x, warm above20%, device peak
above2x, host extra256MiB or2x, and continuing warm growth. A trigger requires
attribution and a recorded lifecycle/capacity disposition; it cannot be waived
by passing numerical tests. Save every run in the existing numbered campaign
root and preserve all failures. Stop for invalid provenance or exhausted budget.

Review: timing different outputs would falsely credit an inaccurate baseline;
therefore every arm receives independent numerical validation, and failed-arm
costs cannot support ranking. Timing inside repeated public recompilation would
confound SVD with owner lifecycle; build one explicit operand-bound outer filter
function for each arm and state that boundary in the result. Public standalone
call overhead remains covered by terminal endpoint costs, not this microstudy.

Commands use the established interpreter and runner:

```text
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts/run_filter_repair_campaign.py matrix --stage tests --test-batch svd_cost_cpu --test-timeout-seconds 300 --repeat 0
```

The CPU pilot is repeat 0 and counts toward the three observations if sources
stay unchanged. After inspecting its raw numerical and resource evidence, use
repeat 1 and 2, then the corresponding `svd_cost_gpu` batches with trusted GPU
access. Pin the selected GPU index for the subsequent GPU repeats; match its
UUID in the result analyzer. All inputs are fixed deterministic fixtures, so
random seeds are not applicable. No target tuning is performed.

The analyzer independently recomputes numerical eligibility from archived
outputs, checks fresh-process coverage, source/environment agreement and GPU
growth/sharing provenance, and suppresses comparison ratios for failed arms.
Its adverse checks cover false pass flags, nonfinite output, changed rank or
support, wrong shapes and shared-device preflights. A one-repeat pilot is
explicitly labeled as such and cannot be substituted for the complete cohort.
