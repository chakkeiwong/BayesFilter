# Fixed fitted-APF matched cost and memory screen

Numerical qualification through04857 permits a descriptive CPU cost screen of
the repaired fixed-fit controller at4629f3bd8 against the original archived
adapter/kernel at4f0dfeb3d. Use exactly the Gaussian/nonlinear FP64 fixture and
live seed labels from `filter_gradient_fitted_apf_execution_20260929.md`. Both
arms include seed-table/stream handling, complete fitting, final analytical
score and the adapter's existing host result formatting. This is the callable
reached by the public score-study endpoints; physical-data generation and the
separate oracle are outside this measured scope. No reduced numerical path or
fixed-coefficient-only proxy is allowed as the timing comparator.

Use one fresh process per arm in this fixed order: Gaussian original,
Gaussian enclosing, nonlinear enclosing, nonlinear original. Instantiate the
same imported modules in every worker; select the archived original factories
only in the original arm. Do not compile a comparison owner in that worker.
Record adapter construction, first synchronized call and their sum, then three
conditioning and30 synchronized warm calls. Synchronize all final arrays and
verify exact replay without numerical comparisons inside the timed section.
Compare full shared output and fit/history records afterward at the already
declared FP64 tolerance2e-10, with exact seed/status/metadata identities. A
numerical mismatch vetoes interpretation and triggers localization.

Record VmRSS/VmHWM/RssAnon/RssFile/RssShmem, smaps_rollup, rusage and read order
before setup, after setup/cold/warm and after ordinary Python collection. These
are non-atomic observations; rusage alone is not a guaranteed peak. No forced
cache clearing or claims of native executable eviction. Record complete source
closure, fixture and exact archived reference hashes, environment, affinity,
platform, one-trace reuse, seed labels and commands. Primary measurements must
precede HLO export, comparison compilation and JSON serialization. The adapter's
ordinary result formatting stays inside both arms' measured boundary.

Allocate6 workers/1800 CPU seconds, no GPU seconds, under the existing total:
four measured arms, combined readback/policy and one localized retry. Use
`run_filter_repair_campaign.py test --group fitted_apf_cost_{model}_{arm}_cpu
--device CPU --test-timeout-seconds 300`; model is gaussian/nonlinear_scalar,
arm original/enclosing. Readback group is `fitted_apf_cost_readback_cpu`.
Use new numbered campaign artifacts and one numerical worker at a time. GPU
performance/capacity waits for unshared non-display hardware; prior GPU
numerical qualification does not provide that evidence.

The question is trustworthy before/after cost accounting, not a statistical
speed ranking. Cold ratio>1.25, warm ratio>1.10 or extra sampled host RSS>64MiB
are descriptive attribution triggers, not thresholds granting acceptance below
them. Replicated owner/capacity evidence is still required for terminal cost
acceptance. Budget exhaustion, changed input/source/environment, invalid
results or missing provenance stop this cohort for repair. No changes to
numerical source during the cohort, scientific controls, packages/system/cache,
HMC/training, canonical claims or main merge.

Skeptical review: importing a reference per invocation would repeatedly create
owners and bias its cost; load it once and preserve its original factory cache.
Counting only the final kernel would omit the repaired fitting work. Measuring
both arms in one process would contaminate compiler residency. This plan avoids
all three. Small N/T makes Python overhead visible but cannot predict large
filters, and a single process per arm cannot establish a general speed ranking.
The seed/fixture are unchanged mechanics hypotheses, not tuned defaults. Review
passes for this bounded accounting screen; acceptance remains a later gate.
