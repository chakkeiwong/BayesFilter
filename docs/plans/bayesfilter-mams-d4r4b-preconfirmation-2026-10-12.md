# D4R4b preconfirmation: reference compatibility and pilot-window information

Date: 12 October 2026. This executes the next dependencies in the
[generic repair](bayesfilter-generic-trajectory-adaptation-repair-2026-10-11.md)
after the [four-model cost result](bayesfilter-mams-d4r4a-full-model-pricing-result-2026-10-12.md).
It does not authorize a posterior comparison before its reference checks pass.
NUTS remains excluded. The existing MAMS proposer and public tuning authorities
retain their meanings; R-hat, ESS and MCSE remain posterior-only diagnostics.

## Questions and evidence roles

1. Do the archived independent posterior references still describe the exact
   finite-filter targets in the chosen committed source, despite broad package
   and tuner-source changes? The comparator is each archive's actual integrand,
   data, parameter chart, quantity definitions, two final integration orders
   and uncertainty record. A bridge requires an audited dependency argument
   and unchanged relevant numerical source, supported by current numerical
   parity; passing a few probes alone is insufficient.
2. Were the D4R4a length updates unavailable because two 64-draw windows with
   lag 16 lacked information? The comparator is the unchanged shared proposer
   under a predeclared window ladder on the same four full-data models and
   geometry hypotheses. The engineering endpoint is a complete record of
   reliable/unreliable estimates, generated/omitted proposals, numerical status,
   actual work and elapsed time. Usable estimates nominate proposals; they do
   not prove mixing or qualify a tuning candidate.

| Diagnostic | Role |
| --- | --- |
| Archived source, observation, chart, horizon, quantity and integration-rung identity | Pass criterion for reference compatibility; any unresolved mismatch is a continuation veto for dependent posterior comparisons |
| Changed likelihood/score/quantity math or unsupported source exclusion | Reference-bridge veto; regenerate the affected reference or investigate before comparison |
| Native finite/status/movement checks | Candidate numerical veto; shared target/source failure stops affected execution |
| Pilot estimate availability, lag limitation and adjacent-window agreement | Mechanism diagnostic and trigger for the prespecified longer rung; never tuning membership or posterior evidence |
| Pilot acceptance, proposed durations, wall time and work | Descriptive planning diagnostics, not rankings |
| Aggregate allocation exhaustion | Resource stop with full inventory; no automatic budget increase |

The research direction is not rejected when a pilot remains unreliable or a
candidate is numerically rejected. A corrupted archive, changed target or
unexplained dependency is a real continuation veto for the affected evidence.
All prior attempts, proposals and costs remain preserved.

## B0: versioned reference compatibility

All archived source copies currently match their recorded digests: 133 files
for PP-SGQF, 132 for PP-UKF, 133 for SIR and 131 for structural. Against
`9c3393669`, nine broad source files differ in each archive; diagnostic
reference modules/drivers are also absent from that committed cost snapshot.
The shared working tree is not the baseline for this audit.

Create a separate derived compatibility report. Never modify an old reference,
rewrite its source hashes or add a caller flag that simply skips strict checks.
Verify the archived code/data bytes first, then classify every old/current
source difference with exact file/function anchors. The candidate exclusions
are tuner/export changes and unused NeuTra training code. The changed
experimental SVD wrapper forwards `jit_compile`. The transitive trace shows
that the custom-gradient route **does call that wrapper**; it is not unused.
The current caller omits the argument, and both wrapper and callee default it
to `True`, so forwarding the default leaves this actual target route unchanged.
Verify those exact caller/default definitions in both archived and current
source before accepting that equivalence. Package imports and side effects
must be inspected, not inferred from filenames.

Require identical actual target/data/quantity computation or an explicit
mathematical equivalence argument. Compare full-horizon native values and
scores with the archived independent reference at a predeclared finite set
including the original reference parity points and current prepared starts.
Existing reference tolerances retain their original provenance; do not tune
them to the new outputs. Check the empirical refinement conditions from the
actual final two rungs, including complete eight-scramble rows and all declared
quantities. Preserve the absence of a rigorous integration-error bound.

A new diagnostic reader may consume a successfully bridged reference only
when its exact current-source/data/quantity identity matches. It cannot issue
a tuner artifact or make an old source snapshot current by assertion. Reject
wrong target or data, missing archive bytes, altered numerical dependencies,
unknown source differences, quantity-chart changes, missing integration orders,
incomplete replications and mismatched bridge identity in regression tests.
If compatibility cannot be established, report the specific affected target
and price a fresh reference; do not weaken the reference criterion.

## B1: bounded window ladder on all four full models

Use the same shared proposer and warm-start-only saved factors as D4R4a. Keep
initial epsilon 0.2, requested duration `sqrt(d)`, domain `[.001,4]`, count cap
128, six four-transition epsilon probes, 32 discarded settling transitions,
two frozen windows, coefficient 0.3, acceptance target 0.9, factor-three window
agreement, duration neighborhood `[1,2]`, maximum request `8*sqrt(d)` and at
most four proposals. These remain explicit hypotheses, not numerical defaults.
Only the diagnostic window length and maximum lag vary.

The first rung uses 256 draws/window and lag 64 on PP-SGQF, PP-UKF, SIR and
structural, in that order. If either source or slow-observable length estimate
is unavailable, that target enters a second rung with 1024 draws/window and
lag 256. The factor-four ladder is a convenience-selected information/cost
experiment preserving the estimator's existing `draws >= 4*lag` requirement;
it is not a calibrated stopping rule. Each rung restarts the same declared
preparation with a new recorded seed block. Never concatenate independently
reset windows and call them a continuous chain. Every within-rung frozen
window continues from its predecessor, as the existing proposer already does.

Price all frozen proposals for 32 transitions/chain after each rung, preserving
the same separation of proposal and price streams. Retain every completed,
omitted, failed and unstarted row from both rungs. No result here is a verified
tuning member. No posterior reference values, R-hat, ESS or MCSE may choose a
rung, nominate a candidate or change any search membership.

The first rung uses seed block 5400; the second uses 5600 with original target
indices preserved when a subset is run. These are disjoint convenience seeds,
not an uncertainty design or common-random-number claim. Extend the cost
consumer with explicit window/lag arguments and per-target seed indices,
recorded in its manifest and exact pilot configuration. Preserve the old
64/16 invocation as a historical executable configuration. Unit checks must
reject incompatible window/lag shapes and verify seed preservation for subsets.
The existing four-model tiny-path smoke remains the prelaunch source guard.

## Budget, commands and phase refresh

The settled balance is 151,019.615518 aggregate worker-seconds after the
preexisting overhead reserve. Suballocate at most 3,600 local diagnostic/test
seconds and two 7,200-second GPU supervisor allocations, including localized
recovery. This leaves 133,019.615518 seconds unallocated while reserved.
The bounds are research-allocation choices, not deadlock detectors or runtime
predictions. D4R4a measured about 722 total pilot seconds across four targets;
the first rung increases base diagnostic work about threefold, and the second
about elevenfold relative to that short design. New learned longer-duration
siblings can add substantial cost; the two caps preserve that uncertainty.
Run all first-rung targets before second-rung targets. Use no arbitrary
per-target kill when machine load varies.

All launches use a new versioned output under
`N5/generic-adaptation/preconfirmation-01/`, a committed immutable source,
trusted GPU/XLA execution and verified memory growth. CPU reference checks hide
GPUs and are explicitly independent diagnostics. The committed source must
contain the exact consumer and tests used by preflight; the earlier SIR
working-tree/snapshot mismatch must not recur. Before launch record the final
source commit, exact command and environment in the execution record.
GPU 1 is preferred. If occupied, an idle GPU 0 of the same hardware class may
execute this cost-only phase after a trusted readiness check; preserve the
actual device identity and contention context. This does not authorize a
cross-device speed ranking or change the aggregate allocation. The terminal
auditor requires the explicit expected host GPU. On 12 October, GPU 1 was
occupied by PID 1990789 from a separate NeuTra task; the trusted fallback probe
passed for idle GPU 0 (the same RTX 4080 SUPER class) with memory growth.

The implementable first-rung command is the existing cost consumer with new
explicit arguments (after its focused checks and scoped commit):

```bash
env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/price_mams_adaptation_full_models.py \
  --geometry-root <N5-root> --output <fresh-root>/window-256 \
  --targets NL-PP-SGQF NL-PP-UKF NL-SIR NL-STR --seed 5400 \
  --fixed-window-transitions 256 --max-autocorrelation-lag 64 \
  --plan docs/plans/bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md \
  --max-wall-seconds 7200
```

The source bridge and cost ladder are independent: a reference-provenance veto
blocks posterior comparison, not this explicitly cost-only diagnostic. At
terminal, charge occupied supervisor time once, audit all source/seed/work and
candidate inventories, release unused reserves and freeze the next actual
qualification/posterior design from measured costs. That design must fund
adequately searched fixed and randomized HMC comparators, the scoped proposed
methods, full model-coordinate/tail quantities, every retained member's
disposition and fresh paired-seed uncertainty. Use the existing public tuner;
reserve independent acceptance error for primary and repair candidates before
qualification. No hidden nominee or success-only cost average is allowed.

The D4R3 oscillation/antithetic, calibrated tail-quantity and model-coordinate
funnel controls remain prerequisites for a robustness claim. Finish these
focused controls before interpreting full-model results as a general repair.
N5 comparative confirmation and D4R5 guide/consumer consistency remain open;
LAPS and release promotion are not closed by this preconfirmation phase.

## Skeptical pre-execution audit

The comparison is now explicit: the archived broad closure is not mistaken
for an altered mathematical integrand, and a successful few-point parity
check is not enough to waive unknown dependencies. The current short-window
fallback is not described as learned adaptation. The first rung covers all
four models before selective longer work; selection uses only predeclared
pilot information availability and preserves every original record. All
numbers are inherited hypotheses, measured costs or disclosed allocation
choices, and none becomes a global default. The command must use the tested
committed snapshot, never the dirty checkout. The total bounds and true
continuation vetoes are explicit. Review verdict: suitable to implement and
execute within the existing authorization and budget, with posterior promotion
held until its separate reference and confirmation conditions pass.

Implementation review found that an explicit subset index could otherwise
name a different model while still passing range and uniqueness checks. The
consumer now requires each explicit index to equal that target's original
four-model position. The isolated four-model CPU suite passed all seven tests;
the narrow index regression passed after this correction. This closes the
seed-identity flaw before any rung launch.

Pre-mortem: longer windows may return usable but biased local estimates while
all chains remain in one region. The existing mixture negative control,
dispersed-start analytic controls and later independent references distinguish
that failure; this phase cannot claim global exploration. Conversely, a
lag-limited estimate or a clipped long-duration hypothesis is a bounded
diagnostic outcome, not evidence that the corrected sampler is invalid.
