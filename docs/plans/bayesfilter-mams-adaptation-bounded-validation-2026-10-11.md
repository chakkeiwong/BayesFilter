# Bounded MAMS adaptation validation

Terminal status, 12 October 2026: the eight-target run completed within the
two-hour cap. Its actual evidence, negative mixture control, test outcomes and
settled budget are in the [result note](bayesfilter-mams-adaptation-feasibility-result-2026-10-12.md).

This executes D4R3 and prices the first D4R4 paths. The question is whether the
same discarded proposer survives contrasting target geometries, preserves
fresh qualification and every verified member, and exposes inadequate
posterior evidence honestly. It is an engineering and descriptive feasibility
run, not a method-ranking experiment. NUTS is excluded.

The comparator is the independently specified target law and existing public
tuning/retained APIs. The earlier nine-point SGQF grid is development context;
this run does not compare its costs against an adaptive method. Formal paired
comparisons remain a later priced design.

## Design and default audit

Use the existing exact Gaussian, rotated Gaussian (condition 25, angle 0.6),
Student t (5 degrees of freedom), exact noncentered funnel with unit scale,
analytic partially whitened funnel, and separated mixture (separation 5,
weight 0.3). The rotation and mixture are inherited validation fixtures; the
larger Gaussian condition is a declared stress hypothesis. The partial funnel
has the analytically specified residual curvature in the existing fixture.
LGSSM and nonlinear accumulation use their actual filter adapters and data.
No map is trained and no exact nonlinear likelihood claim is made.

The initial epsilon 0.1 and mean duration sqrt(d), domain [0.001, 4],
maximum count 128, six four-transition epsilon probes, 32 settling draws,
two 64-draw windows with 16 lags, factor-two epsilon bracketing and duration
factors [1, 2] are explicit bounded feasibility hypotheses. They are not
defaults or calibrated choices. The source-inspired acceptance target 0.9
and length coefficient 0.3 have the provenance in the parent plan. Unavailable
or conflicting IACT estimates retain the declared neighborhood.

Keep up to four primary proposals (two neighborhood hypotheses plus the two
diagnostic hypotheses when available) and reserve one repair for each. Use the
existing independent 64-transition acceptance trials, 32 repetitions per
stage, floor 0.5, separate search/verification alpha 0.05 and no endpoint
efficiency gate. The eight-candidate bound includes children. Pilot work is
capped at 2 million value/score/status row evaluations per target; qualification
at 40 million. Eight candidates times two stages times 32 trials times 64
transitions times four chains times 258 worst-case value/score/status calls
gives 33,816,576; the 40-million ceiling leaves an explicit retry allowance.
These are convenience cost ceilings derived from the maximum
count, four chains, trial counts and reserve; failure to finish is incomplete
evidence, not invalid MAMS. Record actual and conservative work separately.

All verified members receive a bounded public posterior assessment. To price
the path, use 256-transition chunks, minimum 512 warm-up and maximum 1024,
with the latest 256 for readiness; retained minimum 256 and maximum 1024.
These intentionally short diagnostic caps cannot promote the usual posterior
policy. R-hat thresholds retain 1.05/1.01; bulk/tail ESS 100 and lugsail
MCSE/SD 0.1 are explicit feasibility thresholds, not release thresholds.
Means and second moments are reported against exact analytic expectations
when known. Mixture mode-mass error is reported even if local diagnostics
pass. No short-chain failure becomes a tuning-membership veto.

## Evidence contract and stops

Pass requires correct duration/work identities, source-bound disjoint streams,
public-tuner export/reload of every verified member, valid finite target and
score, complete member denominators, and honest posterior outcomes. Shared
identity/correction/target defects stop dependent work. Candidate numerical
failure or posterior insufficiency preserves that outcome and permits the
next planned target. Runtime reports and unfinished members remain explicit.
Performance, IACT, posterior/reference errors and delivery proportions are
descriptive only: one seed per target provides no defensible method ranking.

Allocate at most 2 aggregate worker-hours for this diagnostic stage, including
retries and compilation; per-case boundaries get the remaining stage budget.
This is a suballocation of the existing campaign, not additional compute.
Conservatively reserve one additional hour for all unsaved local implementation
regressions since the last ledger snapshot. Together these reservations leave
147533.7646589242 seconds before settlement; retain the original campaign
overhead reserve. No unrelated old run is being resumed or charged twice.
The launcher records exact elapsed time; boundary stops preserve progress.
Do not extend beyond the two-hour stage allocation without replanning the
remaining existing grant.

Use GPU/XLA with verified growth before initialization; CPU execution is only
an explicit smoke or independent reference. Source closure is copied and
hashed before sampling. GPU 0 passed a trusted probe on October 11 at
12:40 UTC; readiness must be rechecked if launch is delayed. Output root:
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/`.
Entry point: `scripts/validate_mams_adaptation.py`, with explicit `--output`,
`--targets`, `--seed`, and `--max-wall-seconds`.

## Skeptical review

The plan avoids a weak method baseline by making no superiority comparison.
The short posterior policy prices and exercises stopping, not release
convergence. Unavailable IACT cannot nominate a numerically privileged
winner. Finite references for the filters remain necessary for D4R4 and are
not replaced by Gaussian moment checks. The mixture explicitly challenges
false reassurance from locally stationary chains. Ordinary elapsed-time
slowness can consume this bounded allocation without invalidating the method.

Review verdict: suitable for bounded engineering/feasibility execution once
focused tests and staged-source checks pass. D4R4's full SGQF/UKF/SIR/structural
paired confirmation, consumer tail/predictive observables and default promotion
remain open. The next allocation must use measured costs and independent
references; completing this stage does not close those tasks.

The execution review found that interruption could omit targets and verified
members from terminal reports. Before launch, write the planned target ledger,
persist every member before posterior work, recover created candidate IDs from
the public tuner's checkpoint, and classify typed pilot/budget failures
separately from shared defects. Expected pilot failure permits other targets;
shared defects stop dependent execution. A standard-library parent process
enforces the total wall ceiling, reserves 60 seconds inside it for cooperative
shutdown (a convenience allowance, not a scientific timeout), and finalizes
all denominators after abnormal child exit. Native TensorFlow calls remain
nonpreemptible until the process boundary. Built operator SHA-256 values and
framework versions are environment dependencies. The launch uses a coherent
source snapshot from the scoped commit, with those operators copied explicitly.
The pilot and retained-member chunk boundaries obey the enclosing allowance.
Their serialized wall caps stay at the fixed 7200-second stage maximum;
measured remaining time enters only the cooperative allowance, so machine
load cannot alter a random stream through a configuration hash.
