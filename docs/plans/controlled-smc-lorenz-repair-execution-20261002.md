# Lorenz controlled-SMC repair execution

Owner instruction: “continue with the repair”, 2026-10-02. This activates R1,
R2 and frozen-input R3 investigation in the root-cause audit plan. The scientific
target, data and inherited compute authorization are unchanged.

The immediate question is whether we can locate and repair numerical disagreement
without changing the fitted objective, and reject an invalid adaptive update at
its first failing operation while preserving healthy outputs. The comparator is
the saved GPU/XLA execution (108/111/121), independent SciPy SVD and Gaussian
algebra, with selected 80-digit calculations if needed. Healthy forward values
must retain the existing 1e-7 + 1e-9*abs(reference) tolerance and exactly identical
ancestors. No tolerance relaxation, ridge or new damping default is authorized by
this numerical repair. Full recursive agreement is required to close R2.

First-failure location, rejected proposal, last accepted policy and successful
refit count are primary R1 criteria. Finite Gaussian factors, fitted rank,
first-failure guards and ancestry parity can veto the implementation. Fit residual,
coefficient errors, spectral margins, mean displacement and runtime explain the
failure; none establishes filter quality. Failure of a candidate triggers the
next specified localization. Invalid inputs, an invalid reference, exhausted
budget or an unreviewed target change stop the affected experiment.

Implement and run these dispatcher phases, in dependency order where necessary:

1. `c1_lorenz_recursion_localization`: frozen 108/111/114 recursion replay,
   compare the first differing elementary operation and propagation. CPU-only
   independent reference; GPU intentionally hidden.
2. `c2_lorenz_first_failure`: stable-signature GPU/XLA consumer checks of healthy
   execution and synthetic invalid inputs, plus the saved failed Lorenz proposal.
   Guarded execution must preserve healthy values and stop before invalid
   weights reach softmax or resampling. Report rejected and accepted state
   separately. Test the actual controller-to-engine call chain.
3. Localized R2 repair and a fresh complete backward/forward check if the first
   phase identifies a repair justified by unchanged algebra. Register exact
   follow-up commands before launch. Continue frozen R3 attribution if numerical
   agreement is resolved or separately interpretable.

The six-attempt tranche has an aggregate cap of 30 CPU seconds and 300 GPU seconds.
Initial GPU allocations are R1 60 seconds, R2 120 seconds, R3 120 seconds; unused
time may be reassigned within the aggregate after recording measured compilation
cost and the next discriminating question. Compilation and failed attempts count.
This is within the existing campaign balances; it is not another 48-hour grant.
Versioned results go under the existing campaign root, beginning at attempt130.
The dispatcher supplies commands, source snapshots, input hashes, environment,
memory-growth policy, device, seeds, wall time and budgets in each manifest.

Default audit: FP64/TF32-off is the existing source-reference scope; XLA is the
GPU default and a stable CPU graph is an independent debugging comparison. The
frozen covariance floor0.001 and trigger are historical hypotheses under test,
not promoted defaults. SVD solves the same full-rank least-squares objective;
source normal equations remain a separate author baseline. Batch-wide rejection
is acceptable for R1, with affected rows reported. NaN or zero placeholders for
unexecuted history are never eligible scientific values; explicit masks identify
which times/refits actually completed.

Skeptical audit (2026-10-02): passed for this diagnostic scope. The old full
validation status can say PASS while all candidates fail; nested candidate
checks, not that envelope status, determine R2. The saved failing initial mean
is already influenced by upstream projections, so this replay cannot blame the
original unconstrained method for that mean. Independent fixed-time agreement
does not prove recursive agreement. Cheap filter adversaries (BPF, observation
APF, exact Kalman in linear fixtures) remain mandatory for later quality claims;
this tranche makes no superiority or complete-replication claim. No training,
Figure 6 efficiency ranking or new protection threshold is selected here.

## After first two attempts

130 localizes coefficient disagreement to backward time12, correction and
polynomial disagreement to10. Matching the GPU correction branch reduces but
does not eliminate it.131 passes31 GPU/XLA guard checks, including the actual
failed time0 proposal. Charged2.971CPU/24.071GPU seconds. Next registered phase
`c2_lorenz_recursion_capture` runs the complete healthy69 controller against its
pre-repair source snapshot and captures the frozen108/111 backward recursion on
GPU/XLA and a stable CPU graph (explicit diagnostic exception). Limit120GPU
seconds. Primary no-fire criteria remain unchanged; backend disagreement is
localization evidence, not an admission. Skeptical audit: healthy small-fixture
parity alone does not cover source-scale refits, hence this full-scope check.

132 found a full-scope no-fire coefficient failure (max A error3.56772e-7),
although both executions completed four valid refits. Guard promotion is vetoed.
The next capture retry records every field, ancestry and stage, then compares
forward and fit kernels on exactly common inputs. Diagnostic PASS means capture
completed; `guard_admissible` separately records the unchanged no-fire criteria.
This is a localized investigation under the same120-second launch cap and
aggregate300GPU cap, not a relaxation of tolerance or an accepted changed result.

Register `c1_lorenz_recursion_sensitivity` (20CPU seconds within aggregate30):
intervene on the saved next-policy input at times13,12,11, preserving valid
Gaussian distributions, and compare downstream errors. Check the analytical
identity d log integral = -E_tilted[dq] against the observed coefficient
difference. For121, evaluate whole-update alpha values0,1/1024,1/256,1/64,
1/16,1/4,.5,.75,.9,.99,1 against exact Gaussian KL, fixed-normal RK4 validity
and centered Bellman model error. These alpha values are diagnostic resolution
points, not selected defaults. There is no candidate ranking or chosen threshold;
holdout calibration remains necessary. The points include the unchanged old
policy and full failed update as controls. Skeptical audit: these saved normals
and clouds are diagnostic/training evidence only, not a fresh validation set.

## Guard candidate disposition after133/134

133 confirms identical healthy ancestry and passing forward outputs, but A/b
coefficient tolerances fail after adaptive recursion. The fail-fast kernel is
therefore an explicit `first_failure_guards=True` candidate; the default kernels
retain their pre-repair arithmetic. This is a failed non-harm evaluation, not a
claim that observability is useless. The default controller's repair is confined
to acceptance bookkeeping: a rejected refit keeps the last valid policy and
particle history and does not increment successful iterations. Full-scope
no-fire validation of this smaller default change is next.

133's comparison of one replica to all four saved replica columns was an invalid
broadcast comparison; those `recursion_comparisons` fields cannot support any
conclusion. Correct this with an exact four-replica capture, assert identical
shapes before comparing, and retain133 as an invalid harness field. Its separate
one-replica healthy comparisons remain valid.

135 is the final registered attempt of this tranche: `c2_lorenz_repair_validation`,
limit120GPU seconds within the unchanged300GPU aggregate. It tests full healthy
controller parity, rejected-refit bookkeeping, exact four-replica backward
capture and CPU-graph comparison. The aggregate diagnostic completion status
must remain separate from recursive numerical admission and full replication.

## Localized arithmetic continuation after135

135 validates the smaller default bookkeeping repair on the full healthy scope
and injected rejection. All saved four-replica GPU backward coefficients and
corrections now reproduce exactly; CPU graph A differs by0.5414, so independent
recursion remains unadmitted. Six attempts consumed6.094CPU/141.106GPU seconds.
Use up to two additional localized attempts under the SAME30CPU/300GPU caps;
this administrative retry change adds no resource authorization.

136 (`c2_lorenz_absolute_bellman`, <=90GPU seconds) evaluates an optional
diagnostic of the exact identity q_new=LS(-log g-L_new,next), fitting only the
future term and adding the analytical Gaussian observation quadratic. The
Gaussian integral uses a Cholesky solve and completed-square value. The same
full basis, frozen cloud, projection and four-replica scope are preserved.
The derivation is in the root-cause result; absolute Bellman equality holds
before projection because old q is in the LS span. After projection the same
coefficient increment enters the same projection routine. This changes the
finite arithmetic, not the intended regression objective. Compare GPU/XLA and
CPU graph at unchanged tolerances; passing only nominates independent SVD
validation. Do not adopt it as default or claim it repairs the635-mean failure.
Skeptical audit:135 showed exact historical GPU reproducibility, so a simpler
restart of that route cannot resolve independent sensitivity; evaluating the
mathematically equivalent target is the planned R2 discriminating repair.

## Final localized diagnostic repair after136

136 remained finite but failed every cross-backend comparison (A maximum
absolute difference0.26001). The absolute Bellman formulation is not admitted;
independent SVD validation cannot rescue its already failed backend criterion.
Use the eighth and final attempt for `c2_lorenz_failure_rows`, <=90GPU seconds,
within the unchanged aggregate300GPU cap. Static review found that a rejected
batch supplied placeholder values to later guards, incorrectly flagging healthy
rows. Prevent later stages from attributing new failures after the batch stops;
likewise do not interpret a skipped QR's placeholders as QR rank failures.

Skeptical audit: repeat all31 prior diagnostic guard checks and add two-replica
injections at proposal, particle draw, resampling and regression input stages.
Require the exact failed row/code, no resampling after rejection, and one trace.
The guarded candidate remains experimental because132/133 failed full healthy
coefficient parity; passing these tests cannot promote it. The accepted default
bookkeeping path and its numerical kernels are untouched by this repair.

137 failed before TensorFlow import: the isolated source snapshot omitted the
shared first-failure test module. Charge0.016630GPU seconds conservatively;
add that dependency to the snapshot and allow one infrastructure retry138 under
the same30CPU/300GPU aggregate and90-second phase total. No numerical evidence
was produced by137. This localized retry is authorized by the campaign policy.
Synthetic guard fixtures use stateless seeds20261002/1 and20261002/137; earlier
repair manifests' generic frozen-input seed description was incomplete for those
fixtures. The exact seeds were already preserved in source snapshots.

## Terminal status

Completed through138; see [result](controlled-smc-lorenz-repair-result-20261002.md).
135 validates the smaller default bookkeeping change in20 checks;138 passes39
focused guard checks. Broad guard admission and full independent recursion remain
failed.136 is a completed diagnostic, not an admitted Bellman reformulation.
Charged6.094010CPU/176.247307GPU seconds under the unchanged30/300 caps.

Scope clarification for134: alpha scales the joint time0 A/b/c increment while
the downstream policy is held fixed. This is not a full damped backward sweep.
No KL threshold, new default, stochastic ranking or empirical replication is
claimed. The next work is high-precision local recursion and complete-update
calibration as specified in the refreshed master; no numerical process is left
running and the current bounded diagnostic stage is closed.
