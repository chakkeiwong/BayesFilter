# Lorenz bounded covariance repair: terminal result
Updated 2026-10-02 after attempt126. Stage complete; NO VALIDATED REPAIR.
The terminal decision and next repair question appear below the preserved history.

Subsequent [code/math audit128–129](controlled-smc-lorenz-root-cause-result-20261002.md)
localizes the extreme mean before time0 projection and explains why the
covariance floor fails to control it. This supersedes the earlier unspecified
mean-failure hypothesis below; historical results and rejections remain valid.
The [repair plan](controlled-smc-lorenz-root-cause-audit-20261002.md) gives the
next ordered phases.

## Historical result after104
The covariance repair was not validated.

Attempt97 created nine independent SVD fixtures. Attempts99/101 found relative
XLA eigenpair residual 7.83e-9 despite FP64; setting the exposed epsilon did not
fix it. Native batched Jacobi refinement in102 reduced the residual to1.57e-15.
The0.001 relative floor passed all fixed-input projection/integral and healthy
non-harm checks; smaller floors failed the integral check. Attempt103 finished
all four saved rows through four refits, with healthy output bitwise unchanged
and every graph tracing once. Maximum relative precision changes58.5/118.7
are large. These are finite-execution observations only.

Attempt104 failed six independent coefficient checks, all row0 refits3/4.
All rank and fixed-input forward checks passed. Maximum A differences were
0.000201 and2.480. The cause is not yet established; localize propagation
through the Gaussian correction before further repair.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain optional eigensolver refinement | Fixed-input residual/projection pass | None in fixtures | Wider shapes/conditioning | Check through consumer | General eigensolver certification |
| Reject covariance repair admission | Full backward parity fails | Six coefficient checks | Cancellation or changed policy | Localize and repair | Replication or efficacy |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Full coefficient parity fails |
| Statistically supported ranking | None |
| Descriptive-only differences | Four frozen rows finish; projection changes large |
| Default readiness | No; baseline unchanged |
| Next evidence needed | Full backward parity, fresh seeds, conditional heuristics and cost |

Evidence: artifacts/controlled-smc-kdm-replication-20260928/attempt97_c1_lorenz_bounded_reference,
attempt102_c2_lorenz_bounded_refined_calibration, attempt103_c2_lorenz_bounded_safety,
attempt104_c1_lorenz_bounded_validation under the same root. Manifests preserve
commands, sources, seeds/designs, device settings, wall time and logs.

After104:104 attempts/ceiling106. Tranche remaining104.466641 CPU/3421.526325 GPU
seconds; inherited357.633577 CPU/116372.826676 GPU seconds. Bounded-stage use
14.774140 CPU/112.317348 GPU seconds of90/1800. No new compute grant.

Post-run skeptical review: fixed-input projection success did not establish
full-recursion parity. Accumulated cancellation is a hypothesis, not a checked
cause. Four frozen rows cannot establish statistical efficacy. Both full
controlled-SMC and KDM replications remain incomplete; iAPF is deferred.

## Continuation through114 (2026-10-02)
105 localized first mismatches immediately after projections.106/107 verified
the algebraically equivalent integral against80-digit raw Gaussian calculations
and the GPU kernel.108 finished every row with healthy output unchanged.
109 removed all third-refit failures but retained three fourth-refit failures.
110–112 found every identical-target SVD/QR and fixed-increment integral check
passing; the80-digit comparison also diagnosed cancellation in the independent
reference.113 failed only because a snapshot dependency was missing;114 repaired
that harness failure and tested the improved reference. Full parity still
failed A,b,c in row0/refit4 (maxA difference0.098895, scaled273.124).

Decision:0.001 remains rejected. Local arithmetic correctness and finite
recursion do not establish global reproducibility. A stronger conditioning
rung is the next candidate, selected solely by numerical agreement/non-harm.
No ranking or scientific efficacy conclusion is supported. The alternative
explanation is accumulated roundoff/sensitivity; the local crossover supports
this direction but does not prove it is the sole cause.

## Terminal result after126 — 2026-10-02

The bounded spectral-covariance repair evaluation is complete. **No candidate
is admitted.** The TensorFlow implementation and its independent references are
executable, but their fixed-input successes do not establish a usable adaptive
filter. No Figure6 or efficacy run was launched from these results.

| Candidate | Healthy trajectory | Failed-design / independent result | Decision |
|---|---|---|---|
| Coupled floor0.001, refined eigenbasis and stable integral | Bitwise unchanged |114: row0/refit4 A,b,c fail full parity; max A difference0.098895 | Reject numerical admission |
| Coupled floor0.01 |117: three activations change healthy output | Finite failed-design run cannot override non-harm failure | Reject |
| Coupled floor0.1 |116: changes a healthy fixed fixture | Full controller ineligible | Reject |
| Activation64epsilon, correction floor0.01 |121: bitwise unchanged, zero activations | Refit4 forward states/weights become nonfinite | Reject |
| Activation0.001, correction floor0.01 |126: bitwise unchanged, zero activations | Refit4 forward states/weights become nonfinite | Reject |

119/120 and124/125 pass independent80-digit Gaussian, shared GPU projection,
activation-boundary and eigensolver checks. Calibration is a necessary check,
not the admission criterion. Both final full-controller runs retain one trace
for each of six graphs and a307898112-byte TensorFlow allocator peak. The
runtime remains native batched TensorFlow/XLA; NumPy is confined to independent
CPU references. The measured failures do not establish Python-loop, retracing
or NumPy use in the runtime call chain.

The strongest newly checked failure mechanism is the forward transition.
122 locates the first nonfinite weights at refit4/time0 and first nonfinite
states at time1 for row0. Policy coefficients remain finite and the minimum
forward precision margin is0.0001631, but the initial cloud reaches637.05 in
absolute coordinate value. In123, both the unchanged author's RungeKutta4.m
and the independent NumPy implementation produce4096 nonfinite coordinates
on this same512-by8 cloud. Both are finite on the preceding refit's cloud
and their finite values agree at1e-10+1e-12|reference|. The original transition
source, exact inputs, command and output are preserved in123.

This is an independently reproduced fixed-step transition overflow caused by
the candidate's extreme proposed states. Positive definiteness is insufficient
to prevent it. We have not proved which projection/linear-coefficient
interaction first creates the extreme mean, nor that the same coordinates
explain every failure of126. It would be wrong to call this an XLA-only fault,
to change RK4 silently, or to infer that the paper's theory is false.

The target of these tests is an explicitly constrained policy under the same
state-space model. It differs from the author's unconstrained full-quadratic
fit whenever protection activates. Paper§6.2, equations43–44 and footnote4
require integrable Gaussian twists and discuss SPD projection; they do not
specify our relative floors or activation rules. The pinned source
cSMC_Resample.m:187–234 constrains its diagonal branch but leaves the full
quadratic branch unconstrained. The transition anchor is RungeKutta4.m:12–21,
with interval0.1 and Lorenz drift in cSMC_VaryF.m:11–16,94–95.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close this bounded spectral-repair stage | No candidate passes complete admission | Healthy-change, full-parity or nonfinite-forward veto for every arm | Can a jointly bounded mean/covariance update retain useful adaptation? | Derive and audit a distinct complete-update constraint | No rejection of controlled-SMC theory |
| Retain optional eigen/integral repairs as research code | Fixed-input identities and high-precision checks pass | Whole-filter admission remains blocked | Wider domains/shapes not tested | Reuse only with a fresh complete-consumer check | No production/default promotion |
| Preserve the published model | Author and independent RK4 reproduce the overflow | Changing step size would change the specified map | Robust behavior of an alternative policy | Keep RK4 fixed in the next candidate | No integrator repair has been adopted |
| Keep Figure6 blocked | Numerical admission absent | Promotion veto; no valid method to benchmark yet | Cost and conditional efficacy still unmeasured for an admitted candidate | Numerical repair first, then fresh seeds and cost | No full empirical replication |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Every tested safeguard rejected;126 nonfinite |
| Statistically supported ranking | None |
| Descriptive-only differences | Activation counts, magnitudes, runtime and finite completions |
| Default readiness | No |
| Next evidence needed | Full independent backward/forward parity; untouched seeds; conditional heuristic controls and cost |

The next scientific question is narrower than simply trying another floor:
can an update keep both covariance and induced Gaussian mean within a
calibrated safe neighborhood of the previous valid policy? Feasibility-only
backtracking was already rejected in78/79 (mean about412 and overflow), so
repeating it is not a new repair. A Gaussian-displacement or KL constraint on
the complete A,b update is a hypothesis requiring a separate derivation,
numerical calibration, source classification and evidence contract. It is
not implemented or validated by this stage. No convenient trust radius should
be adopted without a non-harm calibration and explicit bound/units.

### Provenance and terminal review

Evidence root: artifacts/controlled-smc-kdm-replication-20260928.
Attempts115–118 record the stronger coupled floors;119–123 record the
feasibility trigger and author transition crossover;124–126 record the last
bounded condition trigger. Full independent trigger/margin validation was
ineligible because neither candidate survived; those phases were not run.
118's earlier zero-survivor completion performed zero full comparisons.

Commands used the stable approved dispatcher
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase PHASE`.
Each attempt manifest records the expanded command, source snapshot/hashes,
environment, fixed69/76 designs, wall time, hardware and output paths.
GPU jobs used escalated device access, FP64, TF32 off, XLA and verified growth.
CPU references intentionally hid GPUs.

A historical metadata defect is retained transparently: the top-level
relative_floor_hypotheses field in115–118 repeated the initial grid. Their
commands, calibration files and summaries preserve the actual0.01/0.1 grid.
The dispatcher now records the appropriate floor grid and activation margin;
no old manifest was overwritten. This does not change their verdicts.

Terminal reporting now distinguishes diagnostic completion from admission,
includes nested full-validation results and refuses a zero-survivor launch.
A checkpoint bug caused lexicographic ordering to show99 after126; campaign
attempts are now ordered numerically. Recheck the final checkpoint and report.

Budget after126: bounded stage used73.325226 CPU and359.019730 GPU seconds of
90/1800. Current tranche remaining45.915555 CPU/3174.823943 GPU seconds.
Inherited remaining299.082491 CPU/116126.124294 GPU seconds (about32.26 hours).
The attempt ceiling130 grants no extra compute. No job remains running.

Post-run skeptical review: preserved healthy outputs alone cannot establish a
useful filter; a local high-precision oracle cannot establish recursive
agreement; diagnostic PASS is not candidate acceptance. The strongest
alternative explanation for114 remains accumulated reference/runtime rounding,
and the same-input crossover limits what can be blamed on either solver.
123 gives stronger independent evidence for the later forward overflow.
A complete candidate passing both full independent parity and a finite forward
run would overturn the current rejection for that new candidate. No viable
safeguard remains under this stage's evidence. The research direction remains
open; the full controlled-SMC and KDM replications remain incomplete.
