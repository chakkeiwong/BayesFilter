# NeuTra remaining-gap diagnosis and repair, October 1, 2026

Status: executed and terminally reviewed; three fresh posterior qualifications
passed. See `bayesfilter-neutra-gap-closure-results-2026-10-01.md`. The owner
requested code/math tracing, a repair plan, skeptical review, and execution.
This continues the authorized simpler-model campaign; it does not start q20.

## Checked findings and corrected hypotheses

The unresolved cases are mixture/Gabrié seeds 11 and 37 and wiggle/Gabrié
seed 23. The baseline is the preserved causal-repair terminal review and its
exact saved maps and training states. Historical failures will remain visible.

`TrainingBlock` calls `GabrieProgram` for 4 global-MH/MALA transitions on
64 persistent walkers and then differentiates `-mean(log q_phi(x))` at the
256 detached states. Global MH uses
`log p(y)-log p(x)+log q(x)-log q(y)`; MALA uses the reverse and forward Gaussian
proposal terms. For a fixed map each kernel preserves p, hence their composition
preserves p. This does not establish reversibility of their composition or
finite adaptive-run equilibrium. The Laplace proposal calibrates physical MALA;
it is **not** the global Gabrié proposal, which is the current IAF.

The first draft incorrectly treated missing burn-in/equal initial occupancy as
an implementation defect. Gabrié et al. (2022), section IV.B, equation (11),
explicitly fit the finite walker law rho_k; IV.C explicitly permits initial
samples that are not drawn from the target. Inspected author code:
`.localresources/flonaco-author-20260929/upstream/flonaco/training.py:101-127`
and `sampling.py:207-248`, pinned revision 6b9286b4e581. Paper text:
`.localresources/fab-coverage-followup-20260928/gabrie-adaptive-flows.layout.txt:309-345`.
**No mandatory mass-weighted start or burn-in repair is justified yet.** The
suspect is unmeasured finite training distribution versus optimizer stationarity.

The mixture factorizes as
p(x1,x2)=[phi(x1+5)/3+2 phi(x1-5)/3]phi(x2), with mean (5/3,0) and diagonal
variance (209/9,1). In standardized independent coordinates, an infinitesimal
conditional shift has log-density derivative u_j h(u_<j), and conditional
log-scale has derivative (u_j^2-1)h(u_<j). Their expectations vanish when the
first two moments match. A stack of diagonal-affine layers with permutations
therefore has a stationary forward-KL configuration despite the non-Gaussian
marginal. This is a stationary-point derivation, **not** a proof of a local
minimum or architectural impossibility. The saved Gabrié plateaus have FKL
about .947-.948, matching the development Gaussian baseline .94739. Earlier
SMC and Gabrié seed-23 maps show that this same architecture can escape.

The old finite checkpoints already exist and can be restored. The prior
controller deliberately stopped plateaus rather than continuing without
supported progress. The remaining gap is the absence of a discriminating
plateau experiment, not loss of checkpoint data. Unconditional continuation
until budget exhaustion would repeat that mistake.

Wiggle's saved first member (epsilon .25, L=3) passed numerical health,
R-hat <=1.01 and bulk/tail ESS >=400 at 10,000 retained draws/chain. Only
MCSE/SD failed (maximum .03913 versus .03). Nine other verified members were
not assessed downstream because qualification always chooses index zero.
The separate retry produced 59 receipts: 25 inconclusive conflicts, 13
inconclusive evidence, 9 higher-step, 9 lower-step and 3 trajectory decisions;
none was verified. It is not evidence of invalid scores or no usable kernel.
The frozen map's 1000-point median residual is 4.413, q95 20.736 and max 49.298;
all scale-slope alert fractions are zero. Residual geometry remains real.

## Evidence contract and research intent

Question: does correct finite sampling still leave the IAF near the Gaussian
stationary configuration, can target-specific initialization/continuation
repair it, and can another verified kernel meet wiggle posterior precision?
The exact comparators are the saved failed maps, a current-source continuation
with restored Adam/walkers, and fresh author-architecture initializations with
only the declared conditioner variance changed. Architecture, density, objective,
MALA, identity latent mass and posterior thresholds stay fixed.

Primary engineering checks are actual-consumer state/gradient/diagnostic tests.
Primary candidate screens are the existing physical shape/nonlinear-learning
screen, valid 1000-point probe, frozen Gabrié sampler screen, then sequential
HMC with unchanged R-hat/ESS/precision/reference requirements. Numerical
invalidity, corrupt source/inputs, absent memory-growth evidence and failed
required probes veto promotion. Invalid shared evidence or exhausted budget
vetoes further execution. A failed candidate permits the next declared repair.

Mode occupancy, valley frequency, global/local acceptance, conditional gradient
magnitudes, loss, score residuals and runtime explain failures; they are not
standalone posterior or whitening certification. The measure diagnostic uses
independent fixed-map chains and between-chain means, not an iid SE for all
correlated training rows. Actual training row summaries are descriptive.
No superiority, success probability, universal default or q20 result is claimed.

## Execution and numerical-choice audit

1. Diagnose the two saved mixture plateaus with the **actual** `GabrieProgram`:
   saved walkers and all-left/all-right starts; 256 discarded transitions and
   1024 retained transitions on 64 walkers. These powers-of-two are bounded
   diagnostic hypotheses (roughly 82k states/group), not convergence constants.
   Compare independent-chain means with the existing development reference
   and exact right-mode mass ~2/3 and valley probability Phi(7)-Phi(3).
   Save traces and the existing coarse reference screen. Check a forward-loss
   parameter directional derivative against central differences (1e-5, FP64;
   absolute tolerance 1e-6) using actual inverse-density code. Nonfinite or
   derivative failure is a continuation veto. Measure failure triggers a
   same-kernel longer 1024/4096 check before any optimizer interpretation.
2. Instrument actual Gabrié training batches with physical observable sums and
   counts; never treat correlated rows as iid. Keep 64 walkers, 4 transitions,
   batch 256, LR .001 and width 8 as inherited comparison settings. Preserve
   Adam and walkers in a pure 8192->16384->32768 continuation control for both
   failed seeds. A finite plateau at 32768 ends this control; it does not
   automatically continue to infinity.
3. If fixed sampling passes but the control remains stuck, run fresh matched
   initializations with canonical three-stage, two-hidden-layer ELU IAF,
   author block masks, conditional cap 2 and free scale bias. Compare the
   inherited variance-scale .02 with .2 and, if both fail, 1.0. These are
   initializer hypotheses, not new architecture/defaults: .2 multiplies
   initial weight SD by sqrt(10), 1 by sqrt(50), testing departure from the
   weakly coupled affine regime. Save exact configured maps. Rungs 4096,
   16384,65536 are work limits informed by seed-23's 65536-update pass, with
   plateau screening deferred to 16384 to distinguish early transients.
   If all fail, try width16 at variance .2 (existing capacity candidate), at
   the same rungs; then report unresolved, not an architecture impossibility.
   Run both seeds for a declared arm; stop their training when shape passes.
   Keep each attempt, standard probe and clipping counters. Then the existing
   frozen sampler, RKL refinement (reject and preserve warm if coverage fails),
   and fresh HMC qualification run for a viable candidate in fixed arm order.
4. Repair the HMC consumer to try up to three **verified** members in declared
   order (one earliest member per distinct L before additional same-L members).
   This is trajectory diversity, not efficiency ranking. First select using
   health/R-hat/ESS/precision alone, without opening the final reference. Once
   selected, apply the fresh reference check once; a reference failure ends
   that attempt rather than selecting another member on the same holdout.
   Preserve per-member kernels, seeds, full draws and failures. Use the public
   fixed-transport tuner from the checked capability registry. Initial search
   settings remain those in the earlier plan. If no pair verifies, repeat once
   with base measurement/verification 256 (was 64), rungs 1/2/4: finite extra
   evidence for inconclusive decisions, no acceptance-band relaxation.
5. Freeze a copy of numerical Python sources and launch from that copy on
   assigned GPU1, with TF_FORCE_GPU_ALLOW_GROWTH=true before imports and the
   repository growth verifier. Preserve exact work seeds and source identities.
   No causal claim compares a changed-source HMC stream with an old run merely
   because the top-level seed is equal. CPU tests explicitly hide GPUs.
6. Review every result and update the master next-phase record after each
   phase. Finish with a decision/inference-status table, resource accounting,
   source checks and a reset note. Keep numerical validity, candidate success,
   and research-direction conclusions separate.

The original ledger has 86516.875 GPU process-seconds and 176881.486 conservative
CPU core-seconds remaining. This cycle has a **sub-cap** of 6 GPU process-hours
and 12 CPU core-hours (bounded engineering allocation, not added funding),
including failed workers, at most 24 numerical phase jobs plus final-reference
jobs and two local infrastructure retries. Measured prior costs: 83.29s for
6144 additional mixture updates, 199.94s for 32768 additional Gabrié updates,
106.65s for one wiggle tuning+qualification. Expected cost is roughly 0.5-2
GPU hours, uncertain for new initializations and HMC. Worker limits are 1800s
fit, 1200s diagnostics/HMC, with ledger enforcement; timeout repairs may double
one limit only inside the sub-cap. Artifacts live under
`docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/gap-closure-20261001-r1/`;
phase directories remain uniquely versioned in the shared campaign `attempts/`
with the prefix `gap-20261001-`. The controller records their exact locations.
Environment: existing `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, two CPU
threads, TensorFlow/TFP FP64 benchmark path with XLA, memory growth, no installs.
Commands: `python scripts/run_neutra_gap_closure.py diagnose` followed by
`python scripts/run_neutra_gap_closure.py campaign` (same frozen source copy).

## Skeptical review

The first draft failed review: it presumed that initial occupancy was a bug,
confused saved checkpoints with scheduling, and proposed selecting successive
HMC members on one final reference. Those are corrected above. The revised
plan tests finite sampling before changing it, treats initializer changes as
hypotheses, uses both seeds and a matched inherited initializer, retains all
failed arms, and withholds final-reference values from kernel selection.

Premortem: a useful sampler may coexist with a poor flow; a useful flow may
fail the selected short trajectory; repeated attempts may yield a lucky pass;
new source hashes may change random streams; larger initial weights may create
numerical or clipping failures. The saved sampler traces, measured training
rows, matched controls, per-member results and source freeze answer these
specific risks. A passing three-case repair remains scoped evidence, not
reliability or default readiness. The revised plan passes skeptical review
for this bounded question; unresolved finite candidates are valid outcomes.

## Implementation review before launch

The actual qualification consumer now derives its diagnostic names from the
target, tries at most three verified members with distinct L values first, and
does not supply a final-reference callback to sequential sampling. It opens
the holdout once after selection and stops on a failed final check. Regression
injection at the tuner/member boundary verifies the second member is used after
the first fails, verifies exact epsilon/L, excludes unverified members, and
ensures no holdout read on failed selections. These tests establish consumer
wiring; the GPU run must establish numerical behavior.

Actual training observables accumulate inside the compiled block from the
detached sampler rows, not the unused placeholder pool. Checkpoints preserve
the optional initializer variance and unchanged canonical architecture. A
small numerical directional-derivative regression checks the inverse-density
objective and parameter restoration. The existing real fit/Adam continuation
regressions pass. Initial focused suite: 51 passed (87.90 seconds, explicit
CPU-only reference exception). The inherited temporal-policy implementation
hash matches its saved calibration; its documented within-chain drift
limitation remains, and sequential posterior checks remain mandatory.

The controller enforces the cycle sub-cap through the existing worker resource
limits while restoring the shared campaign's original allocation after each
job. It runs from a copy of all repository Python numerical modules. Repeating
an initializer arm for both seeds is required even when the first seed passes.
A failed finite measure screen blocks an optimizer explanation for that seed;
it does not establish an invalid algorithm. A shape-passing fit whose bounded
downstream checks fail remains unresolved, without an unlimited restart loop.

## Bounded continuation amendment after observed nonlinear progress

The executed width-8 seed-11 arms at variance .2 and 1 have escaped the Gaussian
plateau but reached the 65,536-update ceiling while still improving. The 1.0
arm's paired heldout log-density gain from 16,384 to 65,536 updates is .04939
with standard error .00283; the .2 arm independently receives the existing
`continue_checkpoint` classification. Valley mass remains .02136. This is
different from the .02 Gaussian plateau, which did not improve with extra work.
Treating all of these outcomes as exhausted training would repeat a demonstrated
stopping-rule error. The currently running capacity/initializer grid completes
first; its evidence is preserved and cannot be retrospectively reclassified.

If a mixture seed remains unresolved **at the fit stage**, inspect improving
checkpoints in predeclared order: width8/variance.2, width8/variance1, then
width16/variance.2. Select the first existing `continue_checkpoint` arm, not the
smallest descriptive loss. Restore its exact map, Adam, walkers, LR, clipping
and configuration. Run to 131,072 lifetime updates; continue to at most 262,144
only if the existing paired-progress rule still says `continue_checkpoint`.
The primary pass criterion remains the full existing shape/nonlinear screen;
the frozen-sampler, RKL and final HMC checks remain required afterwards. A
numerical veto, plateau, loss of supported progress, failed downstream candidate
or the finite cap ends this continuation. A previously posterior-qualified seed
is not rerun, and a downstream failure is not relabeled as a training-budget gap.

These powers-of-two are additional finite work hypotheses, justified by the
measured progress and the failed prior work ceiling. The recent 65,536-update
workers cost 352–394 GPU process-seconds; the added 196,608 updates for one seed
are priced at approximately 1,056–1,182 GPU process-seconds plus setup and
downstream checks. The existing 6-GPU-hour, 12-CPU-hour sub-cap and 24 numerical
phase cap remain unchanged. No additional funds or scientific direction are
introduced. Numerical workers use the same frozen Python source; the new
controller and amended plan are archived separately with their hashes.

Skeptical review: this amendment follows measured progress, does not continue
the failed Gaussian controls, does not lower a threshold or select on the final
reference, and has a finite endpoint and progress veto. More forward training
may still leave poor score tails; that is why the standard 1,000-point probe
and downstream HMC remain necessary. No assertion of reliability follows from
eventual success after multiple attempts. The amendment passes review for this
localized repair. Execute with `python scripts/continue_neutra_gap_closure.py`;
it resumes the initial finite program first and then handles this conditional
continuation through the same shared ledger.
