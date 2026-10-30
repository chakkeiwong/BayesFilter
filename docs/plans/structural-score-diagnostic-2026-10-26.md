# Structural Model B score discrepancy diagnostics — 2026-10-26

## Research question

The completed fresh-data run established numerical validity and useful HMC
diagnostics, but it did not identify why the Model B score differs from the
independent references. This diagnostic asks which of three quantities is
responsible:

1. a derivative mismatch between the implemented finite particle program and
   the likelihood value that it actually returns;
2. particle-design and finite-particle variation; or
3. the weighted-particle reset/propagation approximation, including its
   derivative, relative to the nonlinear model's predictive law.

The target remains the declared Model B finite LEDH likelihood with the
production FP32/4032 configuration. This is a diagnostic extension of the
completed campaign; it does not change the HMC target, controls, or default
route.

## Evidence contract

* **Comparator:** for T=2, refined independent Gauss–Hermite quadrature of the
  exact nonlinear state-space likelihood; for T=50, the existing replicated
  bootstrap/Fisher reference is labelled finite-particle and biased-unknown.
  A same-program central finite-difference check is a derivative-consistency
  diagnostic, not an oracle.
* **Primary diagnostic:** on fixed observations and fixed draw design, compare
  the returned analytical score with central finite differences of the
  *returned finite-program likelihood* over a step ladder. A mismatch that is
  stable across steps and precision is an implementation defect.
* **Secondary diagnostic:** compare T=1 and T=2 using the same data prefix and
  draw design. The first increment is computed before any previous reset; a
  discrepancy already at T=1 cannot be caused by the reset at the preceding
  time. A new discrepancy at the second increment nominates reset/propagation
  or finite-particle effects, subject to replication.
* **Secondary diagnostic:** repeat the fixed target with at least four
  independent 4032-particle designs at FP32, preserving the exact data and
  controls. Report mean, sample SD, MCSE, and paired score differences. These
  are uncertainty diagnostics, not evidence of superiority.
* **Reset isolation:** evaluate the same next-observation predictive function
  from the weighted pre-reset cloud and the equally weighted post-reset cloud,
  carrying the existing analytical tangents and weights. This avoids the
  invalid comparison of a weighted cloud with a uniform cloud while dropping
  weight derivatives.
* **Veto diagnostics:** non-finite values, incomplete steps, invalid validity
  traces, missing fixed-design identity, or a derivative check that fails after
  step-size/roundoff analysis veto interpretation of downstream score gaps.
  ESS, acceptance, R-hat, and reset movement are explanatory diagnostics here;
  none is a stop criterion for this investigation.
* **Nonclaims:** these tests do not establish an exact T=50 oracle, posterior
  correctness, or superiority of LEDH over another filter. The T=2 quadrature
  is an exact-model reference only after its order refinement is recorded.

## Trace and implementation audit

Before running, inspect the complete call chain from `StructuralTarget` through
`make_batched_score_filter`, `structural_proposal`, `reset_cloud`, and the HMC
binding. Record whether the claim-bearing endpoint uses the shared analytical
recurrence, whether the score is a total derivative of the returned finite
program, and whether any Python or pfor loop enters the repeated kernel.
Any call-chain mismatch is reported as an engineering defect before numerical
interpretation.

## Execution stages

1. **Static audit and smoke:** verify branch/commit, environment, stable
   `tf.function` signatures, XLA mode, transport chunk selector, and source
   call-chain. Run the existing focused structural tests and a small FP64
   derivative smoke.
2. **Fixed-design derivative ladder:** for T=1 and T=2, use the fresh final
   data/design and production controls. Compare the returned score with
   central differences at steps `1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5` in q.
   Repeat the selected points in FP64 as a diagnostic only. Preserve all raw
   values and validity traces.
3. **Particle-design replication:** execute four independent fixed designs at
   N=4032, FP32, T=2 and T=50, with the same observations and exact controls.
   Use a fresh versioned output directory and do not overwrite the completed
   campaign artifacts.
4. **Reset isolation:** for the first two observations, compare the weighted
   pre-reset predictive expectation with the post-reset uniform predictive
   expectation, including total derivatives. Use this only to attribute a
   diagnosed discrepancy; it is not a replacement filter.
5. **Analysis and result note:** classify each finding as harness/implementation
   defect, numerical defect, finite-particle uncertainty, reset approximation,
   reference limitation, or unresolved. Include a decision table and an
   inference-status table. If a localized implementation defect is found, add
   a focused regression and rerun only the affected diagnostic under the same
   budget.

## Assumptions and budget

The production dtype and particle count are inherited owner directives
(FP32, N=4032); they are frozen for comparability. FP64 is a bounded diagnostic
arm, not a proposed default. Fresh data are the existing Model B confirmation
data, and designs are independent draws generated by the repository helper.
The diagnostic budget is at most 2 GPU-hours plus 30 CPU-minutes, within the
remaining completed-campaign allowance. Each stage has a 20-minute wall-time
stop condition and writes a failure record before any retry. No HMC retraining
or posterior claim is permitted in this plan.

## Skeptical plan review (before execution)

The plan passes review. The same-program finite-difference check is correctly
classified as a derivative consistency test rather than an oracle; the T=50
reference is not promoted to exact; reset isolation retains particle weights
and their derivatives; independent designs are paired on identical data; and
validity failures stop interpretation without turning ESS or energy error into
sampling criteria. The plan does not alter the production target or silently
promote a diagnostic setting.

## Artifacts

Close-out: all reviewed diagnostic stages are complete. The owner requested
a final report, commit, and push on 26 October 2026. See
[the final report](structural-score-final-report-2026-10-26.md). No further
experiment is scheduled; the unresolved T50 attribution is recorded for a
separate future accuracy study.

Results will be written under
`docs/plans/artifacts/structural-score-diagnostic-20261026-01/` with a run
manifest, source audit, raw stage JSON, command log, and a concise result note.
Existing campaign artifacts remain untouched.

## Reviewed reporting repair before the final focused rerun

The first complete FP32 run (attempt 11) and the FP64 T1/T2 run (attempt
13) completed. Review found two reporting defects in the new diagnostic
runner: the T50 ladder was computed but only one step was retained, and
the FP64 manifest inherited a hardcoded `float32` label. Neither changes
the numerical kernels. Preserve the original artifacts, record the latter
correction separately, and run only the T50 derivative ladder again with
all steps, raw perturbed values, and the baseline validity/fallback traces
retained. The existing budget and 20-minute stage bound apply.

Skeptical review: this rerun answers the outstanding long-horizon
derivative question without rerunning HMC or ranking filters. All steps
are disclosed, including unfavorable ones; no post-hoc best-step
promotion is permitted. A mismatch persisting across usable steps
triggers further derivative diagnosis, whereas roundoff growth as the
step shrinks is explanatory. FP64 CPU versus FP32 GPU is a joint
precision/backend check, not an isolated precision experiment. The
short-horizon FP64 result cannot certify T50.

The Fisher reference is checked in latent coordinates `(m_0:T,k_0)`,
where the prior is parameter independent and `k_t` is completed
deterministically. Conditional on the latent m path, only log-beta
enters the observation score. This is a different finite estimator
from differentiating the LEDH numerical likelihood; both aim at the
exact likelihood score as their approximations improve. Finite-reference
bias and long-path degeneracy remain unresolved.

One cheap final refinement repeats the explanatory CPU reset calculation
at 64 Gauss–Hermite nodes, retaining the original 32-node result. Its role
is to check whether the measured reset effect could be conditional
quadrature error. It does not select a production control. If the change
is comparable to the reported reset effect, the attribution remains
unresolved. The same budget and stop condition apply.

## Execution record and final review

**Status: diagnostic plan completed; production target unchanged.**
The source trace and 19-test structural suite completed. The isolated
worktree required a build of the existing Sylvester custom op. Early
diagnostic launches exposed import, device and eager/graph invocation
issues. These were repaired locally without changing production kernels.
Some failed attempts did not preserve complete stderr or elapsed time;
this is an evidence-recording defect, not a scientific result. Their
original manifests and partial evidence remain.

| Successful attempt | Work | Wall seconds |
|---|---|---:|
| 10 | Traced CPU reset isolation | 10.95 |
| 11 | FP32 derivative, refined T2 quadrature, four designs each at T2/T50, reset | 247.15 |
| 13 | FP64 CPU T1/T2 derivative ladders | 504.97 |
| 14 | Full retained FP32 T50 ladder and baseline validity traces | 121.41 |
| 15 | CPU reset conditional-quadrature refinement: 32 to 64 nodes | 14.03 |

Bounding missing failed-run duration by the next sequential launch's
start gives conservative totals of 1,927.3 GPU-enabled process wall
seconds and 802.6 CPU-only process wall seconds (including between-launch
gaps, excluding build/test time). These are not GPU-utilization
measurements. The completed diagnostics remain within the 2 GPU-hour /
30 CPU-minute budget. No HMC or neural training was run.

See `artifacts/structural-score-diagnostic-20261026-01/result.md` for
results and `source-audit.md` for call-chain and reference-score
derivations. `execution-record.json` records command/timing limits;
`metadata-corrections.json` corrects the original FP64 manifest label.
The runner now preserves the full T50 ladder, raw perturbations and
baseline traces, rejects reused output directories, and records future
launch success/failure. These are diagnostic repairs only.

Final skeptical review: FP64 T1/T2 consistency does not certify every
point or T50. The T50 FP32 ladder is roundoff-sensitive, with no
persistent large mismatch at usable steps. One-design T2 reset
attribution survives quadrature refinement but cannot be generalized
to T50 without replication. The reference's complete-data score
formulas pass derivation, but finite-particle bias and historical path
degeneracy remain unmeasured. The T50 persistence gap is unresolved;
good likelihood values and HMC diagnostics do not close it. No method
ranking or new default is established.

### Default and assumption audit at close-out

| Choice | Provenance/status | Failure mode | Diagnostic and limit |
|---|---|---|---|
| FP32 and N=4032 | Owner directive, frozen baseline | Roundoff and finite-N bias | FD ladder, FP64 T1/T2, four-design SD; no N-convergence proof |
| T2 controls: 16 flow/10 reset/epsilon=1/one correction; T50:16/5/1/one | Completed campaign, comparability baseline | Nonlinear prediction distorted despite restored moments | Weighted pre/post prediction; optimality/non-harm not established |
| Saved truth q/data/design | Campaign point, diagnostic choice | Off-truth failures missed | Explicit single-point limit |
| Four fresh designs | Bounded diagnostic sample | Unstable MCSE and invisible bias | Report SD/MCSE; no ranking claim |
| FD step ladder | Numerical diagnostic hypothesis | Truncation versus cancellation | Preserve every step and FP64 refinement |
| 32-node conditional GH | Convenience starting order | Quadrature error mistaken for reset effect | 64-node refinement changes score effect by at most 8.94e-8 |
| FP64 CPU/uniform reset/no XLA | Compatibility diagnostic | Precision/backend confounding | Explicit joint-check qualification; no default change |

Next justified research action is replicated reset attribution and a
stronger T50 reference, followed by a separately reviewed repair
comparison. Those are a new repair phase, not outstanding work in this
completed diagnostic plan. No background process remains.
