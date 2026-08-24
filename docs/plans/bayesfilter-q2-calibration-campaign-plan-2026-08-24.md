# Q2 Calibration Campaign Plan (2026-08-24)

Governing program: `bayesfilter-ledh-canonical-completion-program-2026-08-24.md`
(Q2). This plan pre-declares the per-curve evidence contracts before any
run, per the program's Q2 contract and the Class-C justification rule.
Budget: ~1 GPU-day total, per-process cap 100 minutes, stop after 3
consecutive launch failures. Output root:
`docs/benchmarks/q2_calibration_20260824/` (fresh versioned directory per
launch, never overwritten). Environment: tftwogpu conda env;
`CUDA_VISIBLE_DEVICES=1` (4080 SUPER first per owner directive); TF memory
growth verified before initialization, fail closed otherwise.

## Curve 1 — annealed-SMC k/c response surface (FIRST: harness exists)

Question: which stage count k and flow-prior cap c hold per-step minimum
stage-ESS on the frozen Austria scope, at the production-target lane
(float32 + TF32, GPU), across seeds — so the canonical defaults carry
calibration provenance instead of the probe's single-seed CPU result.

Mechanism under test: the within-step annealed telescope of
`canonical_value_and_diagnostics` (`annealed_resampling=True`,
`temper_stages=k`, `flow_prior_cap=c`) — the claim-bearing filter
implementation, not the probe's inline reimplementation (call-chain rule).

Grid: k in {1, 2, 4, 8} x c in {8.0, inf} x seeds {0, 1, 2}, N=1008,
frozen Austria observations, flow_substeps=16, float32+TF32 on GPU; plus
one float64 CPU anchor cell (k=4, c=8, seed 0) for cross-lane validity.

Evidence contract:
- Hard veto (any cell): non-finite value, non-finite weights,
  `program_valid` false, or crash. A veto stops interpretation of the
  surface, not the campaign (repair trigger: reproduce the cell at
  float64 CPU to classify precision vs harness).
- Promotion criterion (calibration selection rule, pre-declared): the
  recommended default k is the SMALLEST k whose minimum per-step
  stage-ESS fraction is >= 0.30 in every seed at the recommended c.
  Provenance of the 0.30 threshold: the standing annealed-mode gate in
  `test_austria_annealed_mode_holds_takeoff_ess` (between the probe's
  frozen-target 59-88% and the plain-mode measurement). c is selected by
  the same rule; if both c pass, the less intrusive c=inf (no spectral
  cap) is NOT automatically preferred — the cap is a Class-C
  numerics-altering protection, so c=8 retains its recorded repair-arm
  provenance and c=inf must show non-harm (comparable stage-ESS and no
  veto) to be recommended.
- Descriptive only: value differences across k and c (the telescope is
  unbiased for each k; no ranking language), runtimes, ESS spreads
  across seeds beyond the threshold rule.
- Not concluded even if green: posterior correctness, HMC readiness,
  statistical superiority of any k over another beyond the threshold
  screen, transfer of the calibrated k/c to any other model or scope
  (per-scope tuning rule).
- Artifact: JSON per launch with run manifest (commit, command, env,
  device, TF32 status, memory-growth status, seeds, wall time) + a
  result note appended to this plan.

Pre-mortem: (a) f32/TF32 could NaN where the probe's f64 did not — that
is exactly what the hard veto + f64 anchor cell discriminate; (b) the
surface could pass at k=1 making annealing look unnecessary — the
repair-arm evidence (2026-08-21/22) says plain-mode collapses at
takeoff, so a uniformly-green k=1 would indicate a fixture mismatch
(too-easy observations), checked against the recorded plain-mode
baseline before accepting; (c) ESS is a proxy — it nominates defaults
under the stated rule and cannot certify statistical validity.

## Curve 2 — trust-radius model-trust curve (harness to be written)

Predicted vs actual residual reduction across a radius ladder on the
frozen Austria dual-cap scope; acceptance is non-harm (healthy
trajectories identical, pathological ones bounded and flagged), never
primary-metric improvement. Deferred to the next launch window in this
campaign; contract to be finalized in this file before its first run.

## Curve 3 — LM-damping bias-vs-robustness curve (harness to be written)

Same acceptance form as Curve 2. Deferred; contract before run.

## Curve 4 — relative-ridge derivation (analysis + small verification run)

Worst-lane effective-epsilon x safety factor, relative form
delta*tr(C)/d; derivation note first, then a verification cell. Deferred;
contract before run.

## Curve 5 — dual-cap constants (0.98/8, 2.0)

Attach owner rationale from the 07-xx notes if it exists; otherwise run
the Curve-2 protocol at those constants. Deferred; contract before run.

## Curve 6 — float32/TF32 production-lane decision

Consolidates the Part-5 battery evidence plus Curve-1's cross-lane
anchor; decision note, no new large runs expected. Deferred.

## Curve 7 — Austria Fisher-identity gate (GPU replication)

The CPU Fisher gate exists (with the 2026-08-24 non-vacuity guard); this
replicates at GPU scale. Deferred; contract before run.

## Skeptical audit (pre-launch, Curve 1)

Baseline is the recorded repair-arm and probe evidence, not a strawman;
the promotion criterion is a threshold screen with recorded provenance,
not a proxy-metric ranking; stop conditions and budget are stated;
the harness calls the claim-bearing filter entry point; the fixture is
the frozen Austria target used by the passed probe contract; f32/TF32
vs f64 is discriminated by the anchor cell. No material flaw found;
proceeding with a single-cell smoke to measure wall time before the
full grid launch.

## Curve 1 — RESULT (2026-08-24)

Artifacts: `docs/benchmarks/q2_calibration_20260824/`
`kc_surface_f32tf32_full_1787545211/result.json` (24 cells),
`kc_surface_f64cpu_anchor_1787545242/result.json`,
`kc_surface_f64cpu_veto_classify_1787550172/result.json`.
Commit: 5a0a4684 lane. Wall: ~7.5 min GPU grid + 2 CPU cells.

Observed surface (min per-step stage-ESS fraction, N=1008, 3 seeds):

| k | c=8 (seeds 0/1/2)     | c=inf (seeds 0/1/2)      |
|---|------------------------|--------------------------|
| 1 | .010 / .011 / .014     | NaN / NaN / NaN (veto)   |
| 2 | .092 / .095 / .122     | NaN / NaN / NaN (veto)   |
| 4 | .444 / .501 / .510     | NaN / NaN / NaN (veto)   |
| 8 | .811 / .826 / .843     | .711 / .645 / NaN (veto) |

f64 CPU anchor (k=4, c=8, seed 0): 0.454 — agrees with the f32/TF32
cells at the same coordinates (cross-lane validity at the anchor).

Veto classification (pre-declared repair trigger): k=4/c=inf at f64 CPU
is VALID but ESS-collapsed (0.040). So the uncapped flow prior fails in
two separable ways: NaN on the f32/TF32 production lane (precision-lane
effect) and a lane-independent ~10x stage-ESS degradation at f64. The
k=1/c=8 cells reproduce the recorded plain-mode takeoff collapse
(~1%), confirming the fixture discriminates (pre-mortem check b).

Decision table:
- Decision: calibrated defaults k=4, c=8 for the frozen-Austria
  canonical annealed lane (per-scope; not transferable per tuning rule).
- Primary criterion: PASSED — k=4 is the smallest k with min stage-ESS
  fraction >= 0.30 in every seed at c=8 (rule pre-declared above).
- Veto status: hard vetoes confined to the c=inf arm; the c=8 arm is
  veto-free in all 12 cells. The vetoes are themselves the Class-C
  evidence: the OFF setting of the spectral cap (c=inf) is now
  MEASURED as harmful (NaN on the production lane; 10x ESS loss at
  f64), so the cap's on-state carries calibration provenance, not
  convenience provenance.
- Statistical status: k=4 and k=8 both pass the screen; k=8 is
  descriptively higher-ESS at ~2x wall cost; no superiority ranking is
  made (3 seeds, no uncertainty analysis; the selection rule is a
  threshold screen, not a ranking).
- Main uncertainty: 3 seeds; tail behavior of stage-ESS beyond min;
  transfer to other models/scopes is explicitly NOT concluded.
- Next justified action: Curves 2-5 (trust-radius, LM damping, ridge,
  dual-cap constants) under their to-be-finalized contracts; Curve 6
  gains the cross-lane anchor row from this curve.

## Curve 5 — RESULT (2026-08-24): owner rationale located and attached

The dual-cap constants (pairwise_particle_rms_cap=2.0,
coordinatewise_standardized_cap=0.98, cap power 8, pairwise steps 4) are
recorded in `bayesfilter-genut-dual-cap-monograph-ready-spec-2026-08-07.md`
as an owner-directed algorithm/default family policy: "the strongest
single maintenance compromise across the tested LGSSM, KSC SV,
predator-prey, and Austria SIR scopes", explicitly NOT a claim of
universal statistical superiority, with scope-specific tuning still
required for pairwise strength and route controls. Under the Class-C
justification rule this is a recorded owner rationale with cross-scope
evaluation provenance — the justification form the registry's C4 entry
required. Residual (recorded, not blocking): the value-level ladder
(e.g. 0.98 vs alternatives) has no dedicated calibration curve; if a
future failure implicates these constants, the Curve-2 non-harm
protocol applies to them at that scope. Registry C4's dual-cap row can
be marked closed-by-provenance.

## Curve 6 — RESULT (2026-08-24): production-lane decision note

Decision: the production target lane for the canonical LEDH annealed
program on the frozen Austria scope is float32 + TF32 on GPU, with the
spectral flow-prior cap MANDATORY on this lane (c=8 calibrated, Curve
1). Evidence consolidated: (i) owner directive (repo CLAUDE.md default
execution target); (ii) Part-5 battery — f32/TF32 arm green across the
historical disease classes and fail-closed under the exact historical
NaN-escape combo; (iii) Curve-1 cross-lane anchor: f32/TF32 k=4/c=8
stage-ESS (0.444-0.510) agrees with f64 CPU (0.454) at the anchor cell;
(iv) Curve-1 veto classification: the f32/TF32 lane NaNs without the
cap where f64 degrades but survives — so the lane decision and the cap
requirement are coupled, and the cap is part of the lane definition,
not an optional extra. f64 CPU remains the reference/anchor lane.
Not concluded: posterior correctness, HMC readiness, cross-scope
transfer (per-scope tuning rule stands).

## Curves 2+3 — finalized contracts (2026-08-24; runs pending)

Shared harness design (recorded so the next session starts mechanically):
wrap `ledh_canonical_filter_tf._restore_cloud_primal` in the runner
(diagnostic instrumentation only), record each step's real
(children, weights) correction-input cloud from a frozen-Austria
canonical run at the calibrated k=4/c=8 defaults, and select the
minimum-stage-ESS (takeoff) step's cloud as the calibration fixture.
The ladder then drives one diagonal shape iteration per setting on that
frozen cloud. Faithfulness gate before any curve is read: the runner's
mirrored (J, c, displacement) computation must reproduce
`_shape_iteration_jvp`'s actual output on the fixture (parity check,
call-chain rule); a mismatch is a harness veto.

Curve 2 (trust radius): ladder radius in {0.1, 0.2, 0.5, 1.0, 2.0} plus
the off arm (0.0 = uncapped, mandatory zero comparator). Per rung
compute predicted reduction ||r||^2 - ||r - J c_capped||^2 and actual
reduction ||r||^2 - ||r_after||^2; model-trust ratio rho = actual /
predicted. Calibrated radius = largest rung with rho >= 0.75 in every
seed (threshold provenance: the "very successful step" constant of
standard trust-region methods, Nocedal & Wright Numerical Optimization
Alg. 4.1); the current default 0.5 is a warm start, not privileged.
Non-harm acceptance: at the calibrated radius, post-step residual norm
must be <= the uncapped arm's on the healthy fixture (the cap must not
degrade a healthy step), and the capped arm must be finite/valid where
any arm is. Descriptive only: everything else.

Curve 3 (LM damping): ladder damping in {0, 1e-3, 1e-2, 1e-1, 1.0}
(zero arm mandatory). Per rung record post-step residual norm (bias
axis) and max scaled-system condition (robustness axis). Calibrated
damping = smallest rung whose max scaled-system condition <= 1e4;
derivation of the threshold: the production lane is float32
(eps ~ 1.19e-7) and a linear solve loses O(cond * eps) relative
accuracy, so cond <= 1e4 keeps coefficient accuracy ~ 1e-3, the same
order as the correction strength scale. If the zero arm already meets
the condition bound on the fixture, the calibrated value is 0 WITH this
artifact as its Class-C justification (an off setting needs evidence
too); the current default 1e-2 is a warm start.

Curve 7 (Austria Fisher GPU): replicate the CPU Fisher-identity gate
class on the Austria model at GPU scale — simulate observations from
the model's own RK4 simulator at theta0 (independent numpy path per the
harness convention), replications >= 40, directions 0..2, gate
|mean score| < 3*SE + bias slack with the 2026-08-24 non-vacuity guard.
Runs on the GPU because the replication cost rides there (score lane
stays float64; GPU f64 accepted for this diagnostic, recorded in the
manifest).

Curve 4 (relative ridge): derivation note first (worst-lane effective
epsilon: TF32 unit roundoff 2^-11 ~ 4.9e-4 on the production lane;
relative form delta * tr(C)/d; safety factor derived from the Gram
accumulation error bound, N-aware), then a non-harm verification cell:
absolute-1e-5 vs relative form on the frozen fixture — identical
healthy-path outputs within declared tolerance, bounded flagged
behavior on the pathological fixture. Changing the shipped ridge is
Class C and lands only with that verification green.
