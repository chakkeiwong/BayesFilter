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
