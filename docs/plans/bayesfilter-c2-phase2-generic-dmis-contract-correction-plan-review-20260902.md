# Skeptical Review: C2 DMIS Contract-Correction Re-run

Date: 2026-09-02  
Reviewed plan: `bayesfilter-c2-phase2-generic-dmis-contract-correction-plan-20260902.md`  
Verdict: PASS FOR ONE BOUNDED CORRECTIVE RUN

## Review question

Does the corrective plan restore the originally declared estimator and gate,
avoid post-hoc promotion, preserve the mathematical target, and produce an
artifact that can distinguish a contract failure from a numerical candidate
failure?

## Findings

### 1. Contract provenance

The parent plan at the recorded run commit names plain complete-DMIS as the
precision estimator. The existing driver instead aliases the CV estimate as
`log_normalizer`. The correction states this deviation directly and does not
retroactively call attempt07 a valid promotion run.

Disposition: resolved by a fresh attempt and explicit metadata.

### 2. Mathematical target and denominator

The correction changes no target expression, reference density, component
weights, row masses, or complete-mixture denominator. It therefore tests the
same finite carried target and isolates the estimator-selection error.

Disposition: pass.

### 3. Leakage and reproducibility

Alpha selection remains restricted to the two-scramble calibration partition,
but the corrected selection statistic is now plain-DMIS as required by the
parent contract. Claim rows, seeds, row counts, snapshots, and proposal
families remain fixed. A fresh attempt directory prevents overwriting prior
evidence.

Disposition: pass.

### 4. Interpretation discipline

The CV estimate remains visible as an explanatory diagnostic. A plain-DMIS
precision failure vetoes this candidate at the frozen scope but cannot be used
to infer a TT-basis or recursive-map cause. The recursive stage remains
untested and is not silently relabeled.

Disposition: pass.

### 5. Artifact and resource controls

The run is bounded to one GPU attempt within the existing budget. The manifest
must capture source hash, command, environment, GPU placement, memory growth,
seeds, snapshots, and output hashes. No package, model, or default-policy
change is part of the correction.

Disposition: pass.

## Pre-run decision

The plan is executable after the code check confirms that calibration and
`precision_pass` use `plain_log_normalizer`. The only permitted conclusion is
whether the originally declared frozen DMIS screen passes; no recursive or
posterior claim follows.

## Post-run review fields

Record the exact attempt directory, plain-DMIS half-widths, CV diagnostics,
validity checks, and whether the result is a contract-faithful execution. If
the gate fails, the next action is a separately reviewed tail-aware proposal
experiment, not an unplanned basis or recursion rewrite.

## Post-run disposition

The source check passed before attempt09: calibration and the claim ladder use
`plain_log_normalizer`, and the result records
`plain_complete_dmis_log_normalizer` as the precision estimator. Attempt09 is
therefore contract-faithful. Its validity checks pass, but the plain-DMIS
precision screen fails, so the candidate remains unresolved and the recursive
stage stays gated. Attempt08 is retained as a pre-correction duplicate and is
not used for the contract-faithful conclusion.

Post-run verdict: `CONSISTENT; PLAIN_DMIS_PRECISION_VETO`.
