# LEDH Canonical Completion Program (Q-phases)

Date: 2026-08-24. Successor to the discharged execution plan
(`bayesfilter-ledh-canonical-rebuild-execution-plan-2026-08-21.md`,
P0-P7 + Parts 4-5 complete). This document governs everything that
remains; the ledger there stays the append-only history, this program is
the forward contract. Rules R-A..R-G of the execution plan carry over
verbatim (no mid-phase design questions; test-first; declared tolerances;
append-only ledger; session-resumable).

## Phase inventory

### Q1 — Score-path completion (CPU, mechanical, oracle-gated)

Content:
1. S6/S7 tangent wiring: the analytical score currently gates on the
   reset-none slice (honestly documented in the gate docstrings). Wire
   the Contract-E reset tangent (`_restore_cloud_batch_jvp` pattern) and
   the dual-cap correction tangents (`higher_moment_shape_jvp`, already
   hand-derived) into the score recursion so the FULL pipeline
   (flow -> weight -> reset -> correction) carries the score.
2. Annealed-mode score extension: stage-weight tangents through the
   within-step annealing telescope (same Gaussian-density tangent
   machinery, gated per stage).
3. dlgssm q/r-direction score threading (density-callback scale
   parameters; recorded refinement debt).
4. Austria reduction slice (kappa -> 0 diffusion limit) — the last
   non-GPU registry pending.

Evidence contract: every new tangent gated vs the autodiff oracle at
rtol 1e-4 float64 before entering the assembled score; full-recursion
gate rerun after each wiring step; no autodiff ships (C-9 standing).
Budget: CPU only. Exit: registry shows zero non-GPU pendings; the score
entry point covers the full canonical per-step program.

### Q2 — P6 calibration campaign (GPU)

Content: the R6 protocol from the execution plan, unchanged in substance:
- trust-radius model-trust curve (predicted vs actual residual reduction
  across a radius ladder) on the frozen Austria scope;
- LM-damping bias-vs-robustness curve;
- relative-ridge derivation (worst-lane effective epsilon x safety
  factor, relative form delta*tr(C)/d);
- dual-cap constants (0.98/8, 2.0): attach owner rationale from the
  07-xx notes or run the same protocol;
- annealed-SMC stage-count k and flow-prior cap c response surfaces
  (multi-seed extension of the passed probe contract);
- float32/TF32 production-lane calibration (which precision/mode
  configuration is the production target, with the battery evidence);
- Austria Fisher-identity gate (rides here for GPU replication cost).

Evidence contract: pre-declared per-curve acceptance criteria (non-harm
form, never primary-metric tuning); multi-seed with declared seed sets;
statistical-discipline language (no ranking without uncertainty); every
calibrated value lands in the contract registry with its calibration
artifact as provenance (Class-C justification rule). Budget: ~1 GPU-day;
per-process cap 100 min; 3-consecutive-launch-failure stop. Exit:
canonical defaults carry calibration provenance; zero unjustified Class-C
values (including zeros) anywhere in the canonical lane.

### Q3 — Full six-model leaderboard (GPU; gated on Q2)

Content: six models x {canonical LEDH (calibrated), bootstrap PF, UKF
Gaussian filter, SGQF where the repo comparator exists}, value AND score
cells, multi-seed, on GPU at claim scale (N=1008-class); artifacts carry
the G-5 conformance stamp; slice-1/2 CPU tables become the template.
Evidence contract: hard vetoes first (finite/valid/identity), exact
references where linear, descriptive tables with per-seed spread, ranking
language ONLY where a pre-declared uncertainty analysis supports it.
Budget: ~1 GPU-day. Exit: the owner-facing leaderboard report with the
inference-status table (hard vetoes / viable / statistically supported /
descriptive-only / next evidence).

### Q4 — Merge and integration (OWNER-GATED)

Content: merge `worktree-ledh-canonical-rebuild` into main; coordinate
with the sibling agent's branch (the one overlap: both touch LEDH
territory; the deletion lands repo-wide); wire the conformance suite into
the repo's standing test cadence; refresh AGENTS.md pointers from
"rebuild in progress" to "canonical lane is main".
The merge decision, its timing relative to the sibling branch, and any
main-branch conflict resolutions are the owner's. Everything in Q1-Q3 can
run pre-merge on the branch; nothing in Q4 blocks Q1-Q3.

### Q5 — Successor programs (OUT OF SCOPE here, listed for the map)

NeuTra training and HMC campaigns on the canonical targets; posterior
correctness and default-readiness claims; cross-model HMC readiness.
These are their own programs with their own contracts — this program's
nonclaims explicitly exclude them (unchanged from the invalidation
notice's standing nonclaims).

## Dependency graph

Q1 (CPU) and Q2 (GPU) are independent and can interleave; Q3 requires Q2
(calibrated defaults); Q4 requires owner action and is independent of
Q1-Q3 ordering; Q5 requires Q3 + Q4.

## Governance carry-overs

- Referent-taxonomy testing regime (registry meta-test) governs all new
  code; every new model or lane needs its four-class registry row.
- Fidelity tally continues (currently 5/5 found-and-fixed); any new
  infidelity gets a gate in the class that SHOULD have caught it.
- The vendored-reference file is frozen evidence (do not edit).
- Class A/B adopt-by-default and Class C justification rules (global
  policy) apply to every calibration decision in Q2.
- Stop conditions: hard veto (identity/hash/env), 3 consecutive launch
  failures, per-process caps, or any change to scientific targets —
  everything else proceeds and is ledgered.
