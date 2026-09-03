# LEDH Conformance And Anti-Regression Test Plan

Date: 2026-08-21
Authority: owner directive 2026-08-21, item 3.
Target failure classes, from the consolidated registry (E1-E3): (i) lane
forks — a claim-bearing lane silently implementing less than the canonical
algorithm (NeuTra-lane class); (ii) wiring omissions — a documented
algorithm step existing as code (or prose) but not executing in the lane
(UKF class, identity-placeholder class).

Design principle: every test here is EXECUTABLE evidence bound to a
claim-bearing ENTRY POINT — never to a function's existence. Narrative
audits are advisory; these gates are the authority (call-chain audit rule,
global policy 2026-08-20).

## Layer 1 — Contract conformance suite (the anti-UKF-omission layer)

Source of truth: the P0 machine-readable step registry
(`ledh_alg1_contract.py`), transcribed from ch19c. One test module per
contract step, parameterized over EVERY registered claim-bearing entry
point (the registry of entry points is itself part of the contract; adding
a lane without registering it fails a discovery test — see Layer 4).

- C-1 UKF-predict executes: run the entry point on a small nonlinear
  fixture with instrumentation that records per-particle predicted
  covariances; assert (a) they exist per particle, (b) they differ across
  particles with different ancestors, (c) they equal an independently
  computed sigma-point prediction on the same ancestors (tolerance
  declared). An identity or shared matrix FAILS (b)/(c) — this test is
  impossible to pass with the 2026-08 placeholder wiring.
- C-2 Flow consumes the predicted covariance: perturb one ancestor's
  P^i; assert that particle's flow output changes and others' do not
  (sensitivity wiring test). Placeholder/shared covariance fails.
- C-3 Dual-state flow discipline: zero-noise anchor and actual particle
  tracked separately; assert the linearization state is the anchor (perturb
  the actual particle's noise draw; linearization coefficients must not
  change).
- C-4 Theta-product: forward log-det equals the sum over substeps of
  log|det(I + eps*A)| recomputed independently; and equals the numerical
  log-Jacobian-determinant of the realized map on a 2D fixture.
- C-5 PF-PF weight identity: assemble the weight from independently
  evaluated densities (transition at post-flow, proposal at pre-flow,
  theta) and assert equality with the lane's weights.
- C-6 UKF-update executes and covariances recurse: P_k^i differs from the
  predicted P^i in the observed subspace; next step's prediction consumes
  P_k^i (two-step chaining test).
- C-7 Triple resampling: permute ancestry; assert states, covariances, and
  weights moved TOGETHER (inject a marked covariance, follow it).
- C-8 Reset and correction surface: entry point exposes and routes the FULL
  dual-cap trust-region control surface on BOTH value and score paths
  (signature + behavioral: each control, when enabled on a discriminating
  fixture, changes the output; a dead parameter fails).
- C-9 Analytical-score-only: claim-bearing score path contains no autodiff
  (static check: no ForwardAccumulator/GradientTape/gradient call in the
  call graph of the registered score entry; allowed only inside modules
  namespaced `*_oracle_*`).
- C-10 No-placeholder guard: model callbacks registered for claim-bearing
  use must declare provenance for covariance/Jacobian inputs
  (`derived_from` field: "sigma_point", "analytical_jacobian", ...);
  literal identity/constant matrices without a reviewed-exception marker
  fail at construction time (runtime guard + test that the guard fires).

## Layer 2 — Cross-lane parity suite (the anti-fork layer)

- P-1 Batch-vs-single parity: for every fixture and every registered
  control combination, batch-size-1 output of the batch lane vs the
  single-cloud lane; declared tolerance; run in CI on CPU fixtures.
  (Generalizes the 2026-08-20 parity oracle from one function to the whole
  per-step program and the score.)
- P-2 Capability-surface equality: the batch lane's value AND score entry
  points must accept exactly the control surface of the single-cloud lane
  (introspection test). A fork that drops a capability fails before any
  numerics run.
- P-3 Value/score program identity: within each lane and mode, the value
  returned by the score entry equals the value entry bitwise (eager) or
  within the declared mode gate (compiled modes) — the 2026-08 within-mode
  identity gate, now permanent regression.
- P-4 Oracle parity for the analytical score: vs forward-autodiff oracle on
  small fixtures at tight tolerance; vs FD as explanatory. Runs per commit
  on CPU; GPU/XLA arms nightly.

## Layer 3 — Algorithm-level statistical gates (the does-it-work layer)

- S-1 Linear-Gaussian exactness: on LGSSM fixtures the full pipeline's
  value matches the Kalman filter likelihood within declared tolerance
  (flow with exact covariances + exact UKF = Kalman); score matches the
  analytical Kalman score.
- S-2 ESS health gate: per-step ESS is a MANDATORY artifact field for every
  claim-bearing run (Class A). A regression test asserts the instrumentation
  exists and is finite; campaign gates (not unit tests) set scope-specific
  ESS floors.
- S-3 Degeneracy discriminator (regression-pinned): on the frozen Austria
  smoke scope, canonical-lane minimum ESS must exceed the recorded
  bootstrap-lane and identity-covariance-lane baselines (pinned constants
  with provenance). Detects silent proposal-quality regressions.
- S-4 Guard behavior: injected near-singular covariance and tangent-poison
  fixtures produce program_valid=false, never NaN escape or silent
  acceptance (fail-closed regression, generalizing the 2026-08 invariants).

## Layer 4 — Governance enforcement tests (the anti-silent-lane layer)

- G-1 Entry-point discovery: repo-wide scan for modules matching LEDH lane
  patterns; every hit must be in the registry as canonical, oracle, or
  scaffold-with-expiry. An unregistered lane fails CI. (Modeled on the
  existing NeuTra route-ledger discovery guard, which worked.)
- G-2 Scaffold expiry: `*_scaffold_*` modules carry a declared removal
  phase; test fails when the phase's result note exists but the scaffold
  still does.
- G-3 Conformance matrix generation: CI regenerates the matrix (contract
  steps x lanes, each cell the passing test id or ABSENT) and fails on any
  ABSENT cell for a claim-bearing lane; the matrix file is committed so
  humans read what CI enforces.
- G-4 Invalidation-notice integrity: the AGENTS.md rule and the notice file
  exist and cross-reference (guards against accidental deletion during doc
  sweeps).
- G-5 Historical-result quarantine: benchmark aggregators refuse input
  artifacts whose manifest lacks the post-rebuild conformance stamp
  (schema field `alg1_conformance: <suite version>`); stamped only by runs
  through registered canonical entry points.

## Execution and cadence

- Per-commit (CPU, minutes): C-1..C-10 on small fixtures, P-1..P-3, S-1,
  S-4, G-1..G-5.
- Nightly/campaign (GPU): P-4 GPU arms, S-3, compiled-mode identity gates
  (eager/graph-meta-off/XLA, TF32 arms).
- Suite versioned; the version is the conformance stamp of G-5, so
  strengthening the suite automatically re-quarantines older stamps.

## Honest limits

These tests bind code to the ch19c contract; they cannot detect the
contract itself mis-transcribing Li(2017) — that remains a human/literature
audit (postdoc-standard reading policy applies), aided by the June 2026
Alg1-UKF fixture campaigns. They also cannot certify posterior correctness
(S-1 covers the linear-Gaussian slice only); that stays with the
statistical evidence policies.
