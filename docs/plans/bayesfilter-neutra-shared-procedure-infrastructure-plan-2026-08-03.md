# NeuTra Shared Procedure Infrastructure Plan

Date: 2026-08-03
Status: `ACTIVE_ENGINEERING_INFRASTRUCTURE`

## Question

Can one shared top-level Python NeuTra broad-grid procedure — with the tuning
procedure as an explicit reviewed variant — execute the generic operational
route and the PP-style state-continuing epsilon-repair route on three different
frozen-transport targets (PP-UKF, KSC gaussian-sum SV, LGSSM-EXACT) without
changing either route's reviewed semantics?

This is an engineering-infrastructure validation, not a model-science
campaign. No posterior, convergence, ranking, or default-readiness claim will
be made from these runs.

## Mechanism under test

- New shared modules:
  - `bayesfilter/inference/neutra_shared_procedure.py` (single orchestration
    authority; explicit `procedure_variant`; normalized artifact metadata;
    shared sequential-handoff extraction),
  - `bayesfilter/inference/neutra_state_continuing_broad_grid.py` (the PP
    repaired route extracted from
    `docs/benchmarks/run_pp_ukf_state_continuing_epsilon_repair_20260721.py`
    with mechanics preserved: per-L dual averaging, frozen post-adaptation
    epsilon verification, bounded bracketed epsilon repair with state
    continuation, fresh final screens, guards from the calibrated parent
    state).
- `run_neutra_frozen_transport_broad_grid_cell` and
  `run_neutra_broad_grid_sequential_cell` become delegating wrappers.
- Health telemetry contract becomes explicit per target via
  `required_status_keys` (default remains the strict five-key UKF contract;
  LGSSM declares the two-key exact-target contract because its target has no
  innovation-floor telemetry — this is a category fix, not a weaker gate).

## Success criteria (engineering)

1. All focused test suites pass: new shared-procedure tests, operational
   broad-grid contract, end-to-end contract, KSC target tests.
2. KSC generic rerun through the shared procedure with the attempt04 root seed
   `(20260803, 2881)` reproduces the attempt04 dispositions (unique viable
   primary `L=25`, compatible coverage `L=24`) — descriptive consistency check;
   exact float equality is not required on GPU.
3. PP-UKF state-continuing run executes the full repaired protocol
   (adaptation 96 + post 32, calibration region (0.68, 0.72), max 3 repairs,
   final screens 96/8, guards from calibrated state) and emits
   `procedure_variant = state_continuing_epsilon_repair_v1` with
   `state_continuation_performed` and `epsilon_repair_performed` true.
   PP prior tuned epsilons are warm starts only.
4. LGSSM-EXACT generic run executes with the two-key status contract and emits
   a complete barrier result.
5. Every artifact records the exact procedure variant, and the emitted variant
   matches the one selected on the command line.

Dispositions themselves (which L survive) are explanatory only on all three
targets; they cannot veto the infrastructure unless a barrier fails for an
infrastructure reason.

## Budget and artifacts

- Total GPU budget: at most 2 hours (projection: KSC generic ~3 min, KSC
  state-continuing ~6 min, LGSSM generic ~10 min, PP state-continuing ~50 min
  from the 2026-07-21 campaign precedent).
- Output root: `docs/plans/artifacts/bayesfilter-neutra-shared-procedure-20260803/`
  with one fresh subdirectory per run.
- All tuning draws discarded; no retained sampling; `launch_sequential` off.

## Non-claims and stops

- No claim about which (L, epsilon) pairs are good for any model.
- No change to the reviewed generic operational broad-grid semantics.
- Stop and record if any run exceeds twice its projection or the total budget.
- The pre-existing NeuTra route-ledger debt (20 unledgered routes) remains
  open and is not worsened: the new modules match the discovery profile of the
  existing tuning modules (not qualifying routes); ledger repair remains a
  separate task before any claim-bearing sequential result.
