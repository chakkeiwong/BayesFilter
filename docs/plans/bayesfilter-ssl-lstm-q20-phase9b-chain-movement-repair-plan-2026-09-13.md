# Phase 9B Chain-Movement Repair And Test Plan

Date: 2026-09-13
Governing program: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`
Execution source: isolated worktree `/tmp/BayesFilter-phase9b-execution`, pinned to
commit `b603363d` until a reviewed repair commit is created.

## Observed failure

The preserved strict P1 attempt failed in warmup chunk 3. Chain 3 had
`acceptance_rate=0.0` and `all_states_moved=false`; the other chains moved.
The chunk contained finite sampled states, target values, and log-acceptance
values. There were 86 finite log-acceptance values below `-1000`, and the
maximum absolute `delta H` was `3744.6008878593225`. The movement predicate is
`_chain_moved(initial_state, samples)`, which requires at least one changed
state coordinate for every chain. This is a valid hard sampler-health veto.

The evidence does not yet distinguish among an overly large tuned step size,
state-local numerical instability in the transport/gradient, or a chain-specific
seed/initial-state interaction. It does not justify relaxing the movement veto,
discarding chain 3, or changing the scientific target.

## Research question and evidence contract

Question: why does the strict fixed-transport HMC chain become completely
rejected in a later warmup chunk after earlier chunks move?

Baseline: the exact failed strict chart, handoff, initial-state bank, seeds,
dtype, XLA mode, and four-chain controller from the preserved attempt.

Primary diagnostic: reproduce the failure at the same chunk boundary and
identify whether proposal energy, target value/score, target-status telemetry,
or acceptance handling first becomes invalid or extreme for chain 3.

Hard vetoes: nonfinite state/target/score/status, source or artifact mismatch,
changed seeds/chart/handoff, missing per-chain telemetry, or a failed recovery
comparison. The existing `chain_without_movement` veto remains active.

Explanatory diagnostics: acceptance rates, finite extreme `delta H`, gradient
norms, proposal-versus-accepted state differences, and runtime. They cannot
establish convergence or method ranking.

This work may establish a localized repair trigger or candidate failure only;
it cannot establish posterior validity, convergence, superiority, or default
readiness.

## Bounded diagnostic sequence

1. Verify the isolated worktree is clean, record `git rev-parse HEAD`, the full
   source closure, plan hash, environment, GPU UUID, memory-growth receipt,
   XLA/TF32 settings, chart hash, handoff hash, initial-state hash, and exact
   failed seeds. No worker may run from the dirty main checkout.
2. Replay the strict warmup prefix with the preserved chart and handoff. Use
   the same four chains and seeds, and archive each chunk's accepted state,
   proposed state when exposed, target value/score, target-status telemetry,
   log-acceptance ratio, `delta H`, and gradient finiteness/norm. Confirm exact
   equality through chunks 0-2 before diagnosing chunk 3.
3. Run the same prefix and failing chunk with XLA disabled as a diagnostic
   comparator. This is not promotion evidence. If the failure differs, compare
   proposal target/score and gradient outputs at the same chain-3 state.
4. Run a fixed-step-size ladder using the same failed inputs and seeds:
   `0.0275`, `0.01375`, `0.006875`, and `0.0034375`, with three leapfrog
   steps. Record whether chain 3 moves and whether extreme finite `delta H`
   disappears. Do not select a replacement from this one short diagnostic.
5. If step-size reduction does not localize the issue, rerun chain 3 alone and
   then all four chains with the same seed. Compare chain-major versus batched
   execution, proposal telemetry, and target-status fields. A difference is an
   implementation defect requiring repair, not evidence to exclude the chain.
6. Run CPU/reference value-score checks at the exact failing state and proposed
   state. Compare values, analytic scores, SPD/eigen residuals, and status
   flags. Preserve the existing tolerances; do not replace invalid values with
   finite sentinels.

## Repair decision

- If the proposal is finite but `delta H` is too large at the selected step
  size, classify this as tuning failure. Repair scope-specific tuning, rerun
  the public tuner, and require a fresh untouched health screen.
- If XLA and non-XLA disagree at identical inputs, classify this as numerical
  or compiler-path failure. Repair the numerical path, add an exact regression,
  and repeat fixed-bank and trajectory checks before tuning.
- If proposal/accepted telemetry disagrees with the target or score reference,
  classify this as an implementation/trace-contract failure. Repair telemetry
  and fail closed until the trace is complete.
- If the exact failure reproduces with valid finite mechanics across backends,
  retain the movement veto and reject this strict candidate for the current
  scope. Continue only with the planned fresh scope-specific repair/tuning;
  do not reject the research direction.

## Acceptance tests before retry

- Source closure is frozen in the isolated worktree and verified before launch
  and before every chunk; any drift stops before GPU work is interpreted.
- Focused CPU tests cover movement, per-chain proposal telemetry, source
  closure, and exact replay. The execution worktree is clean and its commit
  hash is recorded in every manifest.
- The diagnostic reproduces or explains chunk-3 behavior without changing the
  target, seeds, movement veto, memory-growth policy, XLA default, or budget.
- A fresh public tuning artifact matches the exact scope and is used by the
  retry. Failed prior P1 outputs remain historical and are not warm starts.
- A new versioned P1 attempt passes recovery canaries and complete health
  screens before any P2 decision.
