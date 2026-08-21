# Generic NeuTra adaptive five-stage repair plan (2026-08-15)

## Research intent ledger

| Field | Predeclared statement |
|---|---|
| Main question | Does bounded adaptive joint consolidation repair the fixed five-stage recipe on the correlated Gaussian and banana targets, and does carrying selected Adam moments help or hurt when variable groups expand? |
| Mechanisms under test | A held-out plateau scheduler in stage four, and an explicit `carry_selected` optimizer-state policy that restores the selected model and Adam state between cumulative phases. |
| Expected failure mode | Fixed early-stage spending may leave too little joint consolidation; alternatively, carried global Adam iteration may make newly activated zero-moment variables adapt poorly. |
| Promotion criterion | The untouched 131,072-draw known-law gate for each target and seed. Held-out reverse-KL selects checkpoints only. |
| Promotion veto | Any failed coordinate mean, coordinate second moment, or adjacent cross-moment screen. |
| Continuation veto | Broken controller mechanics, nonfinite state/loss/gradient, invalid exact-law harness, CPU/scalar fallback, missing GPU memory growth, invalid artifact, or campaign time cap. |
| Repair trigger | Adaptive arms reach their cap while improving but fail only localized moment screens; this nominates a larger or differently allocated target-specific budget, not default promotion. |
| Explanatory diagnostics | Held-out loss/checkpoint history, actual versus selected updates, LR reductions, stop reason, clipping, ESS fraction, log-ratio SD, and failed-screen counts. |
| Must not conclude | Passing does not establish universal staging, SSL-LSTM transfer, multimodal coverage, HMC correctness, posterior correctness, or default readiness. |

## Evidence contract

The exact baseline is matched cold-start joint reverse-KL on the identical
three-block dense IAF, initialization, batches, LR grid, target, float64 GPU/XLA
path, and 3,000-update ceiling. The historical fixed five-stage 1,000-update
results remain contextual evidence and are not rerun as if they had the same
budget.

Three fresh arms run on each target:

1. `adaptive_reset`: adaptive stage four with phase-local Adam resets;
2. `adaptive_carry`: the same schedule with selected Adam state carried between
   cumulative phases; and
3. `cold`: independently tuned joint training with all 3,000 updates available.

The staged ceiling is `100 + 300 + 3*100 + 2300 = 3000` selected-path updates.
Cold receives 3,000. Tuning optimizer work is not equal because each staged
phase tunes three rates; artifacts report actual tuning work separately. This
comparison can establish known-law viability and identify a mechanism, but it
cannot support a runtime-efficiency claim.

Two seeds run for every target/arm. Model conclusions remain separate; there is
no joint pass or ranking. Continuous metrics are descriptive unless the hard
known-law screens establish only pass/fail viability.

## Generic API repair

`NeuTraStageSpec` gains an optional generic adaptive policy. The controller must:

- check a deterministic held-out callback only at declared checkpoints;
- reduce the current LR after a declared number of checks without the declared
  minimum improvement;
- stop after the declared reduction limit and minimum updates, or at the hard
  update cap;
- retain the best checkpoint independently of scheduler tolerance; and
- record checkpoint history, executed updates, LR reductions, and stop reason.

`train_neutra_five_stage` gains an optimizer-state policy:

- `phase_reset` preserves current behavior;
- `carry_selected` builds Adam slots for the complete transport, restores the
  selected optimizer state for every phase candidate, updates only active
  variables, and carries the optimizer state paired with the selected model
  checkpoint.

Every LR candidate must begin from the identical incoming model and optimizer
state. Inactive variables may not change. Stage-five validation may not mutate
model state.

## Default and numeric audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Three-block `(32,32)` ELU dense IAF | Prior known-law campaign | Same architecture passed the Gaussian under cold training | Could underfit banana | Cold 3,000-update arm and terminal loss | Reviewed baseline, not universal default |
| LR grid `2e-4, 5e-4, 1e-3` | Prior target campaign | Contains the successful historical rates | Optimum may lie outside grid | Boundary selections and terminal slope | Reviewed bounded hypothesis |
| Batch 4,096 | Prior GPU/XLA campaign | Batch-native and previously finite | Gradient noise or unnecessary cost | Loss/checkpoint stability and clipping | Reviewed baseline |
| 100-update check interval | Inherited low-dimensional campaign cadence | Gives 23 possible joint checks | Could miss faster plateau behavior | Checkpoint history | Convenience hypothesis |
| Patience 4 checks | Existing repository plateau-training precedent | Requires 400 updates without material progress before a drop | Stops too late or early | Stop reason and retained best update | Transferred hypothesis, not default |
| Absolute minimum improvement `1e-5` | Convenience choice below observed `~1e-2` per-check improvements | Avoids treating numerical-scale changes as progress | Loss scale is target dependent | Compare raw checkpoint deltas | Unproven hypothesis |
| LR reduction factor `0.5` | Existing repository plateau-training precedent | Conventional bounded reduction | Too mild to consolidate | LR history and cap behavior | Transferred hypothesis |
| Three reductions | Compute-bounded convenience | Allows four LR levels before stopping | Premature stop | Terminal slope and selected update | Unproven hypothesis |
| Joint cap 2,300; total ceiling 3,000 | Derived from matched-budget equation | Gives most budget to the diagnosed joint-consolidation gap | Early stages may still waste budget | Per-phase selected updates | Reviewed campaign ceiling |
| Two seeds | Bounded replication choice | Detects gross seed instability while keeping campaign small | Insufficient for continuous ranking | Per-seed pass/fail disagreement | Diagnostic replication only |

## Skeptical plan audit

| Risk | Disposition |
|---|---|
| Wrong baseline | Repaired with identical architecture/initialization and a matched 3,000-update cold ceiling. |
| Proxy promoted to correctness | Vetoed: held-out loss only schedules/selects; untouched known-law screens decide viability. |
| Adaptive run receives hidden extra budget | Vetoed: selected-path ceiling is exactly 3,000; actual tuning work is reported and no speed claim is allowed. |
| Carry arm conflates optimizer state with new batches | Vetoed: stateless training batches are shared across LR candidates and policies by phase/update. |
| Zero gradients move inactive variables through Adam momentum | Vetoed: apply gradients only to active variables even though full slots are prebuilt. |
| Newly active variables inherit global Adam iteration | Retained as the mechanism under test and explicit possible failure mode. |
| Failure incorrectly rejects staging | Vetoed: distinguish controller failure, recipe failure, and under-budgeted terminal improvement. |
| Stale historical thresholds | Exact-law harness is rerun through the same checked 99.9% interval implementation; exact transforms pass focused tests. |
| Concurrent GPU load invalidates science | Runtime is descriptive only; memory growth and device provenance are recorded. Nonfinite/resource failures invalidate the affected cell. |

Audit verdict: the plan answers the repair question without changing the target,
architecture, or validity gate. Execution may proceed after focused controller
tests pass.

## Campaign and stop conditions

- Targets: correlated Gaussian d=16 and banana d=16.
- Arms: `adaptive_reset`, `adaptive_carry`, `cold`.
- Seeds: 0 and 1 for every cell.
- Batch: 4,096; audit: 131,072; TensorFlow float64; GPU/XLA; TF32 off.
- Maximum cells: 12.
- Campaign wall-time cap: 45 minutes.
- Stop immediately for a controller invariant failure, nonfinite artifact,
  invalid exact-law check, GPU memory-policy failure, or time-cap exhaustion.
- A candidate known-law failure is not a campaign continuation veto.

## Planned artifacts

Output root:
`docs/plans/artifacts/neutra-generic-adaptive-five-stage-repair-2026-08-15/`

Each cell records the exact command, commit, environment, seed, GPU/memory
policy, architecture, schedules, actual and selected updates, validation, state,
wall time, and hashes. The campaign writes a summary, result note, and reset
memo.
