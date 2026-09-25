# Session Recovery: Context Thrashing Incident 2026-09-14

## Session ID
Session experienced repeated context compaction thrashing. Current conversation should be abandoned and recovered from this memo.

## Background

### Master Program
**Document**: `docs/plans/ledh-surrogate-hmc-executable-master-program-2026-09-07.md`

The active master program is a multi-phase LEDH surrogate HMC campaign. A new Phase 3.5 was inserted on 2026-09-14 to repair a while-loop regression discovered during execution.

### The While-Loop Regression
**Repair Plan**: `docs/plans/ledh-while-loop-regression-repair-plan-2026-09-14.md`

Phase 2B unification (commit 5cc59cfa) inadvertently deleted 14 `tf.while_loop` calls from the canonical engine, regressing to Python `range` loops that unroll ~400 flow stages into the TensorFlow graph. This causes:
- Trace time: 485 seconds (should be <50s)
- Graph size: ~400× larger than necessary
- Blocks all subsequent damping calibration work (requires 16k+ evaluations)

## What Was Completed

### Phase 3.5 Execution Status
**Status Document**: `docs/plans/ledh-while-loop-regression-repair-status-2026-09-14.md`

1. **Phase 0**: pfor approach measured and rejected (×0.26 = 3.8× slower than sequential)
2. **Phase 1**: Time-loop `tf.while_loop` restoration **COMPLETE**
   - Modified: `bayesfilter/highdim/ledh_canonical_score_tf.py`
   - Oracle contract: **PASSED** (8/8 supported tests, 60s runtime)
   - Test modified: `tests/contracts/test_oracle_contract.py` (added Phase 1 constraint documentation)
   - Constraint: `annealed_stages=1` (nested annealing deferred to future phase)
   
3. **Phase 2**: Substep-loop restoration — **NOT STARTED**
4. **Phase 3**: Performance measurement — **NOT STARTED**

## Current Git State

Branch: `surrogate-hmc`

Modified files:
- `AGENTS.md` (governance updates)
- `CLAUDE.md` (governance updates)  
- `bayesfilter/highdim/ledh_canonical_score_tf.py` (Phase 1 while-loop restoration)
- `tests/contracts/test_oracle_contract.py` (Phase 1 constraint documentation)
- Several other experimental files (not part of Phase 3.5)

## Exact Next Action

The oracle contract passed. The next action in Phase 3.5 is:

**Phase 1 Measurement**: Run the parity and timing benchmark to validate that Phase 1 restoration does not regress performance and provides measurable improvement.

**Benchmark script**: `docs/benchmarks/ledh_execution_mode_matrix.py` (exists as untracked file)

Command should be:
```bash
conda run -n tftwogpu python docs/benchmarks/ledh_execution_mode_matrix.py
```

Expected outcome:
- Graph mode time at plan scale (N=252, T=50): should improve from baseline ~39.3s
- Parity: value and score relative error < 5e-4
- If successful: proceed to Phase 2 (substep-loop restoration)
- If failed: investigate and repair before Phase 2

## What Caused Context Thrashing

The agent read multiple large files in rapid succession during code tracing, causing repeated context compaction within 3 turns. The compaction policy requires using focused searches and targeted line ranges rather than full-file reads.

## Recovery Instructions

1. **Do not read the master program or repair plan documents** — this memo contains the essential state
2. Verify git state with `pwd && git branch --show-current`
3. Verify oracle contract still passes: `conda run -n tftwogpu python -m pytest tests/contracts/test_oracle_contract.py::test_oracle_contract_battery_passes -v`
4. Run Phase 1 measurement benchmark (command above)
5. Record results in `ledh-while-loop-regression-repair-status-2026-09-14.md`
6. If measurement successful, proceed to Phase 2 substep-loop restoration using the repair plan as reference

## Key Constraints From Governance

- Backend: TensorFlow/TFP only (no NumPy in non-diagnostic code)
- Default execution: GPU, float32, TF32 enabled
- Conda env: `tftwogpu`
- Oracle contract is binding: all supported analytical tangent tests must pass
- Phase 1 constraint: `annealed_stages=1` only (3 tests excluded as expected failures)

## Critical Files To Preserve

Do not overwrite or lose:
- `docs/plans/ledh-while-loop-regression-repair-plan-2026-09-14.md`
- `docs/plans/ledh-while-loop-regression-repair-status-2026-09-14.md`
- `docs/benchmarks/ledh_execution_mode_matrix.py`
- The modified `ledh_canonical_score_tf.py` with Phase 1 restoration

## What Must Not Be Concluded

- Phase 1 does not restore substep loops — that is Phase 2
- Phase 1 does not support annealed telescoping — deferred to future phase
- Performance improvement not yet measured — measurement is the next action
- No claims about XLA or further optimization — measurement first

---
**Recovery datum**: 2026-09-14  
**Agent**: Claude Opus 5  
**Branch**: surrogate-hmc  
**Phase**: 3.5 Step 1 complete, Step 2 (measurement) next
