# Phase 4 Fixture Validity Repair

Date: 2026-07-13

Status: `REPAIRED_BEFORE_NUMERICAL_OUTPUT`

The first focused command failed before computing any streaming or Contract E
result because the frozen fixture supplied a batch-vector `epsilon`, while the
finite total-derivative helper's existing validated API accepts scalar
`epsilon` only. This was an invalid fixture/environment-interface assumption,
not candidate evidence.

The fixture was refrozen with scalar `epsilon=1/2`. No output, residual, error,
mass, derivative, or comparison value was available or used to choose it. The
value is the first batch's already predeclared epsilon; all two-batch particles,
weights, directions, cotangents, ridge, residual design, steps, and chunk
tilings are unchanged.

Separately, the standalone algebraic test used decimal literals that were not
binary exact despite the intended exact identity role. Those literals are
replaced with dyadic fractions and power-of-two row masses before rerun. This
repairs the test arithmetic; it does not set a tolerance or alter the streamed
fixture.
