# Consistent fixed particle design across arithmetic modes

This is phase 2 of the authorized computation master. The question is whether
SIR really requires expensive FP64 recursion once FP32 and FP64 select the same
fixed proposal ancestors. The saved SIR T10 inputs show one ancestor mismatch at
observation 1 and two at observation 7, with zero mixture-component mismatches.
For one uniform, FP32 rounds 4032u to 1137 whereas its exact value is
1136.9999885559082. Flooring selects adjacent ancestors. These locations match
the abrupt local proposal differences. Evidence: sir-local-arithmetic-ancestor-check.json.

The intended draw uses b=sum(u0>=cumsum(beta)) and j=floor(N*u1), with a fixed
uniform design independent of theta. Compute these discrete indices in FP64
through one shared helper in both primal/directional and all-parameter draw
paths. For a representable FP32 uniform and integer N<=2^24, the product has at
most 48 significant bits, so binary64 represents it exactly. The continuous
filter kernels remain in their requested precision. Evaluate cumulative beta in
the same binary64 arithmetic in both lanes. Equal represented uniforms and beta
then select equal indices. This changes the old FP32 realized particle design
at boundary cases; it does not preserve the old FP32 finite scalar there. It
implements the stated inverse-CDF law consistently and leaves the FP64 rule
unchanged. Do not describe old precision gaps as pure continuous roundoff.

Early checks: CPU XLA tests at observed boundaries and generic mixture boundaries;
exercise both actual draw consumers and their tangents; retain ordinary-case
parity. Add the helper to master source snapshots and scope source identity.
Repeat the saved SIR T10 stage diagnostic with identical data/controls. Then run
all five requested full horizons and off-centre positions, N4032/K2016,
FP32/TF32off versus the same corrected finite program in FP64. The user criterion
is <=1% scaled error in final likelihood and score, with finite/health vetoes.
A full-FP32 failure rejects that scope and triggers the planned localized
arithmetic repair; it does not stop other models or authorize relaxing 1%.
If full-FP32 SIR passes, measure actual consumer cost before retiring its mixed
exception. Refresh scope-specific offline tuning on fresh partitions for all
changed scopes before selected-consumer checks. Retain mixed arithmetic where
required by the evidence. No fixed-count posterior or independent validation
phase is introduced.

Primary promotion: exact discrete-index agreement, correct analytical draw
parity, then full-horizon final-output arithmetic within 1% and actual consumer
parity. Invalid states, stale source/data, or missing comparisons veto promotion.
Runtime differences, intermediate particle tangents and condition margins are
explanatory; a few timings cannot establish a statistical speed ranking.
FP64 is an arithmetic comparator, not an exact model oracle. No posterior
convergence, canonical Contract-E admission, or cross-model scientific accuracy
is established by these checks. Existing exact Kalman/integration references
remain separate. This repair compares the same formula, so no new heuristic
method ranking is made.

Budget: initial repair/precision envelope 10800 worker-wall seconds within the
existing 86400-second ledger, single GPU worker; compilation and failed attempts
count. Stop for corrupt inputs, invalid comparisons or exhausted budget. A local
harness failure may be repaired and retried. Every launch gets a new directory
and a refreshed next-phase plan. Remaining tuning/consumer phases use the same
campaign ledger and the main plan's remaining budget.

Default audit: FP32/N4032/1% are owner choices; FP64 index selection follows the
exact-floor derivation and is tested at boundaries; same saved design is a
comparison requirement; T10 is localization only; prior controls are diagnostic
baselines until fresh scope tuning. The principal risk is mistaking matched
branches for a complete SIR repair: residual transport/reset roundoff must still
be measured at T50. The source dependency changes invalidate old selected-source
admission. Preserve historical runs and snapshot the pre-repair source.

Skeptical audit passed: the comparator shares the discrete design, the change to
the FP32 finite program is explicit, both draw call chains are tested, and the
full requested horizons and real consumers remain required. No default precision
or ridge policy is silently changed. The next action is implementation and CPU
regression, then the bounded GPU diagnostic.
