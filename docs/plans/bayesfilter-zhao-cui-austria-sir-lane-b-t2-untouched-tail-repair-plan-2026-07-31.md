# Zhao-Cui Austria SIR Lane-B T2 Untouched Tail Repair Plan

Date: 2026-07-31

Status: `ACTIVE_LOCALIZED_REPAIR`

## Trigger

The frozen 16,384-row untouched stream failed before artifact creation. Exact
retained column `12287` maps to a finite previous state with maximum magnitude
about `3124.64`; the source-faithful four-substep Austria `sir_step` then
overflows FP64 inside RK4. A 100-digit diagnostic evaluation of the same fixed
polynomial remains finite, with final infectious magnitudes from about
`10^152` through `10^2573`. This is a numerical-representation failure, not a
selected-TT, proposal-correction, or importance-weight failure.

## Evidence Contract

| Field | Contract |
|---|---|
| Question | Can the frozen untouched value estimate be completed without changing the source RK4 algebra, proposal law, samples, denominator, selected TT, or value scalar? |
| Baseline | Direct FP64 author `sir_step` for ordinary rows; exact B2 retained proposal; selected T2 identity `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`. |
| Candidate | TensorFlow signed-log evaluation of the same four-stage/four-substep polynomial only for rows whose direct FP64 state is nonrepresentable. |
| Promotion criterion | Ordinary-row signed-log transition parity; every nonrepresentable row has a certified standardized infectious residual above the FP64 square-overflow threshold by a predeclared log margin; all rows remain in the Monte Carlo denominator; fresh hashes and deterministic replay. |
| Veto | Clipping, dropping, resampling, changing endpoint bounds, changing RK4 stages/substeps, changing the proposal correction, using the failed partial cloud, uncertified tail rows, or any finite non-tail mismatch. |
| Repair trigger | Signed-log cancellation ambiguity, insufficient overflow margin, more than the bounded tail workload, or parity failure. |
| Continuation veto | The real RK4 polynomial is undefined, the overflow row cannot be certified as FP64 zero likelihood, or the repaired estimator changes the stated scalar. |
| Nonclaim | No exact-real zero density, analytical score at the zero row, T5/T10/T20, HMC, production KR, posterior, or scientific validity. |

## Mathematical Repair

Represent each real intermediate as `(s, ell)` with sign `s` and
`ell=log(abs(x))`. Addition uses signed log-sum-exp/log-difference-exp;
multiplication adds log magnitudes and multiplies signs. Apply exactly the
author operation order:

1. four RK stages with step `0.005`;
2. the source fourth-stage half-step convention;
3. four complete substeps;
4. add the frozen Gaussian innovation;
5. subtract sealed observation `y2` on infectious coordinates.

For observation scale `10`, a row is an FP64 extended-real zero-density row
only if at least one standardized residual has

`log(abs(residual/10)) > 0.5*log(sys.float_info.max) + 20`.

The extra margin is a certification buffer, not a likelihood threshold used
for ordinary rows. Ordinary rows continue through the existing TensorFlow/TFP
likelihood. The transition density cancels between target and proposal because
the frozen innovation still defines the sampled transition; no transition term
is approximated in the importance weight.

## Skeptical Audit

- Wrong baseline: blocked by binding the author source half-step RK4 route and
  the selected T1/T2 identities.
- Proxy promotion: the overflow certificate only permits artifact creation; the
  untouched same-scalar gate remains primary.
- Hidden assumption: exact-real Gaussian density is positive. This repair
  claims only the already-declared FP64 extended-real finite program.
- Unfair data handling: the failing row remains in the denominator with zero
  numerator; no row is dropped or replaced.
- Stale context: selection is frozen and the failed attempt read no successful
  untouched value artifact.
- Non-answering command: parity tests answer algebra preservation; the fresh
  one-shot result answers the untouched scalar.

Audit verdict: `PASS_FOR_EXECUTION`.

## Budget

- engineering tests: 20 minutes;
- one fresh CPU-hidden untouched replay: 90 minutes;
- one GPU/XLA read-only value claim: 10 minutes;
- fresh versioned outputs only; no HMC or score execution.
