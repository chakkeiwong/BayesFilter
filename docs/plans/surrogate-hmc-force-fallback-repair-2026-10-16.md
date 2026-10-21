# Surrogate-HMC force fallback repair and range-bearing retry

The owner requested repair of the master and restoration of range bearing to
the queue on 2026-10-16. The question remains whether the five saved-model
surrogate-HMC runs achieve the declared cumulative sampling diagnostics within
the authorized budget. This is a controller repair, not a new filtering method.

## Evidence contract and mathematical scope

The comparator is the saved range-bearing 16-flow-step run in
`artifacts/surrogate-hmc-60h-continuation-20261016-01/range_bearing/`.
It stopped after 250 warmup transitions solely on `no_force_fallback`;
endpoint validity, finite arithmetic, energy, movement and divergence checks
passed. The kernel already replaces an invalid proposal force by zero.

For a frozen deterministic position field F, a kick
K(q,p)=(q,p-h F(q)) is a translation in p conditional on q, and a drift
D(q,p)=(q+h M^-1 p,p) is a translation in q conditional on p. Each preserves
volume and has its inverse at -h. The symmetric kick-drift-kick composition,
followed by momentum reversal, is therefore an involution. The argument also
applies to the existing deterministic zero extension on invalid-force regions;
smoothness of F is unnecessary for the translation argument. Consequently MH
with the independently evaluated endpoint Hamiltonian targets the declared
endpoint density. Force accuracy affects efficiency. It does not require
automatic rejection of a finite proposal or termination of the entire run.
Floating-point reversibility is checked to numerical tolerance, not asserted
as exact real-arithmetic execution.

Apply warning semantics only to the repository's frozen, deterministic,
position-only, endpoint-corrected force runner. All other runners retain the
existing fallback health veto. Endpoint failures, nonfinite transitions,
divergences, energy-limit violations, inconsistent saved state, and immobile
chains remain hard checks. Fallback counts, rates and counts by chain are
explanatory diagnostics and potential efficiency-repair signals. They are not
convergence criteria. Keep them in progress and terminal artifacts, including
warmup, replay/resume and retained continuation paths.

The numerical pass criterion remains adaptive warmup readiness and cumulative
retained modern R-hat <=1.01, subject to health checks, maximum 20,000 retained
draws per chain and budget. No independent validation chain is added. A pass
does not establish exactness of an approximate nonlinear endpoint likelihood,
posterior agreement with an independent oracle, or superiority over another
sampler. Exact Kalman and other existing references are unchanged. This repair
does not rank methods; heuristic superiority is not a claim under this contract.

## Execution and defaults audit

Preserve all prepared data, priors, force/endpoint implementations, particles
4032, chunks2016, FP32/TF32off, epsilon1/five OT iterations and range-bearing
16 flow steps. These are the owner-approved campaign settings, not newly
validated universal defaults. Source/data selection checks remain mandatory.
The controller files are outside the prepared filter selection's source closure;
reusing that selection does not conceal a changed filter. New run manifests
must bind the repaired controller sources and resolved fallback policy.

Regression checks must demonstrate fallback-only continuation, retained warning
telemetry, unchanged hard failure behavior and strict handling of other runner
semantics. Exercise the actual kernel's zero-force fallback and endpoint
correction, plus fresh/resume/retained consumer wiring. Queue tests must show
that the original supervisor and worker finish before the retry can write the
shared budget, and that attempt/time limits and source mismatches stop launch.
These CPU-only checks are engineering evidence, not a GPU posterior result.

Queue the retry after the running predator-prey and queued SIR phases. Do not
interrupt them. Use a fresh output directory under the current continuation
root; preserve the original failure. Cap the retry at the unused part of the
original range allocation (28,800 minus charged range worker seconds), and at
the remaining 216,000-second total and six-attempt limit. Waiting consumes no
GPU-worker budget. A source/data mismatch, unresolved active ledger entry,
exhausted budget, or corrupt parent termination prevents launch. No new budget
is implied. The already running PP process keeps its loaded controller; future
workers load the repaired one, and their source manifests record the difference.

The exact entry point will be
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_60h_continuation.py enqueue-range-bearing`.
GPU launch uses trusted/escalated execution. The queue record and retry manifest
live under the current continuation artifact root. CPU regression logs and the
repair result note live in
`artifacts/surrogate-hmc-force-fallback-repair-checks-20261016-01/`.

## Skeptical pre-execution audit

The earlier rule confused proposal-force failure with failure of the endpoint
density. Removing endpoint checks would be wrong; only the force-specific veto
changes. The force must stay frozen and deterministic, with no parameter-time
retuning. All call-chain consumers, not just the first warmup chunk, must use
the same policy. Tuning already records fallbacks without treating them as a
standalone rejection. The largest practical risks are poor movement despite
valid MH correction, excessive retry cost, incomplete warning archives and
concurrent budget writers. The checks and budget/queue constraints above
address those risks. The plan passes this audit for implementation and bounded
retry; convergence and scientific accuracy remain unproved.

## Implementation review and execution decision

The checked path is `continue-advance` -> inference worker -> public
`tune_hmc_kernel` -> bound deterministic force runner -> shared sequential
controller. The policy is resolved from the runner's algorithm family, force
semantics and numerical runner identity; there is no CLI switch to disable
health checks. Fresh, replay/resume, aggregate loading and retained continuation
all consume that policy. Warning counts are archived by chunk and by phase,
including counts by chain; they count affected chain-transitions, not individual
force calls. The likelihood, force calculation, priors and numerical kernel
were not changed by this repair.

All 62 focused CPU-only tests passed in 31.15 seconds, with two TensorFlow
Probability deprecation warnings. Actual XLA Gaussian transitions with invalid
or nonfinite force matched the explicit zero-field MH kernel; the piecewise
zero-extended proposal reversed within 1e-13 in the tested fixture. Other health
failures still stopped the controller. Replay and continuation retained warnings.
Queue tests covered waiting, unique output, source drift, unresolved workers,
remaining time and attempt limits. Two stale resume tests were corrected to
the already authorized 20000-draw cap; the sampling cap was not changed here.
The first failed test log is preserved alongside the passing rerun.

The implementation passes the bounded repair audit. The queued retry must still
recheck its saved source and preparation identities before starting. PP's loaded
code is not hot-patched: use its startup worker manifest for original controller
provenance, since older terminal manifests hash files present at finalization.
Do not silently relabel that running attempt as using the repaired policy.
No new GPU comparison or posterior result is claimed by these CPU checks.
