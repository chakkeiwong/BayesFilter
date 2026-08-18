# NeuTra Banana HMC Repair Result (2026-08-16)

## Outcome

The terminal three-arm campaign completed in `534.26 s` on GPU 0 with
float64, XLA, TF32 disabled, and TensorFlow memory growth verified before
device initialization. The terminal artifact root is:

`docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3/`

The learned banana transport passed the proposal audit. HMC with the original
iid-normal start bank failed the retained exact-law screen. HMC with a central
start bank passed, and the exact analytic banana transport passed with the
original bank. This rules out a controller-wide or analytic-target failure, but
does not identify a pure start-only cause because each arm was independently
tuned and therefore Arm A and Arm B selected different kernels.

## Evidence Contract

| Item | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-neutra-banana-hmc-repair-plan-2026-08-16.md` |
| Terminal artifact root | `docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3/` |
| Target | 16-dimensional analytic banana, curvature `0.35`, unit Jacobian |
| Learned transport | Seed `15`, 6,000 batch-native reverse-KL updates, root-preserving `(32,32)` dense IAF, peak `LR=5e-4`, fixed r3 schedule horizon |
| Proposal audit | 131,072 draws; learned and analytic arms passed all exact-law screens |
| HMC policy | Four chains, sequential controller, identity z mass, `L=1` forbidden, fresh per-arm tuning grid `L=(3,5,10,15,20,25)` |
| Integrity | 73 terminal artifacts; all SHA-256 hashes passed |
| Git commit recorded | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |

## Arm Results

| Arm | Kernel | Warm-up | Retained convergence | Retained exact-law | HMC status |
|---|---:|---:|---:|---:|---:|
| Learned transport + original iid bank | `L=5`, step `0.836133` | Pass, max R-hat `1.01072` | Pass at 2,000/chain, max R-hat `1.00429` | **Fail**, adjacent cross moments 4 and 6 | Rejected |
| Learned transport + central bank | `L=10`, step `0.770972` | Pass, max R-hat `1.00708` | Pass at 2,000/chain, max R-hat `1.00308` | Pass | Passed bounded screen |
| Exact analytic banana + original iid bank | `L=10`, step `0.764915` | Pass, max R-hat `1.00615` | Pass at 2,000/chain, max R-hat `1.00296` | Pass | Passed positive control |

For the failed learned/original arm, the retained exact-law adjacent cross
moments at indices 4 and 6 were `-0.03718` and `-0.03700`, with standardized
discrepancies `3.36` and `3.29`. Coordinate means and second moments passed.
The HMC health path itself passed: states, target values, scores,
log-acceptance values, movement, and declared energy checks were finite or
valid. Native TFP divergence telemetry was unavailable and is not interpreted
as zero divergences.

The central-start learned arm passed all retained exact-law screens. Its first
three retained checks were not yet ready at 500, 1,000, and 1,500 draws, then
passed at 2,000 draws per chain. The analytic control showed the same expected
progression and passed at 2,000 draws.

## Mathematical Interpretation

The analytic banana map is

`theta_0 = z_0`,
`theta_1 = z_1 + c (z_0^2 - 1)`, and
`theta_j = z_j` for `j >= 2`, with determinant one. Its exact pullback score
is `g_z0 = g_theta0 + 2 c z_0 g_theta1` and `g_zj = g_thetaj` otherwise.
The analytic arm passing with the original start bank validates the target
binding, triangular Jacobian, score pullback, fixed-kernel mechanics, and
sequential controller on this exact law. It does not validate the learned
transport.

The learned/original failure therefore lies in the interaction between the
learned transport and the original HMC execution path, not in the analytic
banana target or the entire HMC harness. The central-bank pass demonstrates a
viable operational repair for this frozen learned transport, but because Arm B
was independently tuned and selected `L=10` while Arm A selected `L=5`, the
experiment cannot separate a start-bank effect from a start-dependent kernel
selection effect.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Learned/original HMC | Retained exact-law screen | Vetoed by two adjacent cross moments | Why original bank plus selected `L=5` fails | Run a matched-kernel start-bank diagnostic if a pure start claim is needed | Learned transport impossibility |
| Learned/central HMC | Sequential gates plus retained exact-law screens | Passed | One bank and one fresh tuned kernel | Refer to the completed fixed-`L=10` two-bank confirmation | Universal start policy or default HMC readiness |
| Analytic/original HMC | Exact-law positive control | Passed | Control is analytic, not learned | Retain as mechanics authority | Learned-transport correctness |
| Banana HMC diagnosis | Three-arm pattern | Partial diagnosis | Start bank and kernel tuning are confounded between A/B | Isolate fixed kernel across banks before claiming start-only causality | Controller-wide failure |
| SSL-LSTM transfer | Target-specific evidence | Not authorized | No SSL-LSTM arm was run | Do not transfer settings | SSL-LSTM readiness |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Learned/original failed retained exact-law agreement; learned/central and analytic/original passed |
| Statistically supported ranking | None; no superiority or ranking claim is supported |
| Descriptive-only differences | Selected `L`, step size, acceptance, runtime, and per-check R-hat progression |
| Default-readiness | Not supported |
| Next evidence needed | Completed matched-kernel comparison and fixed-`L=10` 5,000-draw confirmation; next is predictive-equivalence testing |

## Red-Team Note

The strongest alternative explanation is kernel selection rather than initial
state: Arm A selected `L=5`, while Arm B selected `L=10`, and the original
start bank may have caused tuning to choose a kernel that is valid by HMC
health/R-hat but fails the exact-law moment screen. The exact analytic control
passing with the original bank weakens a controller-wide explanation but does
not distinguish learned geometry from the selected `L=5` dynamics. The
no-retuning cross-over subsequently resolved this: both banks pass with frozen
`L=10` and both fail with frozen `L=5`, so the failure follows the kernel for
this learned transport.

The weakest evidence is any claim that central starts are universally better;
that claim is not made. The matched-kernel result is recorded separately in
`docs/plans/bayesfilter-neutra-banana-hmc-matched-kernel-result-2026-08-16.md`.
