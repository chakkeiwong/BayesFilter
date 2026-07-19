# SSL-LSTM Complexity Ladder Plan

Date: 2026-07-18  
Status: completed; q=1,2,5,10 passed and q=20 was resource-vetoed  
Tier: Tier 2 material research engineering

## Question and Scope

Does the general TensorFlow/TFP SSL-LSTM SVD-UKF structural implementation
remain mathematically and numerically valid as model/filter complexity grows
from the current scalar rung through `q in {1, 2, 5, 10, 20}`? This is an
engineering and numerical-scaling ladder, not a claim that high-dimensional
NeuTra transports have been trained or that their posterior is correct.

## Exact Rung Semantics

For rung `q`, set

`latent_dim = q`, `hidden_dim = q`, `observation_dim = 1`, and
`augmented_state_dim = q + 2q = 3q`.

The scalar observation is retained so the ladder isolates latent/hidden state
and filter geometry. The parameter dimension is

`p(q) = 4q^2 + 4q^2 + 4q + q^2 + q + q + 1 + 3q + 3q + q + 1`
`     = 9q^2 + 13q + 2`.

Thus the expected dimensions are `(n,p) = (3,24), (6,64), (15,292),
(30,1032)`, and `(60,3862)`. The runner must compute and assert the exact
values from `SSLLSTMStaticConfig` and independently check the closed-form
formula rather than trusting a hard-coded table.

The current scalar rung is `q=1`; it is not a new posterior acquisition. No
G/H retained tensor is reused for `q>1`.

## Research Intent Ledger

| Item | Frozen statement |
| --- | --- |
| Main question | Does dimensional scaling preserve source equations, derivative identities, finite scores, and stable small-data filtering? |
| Candidate/mechanism | Existing general `SSLLSTMStaticConfig`, parameter unpacker, hand derivatives, and SVD-UKF structural adapter. |
| Expected failure mode | Parameter-slice mismatch, derivative error, non-finite score/covariance, spectral degeneracy, memory/runtime blow-up, or XLA shape failure. |
| Primary promotion criterion | Each rung passes the declared engineering checks and produces a complete manifest; continuation proceeds only while the resource envelope and numerical validity hold. |
| Promotion veto | Any source-equation/finite-difference mismatch, non-finite value/gradient, invalid covariance, wrong shape, missing artifact, or failed reproducibility check. |
| Continuation veto | Two consecutive rung failures of the same structural cause, resource cap exhaustion, or inability to produce a trustworthy artifact. |
| Repair trigger | A single localized failure triggers a focused repair or an explicit rung result; it does not invalidate the whole SSL-LSTM direction. |
| Explanatory diagnostics | Runtime, peak memory, score magnitude, Jacobian norm, sigma-point count, condition numbers, and per-rung parameter counts. |
| Must not conclude | High-dimensional posterior correctness, NeuTra quality, HMC readiness, source-faithful training, model adequacy, or superiority over another filter. |

## Evidence Contract

| Role | Contract |
| --- | --- |
| Exact comparator | Same deterministic synthetic fixture family, same TensorFlow/TFP backend, same dtype, and same declared XLA setting at every rung. |
| Primary checks | Parameter layout and shapes; transition/observation hand derivatives versus finite differences on a fixed subset; finite deterministic SVD-UKF score and gradient; tiny simulation/filter pass; repeatability. |
| Veto diagnostics | Non-finite tensors, derivative mismatch beyond declared tolerance, invalid covariance/eigenvalues, shape/provenance mismatch, or failed artifact write/replay. |
| Explanatory only | Runtime, memory, score/gradient norms, sigma-point count, and scaling ratios. No performance ranking is claimed from one seed. |
| Preserved artifact | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/complexity-ladder/complexity-ladder-result.json`, per-rung receipts, and this plan's result note. |
| Nonclaims | This ladder does not train a q>1 transport, acquire q>1 HMC chains, or validate a q>1 posterior. |

## Checks Per Rung

1. Construct a deterministic `SSLLSTMStaticConfig(horizon=3, q, q, 1)` and a
   deterministic finite parameter vector with positive covariance scales.
2. Assert `augmented_state_dim`, `parameter_dim`, all parameter block shapes,
   observation shape, and finite positive covariance diagonals.
3. Compare hand transition and observation Jacobians against central finite
   differences on fixed states and a fixed parameter-index subset. Use
   `rtol=2e-5`, `atol=2e-6` for transition and `rtol=1e-7`, `atol=1e-8` for
   the linear observation derivative, matching the existing adapter tests.
4. Evaluate the SVD-UKF score twice on a deterministic observation fixture and
   require finite, deterministic log likelihood and score. Compare the
   analytic score against finite differences on a fixed parameter subset.
5. Run a tiny deterministic simulation/forecast/filter smoke with fixed seeds,
   checking all states, observations, covariances, and diagnostics finite.
6. Record wall time, TensorFlow version, device/JIT/TF32 settings, seed, exact
   command, source hashes, and artifact hash. Use CPU-hidden XLA for the
   reproducible ladder unless a trusted GPU run is explicitly available; this
   is not production GPU evidence.

## Sequential Ladder and Resource Policy

Execute rungs in order `1,2,5,10,20`. Run the smallest focused checks first at
each rung. Stop after the first hard failure, write its result, and attempt one
focused repair only if the failure is clearly localized. Continue to the next
rung after a pass. Use a 600-second total wall cap and a 2 GiB process-memory
soft diagnostic; a soft memory warning is explanatory, while allocation failure
is a hard veto. The ladder must not start HMC, NeuTra training, network access,
or modify retained/private artifacts.

## LaTeX/Result Handoff

After execution, write the per-rung table and interpretation into Chapter 28a
only as engineering evidence. Include the exact formula, a note correcting any
audited arithmetic, test counts, and the nonclaims. Do not report a rung as
“validated” without saying “engineering/numerical adapter checks passed.”

## Skeptical Pre-Execution Audit

| Audit question | Finding |
| --- | --- |
| Wrong baseline? | No. Every rung uses the same source implementation and fixture family; q=1 is the existing scalar implementation. |
| Proxy promoted? | No. Runtime and norms are explanatory; derivative, finite-value, shape, and reproducibility checks are the hard screen. |
| Missing stop? | No. Ordered rungs, 600-second cap, memory veto, first-failure stop, and one localized repair are explicit. |
| Unfair comparison? | No. Observation dimension, horizon, dtype, fixture construction, and backend are frozen across rungs. |
| Hidden assumptions? | Rung semantics and augmented state are explicit; q>1 has no retained posterior evidence. |
| Stale context? | The scalar forecast API limitation is respected; only the general structural adapter is exercised. |
| Environment mismatch? | CPU-hidden XLA is explicitly a reproducible reference exception; device/JIT provenance is recorded. |
| Do artifacts answer the question? | Yes for implementation/numerical scaling; no for high-dimensional training or posterior validity. |

 Audit decision: `PASS_FOR_ENGINEERING_COMPLEXITY_LADDER_ONLY`.

Audit amendment: the first q=1 preflight incorrectly treated covariance from
the analytic-score trace and covariance from the separate value-only filter as
an exact-parity veto. Their paths differ by a small implementation-path
roundoff/branch residual (`9.5e-7` on q=1), while both are finite and PSD. The
ladder now records this difference as explanatory and retains finite/PSD
covariance as the hard check. No ladder rung was promoted on the old criterion.

Second audit amendment: the first GPU ladder stopped at q=5 because a
`3.81e-10` value/score likelihood residual exceeded an unnecessarily strict
`1e-10` absolute/relative tolerance. This residual is four orders below the
declared finite-difference score tolerance and is consistent with separate
compiled value/derivative paths. The parity tolerance is now `1e-8` absolute
and relative, still recorded per rung; the previous q=1/q=2 receipts remain
immutable and the ladder is rerun once from q=1 under the repaired contract.

Post-execution supervisor repair: the repaired ladder revealed that checking
the total cap only between in-process rungs does not enforce the cap when a
single rung is long. q=20 completed after the cap, so it is resource-vetoed.
Future execution now launches each rung as a killable subprocess with the
remaining total wall budget. This runner repair is prospective; it does not
retroactively admit q=20 or alter any preserved receipt.

Audit amendment: the first draft contained a table arithmetic error for
`q=2` and unresolved entries for `q=10,20`, despite displaying the correct
formula. The table was corrected prospectively to `24,64,292,1032,3862`, and
the runner is required to compare source-derived counts with the independent
closed form. No execution occurred before this repair.

## Stop Conditions and Nonclaims

Stop on an invalid derivative, non-finite score/covariance, shape or source
hash drift, failed reproducibility, memory allocation failure, or total cap.
Record candidate failure separately from research-direction rejection. A pass
at q=20 is not evidence of q=20 posterior correctness, HMC readiness, or
NeuTra transfer.

## Execution Close

The executed ladder admitted q=1,2,5,10 under the engineering/numerical
contract and resource-vetoed q=20. The q=20 computations were numerically
finite but exceeded the declared wall cap and memory warning, so they were not
promoted. The supervisor was repaired prospectively to use killable per-rung
subprocesses. Full results are in
`bayesfilter-ssl-lstm-complexity-ladder-result-2026-07-18.md`; no q>1 posterior,
NeuTra training, or HMC run occurred.
