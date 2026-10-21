# M13 endpoint-corrected surrogate-HMC mechanics plan — 2026-09-24

## Research intent

The question is whether the repository's surrogate-HMC kernel can execute a
small, deterministic mechanics run for M13 while using the canonical
analytical DCTR force only for the proposal and a separately declared endpoint potential for Metropolis correction. M13 is the range/bearing LGSSM-derived
fixture with state dimension 4, observation dimension 2, horizon T=20, and
parameters `(lambda_Q, lambda_R)`.

The candidate mechanism is the existing frozen position-only neural-force HMC
map in `bayesfilter/inference/neural_force_hmc.py`. Its force argument is the
gradient of the scalar potential because the leapfrog kick is `p <- p -
epsilon * force`. The canonical target returns the analytical derivative of
finite canonical log likelihood, so the proposal field is
`force = -score_DCTR + theta/(log(2)^2)` for the negative log posterior.

The endpoint is an independent diagnostic function: a fixed-stateless-seed,
FP64, full-support defensive-importance estimate from
`trajectory_importance_reference`, plus the exact independent Gaussian prior.
It is deterministic for a given position and seed, but it is still a finite
Monte Carlo likelihood approximation and is not a qualified R^2 posterior
oracle. It is not a qualified R^2 posterior oracle. The endpoint and force therefore
answer the mechanics question while remaining visibly different numerical programs.

## Scope and defaults

* M13 fixture, calibration partition, attempt 4 (`data_seed=1704`), untouched
  claim data are not consumed.
* Canonical force: particle count 4096, exact transport chunk policy
  (`K=2048`), FP32, TensorFlow TF32 enabled, XLA compiled, fixed design
  partition 9, and the unselected `correction_lm_multiplier=0.01` candidate.
  The control-selection stage found no stable pair, so this candidate is
  diagnostic only, has no HMC tuning authority, and cannot issue a tuning artifact or promotion decision.
* Endpoint: full-support wrapped angular likelihood, FP64 defensive-importance
  estimator, stateless seed `[2026092402, 9101]`, and exact N(0, log(2)^2)
  prior. The endpoint is cached only through the HMC current-potential state;
  every proposed endpoint is recomputed by the independent function.
* HMC mechanics: one chain, initial position `[0, 0]`, identity inverse mass,
  fixed step size `0.002`, two leapfrog steps, 8 warm-up transitions and 16
  retained transitions. These are mechanics fixtures, not tuned values.

## Evidence contract

The primary criterion is a complete finite execution of all 24 transitions with
finite endpoint values, finite force/proposal values, no endpoint numerical
failure, no force fallback, and no divergence. The report must preserve the
per-transition acceptance and Hamiltonian traces, endpoint validity, force
validity, source hashes, seeds, and exact target identities.

A failed finite-status check, force fallback, divergence, endpoint invalidity,
non-deterministic endpoint replay, or missing artifact is a hard veto for this
mechanics stage. Acceptance rate, Hamiltonian error, and endpoint/force value
differences are explanatory diagnostics only; they do not establish mixing,
convergence, posterior correctness, tuning quality, or scientific validity.

The stage does not conclude that the canonical DCTR force is accurate for the
independent endpoint, that M13 controls are tuned, that the finite-box
reference is an R^2 posterior reference, or that any HMC posterior estimate is
admissible. Those questions require the later control, uncertainty, and
endpoint qualification stages.

## Skeptical audit before execution

The baseline is the actual independent endpoint potential, rather than the
DCTR likelihood reused as both proposal and acceptance target. The force and
endpoint use the same raw parameterization and the prior is included in both,
so the sign and target mismatch are exposed. The endpoint's finite importance
estimate is explicitly diagnostic and cannot become a promotion metric. The
short fixed configuration is a mechanics fixture; it is not a replacement for
public HMC tuning. All data, particle/design scope, dtype, TF32/XLA settings,
seeds, and source hashes are recorded. The stage stops on invalid numerical
status or artifact failure and records those failures without converting them
into posterior claims.

## Budget and command

This is a new bounded master-program pass within the already authorized
campaign totals. Its pass reservation is 800 CPU worker seconds and 1200 GPU
seconds; the review job reserves 180 CPU seconds and the worker 500 CPU seconds.
The output root is unique and is never overwritten:
`docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-mechanics-01`.
The master command is:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py \
  --stage closure --closure-stages m13-endpoint-mechanics \
  --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-endpoint-mechanics-01
```

## Expected interpretation

A pass closes only the engineering question that the frozen proposal field and
independent endpoint can be called together with finite, traceable transitions.
A failure identifies whether the endpoint, force, sign, scope, or kernel
contract failed and triggers a focused repair under the same bounded pass. It
does not reject surrogate HMC as a research direction.
