# Supplied initial mass and optional adaptation

The owner requests a supplied initial mass that can be held fixed or adapted.
Scope: standalone Walnutpie TF/XLA API, preserving existing callers and the
identity-mass replication. No public BayesFilter defaults change.

## Audit and design

Existing `initial_state(mass=...)`, `Initialization(mass='supplied')` and
`Config.adapt_mass` already support positive diagonal masses. The main lifecycle
factory does not expose supplied initialization, and documentation/tests do not
fully exercise fixed-vs-adapting runs. Do not implement a duplicate adaptor.

Mass means momentum covariance M: p~N(0,M), kinetic .5 p' M^-1 p. Existing
state stores diagonal M^-1; supplied values must not be confused with Stan's
inverse metric. Provide a new complete factory with initialize/fit operands
q, seeds, mass [chains,dimension], epsilon [chains], model data, and ordinary
warmup/sample continuations. Adapt_mass=False freezes M through warmup while
step adaptation remains active; True initializes upstream position/score
moments from supplied M and adapts only during warmup. Sampling always freezes.

Support an optional dense SPD initial mass without duplicating the tree: use
fixed linear coordinates y=L' q, p=L r for M=L L', transforming model to
log pi(L'^-1 y)-log det L. The internal diagonal metric starts at identity.
Frozen mode yields exactly the supplied physical M; adapting mode uses the
existing diagonal adaptation in this fixed basis, giving physical
M_eff=L diag(m_internal) L'. This is adaptation in a fixed dense basis,
NOT unrestricted dense covariance learning. Expose this distinction and
physical conversion helpers. Input matrix may be shared or chain-specific;
invalid dimensions/nonfinite/nonsymmetric/non-SPD values fail at validation.

## Checks and evidence contract

Engineering question: does the user-supplied metric define the intended kinetic
energy, survive checkpoints and remain fixed exactly when requested? Primary
checks: inverse-mass identity; fixed warmup/sample equality; adaptive warmup
changes only intended moments; sampling freezes metric; split/full continuation;
dense coordinate/gradient/Jacobian algebra; dense identity reproduces existing
diagonal path; invalid metrics rejected or explicit invalid state; single
stable XLA trace and no host callback on CPU and GPU.

Use deterministic and R2P kernels on small anisotropic Gaussian fixtures.
Fixtures are derived SPD examples, not sampler tuning recommendations. Focused
tests are engineering evidence only; do not assert improved funnel convergence
or native speed based on them. Any dense implementation must keep TF model
evaluation and transformations inside XLA. No NumPy in numerical runtime.

Implementation: extend existing API/initializers minimally; add one metric
coordinate module, focused tests and examples. Initial mass is a runtime
operand, not embedded Python data causing per-metric retracing. Existing
checkpoint format stays intact; dense basis is an explicit immutable operand
passed to each call and must remain fixed across checkpoint continuation.

Budget: <=20 minutes focused CPU/GPU validation, at most two localized repairs
per discovered defect. Write results under a fresh supplied-mass artifact root.
Stop on failed algebra/invariants, non-XLA fallback or invalid GPU allocation.

## Skeptical review

Checked wrong inverse convention, loss of mass at fit initialization, warmup
overwriting a frozen input, shape/chain mixing, stale target cache after basis
change, unsupported dense-adaptation claims and hidden NumPy/host paths. Reuse
existing diagonal numerical authority and make dense-basis adaptation explicit.
The separate million-draw campaign snapshots sources before these edits and
uses frozen identity mass. Audit passes subject to focused invariant tests.

Implementation review caught a material coordinate issue before tests: the
paper Definition 10 (p' M^-1 delta_q) is not invariant under applying the
ordinary Euclidean stopping condition in y. Preserve the physical condition
as r' diag(inv_y) L^-1 L'^-1 delta_y. A small private kernel factory changes
only this stopping calculation and reuses all tree/integrator/adaptation code.
Add a negative-control fixture whose physical and naive y stopping signs
differ. Without this correction, a claim of the same dense-mass algorithm
would be wrong despite correct Hamiltonian dynamics.
