# Surrogate-HMC evidence recovery and bounded diagnosis

The owner's question is whether an imperfect DCTR force can support reliable
model-parameter inference across the repository's models. This recovery uses
merged `main` (`6fbcf314`) as the source baseline and the existing isolated
execution branch (`8a4cffa2`). It preserves earlier programs and results.

## Evidence contract and skeptical audit

Recover the program history, actual acceptance target and force, toy evidence,
LGSSM successes and failures, model coverage, and resource accounting. The
deliverable is an evidence map with executable checks for newly discovered
implementation defects, an active continuation plan, and explicit unknowns.

The September 7 executable program targets the frozen finite-particle
posterior. The owner's intended model-inference claim also requires comparison
with the model posterior. These are separate comparisons. Force disagreement
is explanatory and may trigger a repair; it is not itself a rejection of the
surrogate-HMC mechanism. Failure to approximate the model posterior vetoes a
model-inference claim, but does not answer whether HMC samples its finite target.

The audit found a potentially material error in the September 7 Gaussian toy
script: its potential reduces over both chain and coordinate axes, but its
acceptance and state updates act separately on each chain. Before relying on
the recorded PASS, check the actual function on two explicitly different
states against the exact per-chain Gaussian energies. Also check whether the
reported covariance-Frobenius statistic is the asserted Wasserstein distance.
These deterministic arithmetic checks answer the artifact-validity question;
rerunning a short chain would not repair that evidence.

## Bounded actions and diagnostic roles

- Preserve the ignored toy summary from the shared checkout, recording its
  checksum and the distinction between local provenance and a committed run
  manifest. Recover original program snapshots without editing historical text.
- Trace consumer-to-implementation calls statically. Distinguish a static
  source check from an executable endpoint parity check.
- Run at most two focused CPU reference/debug checks, at most 120 wall seconds
  each and four aggregate CPU-worker minutes in total. Hide GPUs explicitly
  before importing TensorFlow and restrict worker threads. No GPU, HMC chain,
  candidate tuning, or held-out claim run is authorized by this diagnostic plan.
- Primary diagnostic criterion: the toy acceptance energies equal independent
  Gaussian energies for each chain. A mismatch invalidates that demonstration,
  not the general surrogate-force argument. The distance identity is a second
  arithmetic criterion; finite outputs and acceptance rates cannot substitute.
- Preserve complete logs and machine-readable results in
  `docs/plans/artifacts/surrogate-hmc-recovery-20260918-01/`.

No scientific promotion or ranking follows from these checks. The cheap
adversaries here are exact per-chain Gaussian energy arithmetic and the exact
one-dimensional Gaussian Wasserstein formula, evaluated on unequal chains and
unequal small variances respectively. Stochastic heuristic comparisons belong
in the later model-inference plan, not in these deterministic checks.

## Assumptions, budget and stops

Historical names such as “exact force,” “canonical,” “W2,” and “complete” are
claims to verify, not accepted defaults. File dates and headings do not prove
which code ran. The explicit CPU exception is appropriate for tiny arithmetic
counterexamples and does not establish GPU/XLA performance or readiness.

The containment ledger reports 1309.310489 GPU seconds and 3187.843224 aggregate
CPU-worker seconds charged against 64 GPU-hours and 16 CPU-worker-hours. Its
three substantive LG1 calibration attempts are exhausted. Verify the follow-up
audit accounting separately. Do not restart that count or launch a fourth
candidate under a different label. The older surrogate campaign's remaining
budget must be reconstructed or marked unknown, not added to this allowance.

Stop the focused checks for missing source, unexpected import side effects,
nonfinite reference arithmetic, or the stated time limit; preserve the failure.
Continue independent evidence recovery. A new inference run requires a concrete
target/reference/tuning protocol and honest treatment of the exhausted attempt
allowance before launch.
