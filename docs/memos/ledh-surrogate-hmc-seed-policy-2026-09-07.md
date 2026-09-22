# LEDH Surrogate-Force HMC Seed Policy

**Date:** 2026-09-07
**Phase:** 0, Task 0.2
**Authority:** Corollary 5.2 and Remark 5.3, `docs/bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex`

---

## Policy

**One master particle-noise realization ω, frozen for the entire HMC chain.**

- Draw ω once, before any chain starts.
- Freeze it: every force and value evaluation in every arm reuses the same ω.
- ω may depend on the chain's initial θ₀, because θ₀ is part of the frozen
  run context. ω must **not** depend on the current θ during leapfrog.
- No reseeding per leapfrog step, per trajectory, per adapter call, or per
  arm. The exact-force and damped-force arms share one ω.

Derive sub-seeds from ω by `SeedSequence`-style hashing, never by
consecutive integers (see [[tf-consecutive-from-seed-is-one-stream]]:
`from_seed(s)` and `from_seed(s+1)` are shifted views of a single Philox
stream, so consecutive derivation produces pseudo-replication rather than
independent draws).

---

## Why the theorem requires this

Corollary 5.2 (note line 953) states that the algorithm leaves
π(θ,p) ∝ exp(−H(θ,p)) invariant, hence preserves the θ-marginal
∝ exp(−U(θ)), **for any quality of force F**. That is the whole value of the
construction: score bias becomes a mixing question rather than a correctness
question.

Remark 5.3 (note line 980) states the price:

> F must be a deterministic function of θ alone: all Monte Carlo seeds inside
> F must be frozen, and F may not depend on momentum or on the trajectory's
> history.

Two propositions supply the invariance, and both are premised on that
determinism:

- **Measure preservation** (line 874): each leapfrog substep preserves
  Lebesgue measure for every measurable F.
- **Reversibility** (line 891): S = F∘Ψ^L is an involution for every F and
  every L. The proof closes with "no palindromic step arrangement is needed
  because F depends only on θ, so the same substep sequence serves both
  directions."

If ω is redrawn during a trajectory, F is no longer a function of θ, the
involution argument fails, and Metropolis–Hastings no longer targets π.

---

## Why θ-dependent seeding is specifically wrong

A tempting shortcut is `seed = hash(θ)`, which is reproducible per θ and so
looks deterministic. It is not admissible, for a stronger reason than
non-determinism.

The note's branch proposition (line 1166) proves that a θ-dependent branch in
the executed program makes θ ↦ L̂^N(θ) **discontinuous** at the crossing
surface, and that "no same-scalar derivative claim holds across the crossing."
The proof is by the implicit function theorem: the crossing set is locally a
hypersurface separating two open regions running different programs, and the
one-sided limits differ.

`hash(θ)` is maximally θ-dependent — every infinitesimal move in θ lands on a
different random stream. So U would be discontinuous everywhere, no gradient
would exist, leapfrog energy error would be unbounded, and acceptance would
collapse. The failure would present as "damping too aggressive" while the real
cause was a discontinuous potential.

The note's own conclusion (line 1189): "θ-independent reset schedules are the
safe policy for the repository's fixed-branch score contract."

Assumption 2 (line 113) makes the same requirement from the other direction:
the manual recursive tangent equals the total derivative of the executed
scalar only on an open set where every Cholesky argument stays positive
definite, every Sinkhorn row mass exceeds its floor, the cost-scale maximum
branch is locally constant, and no validity gate changes state.

---

## Verification tests (Phase 3)

Each test targets one premise. All three are cheap and CPU-only.

### V1 — Determinism

Two adapters constructed with the same master seed must produce bitwise
identical value and force at the same θ.

Premise tested: F is a function of θ.
Failure means: ω is being redrawn, or hidden state leaks into F.

### V2 — Reversibility (involution)

Integrate L steps forward, flip the momentum, integrate L steps back, flip
again. The result must return to the start within integrator tolerance.

Premise tested: S = F∘Ψ^L is an involution (line 891).
Failure means: F depends on trajectory history, call order, a cached cloud, or
accumulated float state. This is strictly stronger than V1 — V1 only probes
the same θ twice in a row, while V2 probes F along a path.

### V3 — No call-count dependence

Evaluate F at θ, then evaluate F at 50 other points, then return to θ. Both
evaluations at θ must agree.

Premise tested: F carries no evaluation counter or cache.
Failure means: hidden mutable state in the adapter.

---

## What this policy does not claim

- It does not make U_N^ω an unbiased estimator of the exact log-likelihood.
  The note is explicit (line 132) that L̂^N is a biased approximation of
  log p_θ(y_{1:T}) at finite N, because the OT/moment reset is not an
  unbiasedness-preserving resampling. Freezing ω fixes the *target*; it does
  not remove finite-N bias.
- It does not establish that results at one ω generalize. Each ω defines its
  own π_N^ω. Single-ω evidence is one draw from a distribution over
  pseudo-posteriors.
- It does not address acceptance or mixing. Remark 5.3 closes by saying "no
  claim is made about acceptance rates or mixing; those are empirical
  questions for a planned experiment." This program is that experiment.
