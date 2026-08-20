# LEDH Results Invalidation Notice

Date: 2026-08-21
Authority: owner directive, 2026-08-21.

## Ruling

ALL prior testing, benchmark, leaderboard, tuning, parity, and admission
results for LEDH-related algorithms in this repository are reclassified as
**HISTORICAL — DO NOT REUSE**. No agent may cite, extend, warm-start from,
or treat as baseline any pre-2026-08-21 LEDH/PFPF/SQMC result without first
re-establishing it under the conformance regime of
`docs/plans/bayesfilter-ledh-canonical-rebuild-plan-2026-08-21.md`.

This applies regardless of the individual document's internal honesty. The
reason is structural, established 2026-08-18..21 (full registry:
`docs/plans/bayesfilter-consolidated-issue-registry-2026-08-21.md`):

1. The NeuTra batch lane contained no particle flow at all (bootstrap
   proposal + OT reset), yet months of campaigns validated it.
2. No production lane implements the Li(2017) Algorithm 1 UKF per-particle
   covariance lifecycle documented as an implementation contract in
   `docs/chapters/ch19c_dpf_implementation_literature.tex`. The flow lane
   runs on placeholder Gaussians (`initial_covariance = transition_covariance
   = eye(18)`, `transition_matrix = eye(18)` for Austria), and the identity
   is also the SAMPLING covariance of the pre-flow proposal noise
   (`ledh_pfpf_genut_initial_rqmc_tf.py:738-741`), so no prior result
   reflects the documented algorithm's proposal distribution.
3. The dual-cap trust-region capability was forked per lane; claim-bearing
   lanes ran reduced reimplementations.
4. A faithful UKF implementation exists
   (`experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_alg1_ukf_tf.py`,
   June 2026 Alg1-UKF campaign) but was never wired into any
   `bayesfilter/` production lane.

## Classification of the invalidated corpus

- Results claiming or implying LEDH-PFPF-OT fidelity (leaderboards,
  admission rows, readiness gates for `ledh_pfpf_ot` arms): **wrong relative
  to the algorithm claim** — the executing lanes did not implement the
  documented algorithm.
- Results honestly scoped to their lane (e.g. `batch_diagonal_candidate`
  campaigns, SQMC variance ladders, compiler-mode localizations): **valid
  for their stated lane, historical for the program** — the lanes themselves
  are deprecated by the canonical rebuild; findings about compilers,
  numerics, and degeneracy mechanisms transfer as engineering knowledge but
  confer no algorithmic validation.
- The June 2026 Alg1-UKF experiments-tree campaign: methodologically
  genuine, but scoped to fixture models and never connected to production
  lanes; historical.

## What survives

- Engineering/diagnostic knowledge: grappler value/JVP program split, TF32
  seeding of Stage D blowups, trust-cap NaN removal (R2), Cholesky guard
  patterns, parity-oracle methodology, degeneracy mechanism analyses.
- The LaTeX algorithm specification (ch19c) — the contract for the rebuild.
- Frozen target fixtures, hashes, and the artifact corpus as provenance.

## Enforcement

The repository `AGENTS.md` carries a binding rule referencing this notice.
The conformance test suite (see
`docs/plans/bayesfilter-ledh-conformance-test-plan-2026-08-21.md`) is the
only path by which a LEDH result may re-enter claim-bearing status.
