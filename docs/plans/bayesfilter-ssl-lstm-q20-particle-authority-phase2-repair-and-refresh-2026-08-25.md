# Phase 2 Repair and Refresh Note

Status: `PASS_GATE_REPAIRED_CANDIDATE_BRANCH`

Use the common inter-phase protocol. The leading alternative explanations for
a failed q=20 authority are, in order, (a) proposal support/tail failure, (b)
mutation not reaching the separated mode, (c) too few fresh particles, and
(d) target/status or bookkeeping defect. The repair must run the cheapest
diagnostic that separates these explanations before increasing particle count.

Required refresh for Phase 3:

- exact M0 protocol hash and target signature;
- accepted/rejected C0/M0 status and why;
- per-seed mass and mode summaries with MCSE/interval classification;
- proposal-tail and bridge findings;
- controls that are frozen versus warm-start hypotheses;
- whether M1-M4 are eligible, auxiliary-only, or blocked;
- remaining wall-time budget and a bounded one-factor arm order.

## Actual result and repair trigger

The N=16 and N=100 runs passed finite/status, density, support, frozen-hash,
beta-one, and finite-mass screens. At N=100, C0 retained 26 negative and 30
positive roots with weighted negative fraction `0.4585`; M0 retained 26 negative
and 21 positive roots with weighted negative fraction `0.6018`. These are
descriptive one-seed values, not a ranking or mode-mass estimate. The identity
kernel is mathematically invariant but leaves resampling-driven diversity only;
the next repair candidate is a scope-specific mutation kernel, not a relaxed
authority gate. Phase 3 proceeds on the M0 candidate while preserving this
limitation.

## Mutation repair execution

The opt-in repair route is
`random_walk_metropolis_symmetric_reference`: a batched Gaussian random walk in
the affine chart followed by the Metropolis ratio for
`(1-beta) log q + beta log p`. Its forward and reverse transition log densities
are equal by construction. The identity route remains the frozen historical
Phase 3/4 input; the repair route is never silently substituted.

| Attempt | Scope | Result | Interpretation |
|---|---|---|---|
| `phase2-mutation-attempt1-n16` | N=16, scale 0.5, one step | merge-mask shape error | implementation repair; preserved |
| `phase2-mutation-attempt2-n16` | N=16, scale 0.5, one step | hard gates pass; acceptance 0--3.1% | scale tuning failure, not invariance failure |
| `phase2-mutation-attempt3-n16-scale005` | N=16, scale 0.05, one step | hard gates pass; acceptance 12.5--21.9% | viable tuning hypothesis |
| `phase2-mutation-attempt4-n100-scale005` | N=100, scale 0.05, one step | hard gates pass in 594.7 s; acceptance 15.25--19.0%; zero invalid proposals | candidate mutation evidence |
| `phase2-mutation-attempt5-n100-identity-paired` | same N/seed/schedule, identity | hard gates pass in 82.6 s | paired descriptive control |

The repaired N=100 branch retained 34 negative and 22 positive unique roots,
weighted negative fraction `0.6671`, and ESS fraction `0.9868`; the paired
identity branch retained 28 and 16 roots, weighted negative fraction `0.5701`,
and ESS fraction `0.9857`. These are one-seed descriptive differences, not a
statistical ranking or mode-mass estimate. The mutation branch changes the
finite particle cloud and therefore cannot be promoted as an authority without
independent seeds, an exact SMC-U bookkeeping fixture, and uncertainty analysis.

## Refresh decision

The mutation route is retained as a named candidate repair. It does not replace
the original identity M0 input for already executed Phase 3/4 claims. Its bank
was propagated through an auxiliary Phase 3 revalidation and a role-limited
Phase 4 GPU screen; both passed hard role checks, with no posterior or HMC
claim. A future campaign should tune mutation scale and run independent seeds
before any M0 authority admission.
