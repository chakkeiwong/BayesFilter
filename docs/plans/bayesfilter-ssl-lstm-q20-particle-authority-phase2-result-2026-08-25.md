# Phase 2 Result

Status: `PASS_GATE_CANDIDATE`

Fresh q=20 pilots ran at N=16 and N=100 for paired C0/M0 arms with a
calibration-selected but claim-run-frozen schedule. All target/status,
density, support, protocol-hash, beta-one, and finite unnormalized-mass
screens passed. The N=100 run took `146.7 s` and is the Phase 3 input.

The M0 route uses a full-support defensive mixture and an identity mutation
kernel. Identity is exactly invariant but has no mixing power. At N=100 the
terminal weighted negative-mode fractions were C0 `0.4585` and M0 `0.6018`,
with both signs represented in each arm. These values are one-seed descriptive
diagnostics; they do not estimate exhaustive mode masses or rank C0/M0.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Keep M0 candidate viable | hard bookkeeping and support screens pass | no hard veto | mutation/mode reachability and SMC-U fixture transfer | run role-separated M1-M4 diagnostics, then mutation repair if needed | no authority admission |

The program continues because the identity-kernel limitation is a repair
trigger, not a real blocker.

## Repaired mutation branch

The same-scope symmetric random-walk Metropolis repair passed at N=100 with
scale `0.05` and one batched step per nonterminal stage. Acceptance was
`15.25--19.0%`, every proposal had valid target/status, and the transition
log-density symmetry residual was zero. Against a same-seed identity control,
the repaired branch had descriptive root counts `34/22` versus `28/16` and
weighted negative fraction `0.6671` versus `0.5701`. This is candidate evidence
that mutation can alter finite-cloud diversity; it is not a mode-discovery
guarantee, an authority admission, or a statistical superiority result.

The repaired branch is preserved separately and was not substituted into the
already completed Phase 3/4 claim path. It is the smallest next candidate for
an independent-seed SMC-U campaign.
