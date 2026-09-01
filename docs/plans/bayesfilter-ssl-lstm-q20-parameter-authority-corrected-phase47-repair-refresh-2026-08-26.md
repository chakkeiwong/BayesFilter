# Corrected Parameter-Authority Phase 47 Repair and Refresh

Date: 2026-08-26  
Source result: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase47-result-2026-08-26.md`  
Branch: `mh_rejuvenation_does_not_reduce_variability`  
Next version: `v3.0-independent-proposal-mutation`

## Interpretation

The valid Phase 47 execution established that the theta-space target/proposal
measure, status handling, pairing, and symmetric local MH implementation are
auditable. The two-step isotropic random-walk arm moved particles in all three
replicates, but did not reduce the declared between-replicate support-spread
vector. The result is descriptive and has no uncertainty-supported ranking.

The first attempt's failure was a harness-only false gate and was repaired in a
fresh root. It must not be pooled with the valid run as scientific evidence.

The negative result weakens the specific explanation "lack of local MH
rejuvenation is the dominant cause". It does not test a nonlocal invariant
kernel: an isotropic step can remain within a mode, while the fixed defensive
mixture already provides a normalized full-support independent proposal.

## Repair decision

Run a new paired diagnostic comparing identity mutation with an independent-
proposal MH kernel. At each nonterminal beta stage, draw `theta' ~ q` from the
declared defensive mixture and use

`log a = min(0, [log pi_beta(theta') - log q(theta')] - [log pi_beta(theta) - log q(theta)])`.

This is the exact independent-MH ratio for the same theta proposal density. It
can cross separated components without changing the target, measure, or
proposal receipt. The arm remains role-limited: no finite-run posterior,
whitening, HMC, LEDH, or default claim is allowed.

## Required gates and stop rules

1. An analytic independent-MH fixture checks the beta-zero identity and finite
   shifted-target movement before q=20 execution.
2. Candidate rows are sampled from the same normalized defensive mixture whose
   log density enters the acceptance ratio; candidate target/status failures
   are rejected.
3. Identity and independent-MH arms share exact initial tensors, resampling
   seeds, schedule, particle count, target signature, and proposal geometry.
4. All retained rows remain finite `[N,4]` theta rows; no internal UKF state is
   passed to the mutation kernel.
5. Raw acceptance, root count, mode mass, ESS, and support-spread values remain
   explanatory; no threshold or ranking is promoted.

An unavailable target/proposal, fixture contradiction, three unrepaired
infrastructure failures, or exhausted campaign budget is a continuation veto.
Poor support or whitening remains a repair trigger.

## Next subplan

`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase48-subplan-2026-08-26.md`
