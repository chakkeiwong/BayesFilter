# Corrected Parameter-Authority Phase 46 Repair and Refresh

Date: 2026-08-26  
Source result: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase46-result-2026-08-26.md`  
Branch: `n512_c_outside_two_bank_scalar_envelope`  
Next version: `v2.9-invariant-mutation-diagnostic`

## Interpretation

The proposal density is not stale: its recomputation residual is exactly zero
for all seven banks. Nonetheless, the third independent N=512 cloud falls
outside the first two N=512 scalar envelope on several raw support fields, and
the pairwise finite-cloud geometry remains variable. This weakens the claim
that larger N alone has stabilized the support draw. It does not prove a
globally incorrect proposal or an objective defect.

The common active mechanism that can explain this result is identity mutation
after resampling. Identity mutation is an invariant reference kernel but gives
no finite-run mode-mixing or duplicate-rejuvenation guarantee. The target,
theta measure, and whitening/HMC/LEDH vetoes remain unchanged.

## Repair decision

The next phase will compare identity mutation with a theta-space
Metropolis-Hastings rejuvenation kernel at the same fixed annealing stages. The
mutation proposal and acceptance ratio will be explicit, target/status calls
will remain batch-native, and the mutation comparison will be a finite support
diagnostic rather than a posterior or whitening claim. No NeuTra trainer state
will be promoted from this phase.

## Required gates and stop rules

1. A symmetric theta-space proposal has a bounded, predeclared scale and its
   MH acceptance ratio uses the tempered target `V_beta` at both endpoints.
2. Invalid target/status endpoints are rejected without contaminating the
   particle cloud; all retained values remain finite and in `theta_R4`.
3. The identity and MH arms use identical initial banks, annealing schedule,
   seeds, particle counts, and target signatures.
4. The mutation kernel is checked on an analytic finite fixture for invariance
   before q=20 execution; fixture failure is a harness veto.
5. ESS, mode mass, root diversity, and downstream transport residuals remain
   explanatory. No threshold is promoted from the mutation diagnostic.

An unavailable target/status API, an exact fixture contradiction, three
unrepaired infrastructure failures, or exhausted budget is a continuation
veto. Poor whitening or incomplete mode mixing is a repair trigger, not a
direction veto.
