# Zhao-Cui Austria SIR Observed-Data Score Reset Memo

Status: `SUPERSEDED_HISTORICAL_APF_RESET_NOT_ACTIVE`

> Superseded by
> `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-reset-memo-2026-07-30.md`.
> The parent-dependent APF compiler described below is not the active next step.

Date: 2026-07-30
Historical terminal state: `T1_BRIDGE_COMPLETE_PROPOSAL_QUALITY_BLOCKED`

The implementation campaign closed before T2. Do not restart from the old
proposal-preflight blocker: the sealed-target T1 trainer, artifact loader,
exact-prior transport, complete source-order compiler, and runner stage now
exist in `bayesfilter/highdim/zhao_cui_austria_sir_proposal_tf.py` and
`scripts/run_zhao_cui_austria_sir_observed_data_score.py`.

The first unsatisfied gate is downstream proposal quality. The terminal
artifact has initial ESS `8/8`, KR round-trip `5.95e-8`, T1 ESS fraction
`0.125`, and log-weight spread `94.8`. Do not advance to T2 by reusing this
artifact, relaxing the ESS gate, using the retained-grid route, or transferring
P76/P77 physical-density settings.

The next smallest discriminating repair is a compiler that lets the current
coordinate map depend on the selected parent, so the TT can operate in
process-innovation coordinates. This is an `extension_or_invention` and needs a
focused reviewed subplan because it changes the compiler/map contract. A
reviewed compartment or neighborhood factorization is the alternative if the
parent-dependent full coupled route remains infeasible.

Before another run:

1. Preserve exact Gaussian initial proposal density equality.
2. Preserve sealed `y1` and latent-preclip target identities.
3. Predeclare T1 ESS fraction `>=0.5` and round-trip `<=1e-4`.
4. Keep L1 `0` as a comparator and positive L1 as the policy-eligible arm.
5. Use fresh calibration/validation seeds and a new versioned attempt directory.
6. Do not interpret held-out cross-entropy alone as proposal-quality evidence.
