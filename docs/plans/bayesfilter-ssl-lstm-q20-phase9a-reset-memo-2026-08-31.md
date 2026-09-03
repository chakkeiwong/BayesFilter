# Phase 9A reset memo

Date: 2026-08-31  
Current state: `PHASE9A_CLOSED_CONTINUATION_VETO_CHART1_BETA0`

The Phase 8 C5 freeze selected the K=2 compact-high, L3 `(0,.5,1)`,
pure-continuation protocol with fixed uniform chart selection.  Phase 9A then
rebuilt two fresh q=20 charts and attempted one active fixed-transport HMC
tuning handoff for each `(beta, chart)` scope.

Preserved evidence is under:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/phase9a-fresh-tuning-preflight/`

Attempt 01 was a runner import defect.  Attempt 02 showed that cap `0.25` was
too small (chart-0/beta-0 acceptance `0.984226`).  Attempt 03 passed the
localized chart-0/beta-0 repair at selected epsilon `0.810010` and acceptance
`0.859967`; its manifest records strict backend, memory growth, one trace per
reusable HMC graph, 306.109 seconds, and `1402670592` bytes (about 1.31 GiB)
peak allocation.  Attempt 04
passed all chart-0 scopes but chart-1/beta-0 requested `1.256879` above cap
`1.0`.  Attempt 05 repeated that scope failure under cap `2.0`: acceptance was
`0.998950` at epsilon `0.628978`, `0.939618` at `1.205189`, and the next
repair requested `2.410379`.  The tuner emitted
`tune_initial_step_size_exceeds_configured_cap`,
`verification_acceptance_outside_pass_band`, and `no_viable_candidate`.

The shared replica-exchange controller was not run because all six verified
handoffs were a prerequisite.  Do not interpret any artifact as whitening,
mode discovery, posterior, convergence, HMC-readiness, superiority, or scaling
evidence.  Do not launch Phase 9B from this state.

The complete result and decision tables are in
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-fresh-tuning-preflight-result-2026-08-31.md`.
Any continuation requires a new, separately audited repair subplan focused on
chart-1/beta-0.  It must choose between longer dual averaging, a fixed
candidate-step grid, or another explicitly justified scope policy, preserve
the target and acceptance contract unless deliberately revised, add progress
failure manifests with wall/device telemetry, and use a fresh output directory.
No additional cap widening is authorized by the closed Phase 9A plan.
