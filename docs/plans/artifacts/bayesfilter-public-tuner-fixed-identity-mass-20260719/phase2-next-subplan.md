# Phase 3 Subplan: Canonical NeuTra Route

Objective: prove that the active NeuTra caller uses only the public
`tune_hmc_kernel` route and the BayesFilter-owned replay handoff.

Entry conditions: Phase 2 close is PASS; the focused tests and route scan pass;
the preserved LGSSM transport and target signatures are available.

Required artifacts: fresh GPU/XLA preflight output, public tuner config/result,
route manifest, and this phase close record.

Required checks: target signature and dimension binding; target acceptance
`0.70`; acceptance band `[0.65, 0.75]`; fixed-identity policy and unchanged
mass signature; no specialized tuner import/call; no campaign-local sampler or
fixed 1,000-draw R-hat gate.

Evidence contract: this phase establishes route and artifact validity only. A
preflight may be non-promoting and cannot support posterior or scientific
claims.

Forbidden claims/actions: do not claim NeuTra correctness from preflight; do
not retrain the preserved transport for the serious LGSSM check; do not use the
specialized fixed-transport tuner as fallback; do not overwrite prior output.

Handoff: preflight emits a valid public tuner artifact and the next phase may
run the preserved-transport LGSSM validation. Stop on target/signature drift,
nonfinite target status, mass mutation, or missing artifact.
