# Phase 3 Close: Canonical NeuTra Public Route

Status: `PASS_ENGINEERING_ROUTE`

The active NeuTra runner now calls public `tune_hmc_kernel` with
`mass_policy="fixed_identity"`, target acceptance `0.70`, acceptance band
`[0.65, 0.75]`, GPU/XLA execution, and BayesFilter-owned retained-kernel replay.
It contains no active import or call to the specialized fixed-transport tuner.

Three fresh preflight attempts were preserved:

- attempt 01 exposed a harness classification bug: a public hard veto was
  incorrectly reported as an engineering pass;
- attempt 02 correctly exposed that TFP bootstrap invokes rank-1 target states,
  while `BatchNativeBoundAdapter` accepted only rank-2 batch states;
- attempt 03 passed the engineering route after rank-1 delegation to the base
  adapter and exact transformed-target binding for public tuning and replay.

Attempt 03 used current target signature
`bd40a828bc4916e5e09a8e6135f315ebc45c06844aed38a506d6296c2642557d`,
TensorFlow 2.19.1, GPU memory growth, TF32, and XLA. Its public tuner ended
`budget_exhausted` with no hard veto, which is expected non-promoting evidence
for the deliberately tiny one-step/tiny-budget preflight.

Additional focused regression: rank-1 adapter calls now match the scalar base
adapter and rank-2 calls match the inspected batch-native binding.

Review limitation: Claude's health probe previously returned
`CLAUDE_PROBE_OK`; substantive review was rejected by the environment privacy
guard. This is reviewer unavailability, not agreement and not a blocker to
trusted local execution.

Nonclaims: Phase 3 establishes route, target, device, and artifact mechanics
only. It does not establish tuning admission, convergence, truth recovery, or
NeuTra scientific validity.

Handoff: Phase 4 may validate a fully trained preserved transport that is bound
to the current target signature and passes its own artifact-integrity checks.
