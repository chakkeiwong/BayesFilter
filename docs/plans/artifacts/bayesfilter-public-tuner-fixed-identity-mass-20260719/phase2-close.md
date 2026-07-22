# Phase 2 Close: Fixed-Identity Public Tuner

Status: PASS

The public `HMCKernelTuningConfig.mass_policy` option now accepts
`windowed_adaptive` (unchanged default) and `fixed_identity`. The fixed policy
constructs identity covariance in the caller coordinates, ignores geometry
hints, skips empirical mass replacement, and carries the policy through Phase
4, Phase 5, Phase 6, and Phase 7. Phase handoffs reject a changed mass
signature; final private and public kernel payloads include `mass_policy` and
the fixed-identity signature is checked before admission.

Evidence:

- `python -m py_compile bayesfilter/inference/hmc_kernel_tuning.py bayesfilter/inference/hmc_tuning.py bayesfilter/inference/neutra_end_to_end.py` passed.
- Focused suite: `212 passed, 1 skipped`.
- Fixed-identity geometry/windowed/migration regressions: `56 passed, 1 skipped`.
- Default adaptive behavior remains covered by the existing geometry, windowed,
  public API, and outer-loop tests.

Review limitation: the Claude health probe returned `CLAUDE_PROBE_OK`; the
substantive packet review was rejected by the environment privacy guard. This
is recorded as reviewer unavailable, not as scientific approval or a numerical
exception.

Nonclaims: this closes the API and handoff implementation only. It does not
establish NeuTra posterior correctness, convergence, sampler superiority, or
default readiness.

Handoff: proceed to the canonical NeuTra route preflight with a fresh output
directory, GPU/XLA enabled, target acceptance `0.70`, acceptance band
`[0.65, 0.75]`, and `mass_policy=fixed_identity`.
