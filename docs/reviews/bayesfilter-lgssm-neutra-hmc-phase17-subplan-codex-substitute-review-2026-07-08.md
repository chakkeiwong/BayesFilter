# Codex Substitute Review: LGSSM NeuTra Phase 17 Subplan

Date: 2026-07-08

## Scope

Claude review was requested for:

`docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md`

The attempted Claude review gate was rejected by the sandbox approval reviewer
as an external-service disclosure risk. No workaround was attempted. This
record is a same-foreground Codex substitute review of the single subplan path.
It is weaker than an independent Claude review and does not authorize any
human, runtime, product, default-policy, or scientific-claim boundary.

## Findings

No blocking issue found.

The repaired subplan now contains:

- exact pre-packaging checks:
  `python -m py_compile ...`,
  `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_gpu_affine_payload_tf.py -q`,
  and a source/config scan for Phase 16/17 tokens and stale Phase 10/11 source
  tokens;
- an exact CPU-hidden packaging command using
  `--phase16-training-state-path` and the Phase 16 GPU/XLA training-state JSON;
- exact JSON parse and field-validation commands for the Phase 17 payload and
  validation artifacts;
- explicit prohibition on new training, fixed-transport HMC mechanics, HMC
  sampling/tuning, external sample generation, stale Phase 10/11 artifacts,
  and `jit_compile=false`;
- clear handoff that Phase 18 owns the fixed-transport HMC mechanics/XLA
  authority and compile/timing/size gate;
- nonclaims that prevent HMC readiness, posterior correctness, XLA compile
  readiness, production readiness, default readiness, and scientific validity
  from being inferred from packaging.

## Local Evidence Checked

- `python -m py_compile bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_gpu_affine_payload_tf.py -q`: passed, `5 passed, 2 warnings`.
- Source/config scan for Phase 16/17 required tokens and stale Phase 10/11
  source tokens: passed with `missing=[]`, `violations=[]`.
- `git diff --check -- bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md`: passed.

## Boundary Verdict

The subplan is consistent, feasible, artifact-covered, and boundary-safe for
Phase 17 packaging-only execution. It does not authorize Phase 18 mechanics,
compile, chain, or HMC sampling work.

VERDICT: AGREE
