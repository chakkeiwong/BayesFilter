# Phase 0 Result: Exact 18D LGSSM Target Extraction And Identity

Date: 2026-07-13  
Status: `PASS_PHASE0_EXACT_TARGET_EXTRACTION`  
Plan: `docs/plans/bayesfilter-lgssm-neutra-knowledge-transfer-and-serious-validation-plan-2026-07-13.md`

## Outcome

The validated 18-parameter `T=120` deterministic LGSSM target is now available
through a reusable fixture-bound adapter. The completed benchmark driver was
not changed, so historical HMC replay identities remain intact.

The new target identity binds:

- complete config payload and file SHA-256;
- fixture artifact hash, file SHA-256, and observations hash;
- source-contract payload and file SHA-256;
- parameter order, coordinate convention, and prior Jacobian convention;
- the two TensorFlow target/source implementation files.

## Evidence

| Check | Result |
| --- | --- |
| Exact target signature | `f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30` |
| Exact adapter signature | `42dc7bad0137fd9c31aa1d618bb4e560f68d1bbe3a7ab4f5ef95e458b2abc985` |
| Pointwise legacy value parity | Bitwise equality at five deterministic perturbations |
| Pointwise legacy score parity | Bitwise equality at five deterministic perturbations |
| Target-status parity | Bitwise equality for every status field |
| CPU/XLA batch compile | Passed, one concrete graph, finite `[2]` values and `[2,18]` scores |
| Raw fixture tampering | Rejected by embedded artifact hash |
| Rehashed changed fixture | Rejected by expected target signature |
| Focused tests | `5 passed` |

## Files

- `bayesfilter/testing/deterministic_lgssm_exact_target_tf.py`
- `tests/test_deterministic_lgssm_exact_target_tf.py`

## Command

```text
CUDA_VISIBLE_DEVICES=-1 \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m py_compile \
  bayesfilter/testing/deterministic_lgssm_exact_target_tf.py \
  tests/test_deterministic_lgssm_exact_target_tf.py

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-neutra \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m pytest -q \
  tests/test_deterministic_lgssm_exact_target_tf.py
```

GPU devices were intentionally hidden. This was a small target-parity and
CPU/XLA compile check, not NeuTra training or serious sampling.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Proceed to trainable transport/controller implementation | Reusable adapter is exactly equal to the completed campaign adapter and identity is artifact-bound | No Phase 0 veto fired | Both adapters use the same checked target implementation | Implement Phase 1 transport and deterministic training controller | Analytic posterior correctness, NeuTra quality, HMC convergence, robustness, or default readiness |

## Handoff

Phase 1 may use only the new target signature for newly trained/frozen NeuTra
artifacts. It must not rewrite historical target or HMC signatures.

