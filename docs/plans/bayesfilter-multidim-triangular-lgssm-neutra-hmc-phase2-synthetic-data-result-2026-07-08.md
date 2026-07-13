# Phase 2 Result: Lower-Triangular LGSSM Synthetic Data Fixture

Date: 2026-07-08

## Decision

`PASS_PHASE2_SYNTHETIC_FIXTURE_VALID`

Phase 2 generated and validated the fixed-truth 4D lower-triangular LGSSM
synthetic fixture specified by the Phase 1 contract. No NeuTra training, HMC,
posterior/reference sampling, algorithmic implementation edit, or scientific
promotion was performed.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the synthetic fixture instantiate the Phase 1 target with recoverability diagnostics and no hidden nonstationarity? |
| Baseline/comparator | Phase 1 contract JSON/result and stationary Lyapunov residual. |
| Primary criterion | Valid data artifact with fixed truth, stationary initial law, hashes, and moment sanity checks. |
| Veto diagnostics | Contract mismatch, nonstationary `A`, invalid covariance, missing seed/truth/hash, weak or degenerate signal, malformed JSON. |
| Result | Pass for synthetic fixture generation; no posterior/HMC gate is passed. |

## Generated Artifacts

| Artifact | Path | Hash |
| --- | --- | --- |
| Data JSON | `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_synthetic_data_v1_seed20260708.json` | file sha256 `sha256:d3944c6c38f40031dbdfa28d17d1ac9650740604f3202b74113502ddcac6ae01` |
| Data payload | same data JSON | canonical payload hash `sha256:84e80352e4293f8c888142a760bd81dafa52de52145d6c769bb0ecb827f7bcb4` |
| Manifest JSON | `docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_synthetic_data_v1_manifest_seed20260708.json` | manifest payload hash `sha256:42711ed8c31b6806644c630860019a99ae7fe1da33b33b809ce4187d2fd1cdb0` |

Hash convention:

- `data_payload_hash` is SHA-256 over the canonical JSON payload before adding
  the `data_payload_hash` field.
- `data_file_sha256` is SHA-256 over the final data JSON file bytes and is
  recorded in the manifest.
- `manifest_payload_hash` is SHA-256 over the canonical manifest before adding
  the `manifest_payload_hash` field.

## Fixture Summary

| Field | Value |
| --- | --- |
| Seed | `20260708` |
| State shape | `[256, 4]` |
| Observation shape | `[256, 4]` |
| Transition matrix | lower triangular |
| Observation matrix | `I_4` |
| Process covariance | diagonal positive |
| Observation covariance | diagonal positive |
| Initial law | stationary Gaussian `N(0, P_inf)` |

## Diagnostics

| Diagnostic | Value | Gate Role |
| --- | --- | --- |
| Stationarity margin | `0.38` | Pass criterion |
| Lyapunov max residual | `2.7755575615628914e-17` | Pass criterion |
| Minimum eigenvalue of `P_inf` | `0.034577890176373464` | Pass criterion |
| Truth process/observation std ratios | `[2.5, 2.3636363636363638, 2.1999999999999997, 2.0]` | Recoverability design diagnostic |
| Empirical state std | `[0.37983696349662177, 0.31077695334086153, 0.23784052906590394, 0.18758399592150113]` | Explanatory only |
| Empirical observation std | `[0.40231490437238693, 0.3279227061920609, 0.25200507016136536, 0.20748785462296693]` | Explanatory only |
| Max abs state | `1.0279111407619268` | Nondegeneracy diagnostic |
| Max abs observation | `1.0796169943915201` | Nondegeneracy diagnostic |

The stationary residual and covariance definiteness checks pass. The moment
diagnostics are finite and nondegenerate. These moment diagnostics are not
posterior evidence.

## Review And Repair Notes

- Same-foreground Codex substitute review of the Phase 2 subplan returned
  `VERDICT: AGREE`.
- The reviewer suggested a non-blocking tightening: explicitly name `H=I` and
  diagonal `Q/R` in required checks. The subplan was patched before execution.
- During artifact validation, the initial hash field name `data_hash` was
  judged ambiguous. The data and manifest were regenerated with explicit
  `data_payload_hash`, `data_file_sha256`, and `manifest_payload_hash` fields.

## Local Checks

- `python -m json.tool` passed for the data JSON.
- `python -m json.tool` passed for the manifest JSON.
- Bounded diagnostic summary was extracted from the generated data JSON.
- No NeuTra training, HMC, posterior/reference sampling, or runtime model
  estimation command was run.

## Decision Table

| Decision | Primary Criterion Status | Veto Diagnostic Status | Main Uncertainty | Next Justified Action | Not Concluded |
| --- | --- | --- | --- | --- | --- |
| Admit synthetic fixture for Phase 3 implementation work | Met | No Phase 2 veto triggered | Posterior recoverability and HMC behavior remain untested | Implement/adapt stationary triangular target helpers in Phase 3 | No posterior correctness, HMC readiness, NeuTra usefulness, global identifiability, product/default readiness, or scientific validity |

## Plain-Language Gate

Claimed target: generate a valid stationary synthetic fixture for the Phase 1
lower-triangular LGSSM contract.

Computed quantity: one deterministic data artifact with stationary initial
law, observations, latent states, truth, hashes, and diagnostics.

Verdict: `correct` for fixture generation under the Phase 1 contract;
`not checked` for implementation score correctness, posterior recovery,
NeuTra, HMC, and scientific validity.

## Next Phase Handoff

Phase 3 may begin. It must implement or adapt the lower-triangular LGSSM
construction and stationary covariance solve against the Phase 1 contract and
Phase 2 fixture, with no runtime `GradientTape` in the admitted route and no
HMC/training.
