# Phase 6 Pre-Runtime Implementation Codex Substitute Review

Date: 2026-07-11

Review type: bounded independent Codex read-only review substituting for Claude
after the binding managed external-disclosure rejection.

## Scope And Boundary

The reviewer inspected the frozen Phase 6 smoke-authority implementation,
controller integration, exact benchmark driver, three Phase 6 command
entrypoints, focused tests, and reviewed Phase 6 subplan. The review covered
pre-runtime correctness only. It did not authorize or execute a smoke, serious
Phase 7, Phase 8, NeuTra, package, network, product, or scientific action.

No HMC transition or worker ran during the review loop. The pending proposal,
proposal manifest, authority, permanent claim, result, progress, log, output
manifest, infrastructure terminal artifacts, and private sample were absent.

## Repair Loop

The complete initial audit returned `VERDICT: REVISE` with eleven findings:
an incomplete implementation inventory; caller-forgeable launch context;
unbounded teardown; accidental serious-diagnostic gating in smoke mode;
asymmetric emergency evidence; unverifiable pathname-replacement evidence;
missing end-to-end collision/race tests; writable claim inode; asserted rather
than observed parent environment; incomplete serious-output protection; and an
import/restore race. All were repaired with focused no-runtime tests.

The bounded frozen review then found and closed three further issues:

1. Proposal bytes could have been written before full live candidate
   verification. Full candidate verification now precedes either terminal
   write, with an injected-failure absence regression.
2. The retained benchmark-driver role was omitted from the loaded-module
   allowlist. The exact proposal-bound role is now required and covered with
   `require_runtime_imports=True`.
3. Normal import resolution could execute unbound `docs` or `docs.benchmarks`
   package parents before loading the retained benchmark. The child bootstrap
   now purges conflicting `docs` trees, synthesizes and bundle-marks those two
   code-free namespace parents, rejects every other unapproved `docs.*`
   module, and requires the markers during child verification. A subprocess
   regression imports the retained driver with poison parent initializers ahead
   of the repository and proves that the poison code does not execute.

The final bounded re-review found no material issue and ended:

`VERDICT: AGREE`

## Final Frozen Evidence

| Path | SHA-256 |
| --- | --- |
| `bayesfilter/inference/hmc_smoke_authority.py` | `9a0d630532edc7d02c41b93e61fca266a2628a68598cfef4f91f1f73d871439b` |
| `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py` | `f54ae939af4e14675f78753a19eec3328963eff5035821c3d9cd66035d30b5cd` |
| `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py` | `d94207a836315ceaa24c11de07e5cd63e61063fb4d114ac38fc16a54c07ad5ba` |
| `scripts/build_hmc_phase6_smoke_authority_proposal.py` | `59d1d7198681a58f1277ec2d6c9c19a2048caa8bf2b9b60f7d91c1fb49b4202f` |
| `scripts/build_hmc_phase6_smoke_authority.py` | `8cef152f5cff7f121c801577ab9ece7e3cae8daad0c65a56988c5d0607b4b653` |
| `scripts/run_hmc_phase6_typed_identity_smoke.py` | `81a824c7491d526c9d7516ae54626a3c7aa8074e794e4940111dada928088cb5` |
| `tests/test_hmc_smoke_authority.py` | `c3f91e70bf2bd6b89c4df60ba8445840f794fd84f675a11a04ed7918d1fe2dbc` |
| `tests/test_deterministic_lgssm_hmc_phase7_tf.py` | `ba6c140e30a84dd54d81bb2759f0c379dca3c15c089298129de5cd8dd9fdcaef` |
| `tests/test_deterministic_lgssm_hmc_tuning_driver.py` | `4dffde880b2f1b93fb09b75d76f264e328f4ebc00a83a2297bd94e846d8045a2` |
| `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md` | `63a9adea5a73a572ab8f922ac73c77f13e57b40265fb7f48e0f6a3a38b105a8e` |

Final checks were deliberate CPU-hidden engineering checks:

- targeted loader repair: `4 passed`;
- focused Phase 2-6/controller gate: `207 passed`;
- combined eight-module gate: `230 passed`;
- only the two existing TensorFlow Probability `distutils.version`
  deprecation warnings;
- Python compilation, scoped tracked/untracked whitespace checks,
  authority-literal scan, bypass/repin scan, and Phase 6 artifact-absence check
  passed.

## Verdict And Residual Risk

The implementation is admissible for proposal-only materialization. It does
not establish that actual `ProcessPoolExecutor` spawn, CPU/XLA compilation, or
HMC mechanics will succeed on this machine; that is exactly the runtime risk
reserved for a separately approved one-use smoke.

`VERDICT: AGREE`
