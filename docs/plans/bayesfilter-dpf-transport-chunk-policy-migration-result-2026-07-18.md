# DPF Transport Chunk Policy Migration Result

Date: 2026-07-18  
Status: COMPLETE

## Outcome

The active repository policy is now
`dpf_transport_exact_divisor_cap3000_v1` and is enforced in code, execution,
artifact admission, and discovery tests.

For particle count `N`, the only eligible row/column chunk extent is:

\[
K =
\begin{cases}
N, & N \le 3000,\\
\max\{d : d \mid N,\ 1 < d \le 3000\}, & N > 3000.
\end{cases}
\]

Row and column chunks must both equal `K`. A large `N` with no divisor greater
than one under the cap fails closed. Required witnesses pass:

| `N` | `K` | Block grid |
|---:|---:|---:|
| 1000 | 1000 | `1 x 1` |
| 1024 | 1024 | `1 x 1` |
| 10000 | 2500 | `4 x 4` |
| 10240 | 2560 | `4 x 4` |

## Root cause and repair

The regression came from copying an `N=32,K=16` lower-rung fixture into later
`N=128` through `N=1024` runs, even though the originating plan explicitly said
the fixture was not a production selection. The June 24 GPU/XLA timing evidence
supported large exact-divisor chunks, but no single repository-owned policy
boundary prevented a later caller from overriding it.

The repair adds:

- binding policy text in `AGENTS.md` and `CLAUDE.md`;
- repository-owned selection, resolution, and validation in
  `bayesfilter/highdim/transport_chunk_policy.py`;
- fail-closed enforcement in Contract E LGSSM preparation, canonical LGSSM,
  latent SIR, low-level Contract E forward/JVP/VJP, and shared streaming
  transport entry points;
- policy-derived defaults rather than fixed row/column default integers;
- migration of current Contract E drivers so they expose no independent chunk
  CLI;
- mandatory policy ID, exact chunk, and block-grid checks in current LGSSM
  aggregation; and
- a closed archival-wrong ledger and AST discovery guard.

## Historical demotion

The ledger
`docs/plans/bayesfilter-dpf-transport-chunk-policy-archival-wrong-route-ledger-2026-07-18.json`
contains 52 old independent or delegated chunk-CLI benchmark paths and 10
additional pre-policy/raw/noncanonical emitters, wrappers, or supervisors.

All 62 paths are preserved only as provenance. They are wrong relative to the
active policy and ineligible for diagnostic, comparison, timing, tuning,
admission, leaderboard, HMC, or scientific evidence for a new run. Each path's
`main()` now raises `ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY` as its first
statement. The old Phase 4 chunk-parity artifact is preserved byte-for-byte but
is no longer recomputed or certified by tests.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Adopt and enforce `dpf_transport_exact_divisor_cap3000_v1` | PASS: selector witnesses, runtime guards, admission guards, and closed discovery inventories pass | PASS: no current fixed `K=16` constant or fixed row/column streaming default remains in the guarded source inventory | The 3000 cap is evidence-backed on the tested RTX 4080 SUPER route, not proved universally optimal for all future hardware | Use current policy-governed drivers; migrate an archival path only by removing its chunk CLI and adding focused tests | No score correctness, Sinkhorn convergence, posterior correctness, HMC readiness, GPU performance, or leaderboard-completeness claim |

## Engineering, numerical, and scientific ledgers

| Ledger | Result |
|---|---|
| Engineering correctness | PASS for focused scope: sources compile; selection, rejection, runtime, discovery, executable demotion, source-hash, and admission tests pass. |
| Numerical validity | PASS for focused mechanics: exact one-block Contract E forward/JVP/VJP and shared dense-vs-streaming tests pass; one manual-vs-autodiff comparison changed from zero to one ULP because the exact traversal reduction tree changed. |
| Scientific interpretation | Policy enforcement only. The timing evidence supports this repository default but does not establish universal hardware optimality or any filter/score claim. |

## Inference status

| Question | Status |
|---|---|
| Hard veto screen | PASS for policy enforcement; contrary settings and pre-policy artifacts fail closed. |
| Statistically supported ranking | None attempted. June timing artifacts remain descriptive configuration evidence. |
| Descriptive-only differences | The exact one-block traversal caused at most the observed one-ULP manual/autodiff fixture difference. |
| Default readiness | PASS for repository chunk configuration policy only. |
| Next evidence needed | New hardware-specific evidence is needed only if changing the 3000 cap or exact-divisor rule. |

## Checks run

All test commands deliberately hid GPU devices with `CUDA_VISIBLE_DEVICES=-1`.
They are CPU-only correctness checks, not GPU evidence.

1. Policy, preparation, canonical LGSSM, and latent-SIR focused suite:
   initial run `41 passed, 3 failed`; failures were two test-spec issues and one
   one-ULP exact-traversal difference. Focused repair rerun: `13 passed`.
2. Shared streaming subset after migrating mechanics fixtures:
   `17 passed, 23 deselected`.
3. Combined Contract E/shared-streaming/aggregation suite:
   `100 passed, 1 failed, 2 deselected`; the sole failure was a discovery test
   requiring the orchestration driver to import the policy ID rather than copy
   its string.
4. Discovery and aggregation rerun: `23 passed`.
5. Final policy, executable-demotion, source-closure, aggregation, and
   Contract E forward/JVP/VJP suite: `40 passed`.
6. Post-ledger final policy/source-closure suite: `30 passed`; generalized
   chunk-CLI/numeric-wiring discovery suite: `16 passed`.
7. `python -m py_compile` passed for current guarded modules and all 62
   fail-closed archival scripts.
8. `git diff --check` passed.
9. Source audit found no active `ROW_CHUNK_SIZE = 16`,
   `COL_CHUNK_SIZE = 16`, `CHUNKS = 16`, latent-SIR canonical chunk constant,
   or fixed row/column `DEFAULT_STREAMING_CHUNK_SIZE` signature.

The two TensorFlow Probability `distutils.version` deprecation warnings are
unrelated. No test failure remains in the final focused scope.

## Run manifest

| Field | Value |
|---|---|
| Git commit at execution | `15170e1573d19b235d96f3ed3525fa2071f58320` |
| Git status | Dirty shared research worktree; unrelated changes preserved |
| Python | 3.11.14 |
| TensorFlow | 2.19.1 |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu` |
| Device status | Intentional CPU-only checks, GPU hidden |
| Seeds | N/A; deterministic policy and focused existing fixtures |
| Wall time | Focused test commands recorded above; longest combined command 519.04 s |
| Plan | `docs/plans/bayesfilter-dpf-transport-chunk-policy-migration-plan-2026-07-18.md` |
| Result | This file |
| Selector SHA-256 | `24434d7f2d731cc5cc57442edf3ab1bd237230d8191d010a286ba884a6b91d7c` |
| Canonical LGSSM SHA-256 | `6201d85642474a9819a1c8972e94bd49cd317cba9a5862145f90252ddcdd0d24` |
| Contract E streaming SHA-256 | `b2208a9e9f65bceaa6a629e69fb8c0edcdeab39d79ba6f0e5c04c45a427ef34a` |

## Post-run red team

The strongest alternative explanation is that the selected cap reflects the
tested GPU, shapes, and June compiler behavior rather than a universal optimum.
That does not undermine the repository policy requested here; it limits the
claim to the current evidence-backed default. The conclusion would be
overturned by reviewed target-hardware evidence supporting another cap or by a
test showing the selected exact traversal changes the mathematical result
beyond declared floating-point reduction effects.

The weakest remaining boundary is that archival modules remain importable for
source inspection and old helper tests. They cannot execute their `main()` or
pass current artifact admission, and the discovery test makes removal of either
guard visible.
