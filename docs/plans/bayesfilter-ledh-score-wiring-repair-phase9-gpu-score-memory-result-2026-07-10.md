# Phase 9 Result: Trusted GPU/XLA Score-Memory Ladder

Date: 2026-07-11

Status: `SUPERSEDED_FD_POLICY_BASIS_HISTORICAL_RAW_MEASUREMENTS_PRESERVED`

## Supersession Notice

This document is preserved as the historical Phase 9 execution narrative, but
its hard-veto and candidate-rejection conclusions under the inherited
`max_abs <= atol OR max_rel <= rtol` policy are superseded. In particular, the
`0.005` thresholds used by four rows were pre-existing CLI defaults, not
calibrated production or HMC tolerances. Freezing them before execution did not
make them scientifically justified.

The original trusted GPU/XLA score, FD, memory, device, and provenance
measurements remain valid raw evidence. Their current decision authority is:

- correction result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`,
  regenerated under the clarified FD-only policy;
- reclassification JSON:
  `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`,
  containing all 11 SHA-bound comparisons;
- correction review:
  `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-codex-review-2026-07-11.md`,
  replacing the prior review of the wrong `2%` RSS/RMS interpretation.

Under the owner-directed FD-only rule,
`max_j(r_j) <= 0.05*sqrt(p)`, nine stored comparisons pass and two fail.
Predator-prey fails Gate B and generalized-SV fails Gate C `T=4`; fixed-SIR,
Actual-SV, and KSC-SV have no stored FD failure under the clarified rule. The
`5%` value mirrors the conventional 95% threshold, but no confidence interval
or coverage calculation is performed. All threshold-based statements below
are historical unless the correction result restates them under this rule.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the nonlinear Phase 9 ladder without admitting any row. Do not run Gate D, aggregation, or Phase 10 execution under the current plan. | Not met for any nonlinear row. Predator-prey failed tiny Gate B FD. Fixed-SIR reached full `T=20,N=10000` but failed full-time FD. Actual-SV, generalized-SV, and KSC-SV failed FD at their first `T=4,N=10000` prefix. | Five row-local same-scalar FD vetoes fired. No shared harness, terminal-artifact, trust, XLA, GPU-placement, finite-output, prepared-input, or observed score-memory veto fired in the decisive shards. | Write a revised, reviewed diagnostic plan that can distinguish compact-score error from float32 finite-difference resolution or their interaction without changing the scalar or post-selecting thresholds. A separate reviewed closeout/leaderboard subplan is required before Phase 10. | No nonlinear score admission, five-seed result, full-time SV memory result, HMC readiness, posterior correctness, exact nonlinear-likelihood correctness, native actual-SV correctness for KSC, runtime or memory superiority, statistical ranking, or rejection of the compact-score research direction. |

## Research Question Verdict

The Phase 9 question was whether each nonlinear compact score could produce a
trusted full-row GPU/XLA/TF32 five-seed artifact, remain within the
`14000 MiB` per-seed score-memory budget, and pass row-matched same-scalar
finite differences under thresholds frozen before execution.

The answer is **no for every current nonlinear candidate**. This is not because
the shared harness failed. Every decisive score process produced finite output
on `/GPU:0` with XLA JIT, TF32 enabled, trusted provenance, reset memory, and a
terminal artifact. Each row was rejected by its frozen FD screen before the
five-seed promotion criterion could be reached.

The result invalidates the current candidates under the declared admission
screen. It does not invalidate the harness, target definitions, prepared
inputs, source artifacts, TensorFlow/XLA execution path, or the broader compact
score research direction. Whether the score recurrence, float32 central-FD
resolution, or their interaction causes each mismatch remains not checked.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Engineering/scientific question | Can each compact nonlinear score route produce valid trusted full-row GPU/XLA memory and same-scalar correctness evidence? |
| Exact baseline/comparator | Central finite differences of the row-matched value-only scalar at the same fixed seed, prepared inputs, target, coordinates, transport settings, and production precision. |
| Primary promotion criterion | Historical/superseded decision: failed for every nonlinear row before Gate D under the original frozen thresholds. The current FD-only reclassification passes three historical terminal comparisons. No five-seed aggregate exists. |
| Promotion vetoes | Predator-prey: Gate B FD. Fixed-SIR: full-time Gate C FD. Actual-SV, generalized-SV, and KSC-SV: first `N=10000` Gate C prefix FD. |
| Continuation vetoes | Each row stopped locally at its first frozen FD failure. No shared continuation veto fired. With all nonlinear rows terminal, nonlinear Gate D has no eligible row. |
| Explanatory only | Runtime, compile time, sub-budget peak values, objectives, score magnitudes, and coordinate error patterns beyond their declared pass/fail role. |
| What will not be concluded | Claims in the decision table, plus no claim that changing FD step, tolerance, or precision would rescue any row. |
| Preserved artifact | Gate A/B results and reviews, decisive Gate B/C JSON/Markdown/log shards, four reviewed Gate C row results, and this consolidated result. |

## Final Nonlinear Row Decisions

All decisive runtime shards use singleton seed `81120`, `float32`, TF32
enabled, `jit_compile=True`, logical `/GPU:0`, and trust basis
`owner_designated_managed_session_visible_gpu_trusted`.

| Row | Terminal rung | Score-memory evidence | FD max abs / atol | FD max rel / rtol | Frozen decision | Next rung |
| --- | --- | --- | --- | --- | --- | --- |
| predator-prey | Gate B `T=1,N=2` | Tiny peak `0.03759765625 MiB`; not `N=10000` evidence | `0.3162194490 / 0.005` | `1.0 / 0.005` | Fail both branches | Gate C blocked |
| fixed-SIR | Gate C full `T=20,N=10000` | Full-time peak `414.44677734375 MiB`; memory pass | `7.853515625 / 0.01` | `0.0566700101 / 0.05` | Fail both branches | Gate D blocked |
| actual-SV | Gate C prefix `T=4,N=10000` | Prefix peak `35.22705078125 MiB`; not full-time evidence | `0.0094842315 / 0.005` | `0.0602924675 / 0.005` | Fail both branches | `T=50` blocked |
| generalized-SV | Gate C prefix `T=4,N=10000` | Prefix peak `35.23095703125 MiB`; not full-time evidence | `0.0151546374 / 0.005` | `0.4427539706 / 0.005` | Fail both branches | `T=50` blocked |
| KSC-SV | Gate C prefix `T=4,N=10000` | Prefix peak `35.22607421875 MiB`; not full-time evidence | `0.0102410018 / 0.005` | `0.0369351506 / 0.005` | Fail both branches | `T=50` blocked |

The fixed-SIR ladder also passed `T=1` and `T=5` by the relative branch of its
frozen OR rule before failing at full `T=20`. Predator-prey was not permitted to
enter Gate C after failing Gate B, and no longer SV prefix was permitted after
the first `N=10000` FD failure.

## Target And Computed Quantity Ledger

| Row | Claimed target | Quantity actually computed | Relationship and verdict |
| --- | --- | --- | --- |
| predator-prey | Score of the realized finite-`N` additive-Gaussian predator-prey LEDH scalar in physical coordinates. | Compact forward sensitivity and float32 central FD of the same prepared-input value scalar. | Mismatch exceeds both frozen thresholds at the tiny Gate B fixture. Candidate rejected; cause not isolated. |
| fixed-SIR | Score of the realized finite-`N` fixed-SIR LEDH scalar in log-scale coordinates. | Compact forward sensitivity and float32 central FD of the same prepared-input value scalar. | Full-time `T=20` mismatch exceeds both frozen all-coordinate thresholds. Candidate rejected despite a full-time memory pass. |
| actual-SV | Score of the transformed `log(y^2)` actual-SV scalar. | Compact forward sensitivity and float32 central FD of the same transformed scalar. | `log_beta` drives failure at `T=4,N=10000`. This is not exact native actual-SV likelihood evidence. |
| generalized-SV | Score of the raw source-route prior-mean generalized-SV scalar. | Compact forward sensitivity and float32 central FD of the same source-route scalar. | `log_tau` drives failure at `T=4,N=10000`. Candidate rejected; cause not isolated. |
| KSC-SV | Score of the KSC log-chi-square Gaussian-mixture surrogate scalar. | Compact forward sensitivity and float32 central FD of the same surrogate scalar. | Both coordinates miss the frozen screen at `T=4,N=10000`. This is not native actual-SV likelihood evidence. |

For every row, correctness of the claimed exact derivative remains unsupported
under current evidence. The finite-difference mismatch is sufficient to reject
admission under the frozen screen, but the approximate comparator does not by
itself prove which component is mathematically wrong.

## Engineering Correctness Ledger

- Gate A implemented the shared terminal score-only/FD-only harness, strict raw
  and aggregate validators, code/source/governance hashing, reset score-memory
  measurement, singleton-seed enforcement, and GPU/XLA/TF32 provenance.
- Final CPU-hidden verification before GPU execution passed `161` tests. The
  final decision-provenance subset passed `35` tests.
- Fixed-SIR and predator-prey graph-extraction failures were bounded
  implementation defects. They were repaired without changing target math,
  frozen FD settings, transport settings, or public APIs, and received local
  substitute reviews with `VERDICT: AGREE`.
- Gate B then produced a common-identity terminal score/FD pair for every row.
  Predator-prey failed numerically; the other four rows were authorized for
  Gate C by the reviewed Gate B result.
- Every decisive Gate C score shard is terminal, finite, trusted, XLA-compiled,
  TF32-enabled, on `/GPU:0`, and below `14000 MiB` at `N=10000`.
- Every decisive FD shard references the exact SHA-256 of its score shard and
  has the same prepared-input fingerprint and row target.
- No Gate D or nonlinear aggregate artifact exists. Their absence is required
  by the stop rules, not an incomplete attempted run.

## Numerical Validity Ledger

- Predator-prey fails the frozen tiny Gate B same-scalar FD rule.
- Fixed-SIR passes the frozen Gate C FD rule at `T=1` and `T=5`, then fails at
  full `T=20`.
- Actual-SV, generalized-SV, and KSC-SV each fail the frozen Gate C FD rule at
  the first `T=4,N=10000` prefix.
- No row changed FD step, tolerance, precision, target, or transport after a
  result was observed.
- Score-memory evidence has no observed hard failure. Only fixed-SIR reached a
  nonlinear full-time `N=10000` score rung; the SV peaks remain prefix-only.
- No nonlinear candidate satisfies the Phase 9 five-seed numerical-validity
  criterion.

## Scientific Interpretation Ledger

- Hard-veto evidence rejects all five current nonlinear candidates from score
  admission under this plan.
- No candidate remains viable for Gate D under the current evidence.
- No ranking is statistically supported. The runs use one seed at each
  decisive rung and were not designed for comparative inference.
- Runtime, memory below the hard budget, objectives, score values, and error
  sizes are descriptive only. They do not establish superiority.
- Candidate rejection is separate from research-direction rejection. A
  revised diagnostic phase may still test whether FD resolution or score math
  is responsible, but it must be predeclared and reviewed.
- The existing owner-directed GPU-oriented LEDH-PFPF-OT TF32 default policy is
  unchanged. These score candidates are not admitted merely because their
  value/transport algorithm is the production default.

## Reviewed Row Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Gate B consolidated result | `cbfe9ab65929745345d32a765c7067a1c9875dc95d18095abfc84f205010c605` |
| Final Gate B review | `ac1c25f3d24cf329abc0cfabf9cc928f0367d95064e778a749b39f0f3fd70312` |
| Fixed-SIR Gate C result | `c6755111f811e863a6743ea0646d88e922556d26805448682754ff707ef739b6` |
| Fixed-SIR Gate C review | `a02dd51f1aa981d8526951eadc3195f35d4d97f3bed5de70b38fa5e76f1ad38b` |
| Actual-SV Gate C result | `f95ffee2fd47562ff27548e38b3e8c7154dbfb300f0b701146f69ab3fc05c5de` |
| Actual-SV Gate C review | `ff01f800a0563185135d7dcf3c1958a75bacf2bb1a3d374c1af9a162f0ac179a` |
| Generalized-SV Gate C result | `4dee41f63b5afe102d01f3ee0c3f3d160e72521878742b9a7dfdf32476e92f8e` |
| Generalized-SV Gate C review | `9175a2fbc66392db5e6bf1bee06a74d112a3a1c7c39d4e14b0e767a9d2b5c4cf` |
| KSC-SV Gate C result | `595690e138459f5d9a266ea953ce4806829e9cdf6cbd9a6e96c719c8d1673a8f` |
| KSC-SV Gate C review | `19677e5ccf67d3e9913cd4d7c9724c47039ea88fb41b09cd6ef58e5c1d37d718` |

## Decisive Runtime Artifact Hashes

| Row and rung | Score JSON SHA-256 | FD JSON SHA-256 |
| --- | --- | --- |
| predator-prey Gate B `T=1,N=2` | `82eb75a8710a6c4219419b5f9c14f670e371554a3c7943a2a3fb5e03f1c28f5c` | `738c59f9967ec86dfc09be7bfb315e4cc9fdfc04a22cec95292527405f1b3127` |
| fixed-SIR Gate C `T=20,N=10000` | `7acf4612b4082533cfa076635f1788015ffae43da94f15eb4e818e57c2036773` | `00944bcb7f756f914b56f920b62709e9c4d9a950b5dffcf8589ac83fd68f0036` |
| actual-SV Gate C `T=4,N=10000` | `6320f04eab3f03157e3c1789de5b1927cefb33c9752e2fb0a7cfe787797f86b7` | `9547b853db09e2974f2dfa2adf8d5d3d19b274b5dfb74dab8358895e8b03bdaf` |
| generalized-SV Gate C `T=4,N=10000` | `3fb140284b74a02efb8fe57562f0f33ee75a1012bd1dd6cdc554be71c59e71d6` | `edc896ef4a41772e29487257b3c6e01c8543b780aacf930fa607dd43479b8b08` |
| KSC-SV Gate C `T=4,N=10000` | `232c28ae76c945efc843f296e412f58ef30d3db38e28e958d47be633c9311dae` | `288f997acb7dcc0440a5bcd653e34ca626884fc1421e35e2c3fd25048bde366d` |

All Gate C shards bind the unchanged runner SHA-256
`fa1e9602023e96e3a1b68e1a6547397eef7cf1f1d4cd95b2c62b9a61fa13e8fe`
and exact-command manifest SHA-256
`ffa2d232c0582d28d13d57a5ca188ad8d9f30e279555093532561aa264559e3d`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus the dirty worktree recorded in every runtime shard |
| Commands | Literal reviewed `gate_b_commands` and eligible prefix-ordered `gate_c_commands` from `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-exact-commands-2026-07-10.json`; score before FD; no hand-modified argv |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; Python `3.11.14`; TensorFlow `2.19.1`; conda env `tf-gpu` |
| CPU/GPU status | Trusted GPU 0; NVIDIA GeForce RTX 4080 SUPER; driver `591.86`; `16376 MiB`; float32; TF32 enabled; XLA JIT; managed-session trust basis |
| Data version | Row-specific admitted forward scalar artifacts dated 2026-07-07, with path and SHA-256 bound in each shard |
| Random seeds | Singleton seed `81120` for every runtime process; seeds `81121..81124` intentionally not run because no row reached Gate D |
| Wall time | Decisive score/FD elapsed seconds: predator-prey `23.2471/9.7577`; fixed-SIR full `532.9583/49.5846`; actual-SV `78.2786/42.0685`; generalized-SV `75.9381/46.0792`; KSC-SV `76.4188/41.4762` |
| Output artifacts | GPU preflight; Gate B live and archived repair shards; Gate C fixed-SIR `T=1,5,20` shards; Gate C first-prefix SV shards; matching Markdown/logs; row results/reviews; this result |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Result file | This file |

The historical dirty file `bayesfilter/linear/kalman_qr_tf.py` and unrelated
`docs/plans/bayesfilter-post-integration-reboot-reset-memo-2026-07-10.md` were
not modified by Phase 9 execution.

## LGSSM Separate-Lane Status

The existing LGSSM `N=10000,T=50`, seed-`81120` compact score-only artifact
remains trusted precedent with a reset peak of about `719.671 MiB`, but it is
from an earlier commit and lacks same-scalar FD. The nonlinear harness does not
accept it, and no LGSSM merge/content-binding repair was authorized in this
execution. LGSSM remains non-admitted. No LGSSM FD, later seed, or aggregate
command was run.

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Every nonlinear candidate has a reviewed FD veto. No observed score-memory, device, XLA, trust, finite-output, or terminal-artifact veto invalidates the shared harness. |
| Statistically supported ranking | None. The design and one-seed terminal rungs do not support ranking. |
| Descriptive-only differences | All runtime, compile-time, sub-budget memory, objective, score, and non-threshold error differences are descriptive only. |
| Default-readiness | No new score default-readiness or score admission. Existing algorithm default policy remains separate and unchanged. |
| Next evidence needed | A revised reviewed multi-arm diagnostic contract that preserves each scalar and predeclares how float32 FD resolution will be distinguished from compact-score error; then the smallest discriminating reruns. |

## Post-Run Red Team

- Strongest alternative explanation: the common `float32` central-FD screens,
  especially step `1e-4` for the three SV rows, may be resolution-limited. The
  generalized-SV identical FD values for two coordinates and predator-prey's
  zero FD for one coordinate make this plausible. It is explanatory only; no
  diagnostic arm was predeclared to establish it.
- Result that would overturn a row decision: a reviewed, predeclared
  production-relevant derivative check of the unchanged row scalar that passes
  without selecting its settings after seeing these results. It would be new
  evidence and would not retroactively change this Phase 9 outcome.
- Weakest evidence: each decisive comparison uses one seed and one frozen FD
  step. The evidence is sufficient for the declared hard screens but not for
  causal attribution or stochastic ranking.
- How a future repair could mislead: changing tolerance, step, precision,
  target, or transport after seeing these failures could manufacture a pass.
  A revised plan must freeze diagnostic arms and distinguish explanatory
  resolution studies from production admission criteria before execution.

## Final Gate Boundary

- Predator-prey Gate C is blocked.
- Fixed-SIR, actual-SV, generalized-SV, and KSC-SV Gate D and aggregation are
  blocked.
- No nonlinear row is admitted.
- LGSSM remains separate and non-admitted.
- Phase 10 has no scoped subplan in the workspace and is not authorized by this
  negative result.
- The next permissible action is a revised reviewed diagnostic or explicit
  closeout/leaderboard subplan. No additional GPU command is authorized by this
  result.
