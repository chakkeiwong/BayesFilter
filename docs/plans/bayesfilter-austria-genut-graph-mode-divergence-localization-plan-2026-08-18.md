# Austria GenUT Graph-Mode Value/JVP Divergence Localization Plan

Date: 2026-08-18

Prior evidence:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`
(section "Current-Source GPU Endpoint Confirmation"),
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-reset-memo-2026-08-18.md`.

Authorization: user requested plan review and execution with minimal stopping
on 2026-08-18. This is a bounded diagnostic localization campaign; it promotes
nothing.

## Research Question

On the frozen Austria scope and current source
(`cubature_genut_batch_tf.py` = `ae8cbfb...a976e`), the GPU eager endpoints
satisfy exact within-mode value identity, but the non-XLA `tf.function` graph
endpoints do not at `T=20` with four correction steps (gap `0.56207275`).
Questions, in decreasing scope importance:

- Q1 (scope-decisive): does the XLA endpoint arm — the repository default
  execution target — pass within-mode value identity on the current source?
- Q2 (reproduction): what is the smallest `(horizon, correction_steps)` graph
  case that reproduces the within-mode value inequality?
- Q3 (mechanism): is the divergence removed when grappler graph optimizations
  are disabled, which would classify it as a compiler-rewrite arithmetic
  difference between the value-only graph and the JVP-carrying graph, rather
  than a difference in the traced primal program?
- Q4 (provenance, closed): can the `606897...` source state whose stale
  attempt06 artifact passed within-mode graph identity be recovered for a
  diff? Answer determined pre-execution: NO — HEAD and the worktree both hold
  `ae8cbf...`; `606897...` matches no committed state in history and was an
  uncommitted session intermediate. Empirical localization is required.

## Candidate Under Test / Mechanism

The mechanism under test is TensorFlow non-XLA graph compilation of the shared
primal: the score graph's primal ops gain JVP consumers during
`ForwardAccumulator` tracing, which may change grappler fusion/rewrite
decisions and therefore FP32 arithmetic, relative to the value-only graph.
The alternative mechanism is that forward-mode tracing itself emits a
different primal op sequence.

## Evidence Contract

- Exact baseline/comparator: within each process and each case, the
  value-only endpoint versus the value carried by the score endpoint —
  the same within-mode identity gate as the memo (`exact_equal`, boolean).
- Primary criteria: per-case boolean within-mode identity. No ranking, no
  promotion criterion — this campaign localizes; it cannot pass the vetoed
  confirmation.
- Hard vetoes (per process): frozen target/adapter/tensor hash mismatch,
  source hash drift from `ae8cbf...`, wrong TF build/device/dtype/TF32,
  unverified memory growth, nonfinite value/score, `program_valid=false`,
  `status != COMPLETE`.
- Explanatory only: gap magnitudes, ULP counts, score vectors and their
  cross-mode/cross-config differences, tracing wall times, grappler-config
  effects, GPU occupancy.
- What will NOT be concluded even if all arms behave as expected: any repair
  promotion, any tolerance, graph-mode acceptance, dual-cap correctness,
  posterior correctness, NeuTra/HMC/tuning readiness, default readiness.
  A passing XLA arm does NOT by itself close the vetoed three-mode
  confirmation; whether graph mode remains a required claim-bearing arm is a
  human scope decision to be taken with these results in hand.
- Artifacts: fresh JSON per process under
  `docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/`
  in new `graph_bisect_attempt*` / `repair_validation_attempt09` directories.
  No overwrites. Terminal interpretation appended to the execution result and
  checkpoint documents.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Shrink horizon before steps | Graph tracing cost scales with unrolled horizon (39 min at `T=20`) | Bounds wall time | Defect may need long-horizon amplification and not reproduce small | Ladder includes `T=10` escalation before falling back to `T=20` probes |
| `T in {2,5}` first rungs | Cheap; CPU evidence showed endpoint gaps emerge by `T=3`–`5` in the analogous eager-defect history | Small trace, fast | Gap magnitude may round to bitwise-equal FP32 at small `T` | Steps-0 controls distinguish "no defect" from "defect below FP32 visibility": any nonzero ULP diff is recorded even when `exact_equal` is true |
| Grappler probe uses `disable_meta_optimizer=True` | Strongest single discriminator | One flag removes all grappler rewrites | Identity could be restored for an unrelated reason (changed layout) — over-attribution | Record full experimental options; classify as "consistent with", not proof |
| Reusing `_endpoint` from the frozen runner via import | Identical arithmetic path and summary code as the vetoed run | Comparability | Wrapper argv/env mismatch could silently change device policy | Wrapper asserts `_EARLY_DEVICE`, memory policy mode, TF32, and identity guards exactly as the base runner |
| Multiple cases per process | Precedent: base runner ran two cases per process | Saves ~1.5 min TF startup per case | Cross-case tracing state contamination | Each case builds fresh `tf.function` objects, as in the vetoed run; per-case wall times recorded |

## Execution Phases And Budget

GPU lane: RTX 5080 (GPU 0), `tftwogpu` python, TF32 on, deterministic ops,
memory growth — unchanged frozen environment. All GPU commands run trusted.

- P1 ladder (graph): cases `(2,0) (2,4) (5,0) (5,4)`; one process; cap 30 min.
- P2 escalation (conditional, only if P1 reproduces nothing): `(10,0) (10,4)`;
  cap 30 min.
- P3 steps bisection (conditional on a reproducing `T*`): `(T*,1) (T*,2)`;
  cap 30 min.
- P4 grappler probe: smallest reproducing case with
  `disable_meta_optimizer=True`, plus the same case default-config in the same
  process ordering as P1 for direct comparability; cap 30 min.
- P5 XLA endpoint arm: the reset memo's exact step-6 command,
  `repair_validation_attempt09/endpoint_gpu0_xla.json`; cap 90 min.
- Total campaign wall ceiling: 3.5 hours. Stop after 3 consecutive launch
  failures. One fresh occupancy probe before the first launch and before P5.

## Stop Conditions

Any hard veto above; OOM or allocator failure; a process exceeding its cap
(kill, preserve `RUNNING` JSON as non-evidence, record); total ceiling
exhausted; or any observation requiring a scientific-target or tolerance
change (forbidden — record instead).

## Pre-Mortem

The campaign could mislead if: (a) small-`T` bitwise equality is read as
"defect absent" when it is merely sub-ULP — mitigated by recording max ULP
and absolute differences for every case, not only the boolean; (b) the
grappler probe restores identity by accident of layout and is over-read as a
mechanism proof — mitigated by "consistent with" language; (c) the XLA arm
passes and is over-read as confirmation closure — explicitly forbidden in the
contract; (d) GPU contention corrupts timing but not arithmetic — timings are
explanatory only.

## Skeptical Audit Result

Audit passed 2026-08-18: comparator is the declared within-mode identity gate
(not a proxy); no promotion criterion exists to be gamed; stop conditions and
per-process caps are explicit; frozen identity guards are inherited from the
reviewed runner; the only new code is a thin argv-compatible wrapper that
reuses the frozen `_endpoint`; escalation order is cost-ordered; and the
scope decision (is graph claim-bearing?) is explicitly reserved for the
human owner. Known residual risk: reproduction may fail at all `T<20`,
leaving only expensive full-horizon probes; the budget covers exactly one
such probe (P4 at `T=20` within the remaining ceiling) before stopping.
