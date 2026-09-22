# SSL-LSTM q=20 Phase 9A full-replay performance and canary result

Date: 2026-09-02  
Controlling subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md`  
Master program: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `M3_TERMINAL_CONTINUATION_VETO_FULL_REPLAY_BLOCKED`

## Verdict

The corrected chart-1/beta-0 canary retry (`attempt-02`) reproduced the
bounded resource failure after the launcher-root repair. The source-owned root
was correct, the target/profile/seed identities matched the subplan, and the
runner pool reused exactly two traced static contracts. Twenty-one full-chain
calls completed in `1645.967775` seconds; the next call was interrupted by the
fixed `1800` second material cap. The outer process ended after the launcher's
120-second termination grace period at `1852.926691` seconds.

This is the second bounded resource failure after the R1 performance repair,
so the M3 continuation veto is satisfied. The six-scope full replay is not
launched, and Phase 9B remains blocked. The result rejects this exact
eight-pair schedule under the current fixed cap; it does not reject the
tempered transport or multi-chart research direction. A future attempt needs a
new reviewed plan with a changed execution design or budget, and cannot reuse
partial calls as tuning evidence.

## Evidence contract

| Item | Definition | Observed result |
|---|---|---|
| Question | Can the declared chart-1/beta-0 eight-pair measured-grid canary complete under the fixed cap after the launcher repair? | No; reproduced bounded resource failure |
| Target | Frozen q=20 target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` | Matched |
| Backend/route | `tensorflow_eigh_strict`; C5 `phase8-k2-compact-high-l3-pure`; `measured_joint_grid_v1` | Matched in manifest |
| Profile | `phase9a_full_replay_canary_v1`, scope `3/1`, fresh `20260902/780xx--785xx` namespace | Matched |
| Primary criterion | Complete all eight screens, sixteen replicated-selection calls, and held-out verification with durable receipts before `1800` s | Failed: only 21 calls completed |
| Hard veto checks | Root/profile/seed identity, finite route, XLA, memory growth, durable manifest | Identity and runtime checks passed until resource stop; completion criterion failed |
| Explanatory diagnostics | Per-call timing, trace role, runner reuse, retracing warnings, allocator/device state | Recorded; no promotion role |
| Nonclaims | No whitening, mode discovery, convergence, posterior correctness, sampler ranking, scaling, production, or Phase 9B readiness | Explicitly preserved |

## Attempt history and immutable artifacts

| Attempt | Root | Result | Role |
|---|---|---|---|
| `attempt-01` | `.../phase9a-full-replay}/attempt-01` | Resource stop plus literal-brace launcher defect | Invalid launch envelope; timing diagnostic only |
| `attempt-02` | `.../phase9a-full-replay/attempt-02` | Resource stop after root repair | Valid corrected retry; supports the M3 continuation veto |

Attempt-02 files are preserved under:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/phase9a-full-replay/attempt-02/`.

The durable manifest reports Git commit
`54201f5cd925ed15036bad8156606b812d53b045`, status checksum
`a00c1b1d54103e405a8ba7b13452539c5400c758ba75478374fa41ea524d3779`, target
signature above, and `failure_classification=resource_or_execution`. The
launcher source checksum after repair is
`a5ca1d89ffc78b10e2c42b12795cfa8b78eabd68affbf5cfb91a65dc95c302bb`.

## Cost and trace evidence

| Measurement | Attempt-02 value | Interpretation |
|---|---:|---|
| Completed full-chain calls | 21 | Eight screens plus thirteen of sixteen replicated selections |
| Completed-call sum | `1645.967775` s | Lower bound before remaining selection and held-out calls |
| Failed call | `21`, `60.612555` s until signal 15 | Cap termination, not a numerical exception |
| Started but incomplete calls | `21` and `22` | No results are usable from either |
| First-trace events | Calls `0` and `8` only | One screen graph and one selection/held-out graph |
| Steady reused calls | 19 | Pool reuse was effective; retracing was not the dominant full-chain issue |
| Material cap | `1800.0` s | Source-owned and unchanged |
| Outer wall | `1852.926691` s | Includes termination grace; not a relaxed scientific cap |

The eight screen calls consumed `280.857626` seconds. The thirteen completed
selection calls consumed `1365.110149` seconds. The observed `L=3` selection
calls were about `58--66` seconds each, while `L=8` calls were about
`154--157` seconds each. The remaining three selection calls and one held-out
call could not plausibly fit in the remaining cap interval. TensorFlow emitted
trainer retracing warnings during chart training; these remain a separate
engineering repair trigger, but the canary's repeated cost is already enough
to establish the declared resource veto.

No candidate was selected, no held-out handoff was issued, and no partial
selection result is promoted. All chart checkpoint and preflight receipts that
preceded the failed scope remain mechanics evidence only.

## Repair and closeout record

The mandatory R2a repair was completed before attempt-02:

- removed the extra closing brace from the launcher output root;
- added an exact source-root/profile regression;
- reran the focused Phase 9A suite (`13 passed`), Python compilation, shell
  syntax, and whitespace checks;
- used a fresh attempt directory while retaining the same target, data, seed
  namespace, GPU0, XLA, TF32, memory-growth policy, schedule, and cap.

The retry reproduced the resource failure, so no local repair remains that can
preserve the current scientific contract and make the schedule fit. Reducing
the grid, dropping replicated selection, omitting held-out verification,
widening the cap, batching candidates, or changing the kernel would be a new
execution design and requires a new reviewed plan. Partial calls cannot be
resumed as fresh tuning evidence.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Close M3 canary | Complete fixed-cap canary | Continuation veto fired on second bounded resource failure | Transient contention could affect absolute timing, but repetition is strong evidence for schedule cost | Keep M3 and Phase 9B blocked; draft a new performance-design plan only with explicit direction | No claim about posterior quality |
| Reject full replay under this contract | Six-scope schedule cannot be reached after valid canary | Fixed-cap resource veto | Exact redesign cost is unmeasured | Preserve attempts and require a new plan for any altered schedule/cap | No rejection of transport mathematics |
| Preserve timing evidence | Durable per-call records and root-correct manifest | No artifact-corruption veto | Short run gives no statistical uncertainty | Use as engineering input, not candidate ranking | No sampler superiority or convergence claim |

## Inference-status table

| Inference class | Status |
|---|---|
| Hard veto screen | Supported: corrected retry reached the predeclared second bounded resource failure |
| Statistically supported ranking | None; no candidate completed selection and no replication comparison exists |
| Descriptive-only differences | Two-runner trace reuse, per-call timing, and retracing warnings |
| Default readiness | Not assessed; the route remains a repository target direction but this schedule is not executable under its cap |
| Next evidence needed | A separately reviewed performance design (or a changed budget) with analytic/frozen-route equivalence, then a new canary |

## Post-run red team

The strongest alternative explanation is transient GPU contention. It is
weakened, but not eliminated, by the near-identical call timings and failure
position across attempts 01 and 02; the device was available on GPU0 before the
retry. A second alternative is hidden retracing or allocator growth. The
recorded two-runner reuse and absence of a numerical/memory veto make those
possible contributors rather than a reason to reinterpret the result as a
scientific failure.

A future redesigned route could pass the cap while silently changing the
measured-grid evidence contract. That would not be comparable to this result;
the next plan must state exactly which calls, seeds, and kernel semantics are
preserved and prove equivalence before timing is considered. The weakest part
of the present evidence is that it is one scope and a short mechanics schedule;
its conclusion is intentionally limited to bounded execution feasibility.

## Real blocker and handoff

`BLOCK_M3_CANARY_RESOURCE_VETO`: after the launcher repair, a fresh corrected
canary again exceeded the fixed material cap. Do not launch
`phase9a_full_replay_v1`, any six-scope replay, or Phase 9B from this plan.

The master program is refreshed to this terminal state. Any continuation must
begin with a new subplan that names the changed performance design or budget,
repeats the skeptical audit, and defines a new evidence contract; no further
retry is authorized by the current M3 plan.
