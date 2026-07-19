# SSL-LSTM NeuTra Phase 7 Retained Admission Plan

Date: 2026-07-16

Last revised: 2026-07-17

Status: `STAGE_A_PASSED_RETAINED_ACQUISITION_AUTHORIZED`

## Objective And Entry

Acquire fresh retained four-chain samples independently for G and H using the
Phase 6 frozen kernels, decide sampler admission before predictive comparison,
and measure cross-replication stability without treating either chart as a
posterior oracle.

Entry artifacts:

- Phase 5 exact target decision `PHASE5_EXACT_TRANSFORMED_TARGET_PASSED`;
- Phase 6 final receipt SHA-256
  `dc340ab2570032a85062d0ec9cd8c9e020c41a133ec9d11b78982502ff08b9b2`;
- G/H immutable transport hashes
  `5e485163...e68505aa` and `afa52cc5...f4fd44a`; and
- identity-mass kernels `epsilon=0.8,L=4` for both charts.

The user's 2026-07-17 instruction to continue authorizes implementation,
focused checks, Stage A, and the timing-derived retained acquisition frozen
below.

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Do independently trained G/H charts produce admitted retained HMC chains and uncertainty-compatible mapped-parameter functionals? |
| Baseline/comparator | G and H are independent replications of the same faithful training procedure and exact target; neither is an oracle or truth baseline |
| Candidate mechanism | Exact fixed-transport HMC in each chart's `z` coordinates, mapped to the common free `theta` coordinates |
| Primary admission | Each chart independently passes retained chain gates in both `z` and mapped `theta` coordinates |
| Hard execution vetoes | Target/artifact/source binding drift, nonfinite retained samples/value/score/core telemetry, all or any chain unmoved cumulatively, positive exposed native divergence, corrupt shard/final-state/manifest lineage, seed reuse, invalid draw order, or resource-cap exhaustion |
| Promotion vetoes | Maximum rank-normalized split R-hat `>1.05`, minimum bulk/tail ESS `<100`, maximum mean MCSE/SD `>0.10`, aggregate or per-chain retained acceptance outside `[0.55,0.85]`, or failed predeclared cross-replication stability |
| Explanatory only | Runtime, RMS jump, unavailable native divergence, initialization memory, marginal summaries, and continuous G/H differences below the stability veto |
| What passing changes | Both admitted and stable charts may hand their independent archives to Phase 8 predictive-moment validation |
| Nonclaims | No posterior oracle/truth, stationarity proof, predictive equivalence, sampler/transport superiority, complete mode or tail coverage, production/default readiness, or broad SSL-LSTM validity |
| Result artifact | One strict-JSON public receipt plus private immutable archive shards under `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-7-retained-admission/`, followed by one Phase 7 result note |

Native divergence unavailability remains explicit and is not a veto or evidence
of zero divergence. A positive count is a hard veto when the kernel exposes a
native boolean.

## Skeptical Pre-Execution Audit

Audit completed 2026-07-17 with one material design repair.

| Audit risk | Finding and disposition |
| --- | --- |
| Wrong baseline/oracle | Repaired: G/H are peer replications on one exact target; comparisons occur only in common mapped `theta`, never between their unrelated `z` charts |
| Proxy promoted silently | Repaired: acceptance is a promotion veto, while runtime, jump size, initialization memory, and canary behavior are explanatory only |
| Missing stop conditions | Repaired: evidence-invalidity, native divergence, nonfinite score/value, lineage failure, and resource exhaustion are hard stops; failed convergence diagnostics normally extend to the next checkpoint |
| Unfair opportunities | Repaired: G/H receive the same segment size, checkpoint ladder, maximum retained draws, and acquisition wall-time cap; one chart's early admission does not reduce the other's opportunity |
| Stale context/source | Phase 5 receipt, Phase 6 final kernel receipt, A0 starts, payload/transport identities, and current target/adapter/loader/HMC source hashes are revalidated before each GPU stage |
| Environment mismatch | Serious work is TensorFlow/TFP `float64`, trusted physical GPU 1, XLA JIT on, TF32 metadata recorded; CPU-hidden execution is limited to tests/reference checks |
| Artifact cannot answer question | Repaired: use the existing one-call retained archive primitive and compose immutable segments; add post-archive value/score checks because finite archived samples alone do not establish a valid gradient-bearing target at retained points |
| New checkpoint implementation risk | Rejected: the existing archive already emits hashed sample, final-state, final-target, and manifest sidecars. Phase 7 composes them rather than adding a competing shared checkpoint API |

The revised plan passes audit because every promotion decision is computed from
fresh retained draws with preserved chain identity, every mechanics artifact is
excluded, and a failed checkpoint triggers the next discriminating segment
unless a true continuation veto fires.

## Immutable Segment And Seed Contract

One archive-runner invocation writes exactly one immutable retained shard.
Phase 7 composes equal-size shards in draw order:

1. Segment 0 starts from the four reconstructed A0 starts and is the only
   segment with burn-in.
2. Segment `k>0` uses exactly the parsed final-state sidecar of segment `k-1`,
   has `num_burnin_steps=0`, and has a fresh predeclared stateless seed.
3. An initial runner owns the fixed burn-in shape. A separate continuation
   runner owns the fixed zero-burn-in shape and is reused for all equal-size
   continuation segments so XLA traces do not grow with segment count.
4. Each manifest records the previous private-manifest and final-state hashes.
   No-overwrite is mandatory; no partial prefix may be admitted.
5. Archived shape is `[draw, chain, parameter]`; diagnostics transpose it to
   `[chain, draw, parameter]`. Mapping to `theta` occurs segment-wise through
   that chart's frozen transport before concatenation.

All Phase 7 seeds are disjoint from Phase 6 seeds, whose maximum used component
was `6901`. Stage A freezes these exact mechanics-only seeds:

| Chart | Initial | Continuation compile | Warm continuation |
| --- | --- | --- | --- |
| G | `(7101,7102)` | `(7111,7112)` | `(7121,7122)` |
| H | `(7201,7202)` | `(7211,7212)` | `(7221,7222)` |

Acquisition seeds use the prospective formulas
`G_k=(8101+10*k,8102+10*k)` and
`H_k=(9101+10*k,9102+10*k)` for zero-based segment `k`. The final frozen
ladder must declare a finite `k` range and verify uniqueness before execution.

## Stage A Timing And Mechanics Canary

For each chart, run three immutable four-draw segments: segment 0 uses two
burn-in transitions; segments 1 and 2 use zero burn-in and exact continuation.
The canary measures initial compile/execute, continuation compile/execute, warm
continuation execution, serialization overhead, device placement, and hash
lineage. Its samples are mechanics-only and are permanently excluded from
retained evidence.

Stage A passes only if both charts have finite samples/value/score/core
telemetry, all four chains move cumulatively, output evidence tensors are on
the trusted GPU, XLA trace counts are one per runner, exact continuation and
hash checks pass, and the command finishes within its declared `1,800` second
wall cap. Acceptance and convergence metrics cannot fail or pass this canary.

Stage A passed. The public receipt is
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-7-retained-admission/timing-canary.json`
with SHA-256
`647be960a5307d564d1777d9cee5488262f3345ac0fd46ae0a5aea05367841ef`.
All six segments passed hash lineage, GPU placement, XLA trace, finite
value/score, and final-target replay checks; all four chains moved in both
charts. The samples remain permanently excluded from retained evidence.

Measured slower warm continuation was `0.455205` seconds per four retained
draws, or `0.113802` seconds per retained draw per chart. The full canary cost
was `833.715102` seconds and was compilation-dominated. Using this measurement,
the retained contract is frozen as follows:

| Field | Frozen value |
| --- | --- |
| Retained draws per immutable segment | `256` per chain |
| Burn-in | `128` transitions in segment 0 only; `0` thereafter |
| Cumulative checkpoints | `256`, `512`, `1,024`, `2,048` draws per chain |
| Maximum segments per chart | `8` |
| Maximum opportunity | `2,048` retained draws per chain for each G/H |
| Per-chart acquisition cap | `1,050` seconds |
| Cumulative acquisition cap | `2,100` seconds (`0.5833` GPU-hours) |
| Stage order | G then H; one chart's admission does not truncate the other's opportunity |
| Output | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-7-retained-admission/retained-acquisition.json` |

The slower warm rate projects `495.263` seconds for both charts' maximum
retained plus burn-in transitions. Adding the entire measured canary compile
and mechanics cost gives `1,328.978` seconds; the `2,100` second cap leaves
`771.022` seconds of margin, more than the required 25%. Compilation for the
new fixed shape is included in the cap. No retuning, candidate search, or HMC
kernel change is authorized.

Exact acquisition command:

```text
CUDA_VISIBLE_DEVICES=1 timeout 2100s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase7_retained_admission_2026_07_17.py --stage acquisition --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-7-retained-admission/retained-acquisition.json --wall-cap-seconds 2100
```

## Retained Admission And Value/Score Audit

At every frozen checkpoint, concatenate complete shards and compute in both
`z` and mapped `theta`:

- rank-normalized split R-hat, requiring every coordinate `<=1.05`;
- rank-normalized bulk and tail ESS, requiring every coordinate `>=100`;
- posterior-mean MCSE/SD, requiring every coordinate `<=0.10`; and
- cumulative movement for each of the four preserved chains.

Reconstruct exact per-chain accepted counts from every segment manifest.
Retained aggregate and per-chain acceptance must remain in `[0.55,0.85]`.
Acceptance outside the band blocks admission at that checkpoint but is not an
execution-invalidity stop: continue to the next frozen checkpoint because the
kernel is already frozen and the observation may stabilize.

Every segment receives a fixed-shape TensorFlow/XLA post-archive audit on all
retained `z` points. It recomputes transformed target values and scores,
requires all values/scores to be finite, and requires the final recomputed
target value to match the archived final-target sidecar within `1e-10`
absolute error. This is an engineering-validity gate, not a convergence or
posterior-correctness claim.

## Cross-Replication Stability

Compare G and H only after each independently passes admission. Use mapped
`theta` draws and the prospectively frozen 14 functionals:

- four coordinate means `E[theta_i]`; and
- ten upper-triangular raw second moments `E[theta_i theta_j]`, `i<=j`.

For each chart and functional, estimate the pooled mean MCSE from its four
preserved chains. Define

```text
z_GH = abs(mean_G - mean_H) / sqrt(MCSE_G^2 + MCSE_H^2).
```

The stability promotion veto fires if any functional has a nonfinite MCSE or
`z_GH>3.0`. The threshold is a conservative simultaneous diagnostic screen,
not a formal equivalence test. No ranking is supported: passing means only
that no predeclared material cross-replication instability was detected at the
available Monte Carlo precision. All continuous differences remain
descriptive. Phase 8 performs the separate predictive-law comparison.

## Public And Private Artifact Boundary

Private manifests expose raw sample/final-state paths and are consumed only by
the Phase 7 harness. Public receipts expose artifact labels, SHA-256 hashes,
counts, diagnostics, decisions, and lineage hashes, but never raw values,
sample descriptors, kernel payloads, or private paths. The result note may cite
the public receipt path and hash only.

## Required Checks And Focused Review

- exact Phase 5/6 target, payload, transport, source, start, and kernel replay;
- seed uniqueness/disjointness and no-overwrite tests;
- exact final-state continuation and manifest/shard/sidecar hash tests;
- burn-in only on segment 0 and XLA runner-reuse tests;
- draw/chain ordering and cumulative per-chain acceptance aggregation tests;
- mapped `theta` shape/finite and post-archive value/score checks;
- strict-JSON nonfinite explanatory encoding;
- no admission from partial G/H or canary samples; and
- one focused native review of the harness, statistical gates, final timing-
  derived ladder/resource cap, and result receipt.

## Stop, Result, And Handoff

Stop acquisition for invalid evidence, a hard sampler veto, exhausted frozen
resource cap, or missing authority. A failed admission checkpoint is normally
a continuation signal to the next predeclared segment, not rejection of the
chart or research direction.

Result interpretation must answer separately:

| Ledger | Required disposition |
| --- | --- |
| Engineering correctness | Valid or invalid archive/target/score/lineage |
| Sampler validity | G admission, H admission, and the exact failed gates |
| Cross-replication | Stable/unstable/not reached; no ranking unless separately supported |
| Scientific interpretation | What remains viable and what is not concluded |

Phase 8 receives only independently admitted and cross-replication-stable G/H
archive hashes, exact kernel/transport/target lineage, mapped diagnostic
receipts, and fresh predictive random-number banks. No Phase 7 run may claim
predictive equivalence, posterior truth, complete mode coverage, or method
superiority.
