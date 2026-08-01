# Canonical LGSSM Balancing And Kalman Repair Result

Date: 2026-07-17

Status: `MECHANICAL_SUBDEFECT_PASS_CANONICAL_AND_SCIENTIFIC_INCOMPLETE`

Campaign ID: `canonical-lgssm-balancing-kalman-repair-20260717`

## Outcome

The LGSSM terminal-balancing defect is repaired.  The candidate construction
enforcing Contract E semantics now requires an explicit positive terminal
balance count.  Primal and manual JVP execute the same count, reset validity
fails closed on both consumed-plan
marginals, telemetry exposes the post-quotient column marginal, and preparation
identity binds the schedule.  A Kalman-blind design/audit selected
`balance_steps=50`.

The repaired finite program is internally derivative-correct at the tested
center: the maximum HMC-coordinate same-scalar FD relative error was
`2.1596e-10`, and manual JVP versus TensorFlow forward AD passed in affected
tests.  This does not establish the scientific score target.

Fresh `T=2,N=1024`, 16-seed Contract E and no-reset comparisons against an
independently differentiated Kalman filter were both `inconclusive`.  No paired
quantity supports a reset-specific error direction.  Therefore the severe old
gradient discrepancy is not reproduced as a Contract E-only wiring defect,
but Kalman equivalence is not established.

`T=10` and `T=50` were not launched.  The measured `T=2` nodes take
`885.02 s` and `951.11 s`; `T=10` cannot plausibly satisfy the frozen
30-minute node cap, and the candidate horizon remains Python-unrolled.  Lowering
particles, seeds, or horizons after observing cost would have changed the
reviewed scientific design.

Finally, the repository production identity factory remains intentionally
empty.  These artifacts are diagnostic-only and cannot support canonical
leaderboard, HMC, or admission claims even though the local balancing subdefect
is repaired.

## Claimed And Computed Quantities

| Item | Claimed target | Quantity actually computed | Verdict |
| --- | --- | --- | --- |
| own-scalar score | total HMC derivative of the exact fixed finite Contract E value | manual total JVP and central FD of the same prepared callable | correct at the tested center and fixed branches |
| Kalman score | differentiated exact LGSSM observed-data likelihood | simultaneous mean relative-error intervals from 16 fixed particle seeds | inconclusive at `T=2`; not checked at `T=10,50` |
| reset effect | Contract E-minus-no-reset absolute Kalman loss | paired simultaneous intervals | all six inconclusive |
| canonical identity | factory-issued non-overridable identity of the actual callable/settings | preparation identity plus source hashes while production factory is empty | unsupported for canonical admission |
| XLA production loop | fixed-state TensorFlow loop rather than horizon graph unrolling | XLA-compiled graph with Python-unrolled horizon | wrong relative to production loop-policy certification |

## Scientific Results

The differentiated Kalman oracle at `T=2` is:

```text
value = -8.862150494354594
HMC score = [
  1.8435713978982007,
 -0.2679662859084861,
 -0.07290903832038856,
  1.546024045068626,
  5.012160301141432
]
```

Contract E simultaneous normalized-error intervals:

| Quantity | 95% Bonferroni/Student interval | Required region | Status |
| --- | --- | --- | --- |
| value | `[-0.0035853, 0.0039797]` | `[-0.001,0.001]` | inconclusive |
| phi1 | `[-0.0099955, 0.0166499]` | `[-0.05,0.05]` | contained |
| phi2 | `[-0.0598487, 0.0510096]` | `[-0.05,0.05]` | inconclusive |
| phi3 | `[-1.0754749, 0.2965130]` | `[-0.05,0.05]` | inconclusive |
| q_scale | `[-0.0277359, 0.0138314]` | `[-0.05,0.05]` | contained |
| r_scale | `[-0.0178792, 0.0145395]` | `[-0.05,0.05]` | contained |

The global screen is inconclusive because all six intervals must be contained.
The no-reset intervals are nearly identical and likewise inconclusive.  All
paired absolute-loss intervals cross zero.  The evidence supports no ranking.

## Engineering Ledger

| Check | Result |
| --- | --- |
| explicit terminal balance | candidate callable factory rejects zero; diagnostic core alone permits explicit historical zero |
| selected schedule | first marginal-only design pass at 50; untouched audit passed |
| consumed row marginal | fresh `T=2,N=1024` maximum `4.2188e-15` |
| consumed post-quotient column marginal | maximum `3.3751e-14` versus tolerance `1.0472e-10` |
| same-scalar FD | pass; maximum coordinate relative error `2.1596e-10` |
| bitwise replay | pass for both fresh arms |
| GPU/XLA | RTX 4080 SUPER, float64, XLA, exact 8192 MiB cap, no memory growth |
| peak TensorFlow allocator | `59,751,424` bytes for each fresh `T=2` arm |
| affected tests | 53 passed: canonical 13, preparation/telemetry 12, streaming 18, campaign 10 |
| time-loop policy | open; `T=2` graph has 13,798 operations and the horizon is Python-unrolled |

Three historical diagnostic harnesses falsely recorded preparation
`balance_steps=0` while executing 1.  Their callers now record 1.  Existing
artifacts remain historical and are not upgraded.

## Inference Status

| Question | Status |
| --- | --- |
| hard veto screen | passed for every executed balanced arm and FD endpoint |
| statistically supported ranking | none |
| descriptive-only differences | particle-count narrowing and small Contract E/no-reset mean differences |
| default readiness | not established |
| next evidence needed | loop-core repair/resource pilot, longer horizons, and a predeclared `T=2` precision/power design |

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| keep `CE-01` open after repairing its balancing/gating subdefect | passed design/audit, derivatives, marginals, preparation identity fields, callers, and tests | production identity factory remains an admission veto | exact route-spec dependency closure | preserve `balance_steps=50` only for this LGSSM campaign evidence and later bind the eligible callable in the repository factory | no full canonical closure, general optimum, or cross-model transfer |
| keep `CE-02` open | `T=2` Kalman screen inconclusive; `T=10,50` absent | no execution-invalidity veto | finite-particle uncertainty and shared proposal/weight error | repair loop feasibility, then finish horizon ladder and precision design | no Kalman equivalence or failure claim |

## Run Manifest

| Field | Value |
| --- | --- |
| git commit | `15170e1573d19b235d96f3ed3525fa2071f58320` with scoped uncommitted campaign changes |
| environment | project TensorFlow environment, TensorFlow `2.19.1` |
| CPU/GPU | CPU-hidden for oracle aggregation/tests; RTX 4080 SUPER for serious XLA nodes |
| data | LGSSM dataset seed `81100`; observation hashes in arm artifacts |
| Phase 2 seeds | `81400..81415` at `N=128,256,512,1024` |
| Phase 3 seeds | `81500..81515` at `T=2,N=1024` |
| serious GPU artifact wall time | approximately `1.52 h` including Phase 1 selected run; below the four-hour allocation |
| output root | `docs/benchmarks/artifacts/canonical_lgssm_balancing_kalman_repair_20260717/` |
| structured run manifest | `docs/benchmarks/artifacts/canonical_lgssm_balancing_kalman_repair_20260717/run_manifest.json` |
| plan | `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-plan-2026-07-17.md` |

Key SHA-256 values:

```text
phase2/N1024 aggregate  29c48cb13a6f8bdfdd13682c8b11941eb86eebc2d37949ccd51565f49204bf65
same-scalar FD          512b322d2772877a0257eec935ff90b9ee41755677ac6175b32038de8eb680f5
fresh T2 Contract E     4539e61beeb813f1c5b646ac3771c3a5d763462dff4ae8e03d3b0da4714e6c7d
fresh T2 no reset       f048e0b2b99934e98ce207b9e7fc3f24f4a33e2eeb239db9973725b669cfa1da
fresh T2 aggregate      f640b695c53c1ce8c086b7969d1914830eb9a7788a020dc5834d5859a3f3175b
```

## Post-Run Red Team

The strongest alternative explanation is shared finite-particle proposal and
importance-weight error, possibly amplified for the small Kalman `phi3` score,
rather than reset error.  Low power under 16 Student-model seeds is also viable.
The present artifacts cannot distinguish those explanations.

An outcome that would overturn the limited mechanical-subdefect conclusion is
a balanced caller test showing primal/manual-JVP schedule mismatch, a consumed marginal
outside its bound, or a non-historical implicit zero path.  None occurred.
An outcome needed to establish scientific equivalence is simultaneous interval
containment at all declared horizons under a valid power/resource design.

The weakest evidence is the transfer of the marginal-only `balance_steps=50`
schedule beyond the frozen design/audit and `T=2` executions.  It is not a
mathematical Sinkhorn convergence theorem, a cross-model default, or an
admission certificate.
