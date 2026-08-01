# Canonical LGSSM Balancing And Kalman Repair Phase 2 Result

Date: 2026-07-17

Status: `PASS_HANDOFF_WITH_SHARED_KALMAN_INCONCLUSIVENESS`

Campaign ID: `canonical-lgssm-balancing-kalman-repair-20260717`

## Outcome

The paired `T=2` particle ladder completed at `N=128,256,512,1024`.  Every
Contract E and no-reset arm passed finiteness, replay, chart, preparation,
identity, and applicable marginal/reset gates.  Both arms remained
statistically inconclusive against the differentiated Kalman oracle at every
rung.  At `N=1024`, all six paired reset-minus-no-reset absolute-loss intervals
also remained inconclusive.

This rejects a narrow reset-specific diagnosis: the current evidence does not
show Contract E causing the disagreement relative to the same prepared
no-reset weighted recursion.  It does not prove that either finite-particle
score is Kalman-correct.

## Particle Ladder

| `N` | Contract E screen | No-reset screen | Global paired reset direction |
| ---: | --- | --- | --- |
| 128 | inconclusive | inconclusive | mixed or inconclusive |
| 256 | inconclusive | inconclusive | mixed or inconclusive |
| 512 | inconclusive | inconclusive | mixed or inconclusive |
| 1024 | inconclusive | inconclusive | mixed or inconclusive |

At `N=1024`, the Contract E simultaneous normalized-error intervals were:

| Quantity | Interval | Declared equivalence region |
| --- | --- | --- |
| value | `[-0.0068231, 0.0019971]` | `[-0.001,0.001]` |
| phi1 | `[-0.0156575, 0.0203939]` | `[-0.05,0.05]` |
| phi2 | `[-0.1085120, 0.0229274]` | `[-0.05,0.05]` |
| phi3 | `[-0.9026845, 0.0005434]` | `[-0.05,0.05]` |
| q_scale | `[-0.0148292, 0.0285334]` | `[-0.05,0.05]` |
| r_scale | `[-0.0043869, 0.0191484]` | `[-0.05,0.05]` |

The no-reset intervals were nearly the same.  No paired quantity supported
either lower or higher Contract E error.  Descriptive narrowing with particle
count is not a statistically supported ranking or convergence rate.

## Resource And Program Evidence

- Total Phase 2 GPU wall time across eight arms: `3337.65 s`.
- `N=1024` Contract E: `1118.88 s` total, `59,751,424` byte allocator peak.
- `N=1024` no-reset: `1096.92 s` total.
- Contract E maximum active row residual: `3.1086e-15`.
- Contract E maximum active post-quotient column residual: `4.7962e-14`.
- Contract E maximum marginal tolerance: `1.0181e-10`.
- The candidate graph contains a Python-unrolled horizon and
  `13,798` operations at `T=2`; this remains an XLA production-loop-policy gap.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Advance with `N=1024` to fresh-seed `T=2` | largest predeclared rung is inconclusive without reset-specific veto | no hard validity veto | wide value, phi2, and phi3 intervals | run separate same-scalar FD diagnostic and fresh-seed paired `T=2` | no Kalman equivalence, reset superiority, HMC, XLA-loop, or leaderboard claim |

## Post-Run Red Team

The strongest alternative explanation is shared finite-particle proposal and
importance-weight error, not Contract E.  Another viable explanation is low
power under 16 Student-model seeds.  The paired evidence cannot distinguish
those explanations.  A later horizon cannot repair a failed or inconclusive
`T=2` center screen; it can only test accumulation after a valid handoff.
