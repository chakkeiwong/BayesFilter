# G2.3 Fixed-Trajectory HMC Leapfrog Ladder (Amendment A3 completion)

Date: 2026-09-01. Commit at planning time: `bf4d697f`.
Governing authority: `docs/plans/hardbound-kink-hmc-master-program-2026-08-21.md`
Amendment A3, which requires an explicit `num_leapfrog_steps` for G2.3
"selected via manual tuning ladder". This plan is that ladder.

## 1. Question

What trajectory length `L` should the G2.3 windowed-dense-mass route use, now
that the kernel is fixed-trajectory HMC at acceptance target 0.70 instead of
NUTS at 0.95?

NUTS chose its own trajectory length adaptively and no longer does. `L` is
therefore a new free control introduced by A3, with no inherited value: the
current `NutsConfig.num_leapfrog_steps = 50` is a placeholder I wrote during the
port, not a tuned quantity, and must not be treated as a default.

## 2. Mechanism under test

Under dual averaging the step size `eps` is driven to whatever holds acceptance
at 0.70, so `L` controls the integration time `L*eps`, which is what determines
how far one proposal travels. Too small and the chain random-walks (high R-hat
at any acceptance); too large and each iteration costs `L` gradients for a
trajectory that has already turned around. The prior NUTS evidence bounds the
plausible region: NUTS saturated at `2^10 = 1024` leapfrog steps while running
step sizes of 5.99e-4 to 9.19e-4. If acceptance 0.70 buys a step size near 5e-2
(the windowed smoke test observed 5.2e-2), the same integration time needs
`L ~ 1024 * 6e-4 / 5e-2 ~ 12`. A geometric ladder over the tens therefore
brackets the expected optimum from below and above.

## 3. Evidence contract

- **Comparator.** Rungs are compared against each other, not against NUTS. The
  NUTS arms are not a baseline here: A3 retired them on policy and performance
  grounds, so a NUTS-vs-HMC speed or mixing comparison is out of scope and will
  not be claimed either way.
- **Primary promotion criterion.** Not in this screen. Promotion of an `L` is
  the G2.3 gate itself (`tests/hardbound/test_phase2_joint_hmc.py`, windowed
  route): `divergences <= 0.001 * 12000`, per-parameter R-hat `< 1.01`, and
  posterior means within `3` posterior sd of fixture truth. The ladder only
  nominates the rung that gate runs at.
- **Screen nomination rule.** Among rungs whose screen max R-hat is below a
  loose eligibility bar of `1.2` (evidence the chain is moving at all), nominate
  the rung with the highest `min(ESS) / total gradient evaluations`. Gradient
  count is `num_chains * (warmup + samples) * L`, so this ranks mixing per unit
  of work rather than mixing per iteration, which would trivially favour large
  `L`.
- **Vetoes.** Nonfinite draws, divergences above the gate's own `0.001` rate, or
  a sampling acceptance outside the wide adapter-failure band `(0.30, 0.98)`.
  Acceptance inside any band is *not* evidence of good mixing. That inference is
  exactly the G2.3 error A3 corrects, where 0.95 acceptance coexisted with
  R-hat 1.03.

  *Veto corrected before execution (2026-09-01).* This clause first read "a rung
  whose achieved acceptance falls outside A3's band `(0.65, 0.75)`". A smoke
  check on a 3-d Gaussian while wiring the diagnostics showed why that is wrong:
  final warmup window acceptance 0.60, sampling acceptance 0.86, against target
  0.70. The windowed handoff freezes `exp(log_averaging_step)`, the *smoothed*
  dual-averaged step size, which is more conservative than the instantaneous
  value, and a smaller step buys higher acceptance. The offset is a property of
  the handoff and appears at every rung, so an A3-band veto on the sampling
  phase would have failed rungs for a reason unrelated to `L` -- the "fails for
  the wrong reason" case in section 7's own pre-mortem. The A3 band is now
  applied where dual averaging actually operates (the final warmup window) and
  recorded per rung as `warmup_acceptance_in_a3_band`, read in the result note
  rather than asserted. The band was not weakened as a convenience: it was
  moved to the phase it describes, and a distinct adapter-failure veto covers
  what the original clause was reaching for.
- **Explanatory only.** Screen R-hat and ESS values, per-window metric
  condition numbers, adapted step sizes, and wall times. Single seed per rung,
  so no ranking beyond the nomination rule is statistically supported.
- **Repair trigger, not an `L` failure.** If per-window condition numbers stay
  at ~1e5 across every rung, the dense metric is not absorbing the geometry and
  the next action is metric work (shrinkage `lambda`, window budget), not more
  `L` search. Tuning `L` against a broken preconditioner would be the way this
  screen passes while misleading us.
- **Continuation.** If the trend is still improving at the top rung, extend the
  ladder upward once (`L = 256`) under the same budget rules.
- **Non-claims.** No optimal-`L` claim, no cross-fixture or cross-model `L`
  inheritance (G2.2 and the Phase 3 harnesses tune their own), no posterior
  correctness claim, no NUTS comparison, and no claim that a passing gate
  validates the frozen warm starts listed in section 4.
- **Artifact.** `docs/plans/hardbound-g2-3-leapfrog-ladder-result-2026-09-01.md`
  with the decision table, inference-status table, and run manifest CLAUDE.md
  requires.

## 4. Default and assumption audit

Everything below is held fixed across rungs so that `L` is the only varying
control. All of it predates A3 and was tuned against NUTS, so each is a
warm-start hypothesis, not a validated default.

| Choice | Value | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|---|
| `initial_step_size` | 1e-2 | NUTS-era G2.3 call site | Dual averaging starts far from the 0.70 optimum and wastes early warmup | Adapted step size per window in screen output | Warm start |
| `mass_shrinkage` | route default | A4 windowed work | Off-diagonals over- or under-shrunk for HMC trajectories | Per-window condition number | Warm start |
| Window buffers | 75 / 50 / 25 | Stan convention, A4 | Slow windows too short to estimate a 337x337 covariance | `pooled_draws` per slow window | Warm start |
| Screen budget | warmup 1000, ns 1000 | Chosen here for cost | Screen R-hat does not predict gate R-hat at 4000/3000 | Nomination rule ranks ESS/gradient, not absolute R-hat | Convenience, declared |
| Gate budget | warmup 4000, ns 3000 | NUTS-era G2.3 | Insufficient for HMC at the nominated `L` | Gate ESS alongside R-hat | Warm start |
| Seed | 20260822 | NUTS-era G2.3 | Single-seed noise mistaken for an `L` effect | All continuous metrics declared descriptive | Frozen for comparability |
| Chains | 4 | Master program | R-hat on 4 chains is noisy | ESS reported with R-hat | Reviewed default |
| dtype / device | float64, CPU | Master program lines 53-54, risk row 263 | None material; float64 GPU is slow here | N/A | Reviewed default |

GPUs are intentionally hidden with `CUDA_VISIBLE_DEVICES=-1` before any TF
import, per the CLAUDE.md CPU-only artifact rule.

## 5. Threshold note (recorded, not amended)

Master program line 191 states G2.3 R-hat `< 1.02`. The gate test asserts
`< 1.01`, with an in-test justification citing the repository standard. The test
is the stricter of the two, so it governs; a result in `[1.01, 1.02)` would pass
the master program as written and fail the test. Both numbers will be reported
so that reading is not left implicit. This plan does not change either.

## 6. Ladder

Rungs `L in {8, 16, 32, 64, 128}`, four chains, warmup 1000, samples 1000,
windowed dense mass, acceptance target 0.70, all other controls per section 4.
Cost is `4 * 2000 * sum(L) = 1.98e6` gradient batches, roughly 20 minutes at the
observed throughput.

Command:

```
CUDA_VISIBLE_DEVICES=-1 conda run -n tf-gpu python -m pytest \
  tests/hardbound/test_g2_3_leapfrog_ladder.py -q -s -m "hmc and extended"
```

The screen asserts only the true vetoes (finiteness, divergence bound,
acceptance band). It does not assert R-hat: a nomination screen that fails on
the criterion it is meant to inform would tell us nothing.

## 7. Pre-mortem

- *Passes while misleading.* Condition numbers stay ~1e5 and the nominated `L`
  merely compensates for a poor metric. Caught by the condition-number repair
  trigger in section 3.
- *Fails for the wrong reason.* Screen warmup 1000 leaves the windowed metric
  underfitted, so every rung looks bad and the ladder appears flat. Caught by
  `pooled_draws` per slow window; the response is a longer screen, not a
  verdict on `L`.
- *Right answer, wrong scope.* A rung wins on ESS/gradient at a budget the gate
  does not use. This is why nomination and promotion are separated and the gate
  runs at the full 4000/3000.

## 8. Budget and stop conditions

Total campaign budget for this ladder: the screen plus at most two full gate
attempts at the nominated `L`, plus one upward ladder extension if section 3's
continuation clause fires. Stop for direction on: a mathematical error in the
survey, a gate failure surviving one diagnosis-and-repair cycle (master program
line 62b), or any action needing a file outside the master program's Section 7
scope.

## 9. Skeptical audit result

Audited before execution per the CLAUDE.md pre-execution rule. Three material
flaws were found in the first draft of this plan and repaired here: the screen
originally used R-hat as its promotion criterion at a budget that cannot support
it (now nomination-only, with the gate as the sole promotion criterion);
acceptance-in-band was originally read as mixing evidence (now an
out-of-band-only veto, with the G2.3 counterexample recorded); and the frozen
controls were originally unlabelled inherited defaults (now an explicit
warm-start table with per-row failure modes). Audit passes with those repairs.
