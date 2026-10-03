# G2.3 leapfrog ladder and gate — result note

- Plan: `docs/plans/hardbound-g2-3-leapfrog-ladder-2026-09-01.md`
- Screen artifact: `docs/plans/hardbound-g2-3-leapfrog-ladder-screen-2026-09-01.json`
- Governing authority: master program `hardbound-kink-hmc-master-program-2026-08-21.md`,
  Amendment A3 (NUTS retired suite-wide; fixed-trajectory HMC at acceptance
  target 0.70, band (0.65, 0.75); trajectory length selected by manual ladder).

## 1. Question

A3 replaced NUTS with fixed-trajectory HMC, which has no tree-doubling to pick
its own integration time. The trajectory length `L` therefore becomes an
explicit tuning choice, and A3 requires it be selected by a manual ladder rather
than inherited. This note records that selection and the G2.3 gate run that
followed.

Planning arithmetic predicted the optimum near `L ~ 12`: NUTS had saturated at
`2^10 = 1024` leapfrog steps at step sizes 5.99e-4 to 9.19e-4, and at the new
acceptance target the adapted step size was expected near 5e-2, giving
`L ~ 1024 * 6e-4 / 5e-2 ~ 12`. The ladder contradicted that estimate.

## 2. Ladder screen

Four chains, warmup 1000, draws 1000, windowed dense mass adaptation,
acceptance target 0.70, `initial_step_size` 1e-2, seed 20260822. Rungs run
in one process; each rung writes its record incrementally so a partial run
survives.

| L | max R̂ | min ESS | min ESS/grad | L·ε | sampling acc | warmup acc (final window) | final cond | wall (s) | eligible |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 4.6043 | 6.1 | 9.553e-05 | 0.2844 | 0.897 | 0.685 | 2.11e+04 | 70.9 | no |
| 16 | 1.6282 | 13.5 | 1.058e-04 | 0.4408 | 0.914 | 0.655 | 5.57e+03 | 91.6 | no |
| **32** | **1.0869** | **115.5** | **4.510e-04** | **0.8414** | 0.898 | 0.670 | 2.47e+03 | 127.2 | **yes** |
| 64 | 4.8474 | 5.0 | 9.792e-06 | 1.5358 | 0.686 | 0.650 | 1.31e+03 | 190.8 | no |
| 128 | 1.2280 | 46.5 | 4.541e-05 | 3.1222 | 0.843 | 0.680 | 1.18e+03 | 331.7 | no |

Nomination rule from the plan: among rungs with max R̂ below the loose
eligibility bar of 1.2, take the highest `min(ESS) / gradient evaluations`.
`L = 32` is the only eligible rung and also carries the highest ESS/gradient, by
4.3x over the next best. Nomination is unambiguous under the stated rule.

The plan's upward-extension clause did not fire. It authorizes one extension to
`L = 256` only if the trend is still improving at the top rung; `L = 128` is
worse than `L = 32` on both R̂ and ESS/gradient, so the trend is not still
improving and no extension was run.

Every rung passed the stated vetoes: zero sampling divergences throughout,
finite draws throughout, and sampling acceptance inside the pathology band
(0.30, 0.98). Warmup acceptance in the final window sat inside A3's (0.65, 0.75)
at every rung (0.650 to 0.685), so dual averaging reached its target and these
rungs are the sampler A3 specifies.

## 3. The response is not monotone in L, and the damage is one coordinate

`L = 64` is worse than both its neighbours: max R̂ 4.8474 against 1.0869 at
`L = 32` and 1.2280 at `L = 128`. A monotone reading of the ladder would have
missed this entirely.

Per-coordinate detail locates it. The 9-vector is `[6 drift parameters, 3 log
noise scales]` — the fixture builds `truth` as
`list(theta_bar_truth) + [log(5e-4)] * 3` with `theta_bar_truth` of length 6, and
`out["rhat"][0]` covers `theta_raw` only.

| L | θ0 | θ1 | θ2 | θ3 | θ4 | θ5 | θ6 | θ7 | θ8 |
|---|---|---|---|---|---|---|---|---|---|
| 32 R̂ | 1.026 | 1.059 | 1.016 | 1.023 | 1.048 | 1.036 | 1.011 | 1.024 | 1.087 |
| 32 ESS | 167 | 151 | 255 | 283 | 241 | 296 | 446 | 226 | 116 |
| 64 R̂ | 1.329 | 1.104 | 1.001 | 1.075 | 1.165 | 1.392 | **2.593** | **3.109** | **4.847** |
| 64 ESS | 15.8 | 42.7 | 1977 | 93.8 | 27.4 | 13.8 | 6.4 | 5.8 | 5.0 |
| 128 R̂ | 1.019 | 1.002 | 0.999 | 1.068 | 1.000 | 1.007 | 1.022 | 1.023 | **1.228** |
| 128 ESS | 4000 | 4000 | 4000 | 4000 | 4000 | 4000 | 4000 | 4000 | 46.5 |

The coordinates that break at `L = 64` are θ6, θ7, θ8 — exactly the three log
noise scales. At `L = 128` the six drift parameters and the first two noise
scales reach ESS saturated at 4000, the full retained draw count, while θ8 alone
lags at ESS 46.5. θ8 is the min-ESS coordinate at every rung, so the entire
ESS/gradient ranking is a referendum on one direction.

**Candidate explanation, not a finding.** Log noise scales in a non-centred
state-space chart sit in a funnel against the latent innovations: the conditional
scale of `eta_raw` depends on the noise scale being sampled. A dense mass matrix
absorbs fixed linear correlation, not varying curvature, so it cannot flatten a
funnel. Trajectory-length periodicity in a stiff direction would also produce
non-monotone mixing at healthy acceptance, since energy conservation — and hence
acceptance — is preserved when a trajectory returns near its start. Both readings
fit the data. Neither is established here: one seed per rung, and the metric
differs across rungs (final condition numbers 2469 / 1310 / 1180), so there is no
single frequency to fit and no replication to separate the two.

## 4. Gate attempt 1 — failed at an unchanged threshold

`L = 32` set at the G2.3 gate call site, warmup 4000, draws 3000, 4 chains, seed
20260822, `initial_step_size` 1e-2, target accept 0.70, windowed dense mass.
Wall time 379 s.

```
max R-hat 1.0163   min ESS 360.0
per-θ R-hat [1.0103 1.0067 1.0038 1.0001 1.0060 1.0034 1.0082 1.0123 1.0163]
per-θ ESS   [1359.9 1577.7 1680.2 1341.6 1394.6 1651.5 1109.8 1380.6  360.0]
sampling divergences 0 / 12000     warmup divergences 44
cond  8.68e4 → 4.37e4 → 3.88e4 → 1.20e4 → 3.53e3 → 1.44e3 → 8.75e2
step  1.69e-3 → 5.29e-2 → 5.59e-2 → 3.71e-2 → 3.68e-2 → 3.55e-2 → 3.50e-2
```

The divergence criterion passed with room (0 against a budget of 12). Six of nine
coordinates cleared 1.01. Three missed: θ0 at 1.0103, θ7 at 1.0123, θ8 at 1.0163
— and θ8's ESS of 360 against ~1400 elsewhere is the same bottleneck the ladder
showed.

This is the best number the campaign has produced. Diagonal adaptation reached
1.048, block-dense single-freeze 1.083, windowed dense under NUTS at acceptance
0.95 reached 1.073, and windowed dense under A3's HMC at `L = 32` and acceptance
0.70 reaches 1.0163.

**Which threshold governs.** Master program line 191 states G2.3 R̂ `< 1.02`; the
gate test asserts `< 1.01`. 1.0163 falls in `[1.01, 1.02)` — it passes the master
program as written and fails the test. Section 5 of the plan recorded this
discrepancy and named the test as governing *before* the gate ran, so treating
1.0163 as a failure is the pre-registered reading, not a threshold chosen after
seeing the number. Under master program line 62 a gate failure grants one
diagnosis-and-repair cycle.

## 5. Repair cycle 1 — passed

Repair was budget at an unchanged threshold: warmup 4000 → 8000, draws 3000 →
8000. `L`, acceptance target, shrinkage, seed, initial step size, chains, and the
1.01 bound were all held fixed. The bound was not smoothed and `L` was not
retuned on the failed gate data; the master program forbids both.

Two diagnostics motivated budget specifically rather than as a convenience. The
slow-window condition numbers were still descending when warmup ended at 4000
(1.44e3 → 8.75e2 across the last two windows), so the metric had not converged.
And θ8's ESS of 360 against ~1400 elsewhere made its R̂ the noisiest of the nine,
so the excess over 1.01 was plausibly finite-sample.

Wall time 778 s.

```
max R-hat 1.0090   min ESS 848.7
per-θ R-hat [1.0022 1.0014 1.0007 1.0002 1.0018 1.0021 1.0075 1.0013 1.0090]
per-θ ESS   [5091.6 4689.4 4898.8 4250.7 4019.2 4983.0 1730.5 3991.6  848.7]
sampling divergences 0 / 32000     warmup divergences 46
window 7 [1650:3250] cond 9.808e+02 pooled  6400 step 3.614e-02
window 8 [3250:7950] cond 7.593e+02 pooled 18800 step 3.687e-02
```

All three gate criteria passed:

| criterion | bound | observed | status |
|---|---|---|---|
| post-warmup divergences | ≤ 0.1% (32 of 32000) | 0 | pass |
| split R̂, all 9 coordinates | < 1.01 | max 1.0090 | pass |
| posterior mean vs fixture truth | < 3 posterior sd | max 1.537 sd | pass |

Posterior means against truth, in units of posterior sd:
`[1.036 0.820 0.370 0.023 1.537 0.144 0.156 0.589 0.577]`. The three log noise
scales recover to -7.610, -7.566, -7.801 against a truth of -7.601.

**The pass margin is thin and rests on one seed.** max R̂ 1.0090 clears 1.01 by
0.0010. θ8's ESS is 848.7, so the Monte Carlo error on its R̂ is not negligible at
that margin, and a different seed could plausibly land above the bound. What is
established is that the gate criteria were met on this run, not that they are met
robustly.

**The prefix trace supports the repair's stated mechanism.** R̂ on prefixes of the
retained draws: 1.0286 (2000) → 1.0132 (4000) → 1.0183 (6000) → 1.0090 (8000),
with θ8 the max at every prefix. The sequence descends overall but not
monotonically — the rise at 6000 is consistent with Monte Carlo noise at this ESS.
A flat sequence would have meant the chains disagreed about the distribution and
no additional draws would help; a descending one means slow mixing that draws do
fix. θ8's ESS scaled 360 → 848.7 for a 2.67x draw increase, close to linear,
which is what a fixed slow autocorrelation time predicts and not what a stuck
coordinate would produce.

**Warmup was still improving at 8000.** Condition numbers descended 9.81e2 →
7.59e2 across the last two windows, so the metric had not converged at the larger
budget either. The step size settled at 3.687e-2, giving `L·ε = 1.18`.

θ8 remains the slowest coordinate by a factor of ~5 (ESS 848.7 against 4000-5100
for the drift parameters), and θ6 is second slowest at 1730.5. Both are log noise
scales. The funnel reading from section 3 is consistent with this and still not
established.

## 6. Decision table

| field | entry |
|---|---|
| decision | `num_leapfrog_steps = 32` set at the G2.3 gate call site; G2.3 passes at warmup 8000 / draws 8000 / 4 chains under A3's fixed-trajectory HMC at acceptance target 0.70 |
| primary criterion status | pass — all three gate criteria met: 0 divergences of 32000 (bound 32), max split R̂ 1.0090 (bound 1.01), max posterior-mean error 1.537 posterior sd (bound 3) |
| veto diagnostic status | no veto fired *among the diagnostics that were recorded*. Zero sampling divergences at every ladder rung and both gate attempts; finite draws throughout; sampling acceptance inside the pathology band (0.30, 0.98) at every rung; warmup acceptance inside A3's (0.65, 0.75) at every ladder rung. **Acceptance was not recorded for either gate run** — the gate call site computes it and discards it — so A3's band is unverified at the gate. See section 10 |
| main uncertainty | the pass margin is 0.0010 in max R̂ on a single seed, with θ8's ESS at 848.7. Monte Carlo error on R̂ at that ESS is not negligible relative to the margin, so seed-robustness of the pass is unestablished |
| next justified action | replicate the passing gate configuration on 2-3 additional seeds to establish whether the pass is robust or seed-specific. If any replicate lands in `[1.01, 1.02)`, the honest reading is that G2.3 sits at the threshold rather than clearing it |
| what is not being concluded | that `L = 32` is optimal; that the funnel or periodicity explanation for θ8 is established; that a G2.3 pass validates the hardbound target, the C1 kink, or the fixture beyond the three stated criteria; that fixed-trajectory HMC is superior to NUTS on this target |

## 7. Inference-status table

| row | status |
|---|---|
| hard veto screen | no hard veto supported among recorded diagnostics. Zero sampling divergences across 5 ladder rungs and 2 gate runs; all draws finite; acceptance inside band **at the ladder rungs only** (not measured at either gate run); no invalid artifact; no failed invariant |
| statistically supported ranking | none. One seed per ladder rung and one seed per gate run. The `L` ordering is a nomination under a predeclared screen rule, not a statistically supported ranking. `L = 32` is a viable candidate that passed the gate, not a demonstrated optimum |
| descriptive-only differences | all continuous ladder metrics: max R̂, min ESS, ESS/gradient, `L·ε`, condition numbers, wall times. The 4.3x ESS/gradient advantage of `L = 32` and the `L = 64` collapse are descriptive. The campaign progression 1.048 → 1.083 → 1.073 → 1.0163 → 1.0090 is descriptive |
| default-readiness | `L = 32` is fit for the G2.3 gate call site, which is the scope A3 asked for. It is not established as a cross-fixture default: master program lines 246-248 forbid cross-fixture tuning inheritance, so G2.1, G2.2, and the validation harness need their own selection or an explicit warm-start label |
| next evidence needed | seed replication of the passing gate configuration; if θ8 is to be improved rather than out-sampled, a targeted funnel diagnostic (θ8 against the `eta_raw` scale it conditions) would discriminate the section 3 explanations, and a reparameterization would be the repair the funnel reading implies |

## 8. Run manifest

| field | value |
|---|---|
| git commit | `bf4d697f23a6c3c0f1757e9ce0073d321562acd6` |
| worktree state | dirty. Modified: `bayesfilter/hardbound/hmc_runner.py`, `bayesfilter/hardbound/validation_harness_tf.py`, `tests/hardbound/test_phase2_joint_hmc.py`, master program (A3 amendment), plus files outside this campaign's scope carried in from prior work. Untracked: `bayesfilter/hardbound/{dense,windowed_dense}_mass_*.py`, `tests/hardbound/test_g2_3_{leapfrog_ladder,warmup_shrinkage_sweep}.py`, `tests/hardbound/test_windowed_mass_smoke.py`, this campaign's plan/screen/result docs |
| ladder command | `CUDA_VISIBLE_DEVICES=-1 conda run -n tf-gpu python -m pytest tests/hardbound/test_g2_3_leapfrog_ladder.py -q -s -m "hmc and extended"` |
| gate command | `CUDA_VISIBLE_DEVICES=-1 conda run -n tf-gpu python -u -m pytest tests/hardbound/test_phase2_joint_hmc.py::test_g2_3_full_c1_fixture_recovery -q -s -m hmc` |
| environment | conda env `tf-gpu`; Python 3.11.15; TensorFlow 2.19.1; TensorFlow Probability 0.25.0 |
| CPU/GPU status | **CPU-only by intent.** `CUDA_VISIBLE_DEVICES=-1` set before import, so `tf.config.list_physical_devices('GPU')` returns `[]` and the log's `cuInit: CUDA_ERROR_NO_DEVICE` is the expected consequence of hiding the device, not a hardware fault. Master program lines 53-54 permit CPU for all phases. Host: AMD EPYC 7773X, 256 logical cores, Linux 6.8.0-40-generic |
| XLA | `jit_compile=True` via `NutsConfig` default; log confirms `Compiled cluster using XLA!` |
| dtype | float64 throughout, per master program lines 53-54 |
| data version | fixture `model_tf.FIXTURE`, `target_id="mf_c1_k40_hardmax"`, horizon 40, simulation seed 20260821 |
| seeds | simulation 20260821; HMC chain seed 20260822 (ladder and both gate runs); ladder init scatter `RandomState(20260826)`; gate init scatter module-level `default_rng(20260821)` |
| wall time | ladder 812 s (5 rungs: 70.9 / 91.6 / 127.2 / 190.8 / 331.7); gate attempt 1 379 s; gate repair 1 778 s. Total ~1969 s (~33 min) |
| artifacts | screen `docs/plans/hardbound-g2-3-leapfrog-ladder-screen-2026-09-01.json`; gate logs `/tmp/g2_3_gate_L32.log`, `/tmp/g2_3_gate_L32_repair1.log` (temporary — the durable record is this note) |
| plan file | `docs/plans/hardbound-g2-3-leapfrog-ladder-2026-09-01.md` |
| result file | this note |

## 9. Post-run red-team

**Strongest alternative explanation for the pass.** The gate passed on the run
whose budget was chosen after seeing a failure at the smaller budget. Even with
the threshold held fixed and `L` untouched, one budget increase was tried and one
passed, so the pass carries an implicit selection: had 8000/8000 also returned
1.013, the honest report would have been a second failure, and the recorded
outcome would differ. The protection against reading this as "ran until it
passed" is that the budget was declared once in the test comment before the run
and the prefix R̂ trace is reported, which shows the descent rather than only its
endpoint. It is not protection against the possibility that a third budget would
have been tried had the second failed.

**What would overturn the conclusion.** A seed replicate of the passing
configuration returning max R̂ ≥ 1.01. That would move the reading from "G2.3
passes at this budget" to "G2.3 sits at the threshold, and the passing run was a
favorable draw". This is the cheapest test of the main uncertainty and is the
recommended next action.

**Weakest part of the evidence.** The `L` selection. Each rung is one seed at
warmup/draws 1000/1000, and the response is non-monotone, which is exactly the
regime where a single-seed ladder can mislead: if `L = 64`'s collapse is partly a
seed artifact, the true response surface may be flatter than the table suggests
and rungs between 32 and 128 were never tried. `L = 32` is defensible because it
passed the gate, which is the promotion criterion — not because the ladder
established it as best.

**A pass on three criteria is not target validation.** G2.3 checks sampler
convergence, divergence rate, and fixture recovery within 3 posterior sd. It does
not check that the C1 hard-max target is the intended mathematical object, that
the hardbound construction is correct, or that the posterior is well calibrated.
Those are separate questions with separate gates.

## 10. Defects found and their disposition

**Non-split R̂ at G2.2 — recorded, not fixed.** `hmc_runner.py:189` calls
`tfp.mcmc.potential_scale_reduction(s)` without `split_chains=True`, while the
windowed route (`windowed_dense_mass_adaptation.py:475`) passes it. G2.3 uses the
windowed route, so its 1.0090 is genuine split R̂ and this note's conclusion is
unaffected. But G2.2 calls `run_nuts` without `dense_mass_windowed`, so it asserts
non-split R̂ against master program line 188, which specifies "split-Rhat < 1.01".
Non-split R̂ is the weaker statistic — it cannot see within-chain trends — so
G2.2's gate is more permissive than the program requires. Not changed here
because tightening it could flip an already-recorded pass on a different gate,
which is outside this repair's scope and needs its own run.

**Ladder docstring overclaimed init parity — fixed.** `_g2_3_target` claimed it
mirrored the gate call site "exactly, including the seeded initialisation
scatter". It does not: the gate draws from a module-level
`np.random.default_rng(20260821)` (PCG64, state advanced by earlier tests in the
file), while the ladder used a fresh `np.random.RandomState(20260826)` (MT19937).
The target and the init *distribution* match; the realized draw does not.
Corrected in place, since R̂ at finite warmup depends on init dispersion and the
claim would have misled a reader reproducing the ladder.

**Gate discarded the fixture-recovery diagnostic on failure — fixed.** The
posterior-mean check sat after the R̂ assertion, so attempt 1 reported nothing
about whether the fixture was recovered. Moved above the assertions along with a
prefix R̂ trace. Observability only, per the Class A rule: no computed result
changed.

**Acceptance was never recorded at the gate — an unverified A3 band, and an
unsupported claim I made about it.** A3 imposes acceptance target 0.70 with band
(0.65, 0.75) unconditionally. The ladder screen measured and recorded acceptance
at every rung; the G2.3 gate call site did not print it, so neither gate log
contains it and the band is **not checked** for the two runs that actually decide
the gate. The runner does compute it — `sampling_is_accepted`,
`sampling_log_accept_ratio`, and a per-window `acceptance` are all returned — so
this is a discarded diagnostic at the call site, the same Class A defect this
campaign already fixed once inside the runner.

The first version of this note asserted "warmup acceptance inside A3's (0.65,
0.75) at every rung **and both gate runs**" and "acceptance inside band" across
"5 ladder rungs and 2 gate runs". The gate half of both statements was
unsupported: no such measurement was taken, and the ladder rungs ran at a
different budget (1000/1000 against 8000/8000), so they are not evidence about
the gate. Corrected above to "not measured". The gate's recorded pass on
divergences, R̂, and posterior recovery is unaffected — acceptance is not one of
the three gate criteria — but A3 compliance at the gate is now **unverified**
rather than established.

The gate print block should emit sampling acceptance and final-window warmup
acceptance, at which point the next gate run verifies the band directly. Not
added retroactively here: reporting a number this note's own runs did not produce
would be the error being corrected.

**A3's "no unwrapping" instruction rests on a false premise about TFP —
intent met, letter not.** A3's observability clause requires the trace to capture
`is_accepted`, `log_accept_ratio`, and `step_size` "directly from
`kernel_results` (no unwrapping)". Inspected directly in TFP 0.25.0:

- On a bare `PreconditionedHamiltonianMonteCarlo`, `kernel_results` is a
  `MetropolisHastingsKernelResults` carrying `is_accepted` and
  `log_accept_ratio` at top level — but **not** `step_size`, which lives one
  level deeper at `.accepted_results.step_size`.
- Under a `DualAveragingStepSizeAdaptation` wrapper, the results are
  `DualAveragingStepSizeAdaptationResults`, which expose **none** of the three;
  all require descending through `.inner_results`.

No-unwrapping access to all three fields is therefore impossible for this kernel
stack, whether or not the adapter is attached. Both routes use a
`while not hasattr(pkr, "log_accept_ratio"): pkr = pkr.inner_results` walk, which
resolves for bare and wrapped kernels and for one- or two-deep adapter stacks.
All three fields are captured, so A3's purpose is served; its stated mechanism is
not, because that mechanism does not exist in TFP 0.25.0. Flagged rather than
silently satisfied, since a future reader checking A3's boxes would otherwise
find the clause marked done and assume no unwrapping occurs.

Separately confirmed in the same inspection: `step_size` on a bare PHMC
bootstraps to the empty Python list `[]` (TFP `hmc.py` lines 748-749), populated
only by the adapter. This is why the windowed route's sampling phase — which runs
without the adapter by design — must emit its frozen scalar step size rather than
trace the kernel field.

**Amendment numbering collides across three schemes — recorded, not fixed.**
`test_phase2_joint_hmc.py` line 246 correctly says "master program Amendment A2",
but lines 270 and 276 say "Amendment A3" (meaning diagonal → dense mass) and
"Amendment A4" (meaning block-dense → windowed) unqualified. The master program
has only A1/A2/A3, where **A3 means the fixed-trajectory HMC kernel** — a
different change entirely — and the windowed convergence doc uses yet a third
sequence ("Amendment 1", "Amendment 2"). A reader following "Amendment A3" from
line 270 lands on the wrong amendment. Left alone because renumbering another
phase's comments is outside this repair; a comment I added during this work
initially compounded the problem with a nonexistent "A5" and now cites the master
program explicitly instead.

**A3 v2-protocol tension — still open.** `bayesfilter/inference/
fixed_trajectory_hmc_tuning_v2.py` is constrained to `mass_policy='identity'` and
so cannot supply tuning for this route's dense windowed metric. Either that
constraint is lifted or A3 should state that identity-mass v2 supplies warm-start
values only. This note used a route-specific manual ladder, which is what A3
mandates for G2.3, so the tension did not block the gate.

