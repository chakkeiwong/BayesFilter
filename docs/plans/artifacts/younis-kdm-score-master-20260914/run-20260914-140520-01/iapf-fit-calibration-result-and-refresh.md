# Recursive-iAPF calibration: underflow veto and bounded recovery

The three-control screening phase completed 20 unique GPU rows across weak and
curved scalar models. Both regimes nominated the wider, tighter `k=2, tau=10`
candidate, but every selected claim row contains density-objective underflow.
The weak candidate also loses descriptively to all four cheap heuristics. It
is ineligible for promotion. Curved comparators remain unrun after exhaustion
of the declared phase budget; the master program is still incomplete.

## Target and evidence

The computed score is the analytical derivative of the finite particle log
likelihood conditional on the fitted twist, realized particle count and
resampling labels. It is compared with the refined numerical model-score
reference. It is not an unbiased marginal-likelihood score. The frozen fit is
identical across the two final replicates; their final random streams differ.

Source: clean frozen revision `f5a4d411a4a0197fe9c1d25c24573337a633c96a` in
`.localresources/worktrees/younis-score-iapf-fit-calibration-20260916`.
The exact commands, environment, data identities, seeds, verified GPU memory
growth, FP32/TF32/XLA settings, trace counts and timings are preserved in
`iapf-fit-calibration-gpu-01/` and `iapf-fit-calibration-gpu-02/`.
The latter includes `driver-used.py`. Offline fitting uses FP64.

`iapf-fit-calibration-verification.json` checks all 20 result digests and five
completed study fingerprints, the saved driver and predecessor manifest,
finite scores, device/trace/memory evidence, the shared fits and independent
final streams, and the complete 64-charge accounting. It is an engineering
verification, not a scientific pass.

| Regime | Baseline calibration / validation squared error | Selected calibration / validation | Selected claim mean squared error |
|---|---:|---:|---:|
| Weak | 0.063722 / 0.502523 | 0.026208 / 0.089579 | 0.365829 |
| Curved | 0.654819 / 1.116853 | 0.012505 / 0.630580 | 0.777204 |

The selected final count is 32 in both regimes. The baseline starts at 16;
curved baseline validation adapts to 32. These comparisons do not match cost.
The original iAPF baseline was not evaluated on the claim datasets, so the
table cannot establish a heldout calibration benefit relative to that baseline.
Both candidates and their numerical settings remain hypotheses.

| Weak claim heuristic | Score squared error |
|---|---:|
| EKF | 0.001653 |
| UKF | 0.001802 |
| Bootstrap, N=32 | 0.119848 |
| Local linear, N=32 | 0.145185 |
| Selected iAPF, mean of two final draws, N=32 | 0.365829 |

Each particle heuristic has one draw. The observed underperformance is the
predeclared promotion veto; these tiny comparisons support no statistical
ranking. The curved heuristic table is missing and cannot be inferred from
the weak regime or from previous datasets.

## Fit diagnosis

Eight iAPF rows, including all four selected claim rows, have a zero floating-
point density loss with a positive normalized shape residual. In each weak
claim, three of six fitted time steps underflow and hit a bound; maximum
relative shape residual is 0.770307. Each curved claim has one underflow among
six fitted time steps, two boundary hits and maximum shape residual 0.917038.

The implementation preserves the amplitude factor in the profiled density
criterion. When that amplitude is tiny, the absolute objective and gradient
can vanish in floating point while the scaled residual remains large. The
absolute-gradient stopping rule can also be satisfied by very small genuine
gradients; satisfying that rule does not demonstrate a useful approximation.
The current adapter records underflow but accepts the fit as valid/converged.
This is a missing numerical guard. The finite fitted probability law and its
branchwise score can still be defined; the result cannot certify fit quality.

The broader mathematical issue remains the scale degeneracy documented in
`iapf-density-fit-audit.md`. Merely widening bounds cannot solve it. The next
repair rejects objective underflow without changing healthy coefficients,
values or gradients. Any subsequent normalized-shape or log-domain optimizer
must state its exact objective and relation to the published density criterion.

## Failures, repair and budget

Launch 1 used 32 charges. Eight weak rows completed, then four comparator
results failed because their study evidence class contradicted their mechanics
role. The numerical work was charged and the failed artifacts retained.
Launch 2 corrected that metadata and reused only revalidated completed rows.
The driver now checks the cumulative worst-case row/fit cost before numerics,
records measured successful work, and charges failed calls conservatively.

Launch 2 used the remaining 32 charges and stopped before starting any curved
comparator. The coordinator preserves the denied row as an interruption with
`numerical_work_started=false`; it is not a numerical attempt. Overall: two
launches, 24 numerical row attempts and 40 recursive fits, exactly 64 charges.
The phase is closed as `partial_budget_exhausted`, not complete.

External process-wall times are 31.74 and 44.01 seconds; driver timers report
31.5251 and 46.8431. Charge the larger measure per launch: 78.5831 of 900
seconds. Timed process CPU usage, including preflight and the three recovery
test commands, totals 95.07 seconds of the 5,400-second allocation. Small
post-run inspections are not separately timed. No timing ranking is made.

Seven CPU-only recovery tests pass. Two earlier test artifacts failed because
selection re-derivation intentionally uses its saved GPU settings; those
failures are retained and are not GPU-health evidence. The CPU test now checks
saved links/digests, while the actual trusted GPU retry re-derives selection.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not promote either selected candidate | Claim baseline comparison and uncertainty missing | Objective underflow in every selected claim; weak heuristic veto | Fit geometry versus finite-particle error | Add the underflow guard, then evaluate admissible controls with complete fresh-data baselines | No rejection of twisting or iAPF as a research direction |
| Close this allocation as partial | 20 unique rows preserved | 64-charge and two-launch limits exhausted | Curved heuristic performance unknown | Continue through the master's separately bounded repair phase | Whole master completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Selected fits underflow; promotion blocked. Weak descriptive heuristic screen also fails. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Calibration/validation errors and two-draw claim means above. |
| Default readiness | No change to defaults. |
| Next evidence needed | Valid fits, original-control claim baseline, curved heuristics, fresh partitions and independent replicated comparisons. |

The strongest alternative explanation is that the density-scale local solver
chooses nearly invisible Gaussian components rather than a useful twist; low
particle counts and adaptive-work differences may compound that problem.
Healthy fits with independent replicated score evidence would overturn the
candidate verdict. The weakest evidence is the tiny heldout sample and absent
claim baseline. Repairing underflow is necessary for a trustworthy calibration
study but does not itself promise better scores or close full-control tuning.
