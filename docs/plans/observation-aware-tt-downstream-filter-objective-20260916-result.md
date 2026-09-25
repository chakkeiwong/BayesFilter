# A08: TT usefulness in corrected Zhao-Cui filtering

Date: 2026-09-16 (Hong Kong; numerical run timestamps are UTC).
Status: COMPLETE. Governing [amendment](observation-aware-tt-master-amendment-08-downstream-filter-objective-20260916.md).

The rerun supports continuing TT development. On the eleven complete d=4
sequences, the selected TT/SGQF proposal has normalized filtering-mean MSE
0.0023614 and mean particle ESS 317.53/512. The earlier scalar guided TT has
MSE 0.0066699 and ESS 199.52/512; the repaired pair TT has MSE 0.0026151 and
ESS 310.55/512. These are favorable observed differences, not a statistically
established ordering. One shared SGQF guide still fails, and the selected
route sometimes uses SGQF directly. Neither issue is hidden by these averages.

The subsequent [timing breakdown](artifacts/observation-tt-downstream-filter-objective-20260916-01/timing-summary.md)
extracts the existing timers for all four principal methods. In d=4, the
selected route spends 54.178 seconds on fitting/selection and an average
0.698 seconds per 20-observation particle run, plus shared guide construction.
The timing note separates observed method-block times from derived per-run
estimates and states the compilation, I/O and cache limitations.

## What was repaired and measured

The master now asks whether the TT supplies a useful conditional proposal for
the corrected filtering recursion. Losing a one-step fitting comparison with
SGQF is a representation diagnostic and repair trigger, not a reason to stop
measuring the filter or reject the TT direction. SGQF remains a guide and
comparator. A08 changes observability and interpretation; it does not claim a
new fitting-algorithm improvement.

The call chain is `run_observation_tt_independent_filtering.py::main` to
`run_observation_aware_tt_complete.py::particle_filter` and its XLA
`filter_kernel`. For proposal q, normalized incoming weights w, transition f
and observation density g, the actual computation is

\[
 u_t^{(i)}=w_{t-1}^{(i)}
 \frac{f(x_t^{(i)}\mid x_{t-1}^{(i)})g(y_t\mid x_t^{(i)})}
 {q_t(x_t^{(i)}\mid x_{t-1}^{(i)})},\quad
 w_t^{(i)}=\frac{u_t^{(i)}}{\sum_j u_t^{(j)}},\quad
 \operatorname{ESS}_t=\frac{1}{\sum_i(w_t^{(i)})^2}.
\]

At the first observation the prior replaces f and incoming weights are 1/N.
ESS is measured before resampling, which occurs at ESS < N/2. The reported
means and log evidence belong to this importance-corrected particle filter.
They are different from the raw retained-TT moments and Gram normalizer.
The latter have not been validated by this run. Fitting-row importance ESS
is a separate quantity and is not substituted for particle ESS.

There is no post-hoc numerical ESS acceptance threshold. ESS, maximum weight,
resampling and parent counts diagnose efficiency and occasional collapse.
Finite outputs, valid references and normalized/evaluable proposals are
necessary validity checks; comparison with simple proposals constrains
promotion, but a failed candidate does not end the repair program.

## Execution and reproducibility

Exact full command:

```sh
bash docs/plans/artifacts/observation-tt-downstream-filter-objective-20260916-01/run_campaign.sh attempt-full-01 full
```

- Full run: 2023.805 seconds, 2026-09-15 20:12:11–20:45:19 UTC.
- Checkout: `surrogate-hmc`, commit `509871fd28e21862250e8f094e8d466b004af7e3`,
  with the dirty source snapshot and dependency hashes recorded in the manifest.
  `source_unchanged=true` at completion.
- Environment: `/home/chakwong/anaconda3/envs/tftwogpu`, TensorFlow float64,
  GPU/XLA, RTX 5080, `CUDA_VISIBLE_DEVICES=1`, verified memory growth.
  TensorFlow peak allocator usage: 309331456 bytes.
- Scope: d=1,4; T=20; 12 sequences each; N=512; four particle replicates;
  degree/rank 3, four sweeps, 128 proximal steps, L1 grid {0,1e-5,1e-3}.
- The wrapper and manifest preserve exact seed formulas. A06 observations,
  fits and particle seeds are reused. Every aggregate MSE and bootstrap
  interval exactly reproduces A06. This is not additional independent evidence.
- `attempt-smoke-01` and `-02` failed at GPU device discovery using UUID
  selectors. `-03` passed on the RTX 4080 SUPER. `-04` passed on the RTX 5080
  used for the full run. Those failures are infrastructure failures, not TT
  failures; all logs are preserved.

The [full manifest](../benchmarks/artifacts/observation_tt_downstream_filter_objective_20260916/attempt-full-01/run_manifest.json),
[raw result](../benchmarks/artifacts/observation_tt_downstream_filter_objective_20260916/attempt-full-01/result.json),
[assembled report](artifacts/observation-tt-downstream-filter-objective-20260916-01/report.json),
and [complete tables](artifacts/observation-tt-downstream-filter-objective-20260916-01/tables.md)
preserve every comparator and observation regime. The raw method files retain
each particle-replicate/time record; the report checks aggregated ESS against
those records. The executed plan is preserved separately because its closeout
clarifies corrected-particle versus uncorrected-TT outputs.

## Downstream results

The following d=4 table uses the same eleven complete sequences for every
method. MSE is squared filtering-mean error divided by stationary coordinate
variance, averaged within sequence and then equally across sequences. ESS
and weight summaries pool equal-sized sequence/replicate/time records.

| Proposal | Normalized MSE | Mean ESS / 512 | Minimum ESS | Largest weight | Resampling fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transition | 0.0066377 | 175.78 | 2.70 | 0.5988 | 0.7989 |
| Stationary prior | 0.0212687 | 75.60 | 1.11 | 0.9489 | 0.9864 |
| SGQF marginal | 0.0060415 | 203.89 | 12.41 | 0.2710 | 0.8625 |
| SGQF joint | 0.0029703 | 311.47 | 3.16 | 0.5548 | 0.3023 |
| Earlier predictive TT | 0.0072739 | 192.13 | 2.69 | 0.6083 | 0.8705 |
| Earlier guided TT | 0.0066699 | 199.52 | 14.34 | 0.2186 | 0.8750 |
| Repaired pair TT | 0.0026151 | 310.55 | 38.77 | 0.1327 | 0.3000 |
| Selected TT/SGQF route | 0.0023614 | 317.53 | 44.39 | 0.1254 | 0.2750 |

The selected route's observed MSE is 64.6% lower than earlier guided TT and
9.7% lower than repaired pair TT. Its root normalized MSE is about 0.0486
stationary standard deviations. These percentages describe this completed
sequence set and have no TT-versus-TT uncertainty interval. In all three
predeclared coordinate-observation regimes its d=4 MSE is descriptively lower
than both those TT comparators. The complete regime table preserves the
actual differences; no d=4 population interval is issued because guide failure
selects the completed sequence set.

In d=1 (all twelve sequences), selected-route MSE is 0.0016702, compared with
0.0016075 for pair TT, 0.0016963 for guided TT and 0.0018482 for SGQF joint.
Its mean ESS is 375.82/512 (73.4%), minimum 55.98, maximum weight 0.1064 and
resampling fraction 0.075. Pair TT's mean ESS is 379.67/512. Thus the
initialization/selection policy does not show a uniform gain over pair TT.

| Selected-route diagnostic | d=1, 12 sequences | d=4, 11 complete sequences |
| --- | ---: | ---: |
| Mean ESS/N | 0.7340 | 0.6202 |
| Mean distinct parents after a step | 497.34 | 455.72 |
| Mean distinct parents when resampling fires | 316.60 | 307.33 |
| Minimum distinct parents | 250 | 200 |
| Mean log-evidence error vs reference (nats) | -0.01384 | -0.01068 |
| Largest absolute sequence-mean log-evidence error (nats) | 0.07673 | 0.13624 |
| Observed log-correction range | [-33.540, 1.781] | [-55.730, 2.380] |

Parent counts describe one resampling step, not full genealogical diversity.
Log-evidence errors are empirical differences, not a proof of unbiased log
evidence. Their d=4 reference also has Monte Carlo uncertainty.

The selected route chose generic TT / SGQF-start TT / exact SGQF on
145 / 59 / 36 d=1 time steps and 10 / 191 / 19 d=4 time steps. These totals
include time zero, where SGQF is fixed (12 and 11 cases respectively).
The route is predominantly TT, especially in d=4, but is not a TT-only
recursive fitting test. No separate new claim about TT-only warm-start
performance follows from this mixture policy.

## Validity, uncertainty and decision

All 24 reference screens pass. Across all methods there are 14880 stored
particle-time records, zero stored nonfinite numeric values and zero recorded
CDF bracket failures; the largest recorded CDF residual is 1.44e-14.
The selected route contributes 960 d=1 and 880 d=4 records. The shared guide
fails on d=4 sequence 8 with non-SPD signed-quadrature covariance, preventing
all six guide-dependent methods from running there. Both unguided methods
finish; their twelve-sequence results remain in the report.

Seven specified d=1 heuristic contrasts have negative simultaneous upper
limits under the predeclared sequence bootstrap. None establishes an overall
TT ranking or superiority over joint SGQF. The two observed d=1 losses remain
statistically unresolved: ordinary observations versus transition have delta
0.00022344, interval [-0.00018662, 0.00063349]; large observations versus
joint SGQF have delta 0.00027871, interval [-0.00050373, 0.00106115]. One
stationary-prior contrast also misses the declared precision bound. The
legacy `candidate_advances=false` and `PROMOTION_VETO` fields preserve that
strict heuristic screen; they do not mean that TT fitting is unusable or that
the TT repair program is rejected.

| Decision | Primary evidence | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue TT development | Useful observed corrected-filter errors and ESS; substantial d=4 differences from scalar TT | No completed-run numerical veto; guide failure remains; heuristic promotion screen not cleared | Selected policy mixes TT and SGQF; d=4 completion selection; no TT-arm interval | Design the next bounded guide-repair and TT-only warm-start comparison before new numerical work | General superiority, default readiness or raw-TT likelihood validity |

| Inference status | Result |
| --- | --- |
| Hard veto screen | One guide failure; no nonfinite or recorded bracket failure in completed arms |
| Statistically supported ranking | Seven specified d=1 heuristic contrasts under the approximate bootstrap; no overall ranking and no TT-versus-TT ranking |
| Descriptive-only differences | All d=4 comparisons, TT-variant comparisons, ESS/weight tails, evidence errors and runtime |
| Default readiness | Not established; fit discrepancy alone is not the rejection criterion |
| Next evidence needed | A prospectively specified guide robustness repair and TT-only recursive warm-start arm on fresh confirmation sequences, with paired TT-arm uncertainty |

Engineering checks: 14 focused CPU reference tests passed (8.67 seconds),
covering paired inference, conditional densities, mixed consumer wiring,
normalization and conditional inversion. GPU smoke and full run passed.
The numerical statistics are therefore checked separately from the limited
scientific interpretation. No new source-faithfulness, HMC or scalability
claim is made.

Post-run skeptical review: the strongest alternative explanation for favorable
d=4 differences is the combination of SGQF fallback, guide-success selection
and Monte Carlo variability. The most vulnerable claim would be extrapolating
these results to a TT-only recursion or higher dimensions. Fresh paired
TT-only comparisons including difficult guide cases could overturn the
interpretation. Those are the next research questions, not grounds for
discarding the completed TT results.

## Manuscript and program repair

The algorithm note already had the correct mathematical use of TT as a
conditional proposal with exact particle corrections. Its closing rejection
language was too broad. The revision now distinguishes promotion from
continuation, defines actual particle ESS, states the selection-policy and
seed-reuse limits, and adds these downstream results. The book chapter also
distinguishes corrected-particle evidence from its uncorrected retained-TT
normalizer. The baseline note is preserved under the A08 artifact root.

The [revised algorithm-note PDF](artifacts/observation-tt-downstream-filter-objective-20260916-01/latex/observation-aware-tt-a08.pdf)
builds to 56 pages. The new objective, equations and comparison table on pages
54–55 were inspected in the rendered PDF. The final build has no overfull
boxes, undefined-reference warnings or pending cross-reference rerun warning.
The larger book builds to 529 pages with no undefined-reference or rerun
warning; its 202 overfull-box warnings remain outside this narrow objective
repair. All 265 prior algorithm-note labels remain, with two additions, and
no counted mathematical environment decreases. These preservation checks do
not constitute a fresh audit of the entire manuscript. Build logs and
`verification.json` are preserved under the A08 artifact root.

The active checkpoint, master and campaign budget are updated at closeout.
The A08 active-work start was not recorded. The budget therefore charges the
verified full-run wall time and reserves the remaining elapsed-time upper
bound conservatively; it does not label possible idle time as measured work.
The budget's `remaining_seconds` is a conservative lower bound after that
reservation, with the pre-A08 ledger preserved beside these artifacts.
No additional experiment is implicitly selected by this result. The next
action is to write the next protocol around the two remaining questions;
new numerical controls and confirmation seeds must be fixed before execution.
