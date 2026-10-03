# M15 result: whole-fit SBC and actual stopping

The bounded campaign completed all planned datasets and stopped/fixed pairs.
No target, candidate-inventory or saved-evidence defect was found. One CPU
worker timed out; the unchanged dataset, source and checkpoint completed in
62.96 additional seconds in a preserved continuation directory. Both attempts
are charged, and the dataset is counted once.

## Execution and interpretation

| Fresh SBC group | Datasets / planned | Independent complete fits | Rank result |
| --- | ---: | ---: | --- |
| CPU normal conjugate | 32 / 32 | 96 | No discrepancy detected |
| CPU beta-binomial | 32 / 32 | 96 | No discrepancy detected |
| CPU LGSSM location | 8 / 8 | 24 | No discrepancy detected |
| GPU normal conjugate | 4 / 4 | 12 | No discrepancy detected |
| GPU beta-binomial | 6 / 6 | 18 | No discrepancy detected |

Each dataset supplied three separately tuned fits. The rank output was always
the first chain's last retained draw, independent of posterior pass status.
All 246 fresh fits happened to pass their declared posterior assessments;
this was not a condition for including their ranks. The tests use parameter,
bounded-radius and data-log-likelihood quantities with the declared
multiplicity. The separate three CPU pilot datasets are excluded from this
table. CPU and GPU experiments are not pooled.

Nonrejection does not establish calibration. Three rank draws give only four
categories, and eight or fewer datasets provide especially weak evidence.
M16 therefore tests sensitivity to explicit defects and larger rank designs.

| Fresh CPU stopping group | Fits / planned | Posterior checks passed | Retained cap |
| --- | ---: | ---: | ---: |
| Normal, dispersed | 4 / 4 | 3 | 1 |
| Normal, remote | 4 / 4 | 4 | 0 |
| Beta-binomial, dispersed | 4 / 4 | 1 | 3 |
| Beta-binomial, remote | 4 / 4 | 0 | 4 |
| LGSSM, dispersed | 4 / 4 | 4 | 0 |
| LGSSM, remote | 4 / 4 | 4 | 0 |

All 24 fixed comparators are available. Retained stopping counts ranged from
2500 to 10,000 per chain, exercising later checks and terminal precision caps.
All these fixtures completed warmup at the 2000-transition minimum. This does
not calibrate a shorter burn-in or establish sensitivity to slow warmup;
M17's difficult geometry is the next discriminating experiment.

The stopped arms covered 45 of 48 reported mean/median targets and the fixed
arms 47 of 48. These totals mix dependent quantities and heterogeneous groups,
so they are descriptive only. The saved group/quantity tables retain exact
binomial intervals: with four datasets, even four covered intervals give a
95% lower limit of only .398. All paired mean-absolute-error intervals include
zero. There is no supported method ranking or nominal stopping-time coverage
claim. Eight precision caps are posterior outcomes, never tuning rejection.
The separate GPU stopping pilots stopped at 4000/10,000/7500 for
normal/beta/LGSSM; the beta precision request remained unmet.

## Integrity, execution provenance and budget

The numerical source was immutable M14 `source-schedule-r1`, identity
`9a8d4da7c5e4706477433a379cacc049445e6233ba6e527d93fdfe93cf8f8b37`.
The terminal audit checked 282 candidate inventories, 43,599 numerical
verification receipts and 20,908 tensor checksums, with no invalid artifact
and no outstanding worker. Verified siblings remain available and unassessed
where the predeclared member rule funded one identity.

Exact commands, source/environment/device/data/seed manifests, attempts and
wall time are in the four run indices and per-job manifests under
`artifacts/hmc-repair-master-2026-09-16/m15-r1/`. The CPU continuation index
points to unchanged completed results in the original root and to the
resumed failed job in `fresh-cpu-continuation-r3`. The original index and
timeout remain preserved. GPU jobs used trusted TF/TFP/XLA with memory
growth verified before logical-device initialization. CPU reference workers
intentionally hid GPUs and used the declared non-XLA exception.

Measured CPU worker cost was 53,834.85 seconds; a conservative 900-second
analysis/accounting charge makes 54,734.85. GPU cost was 18,375.62 seconds.
The terminal ledger leaves 91,734.13026699313 CPU and
30,250.616178131837 GPU worker seconds. M16--M18 preparation checks remain
separately charged to their own phases.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close bounded M15 execution | Complete fit/pair denominators and intact inventories | No shared validity veto | Limited rank/coverage power | Run refreshed M16 | General calibration |
| Preserve precision caps | Actual stopping outcomes saved | Eight CPU posterior failures | Precision at larger counts | Report caps; test difficult geometry in M17 | Tuning failure |
| Continue repair program | Budget and source integrity checked | No continuation veto | Unmeasured diagnostic sensitivity | M16 sequential and SBC power | Default readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No corruption; one resource failure repaired on unchanged streams |
| Statistically supported ranking | None |
| Descriptive-only differences | Coverage totals, errors, stopping counts and runtime |
| Default-readiness | Not established |
| Next evidence needed | Controlled-defect sensitivity, difficult geometry and matched references |

Post-run skeptical review: successful simple conjugate fits may coexist with
poor sensitivity to small defects or unvisited modes. Four-category SBC and
four-dataset stopping groups are the weakest evidence. A sufficiently powered
injected defect that is not detected would limit interpretation of these
nonrejections; a failed independent reference or corrupted receipt would
invalidate the affected evidence. None of the observed caps invalidates the
harness, target or research direction. The next planned phases directly
address the remaining questions.

Machine-readable results: `terminal-summary.json` and
`reconciliation-terminal.json` under the M15 root. The refreshed M16 design
must preserve these limits and use the measured costs before launch.
