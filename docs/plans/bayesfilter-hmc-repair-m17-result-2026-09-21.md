# M17 difficult geometry, routes and matched references

The bounded matrix is complete. All 21 ordinary/prepared/transport fits and
four matched-reference fits finished; every candidate inventory, numerical
receipt and tensor checksum passed the terminal audit. This establishes the
tested mechanisms and preserves difficult posterior outcomes. It does not
establish reliable convergence over arbitrary targets.

Design: `bayesfilter-hmc-repair-m17-design-2026-09-21.md`. Full results and
commands are under `artifacts/hmc-repair-master-2026-09-16/m17-r1/`, with
`terminal-summary.json` and `reconciliation-terminal.json` as entry points.
All numerical jobs use frozen M16 source identity
`b5d2d662171ed61dab32a681693400abc1285ec5c13981a6118d1ce33624b54a`.
CPU references deliberately hide GPUs; GPU jobs use trusted TF/TFP/XLA and
verified memory growth. A first verified identity was selected before posterior
sampling; its siblings remain retained and unassessed.

## Model matrix

Entries give verified members, discarded warmup / retained transitions per
chain, and the selected member's declared posterior-check result.

| Model / route | CPU reference | GPU/XLA |
| --- | --- | --- |
| Rotated Gaussian, ordinary | 21; 2000 / 10000; precision cap | 12; 2000 / 10000; precision cap |
| Centered funnel, ordinary | 0; no verified member | 0; no verified member |
| Noncentered funnel, ordinary | 27; 2000 / 10000; precision cap | 33; 2000 / 10000; precision cap |
| Cauchy, quantiles only | 28; 2000 / 2000; passed | 31; 2000 / 2500; passed |
| Mixture, single-mode starts | 11; 10000 / 0; warmup cap | 12; 10000 / 0; warmup cap |
| Mixture, dispersed modes | 11; 10000 / 0; warmup cap | 11; 10000 / 0; warmup cap |
| Affine Gaussian | 21; 2000 / 1500; passed | 21; 2000 / 1000; passed |
| Affine banana | 5; 2000 / 1000; passed | 2; 2000 / 1000; passed |
| Fixed dense-IAF banana | 2; 2000 / 1000; passed | 2; 2000 / 1000; passed |
| Fixed dense-IAF Dirichlet | 21; 2000 / 1000; passed | 15; 2000 / 1000; passed |
| Student-t, df=5 | 24; 2000 / 1500; passed | Not planned |

The 310 verified candidates remain available. Fifteen selected members produce
retained draws; all fifteen meet the analytic assessor's descriptive tolerance,
but four fail posterior precision at the declared retained cap. Eleven pass
the complete declared posterior checks. Four mixtures stop without retained
draws and two centered funnels return no verified member. These failures remain
in the denominator. Known mode probability participates in the mixture checks;
no result establishes discovery of unknown modes. Cauchy means and variances
are not assessed because they do not exist. The nonlinear maps have fixed
weights and are not learned-transport training evidence.

## Conditional fields and pinned references

All six original position-field attempts executed pilots only because their
zero-repair fixture prevented new epsilon proposals. Preserve these records as
pilot mechanics, not verification coverage. The separately reviewed r2 repair
uses three same-L repairs and reaches pilot, measurement and fresh verification
on all six cells. CPU Gaussian/beta-binomial/LGSSM/banana retain 10/8/7/4
conditional candidates; GPU Gaussian/beta-binomial retain 10/9. Every resumed
prefix is unchanged, and exact-score retained authority is denied. Four
per-model regression tests additionally check independent stage seeds and full
inventories; the seven earlier route tests cover nonlinear replay and field
authority. This is conditional mechanics of the declared .8-score proposal
field, not exact-score or posterior promotion.

| Pinned posteriordb case / device | Verified | Warmup / retained | Mean comparisons | Full assessment |
| --- | ---: | ---: | --- | --- |
| Regression / CPU | 25 | 2000 / 1000 | 6/6 pass | Passed |
| Regression / GPU | 14 | 2000 / 1000 | 6/6 pass | Passed |
| Eight-schools / CPU | 17 | 2000 / 10000 | 10/10 pass | Posterior precision cap |
| Eight-schools / GPU | 9 | 2000 / 3500 | 10/10 pass | Passed |

Comparisons use ten thinned Stan chains at pinned commit
`5545a1dd07ae297c36edecbcd82aa49097b4c385`, matched laws/data/Jacobians,
combined lugsail uncertainty and the predeclared .25-reference-SD margin.
Reference samples never enter tuning. Passing those mean comparisons cannot
override the CPU eight-schools precision failure. No device or coordinate
ranking is supported by one fit per cell. The exact original MacroFinance
target/data/coordinate/reference bundle remains unavailable; these models do
not substitute for it.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete M17 engineering matrix | Public bindings, replay, exclusion and conditional authority checked | No invalid evidence | Single-fit scope | Proceed to M18 parity and integration audit | Universal robustness |
| Preserve posterior failures | Empty, warmup-cap and precision-cap outcomes reported | Those cells cannot support posterior promotion | Geometry and finite budget | Future target-specific repair with fresh design | Failure of HMC as a research direction |
| Retain matched-reference evidence | All mean comparisons pass; 3/4 full screens pass | CPU eight-schools precision insufficient | Finite references and limited replication | Report case-specific evidence | Exact equality or cross-model transfer |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No corrupted numerical evidence; no posterior output in six matrix cells |
| Statistically supported ranking | None |
| Descriptive-only differences | Candidate counts, stopping counts, analytic errors and runtime |
| Default readiness | No tuning, metric, stopping or transport default promoted |
| Next evidence needed | Difficult-geometry repair, stopping coverage, subtle whole-fit defect power and exact consumer reference |

The audit checks 3028 numerical receipts and 3006 tensor checksums, all candidate
inventories, frozen source and pinned inputs. It charges 4024.152771289926 CPU
seconds including 900 inspection/accounting, and 4893.229375969851 GPU seconds.
Remaining authorization is 82393.40737798327 CPU and 18362.589490781014 GPU
worker seconds. No worker remains active in M17.

Terminal skeptical review: short-window tuning compatibility is not posterior
adequacy; all candidate and posterior decisions remain separate. The strongest
alternative explanation for favorable analytic checks is their loose
development tolerance and limited replication. Independent reference uncertainty
is retained, yet this is not a coverage experiment. Difficult-model failures
reject these bounded configurations; they do not invalidate the inspected
targets, harness or next engineering phase. Refresh M18 to finish exact hash
parity, current-source regression/coverage reconciliation and the official book.
