# Phase 8 C2 strict-backend calibration result

Date: 2026-08-30  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-subplan-2026-08-30.md`  
Status: `PASS_C2_STRICT_CALIBRATION_WITHOUT_WHITENING_PROMOTION`

## Question and evidence contract

C2 asked whether the predeclared q=20 capacity/learning-rate hypotheses could
run on the parity-checked `tensorflow_eigh_strict` route at B=32 for 32 fresh
IID Gaussian reverse-KL updates, on two independent initialization roots. A
row passed only with finite/status-valid updates, exact checkpoint replay,
finite held-out diagnostics, and a passing self/cross/reference/declared
learned-map reliability screen. A row could be nominated only when its paired
held-out final-minus-start reverse-KL 95% interval had upper endpoint below
zero. No whitening, mode-discovery, sampler, posterior, or superiority claim
was allowed.

## Execution receipts

The B=8 backend prerequisite passed after a bounded harness repair:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-backend-parity-result-2026-08-30.md`.

The complete C2 manifest is:
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/screen/attempt-02-eight-rows/run_manifest.json`.

The first C2 launch was stopped as an invalid harness attempt after it exposed
an undefined checkpoint-scope seed binding. Its markers remain under
`.../c2-strict-calibration/screen/attempt-01-eight-rows/`; it is not candidate
evidence. The repaired attempt completed in `1074.309582018992` seconds with
zero failed rows.

## Row results

| Architecture | Root | Median update (s) | Start RKL | Final RKL | Paired mean delta | 95% interval | Final score-RMS by coordinate | Stress positive fraction |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| compact-high `(16,16), 1e-3` | 0 | 2.0430 | 188.9144 | 162.9730 | -25.9414 | [-31.4361, -20.4466] | [215.59, 1199.05, 191.97, 14.27] | 0.5781 |
| compact-high `(16,16), 1e-3` | 1 | 2.0308 | 192.3626 | 166.8779 | -25.4847 | [-29.8783, -21.0910] | [233.76, 1221.62, 187.72, 11.61] | 0.4688 |
| compact-low `(16,16), 5e-4` | 0 | 2.0561 | 207.3599 | 194.7498 | -12.6101 | [-15.5065, -9.7137] | [267.90, 1053.42, 216.90, 15.23] | 0.5938 |
| compact-low `(16,16), 5e-4` | 1 | 2.0453 | 147.0572 | 138.4366 | -8.6206 | [-10.4894, -6.7518] | [201.59, 715.68, 183.28, 7.53] | 0.5469 |
| wide-high `(32,32), 1e-3` | 0 | 2.0315 | 180.4535 | 141.3336 | -39.1199 | [-51.5141, -26.7258] | [166.42, 1122.09, 178.95, 12.31] | 0.5625 |
| wide-high `(32,32), 1e-3` | 1 | 2.0411 | 170.4100 | 144.8907 | -25.5192 | [-30.4788, -20.5597] | [199.96, 882.60, 173.04, 11.31] | 0.5781 |
| wide-low `(32,32), 5e-4` | 0 | 2.0349 | 236.5074 | 219.6438 | -16.8636 | [-20.7143, -13.0129] | [266.72, 1031.21, 238.37, 14.37] | 0.5469 |
| wide-low `(32,32), 5e-4` | 1 | 2.0326 | 168.3448 | 157.1007 | -11.2441 | [-13.9098, -8.5785] | [227.76, 931.10, 192.83, 12.52] | 0.5625 |

Every row passed finite/status, beta-0 and beta-0.5 preflight, checkpoint
replay, and the four architecture-level reliability screens. Each architecture
was viable on both roots under the frozen nomination rule. Reliability residuals
were at floating-point roundoff scale (self round trips at most about
`9e-16`, cross round trips at most about `4e-15`, condition proxies between
`1.07` and `1.27`, and all transformed scores finite). Per-row allocator peaks
were `1.701--1.702 GB`, below the 4-GiB cap.

## Interpretation

The primary C2 result is feasibility and candidate viability, not a ranking.
There is no predeclared uncertainty analysis supporting a difference between
the four architectures or learning rates; their update times and paired
intervals are descriptive/within-row nomination evidence. The lower-cost
architecture is the compact `(16,16)` family, but the two compact rates are
not statistically separated, so both remain calibration representatives.

The score residuals are large in every row, especially coordinate 2 (roughly
`716--1222` RMS). The training loss decreases, but the held-out pullback is
far from an IID Gaussian score field. This is exactly why the C2 pass cannot be
promoted to whitening or HMC readiness. The stress sign fractions near one half
also do not establish that both posterior modes were discovered; they only
show the bounded stress bank was not entirely one-sided.

## Decision table

| Decision | Primary criterion | Hard veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close C2 as a viable calibration screen | 8/8 rows valid; 4/4 reliability groups pass; at least one architecture viable on both roots | pass | 256-row held-out bank, 32 updates, and two roots are insufficient for global transport quality or ranking | Retain compact-high and compact-low as blind lineage representatives; run C3 branching/temperature-overlap diagnostics | No whitening, exhaustive mode discovery, posterior correctness, HMC convergence, superiority, or high-dimensional scaling |

## Inference status

| Evidence class | Result |
|---|---|
| Hard veto screen | Pass for C2 implementation/feasibility scope |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss decrease, paired intervals within each row, score residuals, sign fractions, and timing |
| Default readiness | Not ready; no architecture or learning rate is a repository default |
| Next evidence needed | Temperature-overlap and lineage complementarity under an explicit branch protocol, then untouched Phase 9 tuning and sequential HMC |

## Post-run red-team

The strongest alternative explanation is that all four maps remain close to the
Gaussian-prior affine chart and the loss reduction reflects local target
adaptation without learning the global posterior geometry. The large
pullback-score residuals support that concern. The C3 discriminating check is
whether independent positive-temperature branches produce complementary,
reproducible sign-region occupancy and temperature overlap without relying on
the same validation bank. A result that fails there rejects the current
lineage/branching candidate, not the proper bridge or reverse-KL mathematics.

