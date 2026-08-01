# P4 Result: Predator-Prey NeuTra Cells

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `P4_COMPLETE_TWO_MEAN_LEVEL_NEUTRA_CONFIRMATIONS_ONE_SOURCE_ROUTE_BLOCK_CONTINUE_P5`

## Outcome

P4 has three terminal cell states:

| Cell | Terminal state | Binding evidence |
| --- | --- | --- |
| `PP-UKF` | `NEUTRA_CONFIRMED` | target `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30`; repaired R4 result SHA-256 `d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf` |
| `PP-SGQF` | `NEUTRA_CONFIRMED` | target `8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad`; repaired R4 result SHA-256 `a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4` |
| `PP-ZC` | `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH` | the available generic all-axes retained-grid route is an `extension_or_invention` and production-ineligible; no target, HMC, or training was run |

For both admitted cells, the supported claim is only:

> Converged, health-valid, target-bound NeuTra HMC agrees with its same-target
> plain-HMC comparator on the six physical posterior means for this one T=20
> synthetic fixture.

`NEUTRA_CONFIRMED` does not mean full-distribution equivalence. P4 does not
establish covariance, tail, mode, or quantile equivalence; filter exactness;
cross-fixture calibration or robustness; SGQF-versus-UKF superiority; or
production/default readiness.

## Evidence Ladder

| Rung | `PP-UKF` | `PP-SGQF` |
| --- | --- | --- |
| Posterior identity | PF-backed admission result SHA-256 `3771961abd01d3d75a964b5568706f706a56e71aa19f4e8e4a87e1a56b43c8c4` | level-2 SGQF admission result SHA-256 `6eea9ab2f4cf0e5a23262ed450d8b85add27289ab4aeef3a9661b95d46c358c4` |
| Same-target comparator | affine plain-HMC result SHA-256 `4c7e001b181033f4191acf5a6dd841c2dc507c4b25c015ce69817976eec345d5` | Laplace geometry SHA-256 `b54343fdee59c3f86ffb8f8ac69ba0ea31b7a0c780a4f2eb290374df060cabc3`; plain-HMC result SHA-256 `015348e162d35cb062be274eb4b420ee881eb364473b5b7ce5acfdca7c0192ec` |
| Fresh 5,000-step GPU/XLA training | `wide_lr5e3`; result SHA-256 `1650d256577f91d54e6c351545e9a7ef0cb208844dc859f19eecc3b496af27c9`; transport semantic hash `18546c2b30a5e2236e001293f9bbfc71babed47f5592d6821cabe0972990beec` | `wide_lr5e3`; result SHA-256 `de5f7cc35f606fe6d07177d1059d24acc1187e80b4bda42963f9e2823bf64bd4`; transport semantic hash `603a07c420579788e3981aa44dd67892902dc8c32da6ddf7c171918300da6811` |
| Fresh disjoint tuning verifier | step `0.20`; 1,000 burn-in plus 1,000 excluded draws per chain; modern R-hat `1.0054056853`; result root `attempt-03` | step `0.20`; 1,000 burn-in plus 1,000 excluded draws per chain; modern R-hat `1.0013382279`; result root `attempt-02` |
| Fresh post-verifier R4 NeuTra HMC | step `0.20`, 10 leapfrog steps; 2,000 warm-up and 4,000 retained draws per chain | step `0.20`, 10 leapfrog steps; 2,000 warm-up and 4,000 retained draws per chain |

Warm-up was archived separately and excluded from posterior summaries. All
modern R-hat values are the maximum of rank-normalized split and folded
rank-normalized split R-hat.

| Terminal diagnostic | `PP-UKF` | `PP-SGQF` |
| --- | ---: | ---: |
| Maximum modern R-hat | `1.0008110775` | `1.0003275699` |
| Minimum bulk ESS | `27,623.60` | `26,978.49` |
| Minimum tail ESS | `13,394.13` | `12,974.65` |
| Health/status vetoes | none | none |
| Bonferroni six-mean agreement | passed | passed |
| Widest simultaneous upper bound / margin | `0.325` (`a`) | `0.347` (`a`) |
| Repaired R4 wall time | `6,348.86` s | `3,447.57` s |

The 0.10-comparator-SD practical margins and simultaneous rule were frozen
before R4 sampling. The intervals use posterior SD divided by the square root
of the applicable split-chain cross-chain mean ESS. Physical SD, quantile, and
correlation tables in the result artifacts are explanatory only.

## Attempt And Repair Record

| Failure | Classification | Repair and effect |
| --- | --- | --- |
| UKF PF attempt 01 exceeded the 20-minute ceiling | `INFRASTRUCTURE_PF_ORCHESTRATION_TIMEOUT` | replaced pathological categorical sampling with inverse-CDF multinomial sampling; target, seeds, particles, gates, and hardware class unchanged |
| UKF PF attempt 02 passed computation but failed callable inspection | `INFRASTRUCTURE_RECOMPOSITION_CALLABLE_BINDING` | passed the inspectable bound `__call__`; no scientific output reconstructed from the failed attempt |
| UKF identity-mass plain HMC failed warm-up convergence | sampler geometry failure | target-bound affine mass repair; failed warm-up used only for tuning, never inference |
| UKF R4 attempt 01 failed before sampling | `HARNESS_ARTIFACT_REPRESENTATION_MISMATCH` | normalized optional `sha256:` prefixes before equality; semantic hashes were unchanged; focused regression passed |
| P7 attempt 01 found that old P4 confirmations lacked disjoint tuning admission | `EVIDENCE_BLOCKED_TUNING_ADMISSION` | reopened the earliest invalid R4 rung; reused hash-verified short probes only for ordering, then ran fresh disjoint verifiers, warm-up, and retained sampling with new seeds and roots; both cells passed |

The R4 prefix mismatch was representation-only: the training result stored a
bare digest while the manifest stored `sha256:<digest>`. The later tuning-
admission defect was substantive: old short probes could order candidates but
could not admit a fixed kernel. The repair did not relabel old samples. It used
fresh verifier, warm-up, and retained seed domains and immutable attempt roots.

## Budget And Integrity

The two repaired R4 confirmations added `9,796.44` GPU wall-seconds, or
`2.7212` GPU wall-hours, within the existing phase and per-cell ceilings.
Earlier completed P4 manifests recorded `22,868.70` GPU wall-seconds. The total
is still a lower bound because the timed-out PF attempt consumed
more than 20 minutes and the interrupted identity-mass comparator and two
pre-result infrastructure failures have no trustworthy wall-time manifests.
Even after adding those known omissions, P4 remains far below its 120-GPU-hour
phase ceiling and the admitted cells remain below their 40-GPU-hour per-cell
ceilings. No budget veto fired.

The historical phase-close attempt 01 remains preserved. A refreshed phase-
close package is generated after the repaired P7 audit and binds the active
result, terminal ledger, budget accounting, and repaired claim-bearing hashes.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close P4 and continue to P5 | both admitted cells passed convergence, ESS, health, and simultaneous six-mean agreement | clear for `PP-UKF` and `PP-SGQF`; source-route veto remains for `PP-ZC` | one fixture, mean-level comparison, approximate-filter targets | execute reviewed structural target design before any P5 HMC | full-distribution agreement, filter exactness, cross-filter ranking, robustness, calibration, readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Clear for the two admitted target-bound NeuTra runs; source-route veto supported for `PP-ZC`. |
| Statistically supported ranking | None. P4 did not test SGQF versus UKF superiority or a transport-family ranking. |
| Descriptive-only differences | Acceptance, runtime, losses, filter-to-filter summaries, quantile gaps, SD gaps, and correlations. |
| Default-readiness | Not established. Confirmation is fixture-specific and mean-level. |
| Next evidence needed | A separately admitted structural target and negative-control semantics in P5; broader predator-prey claims would require new fixtures and distribution-sensitive criteria. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | typed target and transport bindings, GPU/XLA/memory-growth manifests, frozen/trainable parity, recursive hashes, and focused regressions passed; the hash-prefix defect was repaired and recorded |
| Numerical/sampler validity | fresh disjoint verifier admission, finite status and energies, no declared divergences, moved chains, modern R-hat and bulk/tail ESS gates passed with separate tuning, warm-up, and retained archives |
| Scientific interpretation | supports only simultaneous agreement of six same-target physical posterior means on the one declared fixture; no broader distributional or filter claim |

## Post-Run Red Team And Drift Audit

The strongest alternative explanation is that both comparators and transports
are unusually well matched to this one synthetic trajectory and can agree on
means while differing materially in tails, covariance, or modes. The result
would be overturned at its stated scope by a repeated same-target comparison
whose simultaneous interval exceeds the frozen mean margin; broader claims
would require distribution-sensitive, multi-fixture evidence not collected
here. The weakest evidence is therefore generalization beyond the fixture and
six means, not the recorded convergence or target binding.

Execution review found and repaired five material drifts before close:

1. The runbook's stale phrase `P0-frozen` equivalence rule was false because P0
   left agreement margins unresolved; the R4 rule was frozen prospectively.
2. Mean MCSE prose incorrectly used `SD/ESS`; it now uses `SD/sqrt(ESS)`.
3. `NEUTRA_CONFIRMED` wording was too broad; the claim is narrowed to six
   physical posterior means.
4. Retained sampling originally stopped on convergence alone; it now requires
   both convergence and resolved simultaneous agreement.
5. Training-state digests used inconsistent optional prefixes; equality now
   normalizes representation without weakening semantic hash checks.
6. P7 found that the original R4 kernel nomination was not disjoint tuning
   admission; the active results now bind fresh verifier and downstream draws.

No remaining drift invalidates the P4 artifacts. P5 may begin only with its
separate structural target-design gate; P4 evidence cannot issue or imply a
P5 target signature.
