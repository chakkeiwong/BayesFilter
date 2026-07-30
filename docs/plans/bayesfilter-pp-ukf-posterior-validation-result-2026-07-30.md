# PP-UKF preserved-sample posterior validation result

Date: 2026-07-30

Status: `TWO_EQUIVALENT_EIGHT_INCONCLUSIVE_ZERO_DISAGREEMENT`

Plan: `docs/plans/bayesfilter-pp-ukf-posterior-validation-plan-2026-07-30.md`

Terminal artifact:
`docs/plans/artifacts/bayesfilter-pp-ukf-posterior-validation-20260730/attempt-07/public_result.json`

Run manifest:
`docs/plans/artifacts/bayesfilter-pp-ukf-posterior-validation-20260730/attempt-07/run_manifest.json`

## Result

The preserved samples establish distributional compatibility with the archived
same-mathematical-target affine plain-HMC comparator for `L=12` and `L=17`.
The other eight kernels remain inconclusive under the declared uncertainty
screen. No candidate has evidence of material posterior disagreement, and all
candidate point estimates lie within their practical tolerances.

This was CPU-only TensorFlow post-processing. It launched no HMC and used no
NumPy numerical or decision path. The comparator binds mathematical target
signature `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`
and scope `PP-UKF-six-probit-initial-observation-first-v1`, despite its older
operational wrapper signature.

## Candidate decisions

| L | Retained draws/chain | HMC screen | Established checks | Inconclusive checks | Posterior decision |
| ---: | ---: | --- | ---: | ---: | --- |
| 5 | 1,000 | Passed | 26/30 | 4 | Equivalence inconclusive |
| 9 | 10,000 | Passed | 27/30 | 3 | Equivalence inconclusive |
| 12 | 6,500 | Passed | 30/30 | 0 | Equivalence established |
| 13 | 1,000 | Passed | 24/30 | 6 | Equivalence inconclusive |
| 14 | 2,000 | Passed | 25/30 | 5 | Equivalence inconclusive |
| 17 | 4,500 | Passed | 30/30 | 0 | Equivalence established |
| 18 | 1,000 | Passed | 19/30 | 11 | Equivalence inconclusive |
| 19 | 2,000 | Passed | 23/30 | 7 | Equivalence inconclusive |
| 24 | 1,500 | Passed | 21/30 | 9 | Equivalence inconclusive |
| 25 | 2,000 | Passed | 20/30 | 10 | Equivalence inconclusive |

Each candidate has six parameters and five primary checks per parameter: mean,
population SD, q05, q50, and q95. A check establishes equivalence only when its
chain-aware block-bootstrap 95% interval is wholly inside the predeclared
practical margin. A check supports material disagreement only when the interval
is wholly outside that margin. All non-established checks here are
`inconclusive`, not disagreements.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain `L=12,17` as posterior-compatible kernels; retain the other eight as HMC-valid but posterior-compatibility-inconclusive | `L=12,17` passed all 30 equivalence checks; eight did not establish all checks | No identity, archive, finiteness, HMC-health, or convergence veto; no material-disagreement result | Four-chain reference and candidate Monte Carlo uncertainty, especially tail quantiles; comparator is not an exact oracle | Use `L=12` or `L=17` as representative compatible kernels without ranking them; extend only an inconclusive kernel if there is a concrete need to qualify it | No exact posterior correctness, no rejection of the eight inconclusive kernels, no best candidate, no sampler superiority, no default or production readiness |

## Inference status

| Evidence class | Status | Interpretation |
| --- | --- | --- |
| Hard veto screen | Passed for all ten current archives and the reference identity/archive | All source hashes, shapes, coordinates, target bindings, finiteness checks, and repaired HMC screens passed |
| Statistically supported compatibility | `L=12` and `L=17` | All declared mean/SD/q05/q50/q95 intervals lie within practical equivalence margins |
| Material disagreement | None supported | No interval lies wholly outside a practical margin |
| Descriptive-only differences | Covariance/correlation Frobenius distances, point differences, draw counts, ESS, R-hat, and runtime | These do not rank viable kernels |
| Statistically supported ranking | None | The experiment did not predeclare or support a ranking between `L=12` and `L=17` |
| Default readiness | Not established | A same-approximate-target comparator is not an exact posterior oracle or robustness campaign |
| Next evidence needed | Only if another kernel must be qualified: additional retained draws under a new bounded continuation; for correctness: an independent target/reference audit | Do not rerun tuning or all ten kernels by default |

## Attempts and verification

Attempts 1--3 were harness-only failures before any scientific comparison:
scope-field lookup, standalone import path, and reference hash-ledger path
normalization. Attempt 4 produced the first complete screen. Attempt 5 added
candidate archive provenance. Attempt 6 added the required three-way
equivalence/disagreement/inconclusive interpretation, but its manifest predated
the harness commit. Attempt 7 reran the frozen calculation after commit
`b926a58c`; its scientific payload is identical to attempt 6 excluding only
timestamps and wall time. Attempt 7 is terminal. No attempt launched HMC.

Focused harness tests passed (`5 passed`), covering mathematical target/scope
failure, archive hash/shape verification, reference summary reproduction,
candidate uniqueness, deterministic TensorFlow bootstrap mechanics, and
three-way evidence classification. Python compilation and `git diff --check`
passed.

Terminal hashes:

- `public_result.json`: `8ffb3538bb2c5ad38bab37e617680177fe02893d35a53b91fc950edd44d4c04e`
- `run_manifest.json`: `486ef43982a400e717d2b441560d4f974da400fbef1d1bb7743d98c0d3852657`

## Post-run red team

The strongest alternative explanation is that both current kernels and the
affine comparator sample the same wrong approximate-filter implementation.
This result tests same-target sampling compatibility, not whether PP-UKF equals
the exact predator-prey posterior. A second alternative is that the eight
inconclusive candidates would establish equivalence with more retained draws;
their point estimates are all within tolerance and none supports material
disagreement. The weakest evidence is tail equivalence for short retained
archives, not the engineering integrity or HMC convergence checks.

The result would be overturned at its stated scope by a hash-valid replicated
comparison showing a declared interval wholly outside its practical margin.
Broader correctness would require an independent exact or otherwise justified
posterior reference, not more agreement among HMC kernels targeting the same
approximate implementation.
