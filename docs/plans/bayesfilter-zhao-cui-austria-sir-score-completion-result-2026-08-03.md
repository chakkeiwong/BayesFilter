# Zhao-Cui Austria SIR Score Completion Result

Date: 2026-08-03T00:35:06+08:00

Status: `BLOCK_T3_PROPOSAL_QUALITY`

## Verdict

The Zhao-Cui-derived Austria `T=20` finite-score computation is **not
complete**. The FP64 manual score kernel is implemented and passed its focused
same-scalar, finite-difference, replay, additivity, GPU, and XLA checks. The
selected nine-persistent-guide proposal is nevertheless wrong for the declared
admission scope because three predeclared mixed points in the `0.03` box have
no guide satisfying the frozen proposal-quality gate at T3.

The T3 continuation veto fired exactly as planned. T5, T10, T20 tuning and the
untouched claim were not run. The gate was not weakened and the domain was not
shrunk after observing the failure.

This is rejection of the current analytic proposal design. It is not an
implementation failure of the manual score recursion, invalidation of the
target or data, evidence against the broader Zhao-Cui-derived direction, or an
HMC/posterior result.

## Claimed And Computed Quantities

| Item | Verdict |
|---|---|
| Claimed target | Value and manual total derivative of one frozen nine-branch importance-filter scalar through T20 |
| Quantity actually computed | The same finite scalar and manual score in focused tests and staged T1/T2/T3 proposal screens; no T20 claim scalar was admitted |
| Equality status | Tiny/T2 same-scalar diagnostic equality passed within the five-significant-digit rule; T20 was not checked |
| Source status | `extension_or_invention`; individual squared-TT/KR proposal operations are source-grounded, but the assembled Austria parameter-score route is not in the author Austria example |
| Remaining unproved | T20 proposal viability, untouched T20 score identity, physical-likelihood accuracy, HMC readiness, posterior correctness, default or production readiness |

## Execution Summary

All staged programs used one repository-issued prefix of parent T20 program
`936216cd0a8c4cab2b4551b6d44d99821a662046441c2cfb07200c2e23438fad`.
This repaired a skeptical-audit finding that independently shaped guide-major
random tensors did not guarantee literal cross-horizon prefixes.

| Stage | Status | Selected box | Worst best-guide ESS/N at 0.03 | Worst viable max weight at 0.03 | Minimum branch effective count at 0.03 |
|---|---:|---:|---:|---:|---:|
| FP64 GPU/XLA preflight | Pass | N/A | Bootstrap diagnostic only | Bootstrap diagnostic only | N/A |
| T1 persistent guides | Pass | 0.03 | 0.708192 | 0.006798 | 5.34870 |
| T2 persistent guides | Pass | 0.03 | 0.271639 | 0.022092 | 3.85695 |
| T3 persistent guides | Block | none | 0.045674 | no viable guide at three points | 2.60445 |
| T5/T10/T20 | Not run | N/A | N/A | N/A | N/A |

The proposal gate required all branches finite and, at every theta point, at
least one guide with `ESS/N >= 0.10` and maximum normalized particle weight
`<= 0.10`. All T3 branches and score outputs were finite. The failing points
were:

| Theta | Best ESS/N | Maximum weight of the best-ESS guide | Classification |
|---|---:|---:|---|
| `(0.015, -0.0075, 0.0225)` | 0.0960390 | 0.0516172 | ESS veto |
| `(-0.015, 0.0075, -0.0225)` | 0.0703704 | 0.0526825 | ESS veto |
| `(0.018, -0.021, -0.012)` | 0.0456736 | 0.102963 | ESS and maximum-weight veto |

The first failure is close to the ESS threshold but remains a failure. The
threshold was frozen before this run; proximity is not grounds to change it.

## Engineering Ledger

| Check | Result |
|---|---|
| No NumPy numerical path | Pass; no NumPy import in the claim-owned proposal or runners |
| No Python numerical loop in kernels/tuning | Pass; proposal compilation, time recursion, RK4, and theta sweep use TensorFlow batching and `tf.while_loop` |
| Cross-horizon prefix identity | Pass; exact T2-vs-T20-prefix tensor equality test |
| Focused CPU-hidden tests | `15 passed, 2 warnings in 39.12s` |
| GPU device | NVIDIA GeForce RTX 4080 SUPER |
| Runtime | Python 3.11.14, TensorFlow 2.19.1, TensorFlow Probability 0.25.0 |
| XLA graph | Pass; outer sweeps contain `StatelessWhile`, with no `PyFunc`, `EagerPyFunc`, or `MapDefun` |
| Precision | FP64; TF32 disabled |
| GPU memory policy | Fixed 6,144 MiB logical-device cap configured before logical GPU initialization |
| Peak allocator bytes | 100,834,816 for each staged persistent-guide run |

Host-side tensor materialization was used only for hashes, JSON reporting, and
post-graph pass/fail serialization. It did not feed the XLA numerical path or
select proposal settings.

## Numerical Ledger

The preflight passed T2 finite-difference steps `2.5e-4`, `1.25e-4`, and
`6.25e-5` under

```text
abs(actual-reference) <= 5e-6 + 5e-6*max(abs(actual),abs(reference)).
```

T20 bootstrap mechanics replay was exact and its score-additivity residual was
`3.55e-15`, but its minimum ESS was effectively `1/1008`. It was correctly
classified as a mechanics baseline, not proposal or T20 claim evidence.

The persistent-guide T3 block is hard proposal-quality evidence. The observed
ESS and weight differences among guides are descriptive only and do not rank
methods or establish statistical superiority.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Stop this campaign at T3 | Every calibration point must have a viable guide before later horizons | Fired at three T3 points | Whether a richer XLA-native proposal can cover the same box | Write and review a fresh bounded proposal-repair plan | T20 score completion, physical likelihood, HMC, posterior, default, or production readiness |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Current nine-guide candidate rejected at T3 |
| Viable candidates | This candidate is not viable for the declared box; no enhanced candidate was run |
| Statistically supported ranking | None; guide branches are proposal components, not replications |
| Descriptive-only differences | Per-guide ESS, maximum weights, branch effective counts, and combination concentrations |
| Default readiness | No |
| Next evidence needed | An XLA-native richer proposal that passes the unchanged T1/T2/T3 points before any later horizon or untouched claim |

## Negative-Result Classification

| Question | Answer |
|---|---|
| Harness invalid? | No. Prefix equality, graph audit, finiteness, identity, GPU, and allocator checks passed. |
| Implementation invalid? | No evidence of that. Manual score same-scalar tests passed. |
| Target/data/math invalid? | No. The veto concerns proposal concentration. |
| Current candidate failed? | Yes. The persistent 3x3 kappa/nu rank-one guide family lacks T3 coverage. |
| Research direction rejected? | No. A richer proposal is the declared repair trigger. |

## Post-Run Red Team

The strongest alternative explanation is Monte Carlo variation from the single
frozen seed, especially because one failed point has best `ESS/N=0.0960` near
the `0.10` threshold. That does not rescue the candidate: the claim is about
this exact frozen finite program, two other points fail more substantially, and
post-result seed selection would change the program and tune on observed
failure. A future plan may predeclare multiple proposal seeds or a richer
branch family, but it must preserve these failed points as holdout evidence.

The result would be overturned only by evidence of an artifact or compiler bug
that changes the recorded proposal measures, or by a fresh predeclared proposal
repair passing the unchanged gates. The weakest evidence is physical-likelihood
accuracy, which remains limited and was never the basis of this proposal veto.

## Artifacts

- Preflight: `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase2-fp64-gpu-xla-preflight-01/result.json`
- T1: `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase3-t1-persistent-guide-tuning-01/tuning.json`
- T2: `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase3-t2-persistent-guide-tuning-01/tuning.json`
- T3 blocker: `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase3-t3-persistent-guide-tuning-01/tuning.json`
- Terminal manifest: `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/terminal-block-t3-01/manifest.json`

No leaderboard row was written because the definition of done was not met.
