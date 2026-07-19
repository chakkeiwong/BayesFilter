# SSL-LSTM NeuTra Phase 4 Bounded Training Result

Date: 2026-07-14

Status: `ONE_STAGE_ABLATION_COMPLETE_NONLINEAR_PHASE_NOT_PASSED`

Correction, 2026-07-15: the dense candidate below was not the established
Rotemberg/SGU plain NeuTra procedure. It used one `tanh` stage, no mixing, no
fixed reference translation, 2,000 steps, batch 64, fixed `1e-3` learning
rate, and global clipping. It is retained as a negative ablation. The two
affine candidates remain viable controls, but neither they nor this ablation
complete the nonlinear phase or open the main Phase 5 handoff.

## Outcome

The authorized sequential ladder produced two independent viable frozen
affine controls. The single completed one-stage dense-IAF ablation is rejected:
it remained finite
and replayable but failed the prospective moderate-shell support and scale
saturation gates. This rejects that candidate, not NeuTra or the research
direction.

The shared charge was `3512.999542077072` seconds of the `3600`-second cap,
leaving `87.000457922928` seconds. Dense B and the low-learning-rate repair
pair were not launched because the observed dense-A charge was
`2388.789778603008` seconds and completion was impossible within the remaining
budget. The established three-stage procedure was not tested.

## Candidate Evidence

| Candidate | Decision | Heldout loss upper bound | Original radius | Moderate radius | Saturation | Roundtrip max | Charge (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affine A | `VIABLE_FROZEN_CANDIDATE` | -13.6926 | 1.3153 | 3.0539 | N/A | 8.88e-16 | 577.4181 |
| affine B | `VIABLE_FROZEN_CANDIDATE` | -13.2221 | 1.3578 | 3.0855 | N/A | 4.44e-16 | 546.7762 |
| dense A | `CANDIDATE_NOT_VIABLE` | -18.9384 | 1.8581 | 4.9682 | 0.50 | 1.78e-15 | 2388.7898 |

The frozen gates were inverse radius at most `4.30` for original-neighborhood
and moderate-shell probes, dense saturation fraction at most `0.05`, roundtrip
residual at most `1e-9`, and a paired heldout-loss one-sided 95% upper bound
below zero. Crossing an original-neighborhood or moderate-shell radius
threshold was a promotion veto; continuous radii below the threshold were not
used to rank viable candidates. Every completed candidate passed finite,
reload, exact-resume, heldout-loss, original-neighborhood, and roundtrip checks.
Dense A failed only `moderate_shell_missing_support` and
`dense_scale_saturation_above_cap`.

Far-tail and broad-prior radii were explanatory by prospective contract. They
were not silently promoted into posterior-support gates. Training-loss
differences are also explanatory and do not rank the candidates.

## Artifact And Run Manifest

| Role | Path | SHA-256 |
| --- | --- | --- |
| affine A receipt | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-4/affine-a/result.json` | `b6fb4c58876f94d8590c50966509a6b20f174e1d06a84305962640620dbd3524` |
| affine A frozen payload | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-4/affine-a/frozen-payload.json` | `985ecb2e363fe4d82d3ea948f5cbe19eb1c1c1e76ce849a648d76afda9f2da02` |
| affine B failed environment attempt | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-4/affine-b/failure.json` | `b2f434042183f1549c96e432db83511841909e39e0957d185d1d67ed1e231747` |
| affine B receipt | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-4/affine-b-r2/result.json` | `5000cfc2b6131c0dfde0d0af686f72b7a42e99004563b2f0cf16f488c49eb56a` |
| affine B frozen payload | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-4/affine-b-r2/frozen-payload.json` | `eaa89a5fa51eb246122503f599d67fbeb523036c5478ad20eba12c63528e445a` |
| dense A receipt | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-4/dense-a/result.json` | `df7ccf787b8f0be57911c48a30b2b6061341a92cb3866ddb453b6aba07c24dbc` |
| dense A frozen payload | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-4/dense-a/frozen-payload.json` | `75a52387c1b7a77abc78ff20f871eea307a6cb741fbb95dd95a9c75e2564c3ff` |

All successful runs used commit
`3d353253dc93a102722e00cbca8803a1b3fce7fa`, conda environment `tfgpu`,
Python `3.13.13`, TensorFlow `2.20.0`, `float64`, XLA JIT, TF32 enabled,
soft placement disabled, and the owner-designated managed-session trusted GPU
basis. Training seeds were `2101` and `2102`; heldout seeds were `2201` and
`2202`; the prior-probe role was `3301`.

The first affine-B launch did not initialize CUDA. Its immutable failure
receipt is environment-invalid evidence and makes no candidate claim. Its
generic trust label was overly broad; the runner was repaired after the run so
future failure receipts state that GPU provenance was not established. The
attempt remains in the budget lineage with its recorded `0.0154707670444623`
second charge.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain affine A and B as controls | Passed: two independent frozen controls | No control hard or promotion veto | Controls do not answer the nonlinear-geometry question | Preserve for later matched control use; pause their Phase 5 main handoff | Nonlinear phase pass, posterior correctness, HMC readiness, superiority, or default readiness |
| Reject one-stage dense A ablation | Failed support and saturation gates | Promotion veto only; no hard veto | Established three-stage procedure and a second faithful seed were not observed | Preserve as ablation evidence; execute the source-anchored procedure-parity repair | Failure of NeuTra or proof that affine is scientifically better |
| Stop Phase 4 ladder | Resource stop passed | No budget overrun | Only one dense seed completed | Do not launch truncated dense B or repair | Exhaustive topology comparison |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Affine A/B passed; dense A had no hard veto; failed CUDA launch is environment-invalid only |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss, runtime, within-gate original/moderate radii, all far-tail/prior radii, tail behavior, and continuous A/B differences; crossing the frozen original/moderate radius threshold remains a promotion veto |
| Default readiness | Not established |
| Next evidence needed | Direct `dsge_hmc` procedure parity, mutation rejection, and trusted GPU/XLA timing before a new two-seed nonlinear budget |

## Post-Run Red Team And Handoff

The strongest alternative explanation for the affine pass is that a broad
linear transport satisfies the finite fixed probe bank while still leaving
important nonlinear posterior geometry unresolved. Phase 5 can reject
change-of-variable or mechanics errors, but only later independently admitted
HMC and predictive replication can address exploration. Conversely, dense A's
saturation could reflect the bounded one-stage parameterization or learning
rate rather than evidence against nonlinear transport.

The Phase 5 affine-only main handoff is paused. Affine payloads remain controls
and dense A remains an ablation artifact. The 2026-07-15 procedure-parity repair
must pass and receive a complete two-seed material budget before the nonlinear
lane can return to exact transformed-target preflight.
