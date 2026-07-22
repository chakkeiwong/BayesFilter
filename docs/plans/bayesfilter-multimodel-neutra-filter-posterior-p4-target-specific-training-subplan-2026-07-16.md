# P4 Subplan: Predator-Prey Target-Specific NeuTra Training

Date: 2026-07-16

Status: `READY_FOR_SCREEN_EXECUTION`

## Objective And Entry Conditions

Independently screen and train the only available learned transport family,
plain dense IAF, for the two comparator-admitted predator-prey cells:

- `PP-UKF`, typed signature
  `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30`;
- `PP-SGQF`, typed signature
  `8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad`.

`PP-ZC` remains `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH` and cannot train. The
enhanced learned family is unavailable and is not executed or counted as a
failed candidate.

Each cell has a separate target identity, affine base, recipes, seeds,
heldout evidence, selection, final training, frozen transport, and result.

## Research Intent And Evidence Contract

| Field | Frozen contract |
| --- | --- |
| Question | Which capacity/learning-rate recipe gives an engineering-valid plain dense IAF worth fresh 5,000-step training for each admitted target? |
| Baseline | Fixed target-specific affine transport with an initially identity residual IAF stack |
| Candidate mechanism | Residual dense autoregressive IAF capacity and Adam learning-rate scale |
| Screen promotion | Completed 500 GPU/XLA steps; finite loss/gradient/logdet; all target status valid; frozen reload; common-heldout finite/status valid |
| Selection | Lowest common-heldout mean reverse-KL objective; any recipe within two paired MCSE is statistically indistinguishable, then choose lower parameter count, lower learning rate, declared order |
| Final training | One fresh 5,000-step run using the selected recipe and a seed excluded from screen/heldout/HMC |
| Final engineering pass | Complete frozen artifact; finite diagnostics; all target status valid; frozen reload and trainable/frozen value-score parity; common-heldout valid |
| Promotion vetoes | Target/geometry/hash drift, CPU serious training, non-XLA execution, nonfinite objective/gradient, invalid target status, artifact/parity failure, or seed overlap |
| Explanatory only | Training loss, heldout reverse KL, force norms, runtime, comparator posterior summaries |
| Not concluded | HMC convergence under the transport, NeuTra/plain-HMC agreement, filter exactness, superiority, calibration, robustness, or readiness |

Loss and heldout reverse KL nominate only. Only the later R4 same-target NeuTra
HMC confirmation can establish `NEUTRA_CONFIRMED`.

## Target-Specific Affine Bases

- `PP-UKF`: center and factor from the admitted target-bound empirical affine
  mass artifact under
  `phase-p4/PP-UKF/plain-hmc-affine/attempt-01-20260715T152500Z`. This is a
  target-specific tuned baseline; its failed source warm-up remains tuning-only.
- `PP-SGQF`: center and factor from the admitted SGQF Laplace geometry under
  `phase-p4/PP-SGQF/laplace-geometry/attempt-01-20260715T165000Z`.

No cross-filter affine base, sample archive, frozen transport, training state,
or heldout output may be reused.

## Frozen Recipe Screen

All recipes use three residual IAF stages, ELU, `s_max=1`, initialization scale
`0.02`, batch size `128`, constant learning rate, gradient clip norm `10`,
manual Adam `(0.9,0.999,1e-8)`, and one compiled multi-step `tf.while_loop`.

| Recipe | Hidden layers | Learning rate | Role |
| --- | --- | --- | --- |
| `source_width_lr1e3` | `(18,18)` | `1e-3` | lower-rate baseline |
| `source_width_lr5e3` | `(18,18)` | `5e-3` | inherited LGSSM rate hypothesis |
| `wide_lr1e3` | `(36,36)` | `1e-3` | capacity effect at lower rate |
| `wide_lr5e3` | `(36,36)` | `5e-3` | capacity/rate interaction |

Every screen job runs 500 steps from fresh initialization. Eight common
heldout batches of 128 stateless standard-normal base draws are evaluated in
one GPU/XLA graph per candidate. Screen weights are never resumed into the
final run.

## Seed Ledger

All roots are disjoint from comparator/HMC seeds.

- UKF screen recipes: `(20260716,10001)` through `(20260716,10004)`.
- UKF heldout root: `(20260716,10100)`, generating one `[8,128,6]` tensor.
- UKF final training: `(20260716,10201)`.
- SGQF screen recipes: `(20260716,11001)` through `(20260716,11004)`.
- SGQF heldout root: `(20260716,11100)`, generating one `[8,128,6]` tensor.
- SGQF final training: `(20260716,11201)`.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| three IAF stages | repository plain-family source anchor | insufficient or excessive depth | factorial width/rate screen and downstream R4 | baseline hypothesis |
| widths 18/36 | prior LGSSM protocols scaled above 6D target | no capacity discrimination or overfit | paired common-heldout objective | hypotheses, not promoted defaults |
| rates `1e-3`/`5e-3` | prior protocol bracket | instability or slow learning | 500-step finite/status/loss traces | hypotheses |
| 500 screen steps | repository target-specific protocol | short screen misranks long training | nomination-only language and fresh 5,000-step run | budgeted screen |
| batch 128 | existing batched GPU protocol | Monte Carlo noise | eight common heldout batches and paired MCSE | reviewed baseline |
| fixed affine base | each cell's admitted HMC geometry | local geometry insufficient for global posterior | heldout objective and downstream R4 | target-specific warm start |
| one final seed | phase budget | seed sensitivity remains unknown | explicit nonclaim; R4 can reject transport | bounded campaign choice |

## Required Checks And Artifacts

1. Verify comparator/geometry and identity hash ledgers before every job.
2. Reconstruct and require the repository-issued typed target identity.
3. Enforce GPU memory growth before logical initialization; variables and
   compiled outputs must remain on GPU; XLA must be enabled.
4. Use the repository trainer only. Active training has no NumPy, callback,
   scalar row mapping, Python sample loop, or Python training-step loop.
5. Freeze every completed screen transport so heldout evaluation tests the
   serialized route, then record frozen/trainable parity.
6. Select only after all four rows exist and verify. Report no statistically
   supported recipe ranking unless paired uncertainty supports one.
7. Run one fresh selected 5,000-step training and hash all artifacts.
8. Write per-cell training result and refresh R4 NeuTra HMC subplan.

## Repair, Handoff, And Stops

- Local serialization/reporting failures may be repaired in a fresh attempt
  without changing target, recipes, criteria, or total cell arm budget.
- A nonfinite/status-invalid recipe is `RECIPE_REJECTED`; execute the remaining
  frozen recipes.
- Zero surviving recipes yields `RECIPE_REJECTED` for the only available family
  and a cell-local blocked state, not scientific rejection of NeuTra in general.
- A successful final artifact moves the cell to `TRAINING_ADMITTED` only. It
  must still pass fresh transported-kernel tuning, modern diagnostics, health,
  and simultaneous agreement with its own plain-HMC comparator in R4.
- Stop program-wide only for shared trainer contamination, unavailable trusted
  GPU, corrupted common evidence, or exhausted program budget.

## Skeptical Pre-Execution Audit

Decision: `PASS`.

The protocol does not inherit one LGSSM recipe as a fact, does not use loss as
downstream proof, audits both capacity and optimizer scale, uses common
heldout batches and paired uncertainty, starts each target from its own
admitted geometry, and preserves fresh final training and R4 as independent
gates. The enhanced family remains unavailable rather than silently omitted or
declared failed.
