# NeuTra Control Repair Plan (2026-08-15)

## Research Intent Ledger

| Field | Predeclared statement |
|---|---|
| Main question | Can an independent exact-law nomination partition repair Gaussian learning-rate selection, and which target-specific banana transport factor is most associated with the observed variance-basin failure? |
| Gaussian mechanism | Cold joint reverse-KL training at `2e-4`, `5e-4`, and `1e-3`; nominate from independent exact-law selection screens, then freeze the choice for fresh audit seeds. |
| Banana mechanisms | Matched cold joint training with four architecture/initialization arms: baseline `(32,32)`, full-reverse, init scale `0.02`; identity-biased init scale `0.005`; root-preserving permutation; and width `(64,64)`. |
| Baseline | Existing three-stage `(32,32)` full-reverse transport, 3,000 updates, same batch-native reverse-KL objective and schedule. |
| Primary Gaussian nomination criterion | Both selection seeds pass every exact-law mean, second-moment, and adjacent-cross-moment screen; among multiple passing arms, lowest mean held-out reverse-KL loss. If none pass, minimize mean maximum absolute standardized exact-law discrepancy. |
| Primary banana nomination criterion | For each arm, choose its learning rate using the same two-seed exact-law selection rule; nominate the arm with the lowest mean maximum standardized discrepancy, with all-screen passage taking precedence. |
| Primary fresh criterion | The frozen Gaussian candidate and each nominated banana arm are evaluated on untouched confirmation seeds with 131,072 draws. Viability requires both confirmation seeds to pass every exact-law screen. |
| Hard vetoes | Nonfinite state/output, invalid GPU/XLA/memory-growth provenance, unequal update budget, partition reuse, missing arm, or any fresh exact-law screen failure. |
| Explanatory diagnostics | Reverse-KL loss, maximum standardized moment discrepancy, ESS fraction, ratio SD, terminal gradients, clipping, and runtime. These do not promote a candidate. |
| Nonclaims | No HMC, SSL-LSTM transfer, multimodal coverage, universal architecture, statistical superiority, or default-readiness claim. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| 3,000 updates | Existing equal-work tournament | Keeps repair comparable to prior campaign | Banana may remain undertrained | Terminal loss and cap status | Reviewed baseline |
| LR grid `2e-4,5e-4,1e-3` | Existing target campaign | Covers prior selected and passing Gaussian cold rates | Grid may miss a viable rate | Record all three arms | Reviewed baseline |
| Exact-law selection count 65,536 | Existing selection partition | Gives stable diagnostic intervals without using audit draws | Selection can still overfit two seeds | Fresh seeds 4/5 | Reviewed nomination rule |
| Banana init scales `0.02,0.005` | Current scale and identity-biased hypothesis | Tests whether early basin movement causes variance distortion | Lower scale may slow learning | First 100/500-update loss and final discrepancy | Hypothesis |
| Root-preserving permutation | Existing supported transport option | Tests whether preserving first banana coordinate avoids harmful mixing | May reduce expressivity | Compare with full-reverse baseline | Hypothesis |
| Width 64 | Capacity hypothesis | Tests whether width 32 limits nonlinear shear representation | More capacity may destabilize reverse-KL | Finite/clipping checks and exact-law screens | Hypothesis |
| Two selection and two audit seeds | Existing replicated final policy | Separates nomination from confirmation | Few seeds limit ranking strength | Report all per-seed results | Minimum evidence, not superiority proof |

## Execution Design

### Gaussian LR repair

Run each LR for selection seeds `2,3`, using fresh initialization/training
partitions and a 65,536-draw exact-law selection partition. Select the LR using
the predeclared criterion above. Retrain only the selected LR on fresh audit
seeds `4,5` with 131,072 exact-law draws.

### Banana factor study

For each of the four arms, run all three LR candidates on selection seeds `0,1`
with independent selection partitions. Select one LR per arm, then run each
arm/LR pair on screening seeds `2,3`; these screen all arms and nominate one
arm but are not a final viability claim. Run the nominated arm/LR pair again on
untouched confirmation seeds `4,5`. All arms receive exactly 3,000 updates.
The baseline arm is retained even if another arm has a lower descriptive loss.

No candidate receives extra phase-local tuning. All runs use TensorFlow/TFP,
float64, GPU 0, XLA JIT, TF32 disabled, batch size 4,096, and verified memory
growth. The campaign wall cap is 3,600 seconds and the output root is
`docs/plans/artifacts/neutra-control-repair-2026-08-15/`.

## Skeptical Plan Audit

| Risk | Disposition |
|---|---|
| Selection uses the final audit | Vetoed: selection seeds/partitions and audit seeds/partitions are disjoint. |
| Exact-law selection overstates evidence | Retained: it nominates only; fresh two-seed audit is the promotion/viability gate. |
| Banana arms get unequal work | Vetoed: every LR/arm consumes exactly 3,000 updates. |
| Architecture and LR effects are confounded | Partially controlled: LR is selected within each arm; all arms share the same LR grid and budget. Remaining interaction is reported, not ranked as causal superiority. |
| A passing Gaussian LR proves a default | Vetoed: result is target-specific viability evidence only. |
| Width 64 passes because of more compute | Vetoed: update count and batch are fixed; runtime is explanatory only. |
| Two seeds support a ranking | Vetoed: continuous differences remain descriptive; only hard screen passage is used for viability. |
| Banana failure is declared architecture impossibility | Vetoed: a failed arm triggers another target-specific repair, not rejection of NeuTra. |
| GPU run is launch-invalid | Vetoed: memory growth is configured before logical-device initialization and recorded in every manifest. |

Audit verdict: the plan answers the two repair questions with disjoint
nomination/audit partitions, equal optimizer budgets, explicit target-specific
hypotheses, and no promotion based on proxy loss alone.

## Artifact And Stop Conditions

Record campaign and per-cell manifests, commands, git commit, environment,
device, memory policy, seeds, wall time, result JSON, and SHA-256 hashes.
Stop a cell on nonfinite state, invalid provenance, or budget violation; stop
the campaign only for an infrastructure failure or the wall cap. A fresh
exact-law failure rejects that candidate but does not invalidate the harness
or the remaining arms.
