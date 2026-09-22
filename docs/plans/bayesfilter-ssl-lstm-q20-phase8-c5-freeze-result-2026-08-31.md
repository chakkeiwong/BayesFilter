# Phase 8 C5 calibration freeze result

Date: 2026-08-31  
Status: `PASS_PHASE8_C5_FREEZE_NO_HMC_OR_POSTERIOR_PROMOTION`

Subplan:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-subplan-2026-08-31.md`

## Result

The metadata-only evaluator completed in `0.016365061048418283` seconds.  It
validated the current q=20 target signature, strict square-root backend, C2,
C3A, C3B, C4A, and C4B statuses, and the source hashes of every input before
writing the freeze receipt.  No TensorFlow process, GPU allocation, target
draw, or reserved Phase 9 stream was used.

Freeze manifest:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c5-freeze/attempt-02/freeze_manifest.json`

The initial `attempt-01` receipt was preserved as a pre-closeout provenance
diagnostic. After the subplan status was finalized, the unchanged evaluator
was rerun in fresh `attempt-02`; the terminal receipt hashes the finalized
subplan and has a successful hash round-trip.

The manifest hash round-trips successfully.  Its selected K=2 protocol is:

| Field | Frozen value |
|---|---|
| Representative | `phase8-k2-compact-high-l3-pure` |
| Component count | `K=2` |
| Architecture | `(16,16)`, `tanh`, two stages, learning rate `1e-3` |
| Bridge ladder | `L3=(0,.5,1)` |
| Lineage policy | Pure continuation; no positive-temperature restart |
| Chart selection | Fixed state-independent uniform `gamma=(.5,.5)` |
| Confirmation status | Frozen for Phase 9 tuning only; fresh scope/tuning required |

Both compact C2 learning-rate rows passed the paired within-row nomination
screen on both roots.  Their mean paired changes were `-25.71300309035614`
(compact-high) and `-10.61535308445626` (compact-low).  The first is selected
only by the already declared operational nomination rule after compact-family
parsimony; this is not a scientific ranking or a claim that the learning rate
is better.

The K=4 joint arm is recorded as:

`NOT_RETAINED_FOR_PHASE9`

C4A and C4B establish a valid implementation and resource envelope, but C4B's
independent and joint endpoint banks were unpaired and its row-level objective
contrasts had opposite signs.  Retaining the quadratic arm would therefore
add cost without evidence that answers the Phase 9 question.  The K=4 result
remains preserved as calibration evidence and can be revisited only under a
new reviewed plan.

## Decision

| Decision | Primary criterion | Hard-veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Freeze K=2 protocol | Compact rows valid on both roots; predeclared operational tie-break applied | Pass | Short calibration and large pullback residuals | Write a separate Phase 9 tuning/validation subplan | No posterior or HMC claim |
| Retain K=4 joint arm | At most one optional K=4 representative | Not retained; evidence insufficient for benefit and cost is quadratic | Unpaired endpoint banks and short pilots | Archive C4A/C4B; do not spend confirmation budget on K=4 | No claim that joint training is mathematically invalid |
| Ladder/branch policy | C3B did not establish a statistically supported L5/branching advantage | Pass for protocol freeze | Finite map banks are not chains | Use L3 pure continuation as the parsimonious confirmation hypothesis | No mode-discovery claim |
| Whitening gate | Held-out pullback score residuals must support later scientific use | Closed | Residuals remain large in C2/C4B | Preserve veto; require Phase 9 pullback diagnostics | No IID-Gaussian claim |

## Inference status

| Evidence class | Result |
|---|---|
| Hard veto screen | Pass for metadata/provenance consistency; all prerequisite manifests and statuses are valid. |
| Statistically supported ranking | None. The compact-high choice is an operational tie-break, not a statistical or scientific ranking. |
| Descriptive-only differences | C2 paired nomination means, C3 overlap/diversity summaries, and C4 objective/timing diagnostics. |
| Default readiness | Not ready. The frozen representative is only a Phase 9 tuning hypothesis. |
| Next evidence needed | Fresh scope-specific tuning, at least four sequential chains, retained warmup/sampling diagnostics, ESS/MCSE/R-hat, declared-region travel, and baseline comparison under an audited Phase 9 plan. |

## Between-phase repair and red-team

The only repair was the metadata-only provenance rerun described above; the
selection rule and inputs were unchanged. The strongest alternative explanation
for the compact-high selection is ordinary finite-root optimization variability; the evaluator
therefore preserves the selection basis and its non-ranking label.  The main
remaining implementation debt is bounded TensorFlow retracing observed in the
C4B runner.  Before any production/default claim, a Phase 9 runner must keep
compiled functions reusable and record compile versus steady-state cost.

The freeze would be invalidated by a changed prerequisite hash or status,
target/backend mismatch, accidental use of a Phase 9 stream, or a caller-
stamped gamma/architecture identity.  None occurred.  A later failure of the
frozen candidate would reject that candidate or trigger its declared repair;
it would not retroactively invalidate the bridge or reverse-KL mathematics.

This result does not establish whitening, IID Gaussianity, exhaustive mode
discovery, posterior correctness, convergence, HMC readiness, statistical
superiority, architecture superiority, production readiness, or
high-dimensional scaling.
