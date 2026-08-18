# NeuTra Banana Predictive-Equivalence Follow-up Plan (2026-08-16)

## Research intent ledger

| Field | Predeclared statement |
|---|---|
| Main question | Does the predictive-law MMD excess remain when the same frozen candidate is assessed with a substantially larger retained window and a larger exact-vs-exact calibration bank? |
| Candidate | The unchanged seed-15, 6,000-update root-preserving `(32,32)` dense-IAF transport with identity z mass, `L=10`, and step size `0.7709722545680272`. |
| Comparator | Independent stateless exact analytic banana draws with the same four-chain layout and window lengths. |
| Sample unit | One raw 16-dimensional model-coordinate draw; no temporal horizon is introduced. |
| Primary diagnostic | The same biased multi-bandwidth RBF MMD and stratified moving-block bootstrap used in the prior diagnostic, with 4,096 draws per chain. |
| Window sensitivity | Offsets `0` and `904`, the two maximal 4,096-draw windows in the fixed 5,000-draw archive. They overlap and are not independent replications. |
| Calibration | 128 exact-vs-exact banks per offset, 512 block-bootstrap replicates per bank and candidate, block lengths `32,64,128`, and fixed bandwidths `(2,4,8)`. |
| Promotion criterion | None. This campaign can classify the earlier screen as persistent or plausibly finite-sample/noisy; it cannot promote the HMC candidate. |
| Hard vetoes | Missing/stale candidate archive, wrong frozen kernel/state, nonfinite values, invalid GPU/XLA/memory provenance, malformed windows, or failed exact-vs-exact calibration integrity. |
| Explanatory diagnostics | Candidate/control upper-interval quantiles, per-bandwidth MMD, coordinate and latent moments, block sensitivity, and overlap sensitivity. |
| Nonclaims | No formal equality p-value, no independent-replication claim for the two archive windows, no posterior correctness proof, no training conclusion, no default/production readiness, and no SSL-LSTM transfer. |

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| 4,096 draws per chain | Maximum window supported by the existing 5,000-draw confirmation archive | Overlap reduces effective window replication | Explicitly report overlap and use the second window only as sensitivity | Reviewed diagnostic setting |
| Offsets `0` and `904` | Derived as the two maximal windows that fit the archive; fixed before execution | Window dependence can make results look replicated | Report both windows and do not pool them as independent | Reviewed sensitivity setting |
| 128 exact-vs-exact banks | Four times the prior 32-bank calibration | The 99% empirical envelope is still a high order statistic | Report q95 and q99, all control values, and no formal p-value | Reviewed bounded calibration |
| 512 bootstrap replicates | Two times the prior 256 replicates | Bootstrap quantile Monte Carlo error remains | Record count and treat intervals as descriptive | Reviewed bounded calibration |
| Block lengths `32,64,128` | Retained from the prior diagnostic to isolate sample-size/calibration changes | None of these may represent long memory | Report all three; persistent agreement is stronger descriptive evidence | Reviewed sensitivity grid |
| Bandwidths `(2,4,8)` | Dimension-scaled grid repaired by the pre-run shift diagnostic | A finite grid can miss another discrepancy | Report per-bandwidth values and retain the fixed grid | Reviewed diagnostic hypothesis |

## Evidence contract

| Item | Predeclared value |
|---|---|
| Scientific question | Whether the prior MMD excess survives a larger finite-sample diagnostic under the same frozen HMC candidate. |
| Baseline/comparator | Exact-vs-exact banks using independent stateless seeds and identical chain/window/block/bootstrap geometry. |
| Primary result | Candidate upper-99% MMD versus the exact-vs-exact q99 envelope at each offset and block length; q95 is reported as a secondary calibration description. |
| Veto diagnostics | Any invalid artifact, nonfinite sample, wrong state hash/kernel, malformed window, failed exact-control finite check, or missing memory/XLA provenance invalidates the campaign. |
| Explanatory-only evidence | Point MMD, per-bandwidth values, coordinate/latent moments, q95 versus q99, block sensitivity, and runtime. |
| What will not be concluded | The result is not a formal equality test, not independent replication, and not evidence for or against retraining by itself. |
| Artifact | `docs/plans/artifacts/neutra-banana-predictive-equivalence-followup-2026-08-16-r1/` with copied plan, exact command/source hashes, per-offset JSON, result note, reset memo, and artifact hashes. |

## Skeptical plan audit

| Risk | Disposition |
|---|---|
| Larger window is mistaken for independent replication | Vetoed: the manifest and result explicitly mark offsets as overlapping sensitivity windows. |
| More calibration banks are treated as formal p-value calibration | Vetoed: q95/q99 are descriptive empirical envelopes; no p-value is emitted. |
| Candidate is silently regenerated or retuned | Vetoed: the runner hash-binds the existing confirmation archive and checks seed, update count, state hash, and `L=10`; no training code is called. |
| Memory scales beyond GPU capacity | Controlled: at 4,096 draws the largest sample pairwise matrix is 16,384 by 16,384 float64; the campaign is run on one 30 GB GPU with a bounded time cap and a prior small smoke. |
| Window result answers the wrong question | Vetoed: this is explicitly a persistence/sampling diagnostic for the prior screen, not a new equivalence criterion. |
| Exact controls fail because of the harness | Controlled: every offset has 128 exact-vs-exact banks with the same block/bootstrap construction, and all values are checked finite. |

Audit verdict: the plan is fit for execution as a bounded diagnostic. Its main limitation is unavoidable overlap in the fixed 5,000-draw candidate archive; that limitation is recorded as a nonclaim rather than hidden.

## Execution

1. Update the runner only to expose q95/q99 calibration summaries and the follow-up plan path; preserve the scientific statistic and candidate binding.
2. Run focused tests and a CPU/XLA smoke at 512 draws.
3. Execute the GPU/XLA campaign with 4,096 draws, 128 calibration banks, and 512 bootstrap replicates at offsets 0 and 904.
4. Verify source/artifact hashes and compare numerical rows with the prior result.
5. Write the result and reset memo, then decide whether the screen is persistent or plausibly underpowered/noisy. Do not retrain from this diagnostic alone.
