# LG1 preparation result

The public TensorFlow tuning authority was the original execution blocker. It
has been repaired and qualified; it was a reason to withhold retained sampling,
not a reason to stop authorized implementation. Work continued to independent
LG1 references, scoped control calibration, and separate A/B HMC tuning.

References: `artifacts/surrogate-hmc-all-models-20260919-lg1-reference-master-01/P2-reference`.
All three attempt4 partitions passed the independent stationary joint-Gaussian
likelihood, analytical-score finite-difference, prior/chart and four-grid
quadrature-refinement checks. Maximum log-density difference was2.77e-13 and
score finite-difference discrepancy1.55e-9. Claim posterior mean/sd are
.1702506227484/.3727844203836. These are reference calculations, not HMC results.
The quadrature allowance is empirical rather than a proven numerical bound.

Controls: `artifacts/surrogate-hmc-all-models-20260919-lg1-controls-02/P2-controls`.
All nominal and boundary-stress calculations were healthy at all four geometric
protection levels in both separate tuning partitions. Every adjacent pair passed;
the predeclared first-passing rule selected multiplier1. Maximum normalized
nominal changes for the first pair were.0005574501(calibration) and.0002831039
(validation), below.001. Selection permits the imperfect proposal experiment;
it grants no exact-score or posterior authority. This measured curve supplies
the protection rationale previously missing from the warm-start settings.

Controls01 failed on a Path/string mismatch before evaluating an arm. It consumed
3.859GPU seconds and10.545CPU-worker seconds. The repair also made the no-new-
stress-failure guard rowwise and bound each curve to its actual data checksum.
Controls02 completed in113.996GPU seconds and117.894CPU-worker seconds. Original
failure logs/source are preserved. No inference attempt was consumed by either.
The checksum and rowwise guard tests passed; master/control regression tests18passed.

Scoped tuning follows `surrogate-hmc-lg1-inference-2026-09-19.md` at
`artifacts/surrogate-hmc-all-models-20260919-lg1-tune-02`.
Tune01 failed before numerical tuning because the wrapper used positional
arguments at a keyword-only public API. It consumed4.815GPU seconds; the call was
corrected. Tune02 exact-force A passed fresh verification in23.055seconds.
Surrogate-force B also passed fresh verification in1092.511seconds. Both then
completed retained inference; see the [September20 result](surrogate-hmc-lg1-inference-result-2026-09-20.md).
Their retained conclusions are INCONCLUSIVE, with health/mixing screens passing.
This preparation-stage table records why execution proceeded at that boundary.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Proceed to scoped A/B tuning | Reference and control prerequisites pass | No observed reference/control validity veto | Heldout trajectory mixing and MCSE | Fresh fixed-kernel verification, then budgeted retained comparisons | B/R, C/D, D/R, C/R |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Preparation and scoped A/B verification passed; subsequent retained checks are in the linked result |
| Statistically supported ranking | None |
| Descriptive-only differences | Protection sensitivity and runtime |
| Default readiness | Not established |
| Next evidence needed | Retained posterior equivalence, cheap adversaries, finite-target comparisons and replication |

Red-team note: healthy fixed probes do not establish safety at every HMC state.
The runtime preserves original status and numerical vetoes. A stalled or biased
chain would overturn an inference claim. The later posterior comparisons are
reported separately from these preparation checks.
The broad all-model result remains incomplete. Do not confuse the master's exit2
for incomplete coverage with a failed numerical stage; consult its per-job status.
