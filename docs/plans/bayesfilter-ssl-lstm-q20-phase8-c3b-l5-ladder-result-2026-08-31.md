# Phase 8 C3B L5 temperature-ladder result

Date: 2026-08-31  
Status: `PASS_C3B_L5_OVERLAP_WITH_PAIRED_DIVERSITY_SIGNAL_NO_PROMOTION`

Subplan:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-subplan-2026-08-31.md`

## Result

The repaired C3B attempt completed all eight declared rows (two calibration
architectures, two lineage arms, and two fresh roots) on one GPU with the
strict q=20 backend. The run status is
`PASS_PHASE8_C3B_L5_LADDER`; wall time was 3,080.149 seconds. Every row passed
finite target/status checks, immutable checkpoint replay, proper-bridge
overlap evaluation, learned-map reliability, memory growth, and the 4-GiB row
cap. The largest TensorFlow allocator peak was 2,150,787,840 bytes.

The first launch is preserved separately as a harness failure: all rows
stopped at the beta-0.25/component-1 continuation with an indexing error. The
repair used the component index within the immediately preceding beta slice,
which is the lineage controller's contract. No result from that attempt was
used.

Primary manifest:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/attempt-02/run_manifest.json`

## Evidence

| Architecture | Arm | Root | Adjacent acceptance (four pairs) | Mean distance | Covariance distance | Sign distance |
|---|---|---:|---|---:|---:|---:|
| compact-high | pure | 0 | 0.4865, 0.4742, 0.4996, 0.5202 | 0.6492 | 5.7253 | 0.0939 |
| compact-high | pure | 1 | 0.5627, 0.3856, 0.4130, 0.5966 | 0.5196 | 5.5813 | 0.0276 |
| compact-high | branching | 0 | 0.4865, 0.4783, 0.5018, 0.5176 | 0.6861 | 5.1322 | 0.0994 |
| compact-high | branching | 1 | 0.5627, 0.3876, 0.4065, 0.6094 | 0.5789 | 7.8758 | 0.0276 |
| compact-low | pure | 0 | 0.5157, 0.5229, 0.5171, 0.5829 | 0.6508 | 6.3416 | 0.0442 |
| compact-low | pure | 1 | 0.4841, 0.5196, 0.5046, 0.4690 | 0.7813 | 8.2324 | 0.0884 |
| compact-low | branching | 0 | 0.5157, 0.5222, 0.5165, 0.5884 | 0.6559 | 6.3752 | 0.0497 |
| compact-low | branching | 1 | 0.4841, 0.5316, 0.5045, 0.4682 | 0.8347 | 8.0459 | 0.0994 |

Across all recorded acceptance values, the L5 minimum was 0.3856 and the
median was 0.5101. The corresponding C3A L3 values were 0.2350 and 0.3482.
This is a descriptive overlap increase under a changed ladder, not a mixing
estimate: the banks are finite map evaluations, not Markov chains.

The arm-neutral seed construction made the C3B comparison paired. The
branch-minus-pure mean-distance changes were `+0.0369`, `+0.0594`, `+0.0051`,
and `+0.0534` for compact-high/root-0, compact-high/root-1,
compact-low/root-0, and compact-low/root-1. Covariance-distance changes were
`-0.5931`, `+2.2945`, `+0.0336`, and `-0.1865`; sign-distance changes were
small (`0`, or at most `+0.0111`). Thus the mean-distance direction is
consistent in this small diagnostic, while covariance and sign summaries are
not.

## Decision

| Decision | Primary criterion | Hard-veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close C3B hard screen | 8/8 rows finite, replayable, reliable, and under cap | Pass | Two roots and short training | Preserve the L5 receipt and continue to an audited C4 joint-feasibility pilot | No posterior or HMC result |
| Finer L5 overlap | Compare adjacent proper-bridge diagnostics with C3A | Descriptively favorable | Banks are not chains; no uncertainty interval for the contrast | Use L5 only as a calibration hypothesis | No improved mixing claim |
| Positive branching | Directional mean-distance contrast across paired roots | Not a promotion gate; covariance/sign are mixed | Four paired rows and finite-bank variability | Keep pure and branching as comparators; do not set a default | No mode-discovery claim |
| Whitening | Pullback score residuals | Closed; C2 residuals remain large | Capacity/objective adequacy is unresolved | Keep whitening and HMC gates closed | No IID-Gaussian claim |

## Inference status

| Evidence class | Result |
|---|---|
| Hard veto screen | Supported for the bounded C3B implementation and artifacts; no target, map, bridge, checkpoint, route, or allocator veto fired. |
| Statistically supported ranking | None. Two roots, short updates, and finite diagnostic banks do not support a ranking. |
| Descriptive-only differences | L5 acceptance, log-ratio means, chart distances, covariance summaries, and sign occupancy. |
| Default readiness | Not established. No scope-specific tuning, retained chains, ESS/R-hat, downstream agreement, or posterior validation was run. |
| Next evidence needed | Measure the `K=4` joint mixture-RKL arm on the same q=20 GPU/XLA route, with explicit `K^2B` work and memory bounds, before any C5 freeze or Phase 9 work. |

## Provenance repair

The C3B runner imports the standard-library portion of
`run_ssl_lstm_q20_phase8_c3_lineage_overlap_2026_08_30.py` for the audited
training and artifact helpers. The original C3B manifest hashes the route
modules and its own runner but omits that imported helper. The helper was
unchanged before the C3B launch and its current SHA-256 is
`e083f8fbc82e1c309e4de79c92012febce074af4b3970789b7f2859a945e4b35`.
A supplemental receipt is required before treating the metadata as complete;
it does not rerun or modify the numerical result.

## Post-run red-team

The strongest alternative explanation for the larger L5 acceptance is simply
smaller beta increments, not better chart quality or replica mixing. The
consistent mean-distance contrast could also be a finite-bank or short-training
effect; the covariance and sign results already show that the contrast is not
uniform. Coordinate-2 sign is a declared diagnostic label, not a proof of a
posterior basin. The result would be overturned as an implementation receipt
by any replay/hash mismatch, nonfinite bridge value, failed learned-map
reliability check, forbidden graph route, or allocator violation; none was
observed. The weakest evidence is the arm contrast itself.

## Nonclaims

This result does not establish IID Gaussian whitening, mode discovery,
exhaustive coverage, posterior regional mass, Markov-chain convergence,
replica travel, HMC readiness, architecture or branching superiority,
high-dimensional scaling, or statistical superiority.
