# C2 Phase 8D frozen-control replication plan

Date: 2026-09-05  
Parent: `c2-exact-likelihood-laplace-mixture-apf-phase8c-20260905.md`  
Status: `READY_AFTER_SKEPTICAL_AUDIT`  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Research question

Does the exact-likelihood K=1 Laplace proposal retain its validity and
descriptive ESS advantage over the transformed UKF, bootstrap, transformed
Student, and stationary heuristics across several *new* C2 observation paths
when all proposal controls are frozen from Phase8B calibration?

This phase tests robustness of the mechanism, not a general-model theorem or a
production default.

## Frozen method and data protocol

- Candidate schedule: `quarter_long`, temperatures
  `(0.25,0.5,0.75,1,1,1,1,1)`, unit step fractions, no ridge or clipping.
- Particle count: `N=8192`; one proposal-randomization branch per fixture.
- Fresh observation seeds: `(424243,424244,424245)` with state seeds
  `(20260906,20260907,20260908)`.
- Five-family comparator set; the stale Phase7 Gaussian-hint snapshot remains
  excluded.
- The exact C2 target, complete conditional denominator, and frozen analytical
  score are unchanged.
- The K=1 initial and direct-UKF random streams are aligned. Later proposal
  streams remain family-specific and are not claimed to be common-random-number
  coupled.

The schedule is checked against independent calibration fixtures by the driver
but is selected with `--fixed-schedule-config quarter_long`; fresh ESS cannot
retune it. Each fixture and run has a unique output directory.

## Evidence contract

| Item | Declaration |
| --- | --- |
| Primary question | frozen-control robustness across fresh observation paths |
| Exact baseline | shared frozen APF evaluator and complete `q` denominator |
| Validity pass | every branch finite; exact/APF identities, score finite differences, density recomposition, and XLA parity pass |
| Promotion veto | any candidate loss to a constructed heuristic on any fixture/time, or any validity failure |
| Continuation veto | target/measure mismatch, corrupted fixture, missing branch record, GPU/memory failure, or budget exhaustion |
| Statistical interpretation | three paths and one branch each provide descriptive replication only; no ranking claim |
| Explanatory diagnostics | ESS by time, weight spreads, score error, stationarity/ascent, runtime, and trace counts |
| Nonclaims | no posterior correctness, unbiased likelihood, generality, adaptive-total gradient, HMC, production, or default readiness |
| Artifact | three `phase8d-seed*-attempt01/` directories plus a machine-readable aggregate and close note |

## Budget and stop conditions

The one-fixture Phase8C N=8192 run took 145.694 s including calibration. Three
fresh fixtures are budgeted at 20 minutes wall time and one GPU-hour, with at
most two localized harness repairs. A failed candidate comparison is evidence
against promotion but does not stop the replication; a failed exact identity or
missing artifact does.

## Default/assumption audit and pre-mortem

The seeds are convenience hypotheses, not representative-data facts. Their
only required property is disjointness from Phase8B/8C. Three paths may be too
few for uncertainty, so all aggregate contrasts will be labelled descriptive.
The fixed schedule may fail outside the calibration distribution; independent
stationarity/ascent diagnostics expose that. A successful ESS table could be
driven by unusually mild observations, while a failure could be caused by GPU
memory or fixture serialization; fixture hashes, validity gates, and a
per-path decision table distinguish those cases.

## Commands

Generate each fixture using the Phase8C TensorFlow generator, then run the
driver with the fixed schedule. For seed `S` and output suffix `K`:

```bash
CUDA_VISIBLE_DEVICES=-1 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/generate_c2_fresh_fixture_phase8c_20260905.py \
  --output docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8d-seedS-attempt01/fresh_fixture.json \
  --state-seed STATE_S --observation-seed S
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true \
MPLCONFIGDIR=/tmp/mpl-c2-phase8d-seedS-attempt01 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py \
  --fixture-path docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8d-seedS-attempt01/fresh_fixture.json \
  --plan-path docs/plans/c2-exact-likelihood-laplace-mixture-apf-phase8d-20260905.md \
  --phase-id c2_exact_likelihood_laplace_phase8d_frozen_replication_v1 \
  --data-label fresh-phase8d-seedS --exclude-gaussian-hint \
  --fixed-schedule-config quarter_long --particle-count 8192 --branch-count 1 \
  --output-root docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8d-seedS-attempt01/run
```

The aggregate reader must report each path separately and then give only
descriptive medians, ranges, sign counts, and validity counts. It must not
convert three paths into a superiority or default decision.

## Skeptical audit

`PASS_FOR_PHASE8D_FROZEN_REPLICATION`.

The plan freezes the only candidate control that Phase8B calibrated, uses
disjoint fresh data, preserves the exact target and denominator, and makes the
small replication count and non-CRN limitation explicit. The five-family
heuristic set is constructed from the filtering problem. The output contract
has a validity veto, a promotion veto, and a separate continuation rule, so a
candidate failure cannot be mistaken for a harness failure.
