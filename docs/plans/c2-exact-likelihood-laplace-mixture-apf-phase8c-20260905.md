# C2 Phase 8C fresh paired Laplace/APF diagnostic

Date: 2026-09-05  
Parent plan: `c2-exact-likelihood-laplace-mixture-apf-phase8-20260904.md`  
Status: `READY_AFTER_PHASE8C_DIAGNOSTIC_REPAIR`  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Question

Does the calibration-selected exact-likelihood K=1 Laplace proposal retain its
finite-program validity and its descriptive ESS advantage on a fresh C2
observation path, rather than only on the already-seen mechanism-regression
path?

## Fixed target and candidate

The target is unchanged:

\[
\gamma_t(j,x)=\bar w_{t-1,j}f_t(x\mid x_{t-1,j})g_t(y_t\mid x).
\]

Every branch uses the exact C2 numerator and the complete conditional proposal
denominator. The candidate uses the Phase8B calibration result
`quarter_long`, K=1, exact Gaussian transition conditional, analytical C2
state score/curvature, and a frozen finite-program score. No ridge, clipping,
fallback, adaptive-total derivative, or small-observation rule is allowed.

The fresh comparator set is:

| Family | Role |
| --- | --- |
| exact-likelihood Laplace K=1 | candidate |
| transformed UKF K=1 | direct predecessor |
| bootstrap conditional | minimal exact-transition heuristic |
| transformed Student `nu=8` | robust data-guided heuristic |
| stationary independence | broad-support heuristic |

The old Phase7 Gaussian-hint snapshot is deliberately excluded because it was
fit on the already-seen observation path. A fresh Gaussian-hint construction
requires its own definition and is outside this run.

## Fresh data and randomness

`docs/benchmarks/generate_c2_fresh_fixture_phase8c_20260905.py` generates a
versioned fixture with TensorFlow stateless draws from the fixed C2 model at
`theta=(0.6,log(0.4))`. The state and observation seeds are declared on the
command line and the resulting JSON and serialized observation tensor are
hashed in the run manifest. The observation path is shared by all families and
branches. Proposal-specific compiler APIs still use distinct stateless key
offsets, so common-random-number pairing of particle paths is not claimed.

## Evidence contract

| Item | Declaration |
| --- | --- |
| Scientific question | fresh-data validity and mechanism robustness of exact-likelihood local geometry |
| Exact comparator | shared frozen APF evaluator with exact `f*g`, realized APF law, and complete `q` |
| Primary validity criterion | every branch finite; exact/APF identities, analytical score finite difference, denominator recomposition, and XLA/non-XLA parity pass |
| Promotion veto | candidate loses to any constructed heuristic at any fresh time, or any declared branch validity fails |
| Continuation veto | target/measure mismatch, missing or corrupted fixture, nonfinite program, missing records, GPU/memory-policy failure, or one-hour cap |
| Explanatory diagnostics | ESS, log-weight spread, time-local residuals, runtime, trace counts, and proposal moments |
| Statistical status | at most three branches; all continuous contrasts remain descriptive without a predeclared uncertainty analysis |
| Nonclaims | no posterior correctness, unbiased likelihood, general-model success, adaptive-total-gradient correctness, statistical superiority, or default readiness |
| Artifact | fresh `phase8c-paired-attemptNN/` containing fixture, plan snapshot, manifest, raw branch records, tensor sidecars, result, and close note |

## Budget and stop conditions

The Phase8B refresh measured 72.947 seconds for one N=1024 branch per six
families. The fresh run is capped at three N=8192 branches, one GPU-hour total,
and two localized harness repairs. A 55-minute wall-clock stop leaves time for
artifact finalization. A failed candidate comparison is not a continuation
veto; a failed exact identity or exhausted budget is.

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| N=8192 | Phase8C parent plan | memory or runtime pressure | N=256 fresh smoke and allocator record | reviewed campaign setting |
| three branches | parent budget | insufficient uncertainty | per-branch records and descriptive label | diagnostic only |
| `quarter_long` | independent Phase8B calibration | schedule may not transfer | fresh stationarity/ascent traces | frozen hypothesis |
| no stale Gaussian hint | data-partition rule | fewer comparators | explicit family list in manifest | required fairness repair |
| distinct proposal random keys | existing compiler APIs | noisy ESS contrast | branch seeds and non-CRN disclosure | known limitation |

The N=8192 smoke exposed two localized diagnostic-comparability defects before
the repaired campaign: the branch-aware heuristic table was needed to avoid
discarding earlier branches, and the Laplace K=1 adapter used a different
initial/K=1 sampling-key offset from its direct UKF comparator. The table now
retains `branch_index`, and the Laplace K=1 streams use the comparator's
`1001`, `5100+41t`, and `5200+43t` offsets. These repairs preserve the target,
proposal family, schedule, data, and budget; the superseded smoke/full output
is retained and the repaired run uses a new directory.

## Pre-mortem

The run could appear successful because the fresh path is unusually benign or
because ESS is dominated by the shared initial cloud. It could fail because the
new fixture has a serialization or shape error, because N=8192 exhausts GPU
memory, or because the fixed schedule does not transfer. The earliest checks
are fixture-law/finite checks, a small fresh N=256 smoke, per-family validity,
and the complete all-time ESS table. None of these checks alone certifies a
posterior or a statistical ranking.

## Commands

Generate the fixture (CPU reference/data-generation lane):

```bash
CUDA_VISIBLE_DEVICES=-1 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/generate_c2_fresh_fixture_phase8c_20260905.py \
  --output docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8c-fixture-attempt01/fresh_fixture.json \
  --state-seed 20260905 --observation-seed 424242
```

Run a bounded fresh smoke first:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true \
MPLCONFIGDIR=/tmp/mpl-c2-phase8c-smoke-attempt01 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py \
  --fixture-path docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8c-fixture-attempt01/fresh_fixture.json \
  --plan-path docs/plans/c2-exact-likelihood-laplace-mixture-apf-phase8c-20260905.md \
  --phase-id c2_exact_likelihood_laplace_phase8c_fresh_paired_v1 \
  --data-label fresh-phase8c-smoke --exclude-gaussian-hint \
  --particle-count 256 --branch-count 1 \
  --output-root docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8c-smoke-attempt01
```

After the smoke passes, launch the claim-free diagnostic ladder:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true \
MPLCONFIGDIR=/tmp/mpl-c2-phase8c-paired-attempt01 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py \
  --fixture-path docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8c-fixture-attempt01/fresh_fixture.json \
  --plan-path docs/plans/c2-exact-likelihood-laplace-mixture-apf-phase8c-20260905.md \
  --phase-id c2_exact_likelihood_laplace_phase8c_fresh_paired_v1 \
  --data-label fresh-phase8c --exclude-gaussian-hint \
  --particle-count 8192 --branch-count 3 \
  --output-root docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8c-paired-attempt01
```

The runner must refuse an existing output directory and record the exact
command, fixture hash, source hashes, seeds, GPU/memory/XLA settings, and
remaining budget.

## Skeptical audit disposition

`PASS_FOR_PHASE8C_FRESH_DIAGNOSTIC_AFTER_BRANCH_AND_STREAM_REPAIR`.

The target, denominator, candidate schedule, and analytical-score boundary are
unchanged. Fresh data removes the Phase8B observation-path reuse. Excluding
the stale Gaussian hint removes the only known comparator leakage. The branch
aggregation and K=1 stream repairs remove two comparability defects found by
the smoke. The run is still deliberately descriptive because three branches
and proposal-specific later random keys cannot support a statistical ranking.
Any driver or fixture defect is a repair trigger under the unchanged budget; a
mathematical or exact-program mismatch stops the campaign.
