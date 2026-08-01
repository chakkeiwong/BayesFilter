# LGSSM NeuTra Sequential HMC Repair Phase R1 Subplan

Date: 2026-07-15  
Status: `READY_TO_EXECUTE`

## Objective And Entry Conditions

Run corrected fresh sequential warm-up and tuning admission for both frozen
NeuTra candidates. Phase R0 passed compile, focused tests, import closure, diff
check, and bounded material plan review. Phase 1-3 identities remain valid; the
historical Phase 4 artifacts remain immutable and incomplete for admission.

## Frozen Execution

Environment before TensorFlow import:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=1 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Commands, in order:

```bash
python docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py \
  sequential-candidate --job-id dense_seed1201
python docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py \
  sequential-candidate --job-id dense_seed1202
python docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py \
  sequential-finalize
```

Candidate seeds are frozen in the repair plan: seed1201 warm-up/retained roots
`(20260715,4101)`/`(20260715,4201)`; seed1202 roots
`(20260715,4301)`/`(20260715,4401)`.

## Evidence And Artifacts

Each candidate must emit a versioned result and separate per-chunk/cumulative
warm-up and retained TensorFlow archives under
`docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/sequential-repair-attempt-01/`.
The result must bind source, target, transport, adapter, kernel, seeds,
environment, command, XLA, CPU-hidden status, wall time, plan, archive hashes,
all readiness/admission checks, caps, and hard vetoes.

Warm-up admission: at least 2,000 transitions per chain and recent 1,000-draw
raw-coordinate modern R-hat `<=1.05`. Retained admission: at least 1,000
cumulative draws per chain and raw-coordinate modern R-hat `<=1.01`. Both caps
are 10,000 per chain. Health/status/finite/movement/energy-error screens apply
to every chunk.

## Forbidden Claims And Actions

Do not reuse old retained draws, include warm-up in posterior summaries,
retune the step size or leapfrog count, retrain, weaken thresholds, overwrite
artifacts, rank candidates descriptively, or claim posterior correctness,
recovery, superiority, calibration, robustness, production readiness, or a new
default.

## Handoff And Stop Conditions

Process the second candidate even if the first reaches a genuine cap. If at
least one passes, freeze its kernel and refresh Phase R2/confirmatory sampling
using the still-unused confirmation seed. If neither passes, close these fixed
kernels as candidate failures without rejecting the target or research
direction.

Stop a candidate for a health/identity/artifact veto or a stage cap. Stop the
campaign only for common target/harness invalidity, corrupted evidence,
six-hour aggregate budget exhaustion, or a boundary-changing repair.

Suitability verdict: `PASS`. The subplan is consistent with R0, answers the
research question, preserves artifact separation, and crosses no new human,
runtime, privacy, funding, external, or scientific-direction boundary.
