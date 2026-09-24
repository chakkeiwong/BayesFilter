# C2 Mixture-UKF/APF Phase 4 Backward-Error Repair

Date: 2026-09-03  
Governing plan: `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Parent attempt: `docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-smooth-defensive-n8192-attempt01/`

Status: `READY_AFTER_SKEPTICAL_AUDIT`

## Question and evidence contract

The first Phase 4 GPU attempt completed 132/132 records but reported five
defensive records invalid because the absolute APF rearrangement residual
exceeded `2e-10`. The question is whether those five failures are genuine
finite-program defects or ordinary scale-dependent roundoff.

The exact target, proposal law, observations, particle count, branch seeds,
calibrated controls, GPU class, and promotion rule are frozen to the parent
attempt. Only the validity diagnostic is repaired. The primary validity
criterion is that all 36 defensive replay records pass finite target/score,
complete-density, support, observation-response, score-parity, normalized
APF backward-error, and final-log-weight recurrence-parity checks. The
normalized APF bound is

\[
 |r|/\max(1,|a|+|q|+|\gamma-a-q|+|\gamma|)
 \le 32\epsilon_{64},
\]

with the absolute residual retained as an explanatory field. Replay branch and
program identities plus finite scientific outputs must agree with the parent
records at relative tolerance `1e-12`.

The promotion criterion remains the predeclared paired 95% interval for the
defensive-versus-K=1 log minimum-ESS ratio with at least 10/12 positive
contrasts, together with no loss to a cheap adversary at the declared salient
times. It is evaluated after validity and cannot be tuned during replay.

## Exact command and scope

CPU tests:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mpl-c2-phase4-repair-tests \
pytest -q tests/highdim/test_c2_mixture_ukf_apf_phase4.py \
tests/highdim/test_c2_mixture_ukf_apf_phase4_repair.py
```

GPU replay:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase4-repair \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py \
  --phase phase4-repair \
  --output-root docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-backward-error-repair-attempt01 \
  --rows 8192 --branches 12
```

The replay evaluates exactly 36 defensive records (`K=1,2,4` over twelve
paired seeds). The 96 nondefensive parent records are inherited by hash for
the combined promotion table; the parent directory is never overwritten.

## Budget and stop conditions

This localized replay is capped at one GPU hour and does not refit the TT
proposal. A missing parent, changed fixture/algorithm source, nonfinite replay,
failed recurrence parity, missing record, or GPU/memory-growth provenance
failure is a Phase 4 continuation veto. A valid replay with a negative ESS or
heuristic result is a candidate/promotion veto and must continue to the
predeclared Phase 5 repair. No larger particle count is opened here.

## Skeptical audit and pre-mortem

Audit result: `PASS_FOR_REPAIR_REPLAY`.

The repair does not relax a scientific accuracy criterion: it corrects a
dimensionally inappropriate test of a tautological subtraction and adds an
independent recurrence comparison. It does not use parent claim outcomes for
control selection. The replay command has a fresh output root, fixed seeds,
explicit GPU memory-growth provenance, and a bounded budget.

The run could still mislead if the replay changes the proposal implementation,
if the parent and replay records differ while only aggregate metrics are
reported, or if the normalized bound hides a genuinely wrong term. The branch
and program identity checks, elementwise final-weight parity, complete-density
checks, and preserved absolute residual expose those cases. Passing validity
does not imply low variance, posterior correctness, unbiased likelihoods, or
default readiness.

## Required artifacts and close

The fresh directory must contain `manifest.json`, `calibration.json`,
`branch_results.json`, `replay_parity.json`, `result.json`, `result.md`, and
`command.txt`. The close note must include the parent hashes, repaired bound,
record counts, validity verdict, promotion/heuristic table, budget consumed,
remaining budget, and the refreshed Phase 5 entry command.
