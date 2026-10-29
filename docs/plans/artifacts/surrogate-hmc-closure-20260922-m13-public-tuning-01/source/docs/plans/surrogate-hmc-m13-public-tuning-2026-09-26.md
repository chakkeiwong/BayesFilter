# M13 public tuning and fresh verification — 2026-09-26

This is the next stage of the master program for testing surrogate HMC. It uses
the repository-issued M13 control selection, the public `tune_hmc_kernel`
artifact authority, and `VerifiedTensorFlowHMCConfig`. The canonical analytical
DCTR field remains a deterministic position-only proposal force; the independent
FP64 full-support trajectory-importance endpoint remains the MH potential.
The consumer never claims that the force is the endpoint gradient.

The selected control artifact must come from the bounded fresh-partition repair
or the original control pass after a common materiality pair has passed. Its
scope, source hashes and evidence are rechecked before tuning. The pilot data
are M13 data1704/T20 (d=4, y=2, two parameters, N=4096, chunk=2048), FP32/TF32
GPU/XLA force, fixed Contract-E reset and full R2 prior support. Confirmation
stream3701 remains untouched.

Nomination uses mass windows128/256,256 adaptation transitions,128 screening
transitions, L in{1,2,4} through `max_leapfrog_steps=4`, target acceptance .7,
and prior-scale geometry only as a warm start. The initial normalized step .2,
10% prior shrinkage and zero additive jitter are scoped hypotheses. The positive
prior term gives C=.9E+.1P0 >= .1P0 for PSD E; the actual regularizer is tested
on healthy and rank-deficient metrics. Four dispersed prior-scale starts are
used. Warmup500 is archived and excluded; fresh verification has1000 draws per
chain and must pass finite-state/endpoint/energy/movement checks and R-hat <=1.01.
A failed nomination or verification is a tuning repair trigger, not a posterior
or research-direction rejection.

The stage has a72,000 CPU-worker/62,000 GPU-second reservation from the approved
continuation. It writes a fresh versioned output and re-loads the verified
artifact before declaring `M13_PUBLIC_TUNING_VERIFIED`. That status qualifies
only the frozen tuning handoff. It does not establish posterior correctness,
ESS beyond the verification checks, speed, superiority, default readiness, or
all-model validity. Retained chunks, MCSE-calibrated posterior comparison, and
constructed cheap baselines remain required.

| Decision | Criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Freeze tuning | Public artifact reload passes all checks | Nonfinite state/endpoint/energy, no movement, R-hat, fallback, provenance | Short fresh verification | Retained posterior and baselines | Posterior correctness |

Exact command is issued by the master after control selection:

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-public-tuning --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-public-tuning-01
```
